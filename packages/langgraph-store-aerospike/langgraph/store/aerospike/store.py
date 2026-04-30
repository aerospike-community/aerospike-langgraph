import asyncio
from collections.abc import Iterable
from datetime import datetime, timezone
from typing import Any

from aerospike_helpers import expressions as exp
from aerospike_helpers.operations import map_operations, operations

# `Result`, `SearchItem`, and `TTLConfig` are part of the documented public
# surface of `langgraph.store.base` (they appear in `BaseStore` method
# signatures and the public `Op` union), but upstream forgot to list them in
# `langgraph.store.base.__all__`. The `noinspection PyProtectedMember` comment
# keeps PyCharm quiet without pulling in genuinely private helpers
# (`_ensure_ttl`, `_ensure_refresh`, `_validate_namespace`).
# noinspection PyProtectedMember
from langgraph.store.base import (  # noqa: PLC2701
    BaseStore,
    GetOp,
    Item,
    ListNamespacesOp,
    NamespacePath,
    Op,
    PutOp,
    Result,
    SearchItem,
    SearchOp,
    TTLConfig,
)

import aerospike
import aerospike.exception  # noqa: F401  # expose `aerospike.exception` submodule for type checkers

SEP = "|"


def _now_utc() -> datetime:
    return datetime.now(tz=timezone.utc)


class AerospikeStore(BaseStore):
    """Aerospike-backed implementation of LangGraph's ``BaseStore``.

    ``BaseStore`` already provides concrete implementations of every
    high-level convenience method (``put``, ``get``, ``delete``, ``search``,
    ``list_namespaces`` and their ``a*`` async twins). Each of those methods
    validates inputs, resolves TTL/refresh defaults via the store's
    ``ttl_config``, and then funnels the work through ``self.batch(...)`` /
    ``self.abatch(...)``.

    Per the LangGraph integration contract, an adapter only needs to
    implement ``batch`` and ``abatch``. Everything else comes for free, so
    this class deliberately avoids overriding the public surface to:

    * keep the adapter small and focused on the Aerospike-specific bits,
    * inherit any future improvements to validation/TTL handling, and
    * avoid importing private helpers (``_ensure_ttl``, ``_ensure_refresh``,
      ``_validate_namespace``) from ``langgraph.store.base``.
    """

    supports_ttl: bool = True

    def __init__(
        self,
        client: aerospike.Client,
        namespace: str = "langgraph",
        set: str = "store",
        ttl_config: TTLConfig | None = None,
    ) -> None:
        self.client = client
        self.ns = namespace
        self.set = set
        self.ttl_config = ttl_config

    # --------------- Aerospike helper functions ------------------

    def _key(self, namespace: tuple[str, ...], key: str) -> tuple[str, str, str]:
        return (self.ns, self.set, SEP.join([*namespace, key]))

    def _build_read_policy_for_refresh(self, refresh_ttl: bool | None) -> dict[str, Any]:
        policy: dict[str, Any] = {}
        if self.ttl_config is not None and self.ttl_config.get("refresh_on_read"):
            policy["read_touch_ttl_percent"] = 100
        if refresh_ttl:
            policy["read_touch_ttl_percent"] = 100
        return policy

    def _get_type_result(self, value: Any):
        if isinstance(value, bool):
            return exp.ResultType.BOOLEAN
        elif isinstance(value, int):
            return exp.ResultType.INTEGER
        elif isinstance(value, float):
            return exp.ResultType.FLOAT
        elif isinstance(value, str):
            return exp.ResultType.STRING
        elif isinstance(value, bytes):
            return exp.ResultType.BLOB
        elif isinstance(value, (dict, list)):
            return exp.ResultType.MAP if isinstance(value, dict) else exp.ResultType.LIST
        return exp.ResultType.STRING

    def _get_op_expression(self, bin_expr, value_expr, operator: str):
        ops = {
            "$eq": exp.Eq,
            "$ne": exp.NE,
            "$gt": exp.GT,
            "$gte": exp.GE,
            "$lt": exp.LT,
            "$lte": exp.LE,
        }

        if operator not in ops:
            raise ValueError(f"Unsupported operator: {operator}")

        return ops[operator](bin_expr, value_expr)

    def _build_path_filter(
        self, path: NamespacePath, bin_name: str, is_suffix: bool = False
    ) -> list:
        """Build a list of expressions to handle wildcards in a NamespacePath."""
        conditions = []
        path_len = len(path)
        size_check = exp.GE(exp.ListSize(None, exp.ListBin(bin_name)), exp.Val(path_len))
        conditions.append(size_check)
        for i, token in enumerate(path):
            if token == "*":
                continue
            algo_index = i - path_len if is_suffix else i
            result_type = self._get_type_result(token)
            match_condition = exp.Eq(
                exp.ListGetByIndex(
                    None,
                    aerospike.LIST_RETURN_VALUE,
                    result_type,
                    exp.Val(algo_index),
                    exp.ListBin(bin_name),
                ),
                exp.Val(token),
            )
            conditions.append(match_condition)

        return conditions

    def _build_filter_exprs_from_dict(self, filter_dict: dict[str, Any]) -> list:
        filter_exprs = []

        for key, condition in filter_dict.items():
            map_key_expr = exp.Val(key)
            if isinstance(condition, dict) and any(k.startswith("$") for k in condition):
                for op, val in condition.items():
                    result_type = self._get_type_result(val)
                    target_expr = exp.MapGetByKey(
                        None,
                        aerospike.MAP_RETURN_VALUE,
                        result_type,
                        map_key_expr,
                        exp.MapBin("value"),
                    )

                    op_expr = self._get_op_expression(target_expr, exp.Val(val), op)
                    filter_exprs.append(op_expr)

            else:
                result_type = self._get_type_result(condition)
                target_expr = exp.MapGetByKey(
                    None, aerospike.MAP_RETURN_VALUE, result_type, map_key_expr, exp.MapBin("value")
                )
                filter_exprs.append(exp.Eq(target_expr, exp.Val(condition)))

        return filter_exprs

    # --------------- Per-op handlers (called from batch) --------------------
    #
    # Each handler implements one Op variant against Aerospike. They are
    # grouped together so `batch` stays a thin dispatch table.

    def _handle_put(self, op: PutOp) -> None:
        p_key = self._key(op.namespace, op.key)

        if op.value is None:
            try:
                self.client.remove(p_key)
            except aerospike.exception.RecordNotFound:
                return
            except aerospike.exception.AerospikeError as e:
                raise RuntimeError(f"Aerospike remove failed for {op.key}: {e}") from e
            return

        # `op.ttl` has already been resolved by `BaseStore.put` via
        # `_ensure_ttl(ttl_config, ttl)`, so it is either `None` (no TTL
        # configured / caller asked for "no expiration") or a positive float
        # in minutes. We map `None` to Aerospike's "never expire" sentinel
        # (-1) so behavior is deterministic regardless of the namespace's
        # default-ttl.
        if op.ttl is None:
            time_to_live: int = -1
        else:
            time_to_live = -1 if op.ttl < 0 else int(op.ttl * 60)

        # `created_at` / `updated_at` live in a `meta` Map bin so we can
        # use `MAP_WRITE_FLAGS_CREATE_ONLY | MAP_WRITE_FLAGS_NO_FAIL`
        # for `created_at`: the value is set on first write and silently
        # left alone on every subsequent upsert.
        now = _now_utc().isoformat()
        ops = [
            operations.write("namespace", list(op.namespace)),
            operations.write("key", op.key),
            operations.write("value", op.value),
            map_operations.map_put(
                "meta",
                "created_at",
                now,
                map_policy={
                    "map_write_flags": (
                        aerospike.MAP_WRITE_FLAGS_CREATE_ONLY | aerospike.MAP_WRITE_FLAGS_NO_FAIL
                    ),
                },
            ),
            map_operations.map_put("meta", "updated_at", now),
        ]
        try:
            self.client.operate(p_key, ops, policy={"ttl": time_to_live})
        except aerospike.exception.AerospikeError as e:
            raise RuntimeError(f"Aerospike put failed for {op.key}: {e}") from e

    def _handle_get(self, op: GetOp) -> Item | None:
        p_key = self._key(op.namespace, op.key)
        read_policy = self._build_read_policy_for_refresh(op.refresh_ttl)
        try:
            if read_policy:
                _, _, bins = self.client.get(p_key, policy=read_policy)
            else:
                _, _, bins = self.client.get(p_key)
        except aerospike.exception.AerospikeError:
            return None

        value = bins.get("value")
        if value is None:
            return None

        ns = tuple(bins.get("namespace", op.namespace))
        k = bins.get("key", op.key)
        # Timestamps live in the `meta` Map bin (see `_handle_put`).
        meta = bins.get("meta") or {}
        now = _now_utc().isoformat()
        created_at = meta.get("created_at", now)
        updated_at = meta.get("updated_at", now)

        return Item(value=value, key=k, namespace=ns, created_at=created_at, updated_at=updated_at)

    def _handle_search(self, op: SearchOp) -> list[SearchItem]:
        if op.query:
            raise NotImplementedError(
                "Aerospike v0.1 does not support semantic/vector search. Use search without query."
            )

        filter_exprs = []
        if op.namespace_prefix:
            filter_exprs.extend(
                self._build_path_filter(op.namespace_prefix, "namespace", is_suffix=False)
            )
        if op.filter:
            filter_exprs.extend(self._build_filter_exprs_from_dict(op.filter))

        policy: dict[str, Any] = {}
        if filter_exprs:
            policy["expressions"] = exp.And(*filter_exprs).compile()

        try:
            scan = self.client.scan(self.ns, self.set)
            records = scan.results(policy=policy)
        except aerospike.exception.AerospikeError as e:
            raise RuntimeError(f"Aerospike search failed: {e}") from e

        # If the caller asked us to refresh TTL on read, we have to re-fetch
        # each matching record with the read-touch policy because `scan`
        # itself doesn't take that policy.
        read_policy = self._build_read_policy_for_refresh(op.refresh_ttl)
        out: list[SearchItem] = []

        for pkey, _, bins in records:
            if read_policy:
                try:
                    _, _, bins = self.client.get(pkey, policy=read_policy)
                except aerospike.exception.AerospikeError:
                    continue
            ns = tuple(bins.get("namespace", ()))
            key = bins.get("key")
            value = bins.get("value")
            meta = bins.get("meta") or {}
            now = _now_utc().isoformat()
            created_at = meta.get("created_at", now)
            updated_at = meta.get("updated_at", now)

            out.append(
                SearchItem(
                    namespace=ns,
                    key=key,
                    value=value,
                    created_at=created_at,
                    updated_at=updated_at,
                    score=None,
                )
            )

        if op.offset:
            out = out[op.offset :]
        if op.limit is not None:
            out = out[: op.limit]

        return out

    def _handle_list_namespaces(self, op: ListNamespacesOp) -> list[tuple[str, ...]]:
        prefix: NamespacePath | None = None
        suffix: NamespacePath | None = None
        if op.match_conditions:
            for condition in op.match_conditions:
                if condition.match_type == "prefix":
                    prefix = condition.path
                elif condition.match_type == "suffix":
                    suffix = condition.path
                else:
                    raise ValueError(f"Match type {condition.match_type} must be prefix or suffix.")

        filter_exprs = []
        if prefix:
            filter_exprs.extend(self._build_path_filter(prefix, "namespace", is_suffix=False))
        if suffix:
            filter_exprs.extend(self._build_path_filter(suffix, "namespace", is_suffix=True))

        policy: dict[str, Any] = {}
        if filter_exprs:
            policy["expressions"] = exp.And(*filter_exprs).compile()

        try:
            scan = self.client.scan(self.ns, self.set)
            records = scan.results(policy=policy)
        except aerospike.exception.AerospikeError as e:
            raise RuntimeError(f"Aerospike search failed: {e}") from e

        all_namespaces: set[tuple[str, ...]] = set()
        for _, _, bins in records:
            ns = tuple(bins.get("namespace", ()))
            if op.max_depth is not None:
                ns = ns[: op.max_depth]
            all_namespaces.add(ns)

        # Sort before paginating: Aerospike scans return records in
        # digest-hash order, which is non-deterministic across calls. Without
        # a stable sort, two consecutive `list_namespaces(limit=N, offset=K)`
        # calls could overlap or skip entries, making pagination meaningless.
        result = sorted(all_namespaces)
        if op.offset:
            result = result[op.offset :]
        if op.limit:
            result = result[: op.limit]
        return result

    # --------------- BaseStore implementation ------------------

    def batch(self, ops: Iterable[Op]) -> list[Result]:
        result: list[Result] = []
        for op in ops:
            if isinstance(op, GetOp):
                result.append(self._handle_get(op))
            elif isinstance(op, PutOp):
                self._handle_put(op)
                result.append(None)
            elif isinstance(op, SearchOp):
                result.append(self._handle_search(op))
            elif isinstance(op, ListNamespacesOp):
                result.append(self._handle_list_namespaces(op))
            else:
                raise TypeError(f"Unsupported operation type: {type(op)}")

        return result

    async def abatch(self, ops: Iterable[Op]) -> list[Result]:
        return await asyncio.to_thread(self.batch, ops)
