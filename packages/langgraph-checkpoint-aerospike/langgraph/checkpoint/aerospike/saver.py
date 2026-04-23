from __future__ import annotations

import asyncio
import builtins
import contextlib
import json
from collections.abc import AsyncIterator, Iterator, Mapping, Sequence
from datetime import datetime, timezone
from typing import Any, cast

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import (
    WRITES_IDX_MAP,
    BaseCheckpointSaver,
    ChannelVersions,
    Checkpoint,
    CheckpointMetadata,
    CheckpointTuple,
)

import aerospike
import aerospike.exception  # noqa: F401  # expose `aerospike.exception` submodule for type checkers

SEP = "|"


def _now_ns() -> datetime:
    return datetime.now(tz=timezone.utc)


class AerospikeSaver(BaseCheckpointSaver):
    def __init__(
        self,
        client: aerospike.Client,
        namespace: str = "test",
        set_cp: str = "lg_cp",
        set_writes: str = "lg_cp_w",
        set_meta: str = "lg_cp_meta",
        ttl: dict[str, Any] | None = None,
    ) -> None:
        self.client = client
        self.ns = namespace
        self.set_cp = set_cp
        self.set_writes = set_writes
        self.set_meta = set_meta
        self.ttl = ttl or {}
        self.timeline_max: int | None = None
        self._ttl_minutes: int | None = self.ttl.get("default_ttl")
        self._refresh_on_read: bool = bool(self.ttl.get("refresh_on_read", False))

    # ---------- config parsing ----------
    @staticmethod
    def _ids_from_config(
        config: Mapping[str, Any] | None,
    ) -> tuple[str, str, str | None]:
        """Returns ``(thread_id, checkpoint_ns, checkpoint_id)`` from a RunnableConfig."""
        cfg = config or {}
        c = cfg.get("configurable", {}) or {}
        md = cfg.get("metadata", {}) or {}

        thread_id = c.get("thread_id") or md.get("thread_id")
        if not thread_id:
            raise ValueError("configurable.thread_id is required in RunnableConfig")

        checkpoint_ns = c.get("checkpoint_ns") or md.get("checkpoint_ns") or ""

        checkpoint_id = c.get("checkpoint_id")

        return thread_id, checkpoint_ns, checkpoint_id

    # ---------- keys ----------
    def _key_cp(self, thread_id: str, checkpoint_ns: str, checkpoint_id: str):
        return (self.ns, self.set_cp, f"{thread_id}{SEP}{checkpoint_ns}{SEP}{checkpoint_id}")

    def _key_writes(self, thread_id: str, checkpoint_ns: str, checkpoint_id: str):
        return (self.ns, self.set_writes, f"{thread_id}{SEP}{checkpoint_ns}{SEP}{checkpoint_id}")

    def _key_latest(self, thread_id: str, checkpoint_ns: str):
        return (self.ns, self.set_meta, f"{thread_id}{SEP}{checkpoint_ns}{SEP}__latest__")

    def _key_timeline(self, thread_id: str, checkpoint_ns: str):
        return (self.ns, self.set_meta, f"{thread_id}{SEP}{checkpoint_ns}{SEP}__timeline__")

    # ---------- aerospike io ----------
    def _put(self, key, bins: dict[str, Any]) -> None:
        policy: dict[str, Any] | None = None
        minutes = self._ttl_minutes
        if minutes is not None:
            minutes = int(minutes)
            if minutes > 0:
                policy = {"ttl": minutes * 60}
        try:
            if policy is not None:
                self.client.put(key, bins, policy=policy)
            else:
                self.client.put(key, bins)
        except aerospike.exception.AerospikeError as e:
            raise RuntimeError(f"Aerospike put failed for {key}: {e}") from e

    def _get(self, key) -> tuple | None:
        try:
            rec = self.client.get(key)
        except aerospike.exception.RecordNotFound:
            return None
        except aerospike.exception.AerospikeError as e:
            raise RuntimeError(f"Aerospike get failed for {key}: {e}") from e

        if self._refresh_on_read and self._ttl_minutes is not None and self._ttl_minutes > 0:
            with contextlib.suppress(aerospike.exception.AerospikeError):
                self.client.touch(key, int(self._ttl_minutes) * 60)

        return rec

    def _read_timeline_items(self, timeline_key) -> builtins.list[tuple[str, str]]:
        """Return timeline entries as ``(iso_timestamp, checkpoint_id)`` pairs."""
        rec = self._get(timeline_key)
        if rec is None:
            return []
        bins = rec[2]
        try:
            items = json.loads(bins.get("items", "[]"))
            cleaned: list[tuple[str, str]] = []
            for it in items:
                if (
                    isinstance(it, list)
                    and len(it) == 2
                    and isinstance(it[0], str)
                    and isinstance(it[1], str)
                ):
                    cleaned.append((it[0], it[1]))
            return cleaned
        except Exception:
            return []

    def _delete(self, key) -> None:
        try:
            self.client.remove(key)
        except aerospike.exception.RecordNotFound:
            pass
        except aerospike.exception.AerospikeError as e:
            raise RuntimeError(f"Aerospike delete failed for {key}: {e}") from e

    # ---------- public API (RunnableConfig-based) ----------
    def put(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:
        thread_id, checkpoint_ns, parent_checkpoint_id = self._ids_from_config(config)
        checkpoint_id = checkpoint.get("id")
        if not checkpoint_id:
            raise ValueError("checkpoint_id is required for put()")

        # `Checkpoint.ts` is a required TypedDict field, but be defensive in case
        # an older serialized format ever omits it.
        ts: str = checkpoint.get("ts") or _now_ns().isoformat()
        checkpoint["ts"] = ts

        cp_type, cp_bytes = self.serde.dumps_typed(checkpoint)
        metadata = metadata.copy()
        extra_metadata = cast(CheckpointMetadata, config.get("metadata") or {})
        metadata.update(extra_metadata)

        meta_type, meta_bytes = self.serde.dumps_typed(metadata)

        key = self._key_cp(thread_id, checkpoint_ns, checkpoint_id)
        rec: dict[str, Any] = {
            "thread_id": thread_id,
            "checkpoint_ns": checkpoint_ns,
            "checkpoint_id": checkpoint_id,
            "p_checkpoint_id": parent_checkpoint_id,
            "cp_type": cp_type,
            "checkpoint": cp_bytes,
            "meta_type": meta_type,
            "metadata": meta_bytes,
            "ts": ts,
        }
        self._put(key, rec)

        latest_key = self._key_latest(thread_id, checkpoint_ns)
        self._put(latest_key, {"checkpoint_id": checkpoint_id, "ts": ts})

        timeline_key = self._key_timeline(thread_id, checkpoint_ns)
        items = self._read_timeline_items(timeline_key)

        items = [(t, cid) for (t, cid) in items if cid != checkpoint_id]
        items.insert(0, (ts, checkpoint_id))
        if self.timeline_max is not None and len(items) > self.timeline_max:
            items = items[: self.timeline_max]
        self._put(timeline_key, {"items": json.dumps(items)})

        cfg_conf: dict[str, Any] = {**(config.get("configurable") or {})}
        cfg_conf.update(
            {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint_id,
            }
        )
        new_config: RunnableConfig = {**config, "configurable": cfg_conf}
        return new_config

    def put_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:

        if not writes:
            return

        thread_id, checkpoint_ns, checkpoint_id = self._ids_from_config(config)
        if not checkpoint_id:
            return

        key = self._key_writes(
            thread_id, checkpoint_ns, checkpoint_id
        )  # Do we need to put task id as key. (Reason: Record limit 8Mb)

        existing_rec = self._get(key)
        existing_items: list[dict[str, Any]] = []
        if existing_rec is not None:
            _, _, bins = existing_rec
            existing_items = bins.get("writes") or []

        now_ts = _now_ns().isoformat()

        for idx, (channel, value) in enumerate(writes):
            idx_val = WRITES_IDX_MAP.get(channel, idx)
            type_, serialized = self.serde.dumps_typed(value)

            new_item = {
                "task_id": task_id,
                "task_path": task_path,
                "channel": channel,
                "idx": idx_val,
                "type": type_,
                "value": serialized,
                "ts": now_ts,
            }
            replace_at: int | None = None
            for i, item in enumerate(existing_items):
                if item.get("task_id") == task_id and item.get("idx") == idx_val:
                    replace_at = i
                    break

            if replace_at is not None:
                existing_items[replace_at] = new_item
            else:
                existing_items.append(new_item)

        self._put(key, {"writes": existing_items})

    def get_tuple(
        self,
        config: RunnableConfig,
    ) -> CheckpointTuple | None:

        thread_id, checkpoint_ns, checkpoint_id = self._ids_from_config(config)

        if checkpoint_id is None:
            latest = self._get(self._key_latest(thread_id, checkpoint_ns))
            if latest is None or "checkpoint_id" not in latest[2]:
                return None
            checkpoint_id = latest[2]["checkpoint_id"]

        key = self._key_cp(thread_id, checkpoint_ns, checkpoint_id)
        got = self._get(key)
        if got is None:
            return None

        _, _, bins = got

        cp_type = bins.get("cp_type")
        raw_cp = bins.get("checkpoint")
        raw_meta = bins.get("metadata")
        meta_type = bins.get("meta_type")
        if cp_type is None or raw_cp is None:
            return None
        try:
            checkpoint = self.serde.loads_typed((cp_type, raw_cp))
        except Exception:
            return None

        if meta_type is None or raw_meta is None:
            return None
        try:
            metadata = self.serde.loads_typed((meta_type, raw_meta))
        except Exception:
            return None

        pending_writes: list[tuple[str, str, Any]] = []
        wrec = self._get(self._key_writes(thread_id, checkpoint_ns, checkpoint_id))
        if wrec is not None:
            _, _, wbins = wrec
            items = wbins.get("writes") or []
            for item in items:
                try:
                    task_id = item.get("task_id", "")
                    channel = item["channel"]
                    type_ = item["type"]
                    serialized = item["value"]
                    value = self.serde.loads_typed((type_, serialized))
                    pending_writes.append((task_id, channel, value))
                except KeyError:
                    continue

        cp_config: RunnableConfig = {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": checkpoint_ns,
                "checkpoint_id": checkpoint_id,
            }
        }

        parent_config: RunnableConfig | None = None
        if bins.get("p_checkpoint_id"):
            parent_config = {
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": checkpoint_ns,
                    "checkpoint_id": bins.get("p_checkpoint_id"),
                }
            }

        return CheckpointTuple(
            config=cp_config,
            checkpoint=checkpoint,
            metadata=metadata,
            parent_config=parent_config,
            pending_writes=pending_writes,
        )

    def delete_thread(self, thread_id: str) -> None:
        """Delete all checkpoint history for ``thread_id`` across the default namespace."""
        checkpoint_ns = ""

        timeline_key = self._key_timeline(thread_id, checkpoint_ns)
        items = self._read_timeline_items(timeline_key)
        checkpoint_ids = [cid for _ts, cid in items if cid]

        latest_key = self._key_latest(thread_id, checkpoint_ns)
        latest_rec = self._get(latest_key)
        if latest_rec is not None:
            latest_cid = (latest_rec[2] or {}).get("checkpoint_id")
            if isinstance(latest_cid, str) and latest_cid and latest_cid not in checkpoint_ids:
                checkpoint_ids.append(latest_cid)

        for cid in checkpoint_ids:
            self._delete(self._key_cp(thread_id, checkpoint_ns, cid))
            self._delete(self._key_writes(thread_id, checkpoint_ns, cid))

        self._delete(timeline_key)
        self._delete(latest_key)

    def list(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> Iterator[CheckpointTuple]:

        thread_id, checkpoint_ns, _ = self._ids_from_config(config or {})

        timeline_key = self._key_timeline(thread_id, checkpoint_ns)
        items = self._read_timeline_items(timeline_key)

        before_id: str | None = None
        if before is not None:
            _, _, before_id = self._ids_from_config(before or {})

        if before_id:
            seen = False
            new_items: list[tuple[str, str]] = []
            for ts, cid in items:
                if not seen:
                    if cid == before_id:
                        seen = True
                    continue
                new_items.append((ts, cid))
            items = new_items

        yielded = 0
        for _, cid in items:
            if limit is not None and yielded >= limit:
                break

            cp_config: RunnableConfig = {
                "configurable": {
                    "thread_id": thread_id,
                    "checkpoint_ns": checkpoint_ns,
                    "checkpoint_id": cid,
                }
            }

            tpl = self.get_tuple(cp_config)
            if tpl is None:
                continue

            if filter:
                ok = True
                for k, v in filter.items():
                    if tpl.metadata.get(k) != v:
                        ok = False
                        break
                if not ok:
                    continue

            yielded += 1
            yield tpl

    async def aget(self, config: RunnableConfig) -> Checkpoint | None:
        if value := await self.aget_tuple(config):
            return value.checkpoint
        return None

    async def aget_tuple(self, config: RunnableConfig) -> CheckpointTuple | None:

        return await asyncio.to_thread(self.get_tuple, config)

    async def alist(
        self,
        config: RunnableConfig | None,
        *,
        filter: dict[str, Any] | None = None,
        before: RunnableConfig | None = None,
        limit: int | None = None,
    ) -> AsyncIterator[CheckpointTuple]:

        def _collect() -> list[CheckpointTuple]:
            return list(self.list(config, filter=filter, before=before, limit=limit))

        items = await asyncio.to_thread(_collect)
        for tpl in items:
            yield tpl

    async def aput(
        self,
        config: RunnableConfig,
        checkpoint: Checkpoint,
        metadata: CheckpointMetadata,
        new_versions: ChannelVersions,
    ) -> RunnableConfig:

        return await asyncio.to_thread(
            self.put,
            config,
            checkpoint,
            metadata,
            new_versions,
        )

    async def aput_writes(
        self,
        config: RunnableConfig,
        writes: Sequence[tuple[str, Any]],
        task_id: str,
        task_path: str = "",
    ) -> None:

        await asyncio.to_thread(
            self.put_writes,
            config,
            writes,
            task_id,
            task_path,
        )

    async def adelete_thread(self, thread_id: str) -> None:
        """Asynchronously delete all checkpoint history for ``thread_id``."""
        await asyncio.to_thread(self.delete_thread, thread_id)
