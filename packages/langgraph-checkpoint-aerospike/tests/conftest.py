# tests/conftest.py
import contextlib
import os

import pytest


def _connect_client():
    import aerospike

    host = os.getenv("AEROSPIKE_HOST", "localhost")
    port = int(os.getenv("AEROSPIKE_PORT", "3000"))
    cfg = {"hosts": [(host, port)]}
    client = aerospike.client(cfg).connect()
    return client


@pytest.fixture(scope="session")
def aerospike_namespace():
    return os.getenv("AEROSPIKE_NAMESPACE", "test")


@pytest.fixture(scope="session")
def client():
    import aerospike

    try:
        c = _connect_client()
    except aerospike.exception.AerospikeError as e:
        pytest.skip(f"Could not connect to Aerospike: {e}")
    yield c
    with contextlib.suppress(Exception):
        c.close()


@pytest.fixture(scope="session")
def aerospike_saver_cls():
    from langgraph.checkpoint.aerospike import AerospikeSaver

    return AerospikeSaver


# --------------- NEW: TTL CONFIG FOR TESTS ---------------
def _test_ttl_config() -> dict:
    """
    TTL configuration for all AerospikeSaver instances in tests.

    These environment variables allow you to easily tweak TTL
    while testing without touching code.
    """
    default_ttl = int(os.getenv("TEST_DEFAULT_TTL_MINUTES", "60"))
    refresh = os.getenv("TEST_REFRESH_ON_READ", "true").lower() == "true"

    return {
        "default_ttl": default_ttl,  # minutes
        "refresh_on_read": refresh,
    }


# ---------------------------------------------------------


@pytest.fixture()
def saver(
    aerospike_saver_cls,
    client,
    aerospike_namespace,
):
    ttl_cfg = _test_ttl_config()
    if ttl_cfg is not None:
        return aerospike_saver_cls(
            client=client,
            namespace=aerospike_namespace,
            ttl=ttl_cfg,
        )
    return aerospike_saver_cls(
        client=client,
        namespace=aerospike_namespace,
    )


@pytest.fixture()
def cfg_base():
    # Reuse this in tests; use any namespace/thread you like
    return {"configurable": {"thread_id": "1", "checkpoint_ns": "demo", "user_id": "jagrut"}}
