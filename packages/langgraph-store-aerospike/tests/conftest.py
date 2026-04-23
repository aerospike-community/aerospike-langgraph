import contextlib
import os

import aerospike
import pytest


@pytest.fixture(scope="session")
def client():
    """Create a single Aerospike client shared across tests."""
    host = os.getenv("AEROSPIKE_HOST", "127.0.0.1")
    port = int(os.getenv("AEROSPIKE_PORT", "3000"))
    cfg = {"hosts": [(host, port)]}
    try:
        c = aerospike.client(cfg).connect()
        print("Connected to Aerospike :heavy_check_mark:")
    except aerospike.exception.AerospikeError as e:
        pytest.skip(f"Could not connect to Aerospike at {host}:{port}: {e}")
    yield c
    with contextlib.suppress(Exception):
        c.close()


@pytest.fixture(scope="session")
def namespace():
    """Default namespace for testing."""
    return "test"  # works with Docker default config


@pytest.fixture()
def store(client, namespace):
    """Create a fresh AerospikeStore instance for tests."""
    from langgraph.store.aerospike.base import AerospikeStore  # adjust import if needed

    return AerospikeStore(
        client=client,
        namespace=namespace,
        set="store_test",
    )
