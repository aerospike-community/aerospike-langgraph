# tests/test_ttl_behavior.py
from __future__ import annotations

import time

import pytest


@pytest.fixture
def short_ttl_saver(aerospike_saver_cls, client, aerospike_namespace):
    """Saver with a small TTL (1 minute) and sliding TTL enabled."""
    ttl_cfg = {
        "default_ttl": 1,  # minutes
        "refresh_on_read": True,
    }
    return aerospike_saver_cls(
        client=client,
        namespace=aerospike_namespace,
        ttl=ttl_cfg,
    )


def test_checkpoint_has_ttl(short_ttl_saver, client, aerospike_namespace):
    """Checkpoint records written via the saver should have a TTL."""
    cfg = {
        "configurable": {
            "thread_id": "ttl-demo",
            "checkpoint_ns": "ttl-ns",
        }
    }
    # Your saver expects checkpoint["id"] to be present
    checkpoint = {
        "id": "ck-ttl-demo-1",
        "foo": "bar",
    }
    metadata = {"m": "v"}

    saved_config = short_ttl_saver.put(cfg, checkpoint, metadata, {})

    # Reconstruct the key using the saver’s own helper to avoid drift
    thread_id = cfg["configurable"]["thread_id"]
    checkpoint_ns = cfg["configurable"]["checkpoint_ns"]
    checkpoint_id = saved_config["configurable"]["checkpoint_id"]
    key = short_ttl_saver._key_cp(thread_id, checkpoint_ns, checkpoint_id)

    rec_key, meta, bins = client.get(key)

    # Aerospike stores TTL in record metadata, not bins
    assert "ttl" in meta
    assert meta["ttl"] > 0, f"Expected positive TTL, got {meta['ttl']}"


def test_ttl_resets_on_read(short_ttl_saver, client, aerospike_namespace):
    """
    With refresh_on_read=True, reading via the saver should bump TTL
    back up (sliding TTL). We don't assert an exact value, but we
    expect TTL after refresh to be >= before, and typically higher.
    """
    cfg = {
        "configurable": {
            "thread_id": "ttl-refresh",
            "checkpoint_ns": "ttl-ns",
        }
    }
    checkpoint = {
        "id": "ck-ttl-refresh-1",
        "foo": "bar",
    }
    metadata = {}

    saved_config = short_ttl_saver.put(cfg, checkpoint, metadata, {})

    thread_id = cfg["configurable"]["thread_id"]
    checkpoint_ns = cfg["configurable"]["checkpoint_ns"]
    checkpoint_id = saved_config["configurable"]["checkpoint_id"]
    key = short_ttl_saver._key_cp(thread_id, checkpoint_ns, checkpoint_id)

    # Let TTL tick down a bit so we can see it decrease
    time.sleep(10)

    # TTL before sliding refresh
    _, meta_before, _ = client.get(key)
    ttl_before = meta_before["ttl"]

    # Read via saver (this triggers touch with same TTL config)
    short_ttl_saver._get(key)

    # Give Aerospike a moment to apply the touch and decrement a bit
    time.sleep(1)

    # TTL after refresh
    _, meta_after, _ = client.get(key)
    ttl_after = meta_after["ttl"]
    print(ttl_before, ttl_after)
    # The TTL after refresh should not be *less* than before by any meaningful amount.
    # Because we slept 5 seconds then 1 second, if touch had no effect,
    # ttl_after would be roughly ttl_before - 1 or less.
    # Allowing equality handles coarse timer resolution.
    assert ttl_after >= ttl_before, f"Expected ttl_after ({ttl_after}) >= ttl_before ({ttl_before})"
