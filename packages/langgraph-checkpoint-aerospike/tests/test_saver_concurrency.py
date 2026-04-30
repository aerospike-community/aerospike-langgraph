"""Concurrency tests for ``AerospikeSaver``.

The upstream conformance suite is single-threaded and the integration
tests won't reliably surface contention bugs, so we cover three
properties directly here:

* ``put_writes`` from N concurrent callers against the same checkpoint
  produces N distinct pending writes (no lost updates).
* ``put_writes`` retries of the same ``(task_id, idx)`` upsert in place
  rather than appending duplicates.
* ``put()`` from N concurrent callers against the same thread/ns
  produces N distinct timeline entries.
"""

from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.base import empty_checkpoint


def _seed_checkpoint(saver, thread_id: str, checkpoint_ns: str = "") -> RunnableConfig:
    """Write an empty checkpoint so subsequent ``put_writes`` has a target."""
    base_config: RunnableConfig = {
        "configurable": {
            "thread_id": thread_id,
            "checkpoint_ns": checkpoint_ns,
        }
    }
    cp = empty_checkpoint()
    metadata = {"source": "input", "step": 0, "writes": {}, "parents": {}}
    new_config = saver.put(base_config, cp, metadata, {})  # type: ignore[arg-type]
    return new_config


def test_put_writes_concurrent_no_lost_updates(saver) -> None:
    """N parallel ``put_writes`` calls must each be visible in pending_writes."""
    cp_config = _seed_checkpoint(saver, thread_id="concurrent_writers")

    n = 32
    barrier = threading.Barrier(n)

    def worker(i: int) -> None:
        # Synchronize at the barrier to maximize contention.
        barrier.wait()
        saver.put_writes(
            config=cp_config,
            writes=[(f"channel_{i}", f"value_{i}")],
            task_id=f"task_{i}",
        )

    with ThreadPoolExecutor(max_workers=n) as ex:
        list(ex.map(worker, range(n)))

    tpl = saver.get_tuple(cp_config)
    assert tpl is not None
    assert len(tpl.pending_writes) == n
    seen_task_ids = {task_id for (task_id, _channel, _value) in tpl.pending_writes}
    assert seen_task_ids == {f"task_{i}" for i in range(n)}


def test_put_writes_retry_overwrites_in_place(saver) -> None:
    """A retry of the same ``(task_id, idx)`` must upsert, not append."""
    cp_config = _seed_checkpoint(saver, thread_id="retry_writer")

    task_id = "retrying_task"

    saver.put_writes(
        config=cp_config,
        writes=[("ch", "v1")],
        task_id=task_id,
    )
    saver.put_writes(
        config=cp_config,
        writes=[("ch", "v2")],
        task_id=task_id,
    )
    saver.put_writes(
        config=cp_config,
        writes=[("ch", "v3")],
        task_id=task_id,
    )

    tpl = saver.get_tuple(cp_config)
    assert tpl is not None

    matching = [w for w in tpl.pending_writes if w[0] == task_id]
    assert len(matching) == 1
    assert matching[0][1] == "ch"
    assert matching[0][2] == "v3"


def test_put_concurrent_no_lost_timeline_entries(saver) -> None:
    """N parallel ``put()`` calls must each leave a timeline entry."""
    base_config: RunnableConfig = {
        "configurable": {
            "thread_id": "concurrent_timeline",
            "checkpoint_ns": "",
        }
    }

    n = 16
    barrier = threading.Barrier(n)

    def worker(i: int) -> None:
        cp = empty_checkpoint()
        cp["id"] = f"cp_{i:04d}"
        cp["ts"] = f"2026-04-29T17:00:{i:02d}+00:00"
        metadata = {"source": "input", "step": i, "writes": {}, "parents": {}}
        barrier.wait()
        saver.put(base_config, cp, metadata, {})  # type: ignore[arg-type]

    with ThreadPoolExecutor(max_workers=n) as ex:
        list(ex.map(worker, range(n)))

    timeline = list(saver.list(base_config))
    assert len(timeline) == n
    seen_ids = {tpl.config["configurable"]["checkpoint_id"] for tpl in timeline}
    assert seen_ids == {f"cp_{i:04d}" for i in range(n)}
