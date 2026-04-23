# tests/test_graph_smoke_live.py
from typing import TypedDict

from langchain_core.runnables import RunnableLambda
from langgraph.graph import END, StateGraph


class S(TypedDict):
    x: int


def _build_graph(checkpointer):
    g = StateGraph(S)
    g.add_node("add", RunnableLambda(lambda s: {"x": s["x"] + 1}))
    g.add_node("double", RunnableLambda(lambda s: {"x": s["x"] * 2}))
    g.set_entry_point("add")
    g.add_edge("add", "double")
    g.add_edge("double", END)
    return g.compile(checkpointer=checkpointer)


def test_graph_smoke_runs_and_persists_checkpoint(saver, cfg_base):
    app = _build_graph(saver)

    # Run once
    out = app.invoke({"x": 1}, cfg_base)
    print(out)
    assert out["x"] == 4  # (1+1)*2

    # Verify a checkpoint exists (latest)
    latest = saver.get_tuple(cfg_base)
    assert latest is not None
    assert isinstance(latest.checkpoint, dict)

    # And timeline has at least one id
    timeline = list(saver.list(cfg_base, limit=5))
    assert len(timeline) >= 1
