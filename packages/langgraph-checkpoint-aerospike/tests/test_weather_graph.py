"""Deterministic integration test for a small tool-using graph.

Mirrors the shape of a real "LLM → tool → LLM" agent, but replaces the LLM
with `FakeListChatModel` so every run is reproducible. The point of the test
is the graph-plus-checkpointer pathway, not the model's output quality.
"""

from __future__ import annotations

import re
from copy import deepcopy
from typing import Any, TypedDict

from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import tool
from langgraph.graph import END, StateGraph


class AgentState(TypedDict):
    messages: list[BaseMessage]
    tool_called: bool
    pending_tool_call: dict[str, Any] | None


@tool
def get_weather(location: str) -> str:
    """Fake weather tool. Returns a pretend forecast for a given city."""
    return f"The weather in {location} is sunny and 72°F (fake data)."


def _maybe_extract_weather_location(messages: list[BaseMessage]) -> str | None:
    last_human: HumanMessage | None = None
    for m in reversed(messages):
        if isinstance(m, HumanMessage):
            last_human = m
            break
    if not last_human:
        return None

    match = re.search(r"weather in ([^?.!]+)", str(last_human.content), flags=re.IGNORECASE)
    if not match:
        return None

    return match.group(1).strip().title() or None


def _make_call_model(llm: FakeListChatModel):
    def call_model(state: AgentState) -> AgentState:
        messages = state["messages"]
        system = SystemMessage(content="You are a helpful assistant.")
        response: AIMessage = llm.invoke([system, *messages])

        location = _maybe_extract_weather_location(messages)
        pending_tool_call: dict[str, Any] | None = (
            {
                "id": "local-fallback-get_weather",
                "name": "get_weather",
                "args": {"location": location},
            }
            if location
            else None
        )

        return {
            "messages": [*messages, response],
            "tool_called": pending_tool_call is not None,
            "pending_tool_call": pending_tool_call,
        }

    return call_model


def _call_tools(state: AgentState) -> AgentState:
    messages = state["messages"]
    pending = state.get("pending_tool_call")
    if not pending:
        return state

    if pending.get("name") == "get_weather":
        args = pending.get("args", {})
        location = args.get("location", "somewhere")
        result = get_weather.invoke({"location": location})
        tool_msg = ToolMessage(
            content=result,
            name="get_weather",
            tool_call_id=pending.get("id", "fallback-tool-call-id"),
        )
        messages = [*messages, tool_msg]

    return {
        "messages": messages,
        "tool_called": True,
        "pending_tool_call": None,
    }


def _make_call_model_after_tools(llm: FakeListChatModel):
    def call_model_after_tools(state: AgentState) -> AgentState:
        messages = state["messages"]
        # Some LLMs can't ingest ToolMessage directly; rephrase as user input.
        converted: list[BaseMessage] = [
            HumanMessage(content=f"Tool `{m.name}` returned: {m.content}")
            if isinstance(m, ToolMessage)
            else m
            for m in messages
        ]

        system = SystemMessage(content="Summarize any tool output clearly.")
        response: AIMessage = llm.invoke([system, *converted])

        return {
            "messages": [*messages, response],
            "tool_called": state["tool_called"],
            "pending_tool_call": None,
        }

    return call_model_after_tools


def _build_weather_graph(checkpointer, llm: FakeListChatModel):
    g = StateGraph(AgentState)
    g.add_node("model", RunnableLambda(_make_call_model(llm)))
    g.add_node("tools", RunnableLambda(_call_tools))
    g.add_node("model_after_tools", RunnableLambda(_make_call_model_after_tools(llm)))

    g.set_entry_point("model")

    def route_after_model(state: AgentState) -> str:
        return "tools" if state["tool_called"] else "model_after_tools"

    g.add_conditional_edges("model", route_after_model)
    g.add_edge("tools", "model_after_tools")
    g.add_edge("model_after_tools", END)

    return g.compile(checkpointer=checkpointer)


def test_weather_graph_tool_path_persists_checkpoint(saver, cfg_base):
    llm = FakeListChatModel(
        responses=[
            "Let me check the weather for you.",
            "The weather in Paris is sunny and 72°F.",
        ]
    )
    app = _build_weather_graph(saver, llm)

    cfg = deepcopy(cfg_base)
    cfg["configurable"]["checkpoint_ns"] = "weather-demo"

    initial_state: AgentState = {
        "messages": [HumanMessage(content="Hi, what is the weather in Paris today?")],
        "tool_called": False,
        "pending_tool_call": None,
    }

    out = app.invoke(initial_state, cfg)
    messages = out["messages"]

    # human → ai (pre-tool) → tool → ai (final)
    assert len(messages) == 4
    assert isinstance(messages[0], HumanMessage)
    assert isinstance(messages[1], AIMessage)
    assert messages[1].content == "Let me check the weather for you."
    assert isinstance(messages[2], ToolMessage)
    assert messages[2].name == "get_weather"
    assert "Paris" in messages[2].content
    assert isinstance(messages[3], AIMessage)
    assert messages[3].content == "The weather in Paris is sunny and 72°F."

    assert out["tool_called"] is True
    assert out["pending_tool_call"] is None

    latest = saver.get_tuple(cfg)
    assert latest is not None
    assert isinstance(latest.checkpoint, dict)

    timeline = list(saver.list(cfg, limit=10))
    assert len(timeline) >= 1


def test_weather_graph_no_tool_path_persists_checkpoint(saver, cfg_base):
    llm = FakeListChatModel(responses=["Hi there!", "I am a test assistant."])
    app = _build_weather_graph(saver, llm)

    cfg = deepcopy(cfg_base)
    cfg["configurable"]["checkpoint_ns"] = "no-weather"

    initial_state: AgentState = {
        "messages": [HumanMessage(content="Hello, who are you?")],
        "tool_called": False,
        "pending_tool_call": None,
    }

    out = app.invoke(initial_state, cfg)
    messages = out["messages"]

    # human → ai (pre-tool, no tool triggered) → ai (final)
    assert len(messages) == 3
    assert not any(isinstance(m, ToolMessage) for m in messages)
    assert out["tool_called"] is False
    assert out["pending_tool_call"] is None

    latest = saver.get_tuple(cfg)
    assert latest is not None
