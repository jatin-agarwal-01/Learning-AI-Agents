from __future__ import annotations

from pathlib import Path
import importlib.util
from typing import Any

import streamlit as st
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage


ROOT_DIR = Path(__file__).resolve().parent
AGENTS_DIR = ROOT_DIR / "AI Agents Formation"

AGENT_CONFIGS = {
    "Agent Bot": {
        "file": "Agent_Bot.py",
        "description": "Simple LangGraph chat agent using gpt-4o-mini.",
        "history_type": "lc_messages",
    },
    "Demo Agent": {
        "file": "demo_agent.py",
        "description": "OpenRouter-based conversational agent.",
        "history_type": "dict_messages",
    },
    "Demo Agent Streaming": {
        "file": "demo_agent2.py",
        "description": "OpenRouter agent with streaming API handling.",
        "history_type": "dict_messages",
    },
    "Drafter": {
        "file": "Drafter.py",
        "description": "Document drafting agent with update and save tools.",
        "history_type": "lc_messages",
    },
    "Memory Agent": {
        "file": "Memory_Agent.py",
        "description": "Chat agent that keeps full message history.",
        "history_type": "lc_messages",
    },
    "Memory Agent 2": {
        "file": "Memory_Agent2.py",
        "description": "Memory agent that summarizes older conversation.",
        "history_type": "lc_messages",
    },
    "RAG Agent": {
        "file": "RAG_Agent.py",
        "description": "PDF-based retrieval agent over the stock market report.",
        "history_type": "lc_messages",
    },
    "ReAct": {
        "file": "ReAct.py",
        "description": "Single-turn ReAct-style tool agent.",
        "history_type": "lc_messages",
    },
    "ReAct 2": {
        "file": "ReAct2.py",
        "description": "Multi-turn ReAct agent with saved conversation history.",
        "history_type": "lc_messages",
    },
    "ReAct 3": {
        "file": "ReAct3.py",
        "description": "Stream-friendly ReAct agent with LangSmith tracing support.",
        "history_type": "lc_messages",
    },
}


def load_module(module_name: str, file_path: Path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module from {file_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@st.cache_resource(show_spinner=False)
def get_agent_module(agent_name: str):
    config = AGENT_CONFIGS[agent_name]
    module_name = f"agent_hub_{config['file'].replace('.py', '').replace(' ', '_').lower()}"
    return load_module(module_name, AGENTS_DIR / config["file"])


def ensure_state() -> None:
    if "agent_histories" not in st.session_state:
        st.session_state.agent_histories = {name: [] for name in AGENT_CONFIGS}
    if "agent_meta" not in st.session_state:
        st.session_state.agent_meta = {name: {"summary_text": ""} for name in AGENT_CONFIGS}


def render_langchain_message(message: Any) -> None:
    if isinstance(message, tuple) and len(message) == 2:
        role, content = message
        with st.chat_message("user" if role == "user" else "assistant"):
            st.markdown(str(content))
        return

    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(str(message.content))
        return

    if isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            text = str(message.content).strip() or "_No assistant text returned._"
            st.markdown(text)
            tool_calls = getattr(message, "tool_calls", None) or []
            if tool_calls:
                with st.expander("Tool calls", expanded=False):
                    st.json(tool_calls)
        return

    if isinstance(message, ToolMessage):
        with st.chat_message("assistant"):
            st.markdown(f"Tool output: `{message.content}`")


def render_dict_message(message: dict[str, Any]) -> None:
    role = message.get("role", "assistant")
    content = str(message.get("content", ""))
    with st.chat_message("user" if role == "user" else "assistant"):
        st.markdown(content or "_No content returned._")


def render_history(agent_name: str, history: list[Any]) -> None:
    history_type = AGENT_CONFIGS[agent_name]["history_type"]
    for message in history:
        if history_type == "dict_messages":
            render_dict_message(message)
        else:
            render_langchain_message(message)


def save_history_to_file(agent_name: str, history: list[Any]) -> Path:
    safe_name = agent_name.lower().replace(" ", "_")
    output_path = ROOT_DIR / f"{safe_name}_log.txt"
    lines = [f"{agent_name} Conversation Log", ""]

    for item in history:
        if isinstance(item, tuple) and len(item) == 2:
            lines.append(f"{str(item[0]).title()}: {item[1]}")
        elif isinstance(item, HumanMessage):
            lines.append(f"You: {item.content}")
        elif isinstance(item, AIMessage):
            lines.append(f"AI: {item.content}")
        elif isinstance(item, ToolMessage):
            lines.append(f"Tool: {item.content}")
        elif isinstance(item, dict):
            lines.append(f"{item.get('role', 'assistant').title()}: {item.get('content', '')}")

    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def run_selected_agent(agent_name: str, module: Any, history: list[Any], prompt: str):
    if agent_name == "Memory Agent 2":
        return module.run_agent_turn(history, prompt)
    if agent_name == "ReAct":
        return module.run_agent_turn(prompt)
    return module.run_agent_turn(history, prompt)


def run_selected_agent_stream(
    agent_name: str,
    module: Any,
    history: list[Any],
    prompt: str,
    on_chunk,
):
    if hasattr(module, "run_agent_turn_stream"):
        return module.run_agent_turn_stream(history, prompt, on_chunk=on_chunk)
    return run_selected_agent(agent_name, module, history, prompt)


st.set_page_config(page_title="AI Agents Hub", page_icon="AI", layout="wide")
ensure_state()

st.title("AI Agents Hub")
st.caption("Choose any agent from AI Agents Formation and work with it from one Streamlit app.")

agent_names = list(AGENT_CONFIGS.keys())

with st.sidebar:
    st.subheader("Agents")
    selected_agent = st.selectbox("Choose an agent", agent_names)
    active_history = st.session_state.agent_histories[selected_agent]
    st.write(f"Messages: {len(active_history)}")

    if st.button("Clear selected agent", use_container_width=True):
        st.session_state.agent_histories[selected_agent] = []
        st.session_state.agent_meta[selected_agent] = {"summary_text": ""}
        st.rerun()

    if st.button("Save selected chat", use_container_width=True):
        log_path = save_history_to_file(selected_agent, active_history)
        st.success(f"Saved log to {log_path.name}")

config = AGENT_CONFIGS[selected_agent]
st.subheader(selected_agent)
st.write(config["description"])

if selected_agent == "Memory Agent 2":
    summary_text = st.session_state.agent_meta[selected_agent].get("summary_text", "")
    if summary_text:
        st.info(f"Conversation summary: {summary_text}")

if selected_agent == "Drafter":
    st.caption("Example prompts: `Create a leave email`, `Update the second paragraph`, `Save as meeting_note.txt`.")

if selected_agent == "RAG Agent":
    st.caption("Ask questions about the bundled `Stock_Market_Performance_2024.pdf` document.")

render_history(selected_agent, st.session_state.agent_histories[selected_agent])

prompt = st.chat_input(f"Message {selected_agent}...")

if prompt:
    with st.chat_message("user"):
        st.markdown(prompt)

    try:
        module = get_agent_module(selected_agent)
        history = st.session_state.agent_histories[selected_agent]
        stream_buffer = {"text": ""}

        with st.chat_message("assistant"):
            placeholder = st.empty()
            placeholder.markdown("Running agent...")

            def on_chunk(chunk: str) -> None:
                stream_buffer["text"] += chunk
                placeholder.markdown(stream_buffer["text"])

            result = run_selected_agent_stream(
                selected_agent,
                module,
                history,
                prompt,
                on_chunk,
            )

            if stream_buffer["text"].strip():
                placeholder.markdown(stream_buffer["text"])

        if selected_agent == "Memory Agent 2":
            updated_history, summary_text = result
            st.session_state.agent_histories[selected_agent] = updated_history
            st.session_state.agent_meta[selected_agent]["summary_text"] = summary_text
        else:
            st.session_state.agent_histories[selected_agent] = result

        st.rerun()
    except Exception as exc:
        with st.chat_message("assistant"):
            st.error(str(exc))
