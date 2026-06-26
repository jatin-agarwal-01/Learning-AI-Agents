'''
- this is an ReAct agent that uses a state graph to manage the conversation flow and tool usage.
- this uses langchain-core and langgraph to create a conversational agent that can call tools and manage its state.
- this uses langsmith to track the conversation and tool usage for analysis and debugging.

'''

import os
from typing import TypedDict 
from typing import Annotated #annotated is a type annotation which provide additional context to your variable or your key without actually affecting the type itself.
from typing import Sequence #to automatically handles the state updates for sequences such as by adding new messages to the chat history 
from dotenv import load_dotenv
from langchain_core.messages import BaseMessage #foundational class for all message types in LangChain
from langchain_core.messages import ToolMessage #passes Data back to LLM after it calls a tool such as the content and the tool_call_id
from langchain_core.messages import SystemMessage #Messages for providing instructions to the LLm
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.messages import message_chunk_to_message
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langchain_core.tools import tool
from langgraph.graph.message import add_messages # add_message is a reducer function 
from langgraph.prebuilt import ToolNode
from langsmith import traceable
from langsmith.run_helpers import tracing_context


# Reducer Function
# - Rule that controls how updates from nodes are combined with the existing state.
# - Tells us how to merge new data into the current state
##### Without a reducer, updates would have replaced the existing value entirely!

load_dotenv()
STREAM_CALLBACK = None

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


@tool
def add(a:int, b:int):
    """This is an addition fucntion that adds 2 numbers together"""

    return a + b

@tool
def sub(a:int, b:int):
    """This is a subtraction function that subtracts 2 numbers"""

    return a - b

@tool
def multiply(a:int, b:int):
    """This is a multiplication function that multiplies 2 numbers"""

    return a * b

tools =[add, sub, multiply]
model = ChatOpenAI(model_name="gpt-4o-mini").bind_tools(tools)


@traceable(name="react_agent_model_call")
def model_call(state: AgentState) -> AgentState:
    system_prompt = SystemMessage(content="You are my AI assistant. Please answer my query to the best of your ability.")
    streamed_response = None
    printed_any_text = False

    print("\nAI: ", end="", flush=True)
    for chunk in model.stream([system_prompt] + list(state["messages"])):
        streamed_response = chunk if streamed_response is None else streamed_response + chunk
        if chunk.text:
            print(chunk.text, end="", flush=True)
            if STREAM_CALLBACK is not None:
                STREAM_CALLBACK(chunk.text)
            printed_any_text = True

    if printed_any_text:
        print()

    if streamed_response is None:
        return {"messages": [AIMessage(content="")]}

    return {"messages": [message_chunk_to_message(streamed_response)]}

def should_continue(state: AgentState):
    messages = state["messages"]
    last_message = messages[-1]
    tool_calls = getattr(last_message, "tool_calls", None)
    if not tool_calls:
        return "end"
    else: 
        return "continue"

graph = StateGraph(AgentState)
graph.add_node("our_agent", model_call)

tool_node = ToolNode(tools=tools)
graph.add_node("tools", tool_node)

graph.set_entry_point("our_agent")
graph.add_conditional_edges(
    "our_agent", 
    should_continue,
    {
        "continue": "tools",
        "end": END
    },
)

graph.add_edge("tools", "our_agent")
app=graph.compile()

def print_stream(stream):
    printed_count = 0
    latest_messages = []
    for s in stream:
        latest_messages = s["messages"]
        new_messages = latest_messages[printed_count:]
        for message in new_messages:
            if isinstance(message, AIMessage):
                continue
            if isinstance(message, tuple):
                print(message)
            else:
                message.pretty_print()
        printed_count = len(latest_messages)
    return list(latest_messages)


def is_tracing_enabled() -> bool:
    return os.getenv("LANGSMITH_TRACING", "").lower() == "true"

def run_agent_turn(conversation_history: list[BaseMessage], user_input: str) -> list[BaseMessage]:
    updated_history = list(conversation_history)
    updated_history.append(HumanMessage(content=user_input))

    with tracing_context(
        enabled=is_tracing_enabled(),
        project_name=os.getenv("LANGSMITH_PROJECT"),
    ):
        return print_stream(
            app.stream(
                {"messages": updated_history},
                config={
                    "run_name": "react_agent_turn",
                    "tags": ["react", "langsmith-traced"],
                },
                stream_mode="values",
            )
        )


def run_agent_turn_stream(
    conversation_history: list[BaseMessage],
    user_input: str,
    on_chunk=None,
) -> list[BaseMessage]:
    global STREAM_CALLBACK
    STREAM_CALLBACK = on_chunk
    try:
        return run_agent_turn(conversation_history, user_input)
    finally:
        STREAM_CALLBACK = None


def save_conversation_log(conversation_history: list[BaseMessage], log_path: str = "logging.txt") -> None:
    with open(log_path, "w") as file:
        file.write("Your Conversation Log:\n")
        for message in conversation_history:
            if isinstance(message, HumanMessage):
                file.write(f"You: {message.content}\n")
            elif isinstance(message, AIMessage):
                ai_text = str(message.content).strip()
                if ai_text:
                    file.write(f"AI: {ai_text}\n")
            elif isinstance(message, ToolMessage):
                file.write(f"Tool: {message.content}\n")
        file.write("\nEnd of Conversation\n")


def main() -> None:
    print("ReAct agent is ready. Type 'exit' or 'quit' to stop.")
    conversation_history: list[BaseMessage] = []

    user_input = input("\nPlease Enter Your Message: ")
    while user_input != "exit" and user_input != "quit":
        if user_input:
            conversation_history = run_agent_turn(conversation_history, user_input)
        user_input = input("\nPlease Enter Your Message: ")

    save_conversation_log(conversation_history)
    print("Conversation saved to logging.txt")


if __name__ == "__main__":
    main()
