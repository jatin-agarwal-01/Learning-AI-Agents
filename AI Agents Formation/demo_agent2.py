'''
-This is an AI agent that works on the OpenRouter Free Models API.
-This Agent prints the response from the API in a streaming manner, allowing you to see the AI's response as it is generated.
'''

import json
import os
from typing import List, TypedDict

import requests
from dotenv import load_dotenv
from langgraph.graph import END, START, StateGraph

load_dotenv()
STREAM_CALLBACK = None


class AgentState(TypedDict):
    messages: List[dict]


def process(state: AgentState) -> AgentState:
    url = "https://openrouter.ai/api/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": "openrouter/free",
        "messages": state["messages"],
        "temperature": 0.7,
        "stream": True,
    }

    try:
        response = requests.post(
            url,
            headers=headers,
            json=payload,
            stream=True,
            timeout=120,
        )

        if response.status_code != 200:
            print(f"\nAPI Error: {response.status_code}")
            print(response.text)
            return state

        print("\nAI: ", end="", flush=True)
        full_response = ""

        for line in response.iter_lines():
            if not line:
                continue

            decoded_line = line.decode("utf-8")
            if not decoded_line.startswith("data: "):
                continue

            data = decoded_line[len("data: "):]
            if data == "[DONE]":
                break

            try:
                chunk = json.loads(data)
                if "choices" not in chunk:
                    continue

                delta = chunk["choices"][0].get("delta", {})
                content = delta.get("content", "")
                if content:
                    print(content, end="", flush=True)
                    if STREAM_CALLBACK is not None:
                        STREAM_CALLBACK(content)
                    full_response += content
            except json.JSONDecodeError:
                continue
            except Exception as exc:
                print(f"\nChunk Error: {exc}")
                continue

        print("\n")

        updated_messages = list(state["messages"])
        updated_messages.append(
            {
                "role": "assistant",
                "content": full_response,
            }
        )
        return {"messages": updated_messages}

    except Exception as exc:
        print(f"\nNetwork Error: {exc}\n")
        return state


graph = StateGraph(AgentState)
graph.add_node("process", process)
graph.add_edge(START, "process")
graph.add_edge("process", END)
agent = graph.compile()


def run_agent_turn(conversation_history: List[dict], user_input: str):
    updated_messages = list(conversation_history)
    updated_messages.append(
        {
            "role": "user",
            "content": user_input,
        }
    )
    result = agent.invoke({"messages": updated_messages})
    return result["messages"]


def run_agent_turn_stream(conversation_history: List[dict], user_input: str, on_chunk=None):
    global STREAM_CALLBACK
    STREAM_CALLBACK = on_chunk
    try:
        return run_agent_turn(conversation_history, user_input)
    finally:
        STREAM_CALLBACK = None


def main() -> None:
    print("LangGraph Agent Initialized!")
    print("Type 'exit' to quit.\n")

    conversation_history = []

    while True:
        user_input = input("You: ")

        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        if not user_input.strip():
            continue

        conversation_history = run_agent_turn(conversation_history, user_input)


if __name__ == "__main__":
    main()
