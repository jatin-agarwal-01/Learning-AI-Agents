'''
-This is an AI agent that works on the OpenRouter Free Models API.
-This Agent prints the response from the API in a streaming manner, allowing you to see the AI's response as it is generated.
'''

import os
import json
import requests
from typing import List, TypedDict
from dotenv import load_dotenv
from langgraph.graph import StateGraph, START, END

load_dotenv()


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
            print(f"\n❌ API Error: {response.status_code}")
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
                    full_response += content

            except json.JSONDecodeError:
                continue
            except Exception as e:
                print(f"\n⚠️ Chunk Error: {e}")
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

    except Exception as e:
        print(f"\n❌ Network Error: {e}\n")
        return state


# Build LangGraph
graph = StateGraph(AgentState)

graph.add_node("process", process)

graph.add_edge(START, "process")
graph.add_edge("process", END)

agent = graph.compile()

print("🤖 LangGraph Agent Initialized!")
print("Type 'exit' to quit.\n")

conversation_history = []

while True:
    user_input = input("You: ")

    if user_input.lower() in ["exit", "quit"]:
        print("Goodbye!")
        break

    if not user_input.strip():
        continue

    conversation_history.append(
        {
            "role": "user",
            "content": user_input,
        }
    )

    result = agent.invoke(
        {
            "messages": conversation_history
        }
    )

    conversation_history = result["messages"]