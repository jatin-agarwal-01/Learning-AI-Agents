import os
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
        "Content-Type": "application/json"
    }
    payload = {
        "model": "openrouter/free",
        "messages": state["messages"],
        "temperature": 0.7
    }
    
    try:
        response = requests.post(url, headers=headers, json=payload)
        response_data = response.json()
        
        if "choices" in response_data:
            ai_message_content = response_data["choices"][0]["message"]["content"]
            print(f"\nAI: {ai_message_content}\n")
            
            updated_messages = list(state["messages"])
            updated_messages.append({"role": "assistant", "content": ai_message_content})
            return {"messages": updated_messages}
        else:
            print(f"\n⚠️ OpenRouter API Error: {response_data}\n")
            return state
            
    except Exception as e:
        print(f"\n❌ Network/Parsing Error: {e}\n")
        return state

graph = StateGraph(AgentState)
graph.add_node("process", process)
graph.add_edge(START, "process")
graph.add_edge("process", END)
agent = graph.compile()

def run_agent_turn(conversation_history: List[dict], user_input: str):
    updated_messages = list(conversation_history)
    updated_messages.append({"role": "user", "content": user_input})
    result = agent.invoke({"messages": updated_messages})
    return result["messages"]


def main() -> None:
    print("🤖 LangGraph Agent Initialized! Type 'exit' to quit.\n")
    conversation_history = []

    while True:
        user_input = input("Enter: ")

        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        if not user_input.strip():
            continue

        conversation_history = run_agent_turn(conversation_history, user_input)


if __name__ == "__main__":
    main()
