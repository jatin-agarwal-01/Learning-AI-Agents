from typing import List, TypedDict
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv # used to store secret stuff like API keys or configuration values

load_dotenv() 

class AgentState(TypedDict):
    messages: List[HumanMessage]

llm = ChatOpenAI(model="gpt-4o-mini")

def process(state: AgentState) -> AgentState:
    response = llm.invoke(state["messages"])
    print(f"\nAI: {response.content}")
    updated_messages = list(state["messages"])
    updated_messages.append(AIMessage(content=response.content))
    return {"messages": updated_messages}

graph=StateGraph(AgentState)
graph.add_node("process", process)
graph.add_edge(START, "process")
graph.add_edge("process", END)
agent = graph.compile()

def run_agent_turn(conversation_history: List[HumanMessage | AIMessage], user_input: str):
    updated_messages = list(conversation_history)
    updated_messages.append(HumanMessage(content=user_input))
    result = agent.invoke({"messages": updated_messages})
    return result["messages"]


def main() -> None:
    user = input("Enter: ")
    while user != "exit" and user != "quit":
        result = run_agent_turn([], user)
        if result and isinstance(result[-1], AIMessage):
            print(f"\nAI: {result[-1].content}")
        user = input("Enter: ")


if __name__ == "__main__":
    main()
