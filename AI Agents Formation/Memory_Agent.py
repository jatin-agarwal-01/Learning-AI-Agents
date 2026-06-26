import os
from typing import Union, List, TypedDict
from langchain_core.messages import HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv 

load_dotenv()

class AgentState(TypedDict):
    message: List[Union[HumanMessage, AIMessage]]

llm = ChatOpenAI(model="gpt-4o-mini")

def process(state: AgentState) -> AgentState:
    """This node will solve the request you input"""
    response = llm.invoke(state["message"])

    state["message"].append(AIMessage(content=response.content))
    print(f"\nAI: {response.content}")
    # print("Current State:", state["message"])

    return state

graph=StateGraph(AgentState)
graph.add_node("process", process)
graph.add_edge(START, "process")
graph.add_edge("process", END)
agent = graph.compile()

def run_agent_turn(conversation_history: List[Union[HumanMessage, AIMessage]], user_input: str):
    updated_history = list(conversation_history)
    updated_history.append(HumanMessage(content=user_input))
    result = agent.invoke({"message": updated_history})
    return result["message"]


def save_conversation_log(conversation_history: List[Union[HumanMessage, AIMessage]], log_path: str = "logging.txt"):
    with open(log_path, "w") as file:
        file.write("Your Conversation Log:\n")
        for message in conversation_history:
            if isinstance(message, HumanMessage):
                file.write(f"You: {message.content}\n")
            elif isinstance(message, AIMessage):
                file.write(f"AI: {message.content}\n")
        file.write("\nEnd of Conversation\n")


def main() -> None:
    conversation_history = []

    user_input = input("Enter: ")
    while user_input != "exit":
        conversation_history = run_agent_turn(conversation_history, user_input)
        user_input = input("Enter: ")

    save_conversation_log(conversation_history)
    print("Conversation saved to logging.txt")


if __name__ == "__main__":
    main()
