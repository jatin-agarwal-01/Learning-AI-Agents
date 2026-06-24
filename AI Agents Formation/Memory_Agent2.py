import os
from typing import Union, List, TypedDict, Optional
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
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

conversation_history = []

user_input = input("Enter: ")
while user_input != "exit":
    conversation_history.append(HumanMessage(content=user_input))
    summary = None
    summary_result =""
    if len(conversation_history) > 6:
        summary = " ".join([msg.content for msg in conversation_history[:-6]])
        conversation_history = conversation_history[-6:]
        msg= [HumanMessage(content=summary), SystemMessage("Please summarize the above conversation in a concise manner.")]
        summary_result = llm.invoke(msg)
    print(f"\nSummary of previous conversation: {summary_result.content if summary_result else 'No summary available.'}")
    result = agent.invoke({"message": conversation_history})    
    conversation_history = result["message"]    
    
    user_input = input("Enter: ")


with open("logging.txt", "w") as file:
    file.write("Your Conversation Log:\n")
    file.write(f"Summary of previous conversation: {summary_result.content if summary_result else 'No summary available.'}\n\n")
    for message in conversation_history:
        if isinstance(message, HumanMessage):
            file.write(f"You: {message.content}\n")
        elif isinstance(message, AIMessage):
            file.write(f"AI: {message.content}\n")
    file.write("\nEnd of Conversation\n")

print("Conversation saved to logging.txt")