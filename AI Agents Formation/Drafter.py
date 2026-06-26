from typing import Annotated, Sequence, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

load_dotenv()

document_content = ""


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


@tool
def update(content: str) -> str:
    """Updates the document with the provided content."""
    global document_content
    document_content = content
    return f"Document has been updated successfully! The current content is:\n{document_content}"


@tool
def save(filename: str) -> str:
    """Save the current document to a text file and finish the process.

    Args:
        filename: Name for the text file.
    """

    global document_content

    if not filename.endswith(".txt"):
        filename = f"{filename}.txt"

    try:
        with open(filename, "w") as file:
            file.write(document_content)
        print(f"\nDocument has been saved to: {filename}")
        return f"Document has been saved successfully to '{filename}'."
    except Exception as exc:
        return f"Error saving document: {str(exc)}"


tools = [update, save]
model = ChatOpenAI(model="gpt-4o-mini").bind_tools(tools)


def our_agent(state: AgentState) -> AgentState:
    system_prompt = SystemMessage(
        content=f"""
    You are Drafter, a helpful writing assistant. You are going to help the user update and modify documents.

    - If the user wants to update or modify content, use the 'update' tool with the complete updated content.
    - If the user wants to save and finish, you need to use the 'save' tool.
    - Make sure to always show the current document state after modifications.

    The current document content is:{document_content}
    """
    )

    if not state["messages"]:
        return {
            "messages": [
                AIMessage(
                    content="I'm ready to help you update a document. What would you like to create?"
                )
            ]
        }

    response = model.invoke([system_prompt] + list(state["messages"]))
    print(f"\nAI: {response.content}")
    if hasattr(response, "tool_calls") and response.tool_calls:
        print(f"USING TOOLS: {[tc['name'] for tc in response.tool_calls]}")

    return {"messages": [response]}


def should_continue(state: AgentState) -> str:
    messages = state["messages"]
    if not messages:
        return "end"

    last_message = messages[-1]
    tool_calls = getattr(last_message, "tool_calls", None)
    return "continue" if tool_calls else "end"


def print_messages(messages):
    if not messages:
        return

    for message in messages[-3:]:
        if isinstance(message, ToolMessage):
            print(f"\nTOOL RESULT: {message.content}")
        elif isinstance(message, AIMessage):
            print(f"\nAI: {message.content}")


graph = StateGraph(AgentState)
graph.add_node("agent", our_agent)
graph.add_node("tools", ToolNode(tools))
graph.set_entry_point("agent")
graph.add_conditional_edges(
    "agent",
    should_continue,
    {
        "continue": "tools",
        "end": END,
    },
)
graph.add_edge("tools", "agent")
app = graph.compile()


def run_agent_turn(conversation_history: Sequence[BaseMessage], user_input: str):
    updated_messages = list(conversation_history) + [HumanMessage(content=user_input)]
    result = app.invoke({"messages": updated_messages})
    return list(result["messages"])


def run_document_agent():
    print("\n===== DRAFTER =====")
    state = {"messages": []}

    first_step = app.invoke(state)
    if first_step.get("messages"):
        print(first_step["messages"][-1].content)

    while True:
        user_input = input("\nWhat would you like to do with the document? ")
        if user_input.lower() in ["exit", "quit"]:
            break
        if not user_input.strip():
            continue

        state["messages"] = run_agent_turn(state["messages"], user_input)
        print_messages(state["messages"])

    print("\n===== DRAFTER FINISHED =====")


if __name__ == "__main__":
    run_document_agent()
