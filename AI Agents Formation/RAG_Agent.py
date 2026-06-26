import os
from operator import add as add_messages
from typing import Annotated, Sequence, TypedDict

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, StateGraph

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

pdf_path = os.path.join(BASE_DIR, "Stock_Market_Performance_2024.pdf")
if not os.path.exists(pdf_path):
    raise FileNotFoundError(f"PDF file not found: {pdf_path}")

pdf_loader = PyPDFLoader(pdf_path)

try:
    pages = pdf_loader.load()
    print(f"PDF has been loaded and has {len(pages)} pages")
except Exception as exc:
    print(f"Error loading PDF: {exc}")
    raise

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
)
pages_split = text_splitter.split_documents(pages)

persist_directory = os.path.join(BASE_DIR, "chroma_db")
collection_name = "stock_market"

if not os.path.exists(persist_directory):
    os.makedirs(persist_directory)

try:
    vectorstore = Chroma.from_documents(
        documents=pages_split,
        embedding=embeddings,
        persist_directory=persist_directory,
        collection_name=collection_name,
    )
    print("Created ChromaDB vector store!")
except Exception as exc:
    print(f"Error setting up ChromaDB: {str(exc)}")
    raise

retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5},
)


@tool
def retriever_tool(query: str) -> str:
    """
    This tool searches and returns the information from the Stock Market Performance 2024 document.
    """

    docs = retriever.invoke(query)

    if not docs:
        return "I found no relevant information in the Stock Market Performance 2024 document."

    results = []
    for i, doc in enumerate(docs):
        results.append(f"Document {i + 1}:\n{doc.page_content}")

    return "\n\n".join(results)


tools = [retriever_tool]
llm = llm.bind_tools(tools)


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


def should_continue(state: AgentState):
    result = state["messages"][-1]
    return hasattr(result, "tool_calls") and len(result.tool_calls) > 0


system_prompt = """
You are an intelligent AI assistant who answers questions about Stock Market Performance in 2024 based on the PDF document loaded into your knowledge base.
Use the retriever tool available to answer questions about the stock market performance data. You can make multiple calls if needed.
If you need to look up some information before asking a follow up question, you are allowed to do that!
Please always cite the specific parts of the documents you use in your answers.
"""

tools_dict = {our_tool.name: our_tool for our_tool in tools}


def call_llm(state: AgentState) -> AgentState:
    messages = [SystemMessage(content=system_prompt)] + list(state["messages"])
    message = llm.invoke(messages)
    return {"messages": [message]}


def take_action(state: AgentState) -> AgentState:
    tool_calls = state["messages"][-1].tool_calls
    results = []
    for tool_call in tool_calls:
        print(f"Calling Tool: {tool_call['name']} with query: {tool_call['args'].get('query', 'No query provided')}")

        if tool_call["name"] not in tools_dict:
            print(f"\nTool: {tool_call['name']} does not exist.")
            result = "Incorrect Tool Name, Please Retry and Select tool from List of Available tools."
        else:
            result = tools_dict[tool_call["name"]].invoke(tool_call["args"].get("query", ""))
            print(f"Result length: {len(str(result))}")

        results.append(
            ToolMessage(
                tool_call_id=tool_call["id"],
                name=tool_call["name"],
                content=str(result),
            )
        )

    print("Tools Execution Complete. Back to the model!")
    return {"messages": results}


graph = StateGraph(AgentState)
graph.add_node("llm", call_llm)
graph.add_node("retriever_agent", take_action)
graph.add_conditional_edges(
    "llm",
    should_continue,
    {True: "retriever_agent", False: END},
)
graph.add_edge("retriever_agent", "llm")
graph.set_entry_point("llm")
rag_agent = graph.compile()


def run_agent_turn(conversation_history: list[BaseMessage], user_input: str):
    updated_history = list(conversation_history)
    updated_history.append(HumanMessage(content=user_input))
    result = rag_agent.invoke({"messages": updated_history})
    return list(result["messages"])


def running_agent():
    print("\n=== RAG AGENT===")
    conversation_history: list[BaseMessage] = []

    while True:
        user_input = input("\nWhat is your question: ")
        if user_input.lower() in ["exit", "quit"]:
            break

        conversation_history = run_agent_turn(conversation_history, user_input)
        print("\n=== ANSWER ===")
        print(conversation_history[-1].content)


if __name__ == "__main__":
    running_agent()
