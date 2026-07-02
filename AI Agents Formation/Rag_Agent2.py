import os
import hashlib
import json
import math
import re
from typing import List, TypedDict

import requests
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langgraph.graph import END, START, StateGraph

load_dotenv()
os.environ["LANGCHAIN_TRACING_V2"] = "false"
STREAM_CALLBACK = None

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


class SimpleLocalEmbeddings:
    """Deterministic local embeddings to avoid paid API usage for vector search."""

    def __init__(self, dimensions: int = 256):
        self.dimensions = dimensions

    def _embed_text(self, text: str) -> list[float]:
        vector = [0.0] * self.dimensions
        tokens = re.findall(r"\w+", text.lower())

        if not tokens:
            return vector

        for token in tokens:
            token_hash = int(hashlib.md5(token.encode("utf-8")).hexdigest(), 16)
            index = token_hash % self.dimensions
            sign = 1.0 if (token_hash >> 8) % 2 == 0 else -1.0
            vector[index] += sign

        magnitude = math.sqrt(sum(value * value for value in vector))
        if magnitude == 0:
            return vector

        return [value / magnitude for value in vector]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed_text(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed_text(text)
embeddings = SimpleLocalEmbeddings()

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
collection_name = "stock_market_local"

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


def retriever_tool(query: str) -> str:
    docs = retriever.invoke(query)

    if not docs:
        return "I found no relevant information in the Stock Market Performance 2024 document."

    results = []
    for i, doc in enumerate(docs):
        results.append(f"Document {i + 1}:\n{doc.page_content}")

    return "\n\n".join(results)


class AgentState(TypedDict):
    messages: List[dict]


system_prompt = """
You are a RAG assistant for the PDF "Stock Market Performance 2024".
Your first job is to decide what kind of user message you received.

If the user is greeting you, introducing themselves, thanking you, or making casual conversation, reply naturally and briefly as a normal assistant. Do not answer with stock market information unless the user actually asks for it.

If the user asks a question about the PDF topic, answer only from the retrieved document context. Keep the answer grounded in that context and cite the relevant document snippets.

If the user asks something that is not covered by the PDF, clearly say that the question is not covered by the provided document and ask them to ask a question related to the PDF.

Never force a stock market answer for unrelated or casual messages.
"""

OUT_OF_SCOPE_MESSAGE = (
    "This question is not covered in the provided PDF file "
    "(Stock Market Performance 2024). Please ask a question related to that document."
)


def is_casual_message(message: str) -> bool:
    normalized = message.lower().strip()
    casual_patterns = [
        r"^(hi|hello|hey)\b",
        r"\bmy name is\b",
        r"^(ok|okay|cool|great)\b",
        r"^(thanks|thank you)\b",
        r"\bthanks buddy\b",
        r"^(bye|goodbye)\b",
        r"^(who are you|how are you)\b",
    ]
    return any(re.search(pattern, normalized) for pattern in casual_patterns)


def has_query_overlap(query: str, docs: list) -> bool:
    query_tokens = set(re.findall(r"\w+", query.lower()))
    if not query_tokens:
        return False

    for doc in docs:
        doc_tokens = set(re.findall(r"\w+", doc.page_content.lower()))
        if len(query_tokens.intersection(doc_tokens)) >= 2:
            return True
    return False


def get_relevant_context(query: str) -> str | None:
    docs = vectorstore.similarity_search(query, k=5)
    if not docs:
        return None

    if not has_query_overlap(query, docs):
        return None

    results = []
    for i, doc in enumerate(docs):
        results.append(f"Document {i + 1}:\n{doc.page_content}")
    return "\n\n".join(results)


def stream_openrouter_response(messages: list[dict]) -> str:
    global STREAM_CALLBACK

    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {os.getenv('OPENROUTER_API_KEY')}",
        "Content-Type": "application/json",
    }
    payload = {
        "model": "openrouter/free",
        "messages": messages,
        "temperature": 0,
        "stream": True,
    }

    response = requests.post(
        url,
        headers=headers,
        json=payload,
        stream=True,
        timeout=120,
    )
    response.raise_for_status()

    print("\n=== ANSWER ===")
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
        except json.JSONDecodeError:
            continue

        if "choices" not in chunk:
            continue

        delta = chunk["choices"][0].get("delta", {})
        content = delta.get("content", "")
        if not content:
            continue

        print(content, end="", flush=True)
        if STREAM_CALLBACK is not None:
            STREAM_CALLBACK(content)
        full_response += content

    print("\n")
    return full_response


def process(state: AgentState) -> AgentState:
    user_message = state["messages"][-1]["content"]
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(state["messages"][:-1])

    if is_casual_message(user_message):
        messages.append({"role": "user", "content": user_message})
    else:
        retrieved_context = get_relevant_context(user_message)
        if retrieved_context is None:
            print("\n=== ANSWER ===")
            print(f"{OUT_OF_SCOPE_MESSAGE}\n")
            updated_messages = list(state["messages"])
            updated_messages.append({"role": "assistant", "content": OUT_OF_SCOPE_MESSAGE})
            return {"messages": updated_messages}

        messages.append(
            {
                "role": "user",
                "content": (
                    f"Question: {user_message}\n\n"
                    f"Retrieved context:\n{retrieved_context}\n\n"
                    "Answer using only this context. If the answer is not in the context, say so briefly."
                ),
            }
        )

    try:
        answer = stream_openrouter_response(messages)
    except requests.exceptions.RequestException as exc:
        answer = (
            "I couldn't reach OpenRouter right now. "
            f"Please check your internet or API key and try again.\n\nDetails: {exc}"
        )
        print("\n=== ANSWER ===")
        print(f"{answer}\n")
    except (KeyError, IndexError, json.JSONDecodeError) as exc:
        answer = f"I received an unexpected response from OpenRouter.\n\nDetails: {exc}"
        print("\n=== ANSWER ===")
        print(f"{answer}\n")

    updated_messages = list(state["messages"])
    updated_messages.append({"role": "assistant", "content": answer})
    return {"messages": updated_messages}


graph = StateGraph(AgentState)
graph.add_node("process", process)
graph.add_edge(START, "process")
graph.add_edge("process", END)
rag_agent = graph.compile()


def run_agent_turn(conversation_history: list[dict], user_input: str):
    updated_history = list(conversation_history)
    updated_history.append({"role": "user", "content": user_input})
    result = rag_agent.invoke({"messages": updated_history})
    return list(result["messages"])


def run_agent_turn_stream(conversation_history: List[dict], user_input: str, on_chunk=None):
    global STREAM_CALLBACK
    STREAM_CALLBACK = on_chunk
    try:
        return run_agent_turn(conversation_history, user_input)
    finally:
        STREAM_CALLBACK = None


def running_agent():
    print("\n=== RAG AGENT===")
    conversation_history: list[dict] = []

    while True:
        user_input = input("\nWhat is your question: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Exiting the RAG Agent. Goodbye!")
            break

        conversation_history = run_agent_turn_stream(conversation_history, user_input)


if __name__ == "__main__":
    running_agent()
