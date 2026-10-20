import sys
import uuid

import dotenv
from langchain.agents import create_agent
from langfuse import get_client, propagate_attributes
from langfuse.langchain import CallbackHandler
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain_text_splitters import MarkdownHeaderTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams

dotenv.load_dotenv()

# Simulate user and session IDs
user_id = f"user_{uuid.uuid4().hex[:3]}"
session_id = f"session_{uuid.uuid4().hex[:7]}"

QDRANT_URL = "http://localhost:6333"
COLLECTION_NAME = "smartphones"

# Initialize LLM, embeddings, and external clients
llm = ChatOpenAI(model="gpt-5.4-mini")
embeddings = OpenAIEmbeddings(model="text-embedding-3-small", show_progress_bar=True)
client = QdrantClient(QDRANT_URL)

langfuse = get_client()
langfuse_handler = CallbackHandler()


def qdrant_vector_store(file_path: str) -> QdrantVectorStore:
    # Check if collection exists to avoid redundant data ingestion on every run
    if not client.collection_exists(COLLECTION_NAME):
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()

        # Split the Markdown catalog into individual documents by model headers
        headers_to_split_on = [("##", "Model")]
        splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
        documents = splitter.split_text(content)

        for doc in documents:
            model_name = doc.metadata.get("Model")
            if model_name:
                doc.page_content = f"Model: {model_name}\n{doc.page_content}"

        # Create the collection and ingest the embedded documents into Qdrant
        client.create_collection(
            collection_name=COLLECTION_NAME,
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
        )

        QdrantVectorStore.from_documents(
            documents=documents,
            embedding=embeddings,
            url=QDRANT_URL,
            collection_name=COLLECTION_NAME,
        )
        print(f"Successfully ingested {len(documents)} smartphones into Qdrant.")

    return QdrantVectorStore(
        client=client,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
    )

@tool("SearchCatalog")
def search_catalog_tool(query: str) -> str:
    """
    Searches the smartphone catalog for models matching user needs.
    :param query: A semantic search query composed of relevant attributes and concepts. Examples:
        - Feature/spec focused: "large battery fast-charging", "high-quality display", "gaming"
        - Brand/tier focused: "OnePlus premium", "flagship Snapdragon Samsung", "Oppo camera", "gaming Asus ROG"
        - Model specific (prioritize if known or deducible from input or chat history): "OnePlus Nord 6", "Samsung Galaxy S24 Ultra"
    """
    try:
        # Perform similarity search in the vector database
        vector_store = qdrant_vector_store("smartphones_catalog.md")
        results = vector_store.similarity_search(query, k=4)
        if not results:
            return "No smartphones found matching those criteria in our catalog."

        # Format the retrieved documents for the agent's context
        formatted_results = []
        for doc in results:
            model_name = doc.metadata.get("Model", "Unknown Model")
            formatted_results.append(f"## {model_name}\n{doc.page_content}")

        return "\n\n---\n\n".join(formatted_results)
    except Exception as e:
        return f"Error searching catalog: {e}"


# This tool is a sentinel to signal the end of the conversation. The agent should call it when the user says goodbye or thank you and is clearly finished. 
# The main loop will check for calls to this tool and break the loop to end the session.
@tool("EndSession")
def end_session_tool():
    """
    Useful when the user says goodbye or thank you and is finished.
    """
    pass


def main():
    # Fetch the managed system prompt from Langfuse (latest version labeled "production")
    langfuse_prompt = langfuse.get_prompt("agent_system_prompt", type="text")

    tools = [search_catalog_tool, end_session_tool]

    # Set up memory to keep track of the conversation history
    checkpointer = InMemorySaver()


    # Initialize a simple tool-calling agent
    agent = create_agent(
        model=llm,
        tools=tools,
        system_prompt=langfuse_prompt.prompt,
        checkpointer=checkpointer,
    )

    # Configure threading for the specific session
    # Allows you to store and retrieve conversation history for each user session
    config = {
        "configurable": {"thread_id": session_id},
        "callbacks": [langfuse_handler],
        # Lets the callback handler link generations (and their metrics) to this prompt version
        "metadata": {"langfuse_prompt": langfuse_prompt},
    }

    print(f"Welcome user: {user_id} to session {session_id}")
    print("Agent: Hello! What kind of smartphone are you looking for today?" )

    while True:
        try:
            user_input = input("User: ").strip()
            if not user_input:
                continue

            session_ended = False

            # Wrap the invocation in a root span; propagate_attributes tags every child observation
            with langfuse.start_as_current_observation(name="main_agent_loop"):
                with propagate_attributes(user_id=user_id, session_id=session_id, trace_name="main_agent_loop"):
                    response = agent.invoke(
                        {"messages": [{"role": "user", "content": user_input}]},
                        config=config,
                    )
                print(f"\nAgent: {response['messages'][-1].content}\n")

                if any(hasattr(msg, 'tool_calls') and any(tc['name'] == 'EndSession' for tc in msg.tool_calls)
                       for msg in response['messages']):
                    # Collect user feedback and attach it to the current trace
                    user_feedback = int(input("Rate the agent's response on a scale of 1-5: "))
                    comment = input("Any additional feedback for us? ")
                    langfuse.score_current_trace(
                        name="user_feedback",
                        value=user_feedback,
                        data_type="NUMERIC",
                        comment=comment,
                    )

                    session_ended = True

            # shut down once the root span has ended, so the trace and its scores are flushed
            if session_ended:
                langfuse.shutdown()
                break

        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    try:
        main()
    finally:
        langfuse.flush()
