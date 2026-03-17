# This is the retrieval pipeline (Part 2) of the text-only RAG chatbot. It has the additional feature of 
# maintaining a chat history, which allows it to have a flowing conversation. It also saves chats so that
# conversations can be maintained across terminal sessions.

import os
import pickle

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

load_dotenv()

persistent_directory = "db/chroma_db"
chat_history_file = "db/chat_history_objs.pkl"
os.makedirs(os.path.dirname(chat_history_file), exist_ok=True)

# Load embeddings and vector store
embedding_model = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001", timeout = 30)
db = Chroma(persist_directory=persistent_directory, embedding_function=embedding_model)

# Set up the LLM
model = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite-preview", timeout = 30)

# In-memory chat history (loaded from disk at startup)
chat_history = []

def save_chat_history():
    try:
        with open(chat_history_file, "wb") as f:
            pickle.dump(chat_history, f)
    except Exception as e:
        print(f"Warning: failed to save chat history: {e}")

def load_chat_history():
    global chat_history
    if not os.path.exists(chat_history_file):
        chat_history = []
        return
    try:
        with open(chat_history_file, "rb") as f:
            chat_history = pickle.load(f)
    except Exception as e:
        print(f"Warning: failed to load chat history: {e}")
        chat_history = []

def delete_chat_history():
    global chat_history
    try:
        if os.path.exists(chat_history_file):
            os.remove(chat_history_file)
        chat_history = []
        print("Chat history deleted (disk + memory).")
    except Exception as e:
        print(f"Warning: failed to delete chat history: {e}")

# Load history at startup
load_chat_history()

def ask_question(user_question):
    print(f"\n--- You asked: {user_question} ---")

    # Step 1: Make the question clear using conversation history
    if chat_history:
        # Ask AI to make the question standalone
        messages = [
            SystemMessage(content="Given the chat history, rewrite the new question to be standalone and searchable. Just return the rewritten question. \
                          If the question is already standalone and needs no additional context, just return the original question itself. \
                          In either case, do NOT answer the question itself."),
            *chat_history,
            HumanMessage(content=f"New question: {user_question}"),
        ]

        result = model.invoke(messages)
        search_question = result.content[0]["text"].strip()
        print(f"Searching for: {search_question}")
    else:
        search_question = user_question

    # Step 2: Find relevant documents
    retriever = db.as_retriever(search_kwargs={"k": 3})
    docs = retriever.invoke(search_question)

    print(f"Found {len(docs)} relevant documents:")
    for i, doc in enumerate(docs, 1):
        # Show first 2 lines of each document
        lines = doc.page_content.split("\n")[:2]
        preview = "\n".join(lines)
        print(f"  Doc {i}: {preview}...")

    # Step 3: Create final prompt
    combined_input = f"""Based on the following documents, please answer this question: {user_question}

    Documents:
    {"\n".join([f"- {doc.page_content}" for doc in docs])}

    Please provide a clear, helpful answer using only the information from these documents. 
    If you can't find the answer in the documents, say "I don't have enough information to answer that question based on the provided documents.
    However, if it's a trivial question like 'Hello' or 'What is 2 + 2?', you may respond without consulting the documents."
    ANSWER:"
    """

    # Step 4: Get the answer
    messages = [
        SystemMessage(content="You are a helpful assistant that answers questions based on provided documents and conversation history."),
        *chat_history,
        HumanMessage(content=combined_input),
    ]

    result = model.invoke(messages)
    answer = result.content[0]["text"]

    # Step 5: Remember this conversation (append and persist)
    chat_history.append(HumanMessage(content=user_question))
    chat_history.append(AIMessage(content=answer))
    save_chat_history()

    print(answer)
    return answer

# Simple chat loop with quit and delete commands
def start_chat():
    print("The chatbot has started. Type 'quit' to exit and 'delete' to delete chat history.")
    if chat_history:
        print(f"Loaded {len(chat_history)} messages from saved history.")
    else:
        print("No saved chat history found.")

    while True:
        question = input("\nEnter your question: ").strip()

        if question.lower() == "quit":
            print("The chatbot has ended.")
            break
        if question.lower() == "delete":
            delete_chat_history()
            print("The chat history has been deleted.")
            continue

        ask_question(question)

if __name__ == "__main__":
    start_chat()

# Example Conversation:
# What was SpaceX's first rocket?
# What rocket came after this?
# Who founded this company?
