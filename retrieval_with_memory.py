# This is the retrieval pipeline (Part 2) of the text-only RAG chatbot. It has the additional feature of 
# maintaining a chat history, which allows it to have a flowing conversation.

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

load_dotenv()

# Load embeddings and vector store
persistent_directory = "db/chroma_db"
embedding_model = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001", timeout = 30)
db = Chroma(persist_directory=persistent_directory, embedding_function=embedding_model)

# Set up the LLM
model = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite-preview", timeout = 30)

# Store conversation as messages
chat_history = []

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

    # Step 5: Remember this conversation
    chat_history.append(HumanMessage(content=user_question))
    chat_history.append(AIMessage(content=answer))

    print(answer)
    return answer

# Simple chat loop
def start_chat():
    print("The chatbot has started. Type 'quit' to exit.")

    while True:
        question = input("\nEnter your question: ")

        if question.lower() == "quit":
            print("The chatbot has ended.")
            break

        ask_question(question)

if __name__ == "__main__":
    start_chat()

# Example Conversation:
# What was SpaceX's first rocket?
# What rocket came after this?
# Who founded this company?
