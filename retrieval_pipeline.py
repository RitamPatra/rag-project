# This is the retrieval pipeline (Part 2) of the text-only RAG chatbot.

from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

load_dotenv()

persistent_directory = "db/chroma_db"

# Load embeddings and vector store
embedding_model = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001", timeout = 30)

db = Chroma(
    persist_directory=persistent_directory,
    embedding_function=embedding_model,
    collection_metadata={"hnsw:space": "cosine"},
)

# Search for relevant documents
query = "What was SpaceX's first rocket?"

retriever = db.as_retriever(search_kwargs={"k": 5})

relevant_docs = retriever.invoke(query)

print(f"User Query: {query}")
# Display results
print("--- Context ---")
for i, doc in enumerate(relevant_docs, 1):
    print(f"Document {i}:\n{doc.page_content}\n")

# Combine the query and the relevant document contents
combined_input = f"""Based on the following documents, please answer this question: {query}

Documents:
{"\n".join([f"- {doc.page_content}" for doc in relevant_docs])}

Please provide a clear, helpful answer using only the information from these documents. 
If you can't find the answer in the documents, say "I don't have enough information to answer that question based on the provided documents.
However, if it's a trivial question like 'Hello' or 'What is 2 + 2?', you may respond without consulting the documents."
ANSWER:
"""

model = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite-preview", timeout = 30)

resp = model.invoke(combined_input)
print("Response from LLM:\n")
print(resp.content[0]["text"])


# List of models:
# gemini-3.1-flash-lite-preview  | RPD 500, RPM 15 | Ideal
# gemini-2.5-flash-lite  | RPD 20, RPM 10
# gemini-3-flash-preview  | RPD 20, RPM 5
# gemini-2.5-flash  | RPD 20, RPM 5
# gemma-3-27b-it  | RPD 14400, RPM 30
# gemma-3-12b-it  | RPD 14400, RPM 30

# Check usage:
# https://dashboard.voyageai.com/organization/usage?tab=free-token
# https://aistudio.google.com/rate-limit?timeRange=last-28-days
