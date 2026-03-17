# Introduction

This is a multimodal RAG chatbot that can answer queries based on knowledge derived from documents in various formats like TXT, PDF, DOCX, PPTX, etc. 

### What Is RAG

RAG stands for Retrieval-Augmented Generation. It involves augmenting an LLM's existing knowledge with supplementary information retrieved from documents. Instead of loading the entire documents (which may be very large) into the LLM's context window, only the relevant parts are retrieved and used to answer the user's query.

### Why RAG is Essential in Today's World

While LLMs are good at answering questions based on general, universal knowledge, they are often unable to answer questions whose answers can only be found in specific documents. For example, a company may wish to have an LLM assist with its software. However, it may have thousands of docs for its products that would certainly overflow the LLM's context window. RAG is essential in such cases for optimal performance.

Most LLMs already implement RAG based systems on their end when documents are uploaded by the user. However, many companies still prefer to create their own RAG systems despite relying on external LLMs for the following reasons:

- Privacy and Compliance: Many operate in sectors where they are not legally allowed to upload all their documents to an external LLM. Eg. finance and healthcare sectors.
- Cost Optimisation: Uploading all their docs to the LLM would result in them being charged on a token-by-token basis, which could be very expensive for large sets of documents. It's more cost-effective to handle the RAG system internally and only send the relevant parts of the documents to the LLM.
- Latency: Sending large sets of docs to an LLM over the internet would incur higher latency than processing the docs on their own and only sending relevant parts over the internet.
- No Limits: LLMs often have limits on the number of documents that can be uploaded and their file sizes.
- More Customisation: 
  - Independence from a specific LLM
  - Can use local or any third-party resouces
  - Complete data control
  - Can create more powerful RAGs
  - Can create agents
  - Can interface with any tool
  - Can work with any data format
  - Can optimise for desired jobs

### Basic RAG Workflow

![Image](https://miro.medium.com/v2/1*3--ogs382Na1U2v3LfVVcQ.png)

- First, the document is loaded and it is partitioned i.e. its elements are extracted. For now, let us assume that only text is extracted.
- Then the text is divided into chunks (parts). This is to ensure that only relevant chunks will be sent to the LLM that are able to answer the user's query, instead of sending the entire document. This chunking is done by a text splitter.
- The chunks are embedded i.e. vector embeddings are generated and stored in a vector database. Unlike regular databases, vector databases store embeddings (numberic representations of text that carry semantic meaning) and enable similarity search. So if the user searches for "cat", a chunk that contains "kitten" will have higher similarity compared to a chunk containing "dog".
- Next, the user's query is also embedded. This embedding is compared against the embeddings stored in the vector database to retrieve the top k relevant chunks based on similarity.
- The top k relevant chunks are sent to the LLM along with the user's query.
- The LLM returns the answer based on the query and the relevant chunks.

# Overview of the Project

This project contains both a text-only RAG chatbot and a multimodal RAG chatbot.

### Tools Used

- LangChain is used, which is a framework to help create applications built on LLMs. 
  - It offers loaders, text splitters (for chunking), and wrappers over LLMs (eg. ChatAnthropic and ChatOpenAI) that provide a high-level abstraction over the models (eg. provides standardised message objects).
  - It also includes a lightweight vector DB called Chroma that's used here to store embeddings.
- The following AI models are used:
  - Google's Gemini 3.1 Flash Lite as the LLM
  - Google's Gemini Embedding 1 as the embedding model
- Pickle is used to store the chat history across sessions.
- The Unstructured library is used in the multimodal RAG for partitioning and chunking.
  - It supports extraction of text, images and tables from multiple file formats.
  - Images are extracted in base64, and OCR is also available using Tesseract.
  - Tables are extracted in HTML format to maintain its structure.
  - It supports chunking by title, which ensures that each section of the document gets its own chunk.
- UV is used as the package manager for the text-based chatbot as it is much faster than pip and includes support for virtual environments, lockfiles, and a number of other features which typically require multiple tools. However, pip is used for the multimodal chatbot as it was tested on Colab, and pip is readily available in Colab without requiring installation and configuration.

### Text-Based Chatbot

- It only supports TXT files that must be placed in the `docs` folder.
- Workflow:
  - Load chat history from previous session (if any).
  - Load documents
  - Split documents (using RecursiveCharacterTextSplitter)
  - Embed chunks and store in vector store
  - Reformulate the user's query (based on chat history) using LLM to make it standalone 
  - Embed user's query and retrieve top k relevant chunks
  - Pass the user's query, relevant chunks, chat history, and system instructions to the LLM to generate the final answer. Save it to chat history.
  - Save chat history to file.

#### How to Run:

1. Clone this repository to your local machine.

```
git clone https://github.com/RitamPatra/rag-project.git
cd rag-project
```

2. Install uv

```
pip install uv
```

3. Install the required dependencies.

```
uv sync
```

4. Create a .env file and add your Gemini API key.

```
GEMINI_API_KEY = <your-api-key-here>
```
You can create a free Gemini API key [here](https://aistudio.google.com/app/api-keys) if you don't have one.

5. Run the program.

```
uv run ingestion_pipeline.py
uv run retrieval_with_memory_persist.py
```

### Multimodal Chatbot

- It is able to process PDF files directly, and other types such as PPTX and DOCX once they are converted to PDF.
- It is able to work with text, tables, and images that are present in PDFs.
- It uses the Unstructured library to handle the unstructured data that's present in PDFs. Internally, it uses Poppler for PDF processing and Tesseract for OCR.
- The PDF is chunked by title, so every section becomes its own chunk (with certain limits to prevent overly large or small chunks).
- Each chunk contains elements such as text, tables, and images. For tables, we have them in structured HTML format so the LLM can parse the info easily. For images, we have them in base64  format since that's how LLMs accept images via API.
- Embeddings only work properly on ordinary English text, so embedding the tables and images directly won't work. So we pass them to the LLM first to get their descriptions. These descriptions are then used for embedding and retrieval of relevant queries.
- Workflow:
  - Extract text, images and tables from the PDF.
  - Chunk by title (each section gets its own chunk).
  - For chunks containing images and/or tables, send them to the LLM and get a description of the contents. This is because we can only embed text. Plus, embedding base64 or HTML would not work with embedding models' similarity search.
  - Embed chunks and store in vector store. The raw images and tables are also stored (without embedding).
  - Reformulate the user's query (based on chat history) using LLM to make it standalone 
  - Embed user's query and retrieve top k relevant chunks. The image/table descriptions are used for the similarity search.
  - Pass the user's query, relevant chunks, chat history, and system instructions to the LLM to generate the final answer. Note that the raw images and text are passed here, not the descriptions. This allows the LLM to generate more accurate answers.
- Note that while the images are OCR'd using Tesseract, we do not use this OCR text. It is more reliable to have the LLM analyse the images and use those descriptions for embedding, since the text within an image alone may not make much sense without the visuals.

#### How to Run:

1. Download `multimodal-rag.ipynb` and run it on Google Colab.
2. Make sure to adjust the file path in the code, which points to the PDF file you're working on.
3. Use Colab Secrets to add your Gemini API key. You can create a free Gemini API key [here](https://aistudio.google.com/app/api-keys) if you don't have one.

### Chat History Limits

- The chat history needs to be managed to ensure that we do not overflow the LLM's context window. 
- Gemini 3.1 Flash Lite has a context window limit of [1M tokens](https://ai.google.dev/gemini-api/docs/models/gemini-3.1-flash-lite-preview) (~4M characters).
- Potential Strategies:
  - Simple Sliding Window: Store a fixed number of messages (eg. 20 messages i.e. 10 interactions). Remove older messages.
  - Token-Based Trimming: Store a fixed number of tokens and remove older messages if the limit is exceeded. Google provides an official [token counting API](https://ai.google.dev/gemini-api/docs/tokens#count-tokens). This method is more precise but also creates an overhead since we have to pass the messages to Gemini and wait for its response before removing older messages. An alternative would be to use the approximation 1 token = 4 characters, [as stated by Google](https://ai.google.dev/gemini-api/docs/tokens).
  - Summarisation: After every k interactions, summarise the older messages (eg. older half). Delete them and only keep the summary for the older messages. This preserves context and uses fewer tokens, but it also creates an overhead since we need to pass a portion of the chat history to Gemini and wait for its response. However, this is more efficient as we only do this once we hit k interactions.
  - Vectorisation: Store older messages in a vector DB and retrieve only top k relevant messages based on the user's query. Efficient but more complex as we need two maintain two vectors DBs (one for docs, one for chat history), and it also creates an overhead since we need to retrieve relevant chunks at the beginning of the interaction and also embed & store the response in the vector DB at the end of the interaction.
- For the multimodal RAG chatbot, chat history limit was implemented using the summarisation technique. After 10 interactions, the older half of the conversation is summarised and the raw messages are deleted, with only the summary being stored. Thereafter, this continues every 5 interactions. This ensures that no more than 10 interactions (i.e. 20 messages) plus the summary are stored in the chat history at the same time. Thus, we remain well within bounds of the context window limit (1M tokens).

### Future Scope

- More advanced text splitters
- Reranking
- Multimodal embedding modals (Google [just announced](https://blog.google/innovation-and-ai/models-and-research/gemini-models/gemini-embedding-2/) Gemini Embedding 2, their first multimodal embedding model, on 10th March 2026)
- Support more file types (eg. XLSX)