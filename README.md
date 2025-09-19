
# HackRx: Intelligent Document Retrieval System

HackRx is a sophisticated, high-performance Retrieval-Augmented Generation (RAG) system built with Python and FastAPI. It is designed to intelligently answer complex questions by analyzing the content of various document types, such as legal contracts, insurance policies, and HR manuals.

The system downloads a document from a public URL, processes and understands its content, and uses large language models (LLMs) to provide structured, accurate answers to user queries.

## Key Features

- **Multi-Format Document Support**: Ingests and processes a variety of file types, including PDF, DOCX, PNG/JPG (via OCR), and EML.
- **Intelligent Text Chunking**: Employs a domain-aware chunking strategy that preserves context by splitting documents along semantic boundaries and section headers.
- **Pluggable Embedding Models**: Easily switch between leading embedding providers like OpenAI (`text-embedding-3-large`) and Google (`gemini-embedding-001`) for optimal performance.
- **High-Speed Vector Search**: Utilizes FAISS (Facebook AI Similarity Search) for highly efficient, in-memory semantic retrieval of relevant document clauses.
- **Advanced Reranking**: Optionally enhances search accuracy with a Cross-Encoder model (`BAAI/bge-reranker-large`) to re-rank retrieved results for maximum relevance.
- **Structured LLM Output**: Leverages powerful LLMs like GPT-4o to generate precise answers in a clean, predictable JSON format.
- **Asynchronous API**: Built on FastAPI, the system is fully asynchronous, ensuring high throughput and responsiveness for concurrent requests.

## Architecture Overview

The HackRx retrieval system is orchestrated as a sequential pipeline, where each component processes the data and passes it to the next stage. The entire process is exposed via a secure FastAPI endpoint.

1.  **API Gateway (`main.py`)**: The `FastAPI` application serves as the entry point. It receives a request containing a document URL and a list of questions via a secure `/hackrx/run` endpoint, authenticated with a bearer token.

2.  **Document Loading & Chunking (`loader.py`)**: The system first downloads the document from the provided URL. It detects the file type and uses the appropriate loader. A custom `smartsplitinsurancedocument` function then intelligently splits the text into overlapping chunks, attempting to keep related sections together for better context.

3.  **Vector Embedding (`embedder.py`)**: Each text chunk is converted into a high-dimensional vector embedding. The system is configured to use either OpenAI's or Google's embedding models. These embeddings capture the semantic meaning of the text.

4.  **Vector Storage & Search (`vectorstore.py`)**: The generated embeddings are loaded into a `FAISS` index. When questions are received, they are also embedded, and FAISS performs a high-speed similarity search (cosine similarity) to find the text chunks most semantically relevant to each question.

5.  **(Optional) Reranking (`reranker.py`)**: If enabled, the top results from the initial search are passed to a more computationally intensive Cross-Encoder model. This model re-evaluates the relevance of each chunk against the query and re-sorts them to improve the quality of the final context.

6.  **LLM-Powered Q&A (`structuredqa.py`)**: The top-ranked, relevant chunks are compiled into a context block. This context, along with the original questions, is sent to a powerful LLM (e.g., GPT-4o) with a carefully crafted prompt. The prompt instructs the model to answer the questions based *only* on the provided context and to return the answers in a structured JSON format.

7.  **Keyword Correction (`keywordmatcher.py`)**: Before finalizing the answers, a utility runs to ensure that critical, capitalized keywords (like policy names or defined terms) from the source document are preserved exactly in the final answer, preventing the LLM from paraphrasing them incorrectly.

8.  **Response**: The final JSON object containing the list of answers is returned to the client.

## Getting Started

Follow these instructions to set up and run the project on your local machine.

### Prerequisites

- Python 3.9+
- A virtual environment tool (e.g., `venv`)

### Installation

1.  **Clone the repository:**
    ```sh
    git clone <your-repository-url>
    cd <repository-directory>
    ```

2.  **Create and activate a virtual environment:**
    ```sh
    python -m venv venv
    source venv/bin/activate
    # On Windows, use: venv\Scripts\activate
    ```

3.  **Install the required dependencies:**
    The `requirements.txt` file includes all necessary packages, including a specific SpaCy model.
    ```sh
    pip install -r requirements.txt
    ```

### Configuration

1.  **Create a `.env` file** in the root directory of the project. You can copy the example below.

2.  **Populate the `.env` file** with your credentials and desired settings.

    ```env
    # A secret token for authenticating API requests
    TEAMTOKEN="your-secret-team-token"

    # API key for the embedding model provider (GitHub-hosted OpenAI or Google)
    GITHUBTOKEN="sk-..."
    GOOGLEAPIKEY="AIza..."

    # --- Embeddings Config ---
    # The embedding model to use (e.g., "openai/text-embedding-3-large")
    EMBEDDINGMODEL="openai/text-embedding-3-large"
    # The provider for embeddings ("github" or "google")
    EMBEDDINGPROVIDER="github"
    # Endpoint for GitHub-hosted models
    EMBEDDINGENDPOINT="https://models.github.ai/inference"

    # --- Chunking Config ---
    CHUNKSIZE=1400
    CHUNKOVERLAP=200

    # --- Retrieval Config ---
    # Set to True to enable the cross-encoder reranker
    ENABLERERANK=True
    ```

### Running the Application

1.  **Start the FastAPI server** using `uvicorn`. The `--reload` flag will automatically restart the server when you make code changes.
    ```sh
    uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
    ```

2.  The API will be accessible at `http://127.0.0.1:8000`. You can view the auto-generated documentation at `http://127.0.0.1:8000/docs`.

### Making a Request

You can interact with the API using any HTTP client, such as `curl` or Postman.

Here is an example `curl` command to send a request to the `/hackrx/run` endpoint:

```sh
curl -X 'POST' \
  'http://127.0.0.1:8000/hackrx/run' \
  -H 'accept: application/json' \
  -H 'Authorization: Bearer your-secret-team-token' \
  -H 'Content-Type: application/json' \
  -d '{
    "documents": "https://hackrx.blob.core.windows.net/assets/Arogya%20Sanjeevani%20Policy%20-%20CIN%20-%20U10200WB1906GOI001713%201.pdf",
    "questions": [
        "What is the grace period for premium payment?",
        "What is the waiting period for pre-existing diseases?",
        "Does this policy cover maternity expenses?"
    ]
}'
```

The expected response will be a JSON object with a list of answers:
```json
{
  "answers": [
    "The grace period for premium payment is 30 days.",
    "The waiting period for pre-existing diseases is 48 months of continuous coverage.",
    "No, this policy does not cover maternity expenses."
  ]
}
```

## Project Structure

```
├── app/
│   ├── __init__.py
│   ├── main.py            # FastAPI application entry point and endpoints
│   ├── config.py          # Pydantic settings and environment management
│   ├── models.py          # Pydantic models for API request/response
│   ├── llm/               # Logic for interacting with Large Language Models
│   │   └── structuredqa.py
│   ├── ragpipeline/       # Core RAG pipeline orchestration and components
│   │   ├── __init__.py
│   │   ├── loader.py
│   │   ├── embedder.py
│   │   ├── vectorstore.py
│   │   └── reranker.py
│   └── utils/             # Helper functions and utilities
│       ├── geminicache.py
│       └── keywordmatcher.py
├── scripts/               # Standalone scripts for testing components
│   ├── testloader.py
│   ├── teststructureresult.py
│   └── testvectorstore.py
├── requirements.txt       # Project dependencies
└── .env                   # Local configuration (not checked into git)
```
