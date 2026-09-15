# DocENT.AI

## Project Overview

**DocENT.AI** is an agentic document intelligence application designed to ingest and analyze academic documents and provide grounded answers, summaries, structured information, STAR-format responses, and reports.

The application uses **LangGraph-based agent orchestration**, **FAISS semantic search**, and **LLM-powered specialized agents** to determine the appropriate processing workflow for each user request.

The current knowledge base is primarily focused on:

* Social Network
* Social Network Analysis
* Big Data
* Big Data Analysis
* Big Data Analytics

DocENT.AI is designed with a modular architecture where a planner determines the user's intent and routes the request to the appropriate specialized agent.

![DocENTAI](https://github.com/bpavann/DocENT.AI/blob/main/images/IMG_7409.JPG)

---

## Key Features

* **Multi-Format Document Ingestion:** Supports PDF, DOCX, CSV, HTML, TXT, PPT/PPTX, and other supported document formats.

* **RAG-Based Question Answering:** Retrieves relevant document chunks from the FAISS vector store and generates answers grounded in the retrieved content.

* **Intelligent Agent Routing:** A Planner Agent analyzes the user's request and selects the appropriate specialized agent.

* **Conversational Agent:** Handles normal conversation and general interactions that do not require document retrieval.

* **Summary Agent:** Generates concise summaries and overviews from document content.

* **STAR Agent:** Produces Situation, Task, Action, and Result formatted responses based on relevant information.

* **Extraction Agent:** Extracts structured information such as tables, metrics, numbers, names, and other requested data.

* **Report Agent:** Generates detailed reports based on document information and agent results.

* **Local LLM Integration:** Uses Ollama for local LLM inference, providing a privacy-focused development workflow.

* **FAISS Semantic Search:** Uses vector embeddings and FAISS for efficient document retrieval.

* **Interactive Streamlit UI:** Provides a chat interface with conversation history, retrieved document context, and agent information.

* **Conversation History:** Supports multiple chat threads with the ability to create, switch between, and delete conversations.

---

## Technology Stack

* **Core Language:** Python 3.x

* **LLM & Agent Orchestration:** LangChain, LangChain-Ollama, LangGraph

* **LLM:** Ollama with locally hosted models such as `llama3.1`

* **Vector Search:** FAISS (CPU)

* **Embeddings:** Sentence Transformers (`all-MiniLM-L6-v2`)

* **Document Processing:** PyPDF, pdfplumber, BeautifulSoup4, python-docx, and other document-processing libraries

* **Web Interface:** Streamlit

* **Environment & Packaging:** uv / Python virtual environment

---

## Architecture

DocENT.AI follows a simple agent-as-node architecture:

```text
                    ┌──────────────────┐
                    │   Streamlit UI   │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │    DeepAgent     │
                    └────────┬─────────┘
                             │
                             ▼
                    ┌──────────────────┐
                    │   Planner Agent  │
                    │                  │
                    │ Understands user │
                    │     intent       │
                    └────────┬─────────┘
                             │
                       LangGraph Route
                             │
          ┌──────────┬───────┼────────┬──────────┬──────────┐
          ▼          ▼       ▼        ▼          ▼          ▼
     Conversational  QA   Summary    STAR    Extraction   Report
          │          │       │        │          │          │
          └──────────┴───────┴────────┴──────────┴──────────┘
                             │
                             ▼
                       State / Result
                             │
                             ▼
                    ┌──────────────────┐
                    │   Streamlit UI   │
                    └──────────────────┘
```

Each specialized agent receives the **DeepAgentState**, performs its task, and returns a state update to LangGraph.

---

## Agents

### Planner Agent

The Planner Agent acts as the routing controller for DocENT.AI.

It analyzes the user's request and selects exactly one route:

```text
conversational
qa
summary
star
extraction
report
```

The selected route is stored in the LangGraph state and used to determine which agent executes next.

### Conversational Agent

Handles normal conversation that does not require information from the uploaded document knowledge base.

Examples:

```text
Hi
Hello
How are you?
Who are you?
What can you do?
```

### QA Agent

Handles questions requiring information from the uploaded documents.

The QA workflow is:

```text
User Query
    ↓
FAISS Vector Search
    ↓
Retrieve Relevant Chunks
    ↓
Build Context
    ↓
Ollama LLM
    ↓
Grounded Answer
```

Retrieved document chunks are also returned to the UI so the user can inspect the supporting context.

### Summary Agent

Handles requests such as:

```text
Summarize this document
Give me the key points
Provide an overview
Summarize Big Data Analytics
```

### STAR Agent

Handles requests that explicitly require a:

```text
Situation
Task
Action
Result
```

formatted response.

### Extraction Agent

Handles structured extraction requests such as:

```text
Extract the numbers
Extract the metrics
Extract the names
Extract the table
Extract the structured information
```

### Report Agent

Handles requests for detailed or formal reports based on the available document information.

---

## Document Retrieval

DocENT.AI uses **FAISS** for semantic document retrieval.

The document pipeline is:

```text
Documents
    ↓
Document Ingestion
    ↓
Text Processing
    ↓
Chunking
    ↓
Sentence Transformer Embeddings
    ↓
FAISS Vector Store
    ↓
Semantic Search
    ↓
Relevant Document Chunks
```

The QA Agent uses the retrieved chunks as the primary source of information when answering document-based questions.

---

## State Management

The DeepAgent workflow uses a shared `DeepAgentState` containing information such as:

```text
messages
user_query
documents
route
agent_result
```

The general execution flow is:

```text
User Query
    ↓
DeepAgentState
    ↓
Planner Agent
    ↓
Route Selection
    ↓
Specialized Agent
    ↓
State Update
    ↓
Final Result
```

This keeps the workflow simple while allowing each agent to operate as a LangGraph node.

---

## Workflow

1. **Query Input**

   The user submits a request through the Streamlit interface.

2. **State Creation**

   The user query is stored in the `DeepAgentState`.

3. **Planner Execution**

   The Planner Agent analyzes the user's intent.

4. **Dynamic Routing**

   LangGraph routes the request to exactly one appropriate agent.

5. **Agent Execution**

   The selected agent performs the requested task.

6. **Document Retrieval**

   For document-grounded tasks such as QA, relevant chunks are retrieved from the FAISS vector store.

7. **State Update**

   The selected agent returns its result and any relevant retrieved documents to the shared state.

8. **Result Display**

   Streamlit displays the generated response and, when applicable, the retrieved document context.

---

## Design Highlights

* **Modular Agent Architecture:** Independent agents handle conversation, QA, summarization, STAR responses, extraction, and report generation.
* **Agent-as-Node Architecture:** Each agent directly integrates with the LangGraph workflow and receives the shared `DeepAgentState`.
* **Dynamic Routing:** The Planner Agent determines the appropriate workflow based on user intent.
* **FAISS-Powered Retrieval:** Semantic search retrieves relevant document chunks for grounded responses.
* **Local LLM Inference:** Ollama enables local model inference without requiring every request to be sent to a cloud LLM.
* **Grounded Responses:** Document-based QA uses retrieved document content as the source of truth
* **Interactive UI:** Streamlit provides a simple chat interface with conversation history and retrieved document context.
* **Conversation Management:** Users can create, switch between, and delete chat threads.

---

## UI Interface

![DocENTAI UI](https://github.com/bpavann/DocENT.AI/blob/main/images/UI_Interface.gif)

The Streamlit interface provides:
* Chat-based document interaction
* Multiple conversation threads
* New Chat functionality
* Clear Chat functionality
* Conversation deletion
* AI-generated responses
* Retrieved document context
* Expandable retrieved-document sections

---

## Acknowledgements
* **Ollama** – Local LLM inference
* **LangChain** – LLM and application framework
* **LangGraph** – Agent workflow orchestration
* **Streamlit** – Interactive web interface
* **FAISS** – Vector-based semantic search
* **Sentence Transformers** – Document embeddings
* **PyPDF / pdfplumber / BeautifulSoup4** – Document processing