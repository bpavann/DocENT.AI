# DocENT.AI

## Project Overview

**DocENT.AI** is an agentic document intelligence application designed to ingest, understand, and analyze academic documents and provide grounded answers, summaries, structured information, STAR-format responses, and detailed reports.

The application combines **LangGraph-based agent orchestration**, **FAISS semantic search**, **Sentence Transformer embeddings**, and **LLM-powered specialized agents** to process user requests and generate useful responses based on uploaded documents.

The current knowledge base is primarily focused on:

* Social Network and Big Data Analytics

DocENT.AI provides a simple conversational interface where users can interact with their academic documents through natural-language questions and requests.

![DocENTAI](https://github.com/bpavann/DocENT.AI/blob/main/images/IMG_7409.JPG)

---

## Key Features

* **Multi-Format Document Ingestion:** Supports PDF, DOCX, CSV, HTML, TXT, PPT/PPTX, and other supported document formats.

* **RAG-Based Question Answering:** Retrieves relevant document chunks from the FAISS vector store and generates answers grounded in the retrieved content.

* **Intelligent Agent Orchestration:** Uses a Planner Agent and LangGraph to determine the appropriate processing workflow for each user request.

* **Conversational Agent:** Handles normal conversation and general interactions that do not require document retrieval.

* **QA Agent:** Answers questions using relevant information retrieved from the document knowledge base.

* **Summary Agent:** Generates concise summaries, overviews, and key points from document content.

* **STAR Agent:** Produces Situation, Task, Action, and Result formatted responses based on relevant document information.

* **Presentation Agent:** Presents information in structured and easy-to-understand formats such as tables, comparisons, differences, and organized results.

* **Report Agent:** Generates detailed and structured reports using available document information and relevant agent outputs.

* **Local LLM Integration:** Uses Ollama for local LLM inference, providing a privacy-focused development workflow.

* **FAISS Semantic Search:** Uses vector embeddings and FAISS for efficient semantic document retrieval.

* **Interactive Streamlit UI:** Provides a chat-based interface for interacting with documents.

* **Conversation History:** Supports multiple chat threads with the ability to create, switch between, and delete conversations.

* **Retrieved Context:** Allows users to inspect the document content retrieved from the knowledge base.

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

DocENT.AI follows a modular document intelligence architecture that combines document ingestion, semantic retrieval, agent orchestration, and an interactive user interface.

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
                         └────────┬─────────┘
                                  │
                                  ▼
                         ┌──────────────────┐
                         │ Document / Agent │
                         │    Processing    │
                         └────────┬─────────┘
                                  │
                    ┌─────────────┴─────────────┐
                    │                           │
                    ▼                           ▼
             ┌─────────────┐             ┌─────────────┐
             │    FAISS    │             │ Specialized │
             │   Retrieval │             │    Agents   │
             └──────┬──────┘             └──────┬──────┘
                    │                           │
                    └─────────────┬─────────────┘
                                  │
                                  ▼
                         ┌──────────────────┐
                         │  Grounded Agent  │
                         │     Results      │
                         └────────┬─────────┘
                                  │
                                  ▼
                         ┌──────────────────┐
                         │   Streamlit UI   │
                         └──────────────────┘
```

The application uses a shared **DeepAgentState** to maintain the user query, retrieved document context, execution plan, processing state, messages, and agent results.

---

## Workflow

1. **Query Input**

   The user submits a request through the Streamlit interface.

2. **State Creation**

   The user query is stored in the shared `DeepAgentState`.

3. **Planning**

   The Planner Agent analyzes the request and determines the required processing workflow.

4. **Document Retrieval**

   When document knowledge is required, relevant content is retrieved from the FAISS vector store.

5. **Agent Processing**

   The appropriate specialized agent or agents process the request using the available document context.

6. **State Update**

   Agent results and relevant document context are maintained in the shared workflow state.

7. **Result Generation**

   The processed results are returned through the DeepAgent workflow.

8. **Result Display**

   Streamlit displays the generated responses and, when applicable, the retrieved document context.

---

## Project Purpose

DocENT.AI was developed to explore how **agentic AI, semantic search, local LLMs, and document intelligence** can be combined into a practical application for working with academic knowledge.

The project focuses on making large collections of academic documents easier to:

* Understand
* Search
* Summarize
* Analyze
* Compare
* Structure
* Convert into reports

---