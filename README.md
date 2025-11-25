# DocENT.AI

## Project Overview
**DocENT.AI** is an agentic AI platform that ingests and analyzes multiple document formats (PDF, DOCX, CSV, HTML, PPT, PPTX) to generate actionable insights, summaries, or structured responses. The system uses a modular architecture and local LLMs to provide a flexible, privacy-focused, and scalable solution for document intelligence.

## Key Features
- **Multi-Format Document Ingestion:** Seamlessly process PDFs, Word documents, spreadsheets, HTML files, and presentations.  
- **RAG QA & Summarization:** Answer queries using retrieved document context (`rag_qa`) and generate concise summaries (`summarize`).  
- **Structured Insights:** Extract actionable insights or STAR-format analysis (`insights` / `star`).  
- **Human-in-the-Loop Validation:** Review and improve AI-generated content (`hitl_validate`).  
- **Report Generation:** Summarize and export reports in different levels of detail (`report`).  
- **Local LLM Integration:** Secure and fast inference using Ollama models.  
- **Monitoring & Logging:** Track workflows and performance in real-time.  
- **Interactive UI:** Streamlit frontend for document upload, querying, and visualization.  

## Technology Stack
- **Python 3.x** – Core language  
- **LangChain, LangChain-Ollama, LangChain-Groq** – LLM orchestration and tool integration  
- **LangChain-Core, LangChain-Community, LangChain-MCP-Adapters** – Tool and agent adapters  
- **MCP, MCP-Use** – Multi-agent orchestration  
- **FAISS (CPU)** – Vector search for RAG workflows  
- **FPDF2, PDFDocument, PyMuPDF, docx2txt, PyPDF** – Document processing  
- **Sentence Transformers** – Embeddings for semantic search  
- **Streamlit** – Interactive UI  
- **FastAPI** – Optional API integration  
- **BeautifulSoup4, jq** – HTML and JSON parsing  
- **UV (UltraViolet)** – Packaging, dependency, and environment management  

## Usage
1. Install dependencies using your virtual environment or `uv`.  
2. Run the MCP server:
   ```bash
   uv run docent_server.py
  ```
3. Start the client to interact with the MCP server:
```bash
uv run docent_client.py 
```


## Project Structure
- `src/agents/` – QA, Summary, Extraction, STAR, HITL, Automation agents  
- `src/vectordb/` – Faiss vector store for RAG workflows  
- `src/tools/` – Export tools, structured processing utilities  
- `docent_server.py` – MCP server exposing all tools  
- `docent_client.py` – Example client to interact with MCP server  

## Usage
1. Upload supported documents through the Streamlit UI.  
2. Query the AI for summaries, insights, or document-specific answers.  
3. Agents dynamically select workflows based on input and document content.  
4. Export results (DOCX, PDF, CSV, PPTX) as needed.  

## Design Highlights
- **Modular Architecture:** Agents, tools, and prompts are decoupled for easy maintenance and extension.  
- **Dynamic Task Execution:** MCP orchestrates tool usage based on the query or document context.  
- **Local AI Inference:** No cloud dependency for sensitive document processing.  
- **Real-Time Monitoring:** Logs and workflow tracking for debugging and observability.  

## Acknowledgements
- **Local Ollama Model** – LLM inference engine  
- **LangChain & MCP** – Workflow orchestration and multi-agent management  
- **Streamlit** – Frontend interface  
- **FAISS** – Vector-based document search  

## UI Interface
<p align="center">
  <img src="/images/UI.png" alt="UI Interface" width="80%" />
</p>
