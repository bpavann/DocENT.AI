# DocENT.AI

## Project Overview
**DocENT.AI** is an agentic AI platform that ingests and analyzes multiple document formats (PDF, DOCX, CSV, HTML, PPT, PPTX) to generate actionable insights, summaries, or structured responses. The system uses a modular architecture and local LLMs to provide a flexible, privacy-focused, and scalable solution for document intelligence.

![DocentAI](/Users/pavankumarb/Documents/My Learning/DocENTmcp/images/IMG_7409.JPG)

## Key Features
- **Multi-Format Document Ingestion:** Supports PDFs, DOCX, CSV, HTML, TEXT, and other formats. 
- **RAG QA & Summarization:** Query documents using semantic search and receive precise answers or summaries.
- **Structured Insights:** Extract actionable data, metrics, or STAR-format analysis.   
- **Report Generation:** Aggregate subagent outputs into detailed reports.  
- **GROQ LLM:** Groq llm integrated with deep_agent.py main file
- **Local LLM Integration:** Uses Ollama or Groq models for secure, cloud-free inference. 
- **Monitoring & Logging:** Track workflows, agent execution, and debugging info in real-time. 
- **Interactive UI:** Streamlit-based interface for queries, visualization, Status Indicators and Debug Info.   

## Technology Stack
 - **Core Language:** Python 3.x
- **LLM & Agent Orchestration:** LangChain, LangChain-Ollama, LangChain-Groq, LangGraph and deep_agents.
- **Vector Search:** FAISS (CPU) for document embeddings and semantic search
- **Document Processing:** FPDF2, PyMuPDF, docx2txt, PyPDF, BeautifulSoup4
- **Embeddings:** Sentence Transformers for semantic search
- **Web Interface:** Streamlit for interactive queries
- **Environment & Packaging:** UltraViolet (UV)

## Usage
### 1️⃣ Installation
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```
2️⃣ Start Streamlit UI
```bash
streamlit run streamlit_app.py
``` 

## Project Structure
- `src/agents/` – QA, Summary, Extraction, STAR.
- `src/vectordb/` – Faiss vector store for RAG workflows (Ingestion-Embedding-VectorDB done manually from data)  
- `data/` - All types of required documents. 
- `deep_agent.py` – Core DeepAgent orchestration logic
- `streamlit_app.py` – Streamlit UI frontend 

## Workflow 
1. Query Input: User submits a query via the Streamlit UI.
2. Document Retrieval: QA subagent fetches relevant documents from FAISS vector store.
3. Query Analysis: DeepAgent analyzes the query to determine which subagents to invoke:
  * QA Agent: Answer direct questions using retrieved documents.
  * Summary Agent: Generate concise summaries or overviews.
  * STAR Agent: Perform Situation-Task-Action-Result analysis for business/event context.
  * Extraction Agent: Extract structured tables, metrics, and key information.
  * Report Agent: Aggregate all subagent outputs into a final report.
4. Subagent Execution: Only required subagents are run; skipped agents are tracked.
5. Result Aggregation: Outputs are collected, displayed in Streamlit with:
  * Status badges (completed/skipped/error)
  * Expandable sections for subagent outputs
  * Debug info including raw messages and workflow logs
6. Final Report: Optionally generated if report subagent is invoked.

## Design Highlights
- **Modular Agent Architecture:** Independent QA, Summary, STAR, Extraction, and Report agents. 
- **Dynamic Subagent Routing:** Intelligent decision-making to run only relevant agents. 
- **FAISS-Powered Retrieval:** Semantic document search for fast and accurate responses.
- **Local AI Inference:** No cloud dependency for sensitive document processing. 
- **Interactive UI:** Clean, expandable sections for results and debug logs and .
- **Real-Time Monitoring:** Logs and workflow tracking for debugging and observability.  

## Acknowledgements
- **Groq / Ollama Local Models** – Core LLM inference engine for reasoning and summarization.
- **LangChain,deep_agents & LangGraph** – Workflow orchestration and multi-agent management  
- **Streamlit** – Frontend interface  
- **FAISS** – Vector-based semantic document search.

## UI Interface
![UIInterface](/Users/pavankumarb/Documents/My Learning/DocENTmcp/images/UI_Interface.gif)
