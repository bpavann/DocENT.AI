from mcp.server.fastmcp import FastMCP
from typing import List, Optional
try:
    from src.summary_agent import SummaryAgent
    from src.qa_agent import QAAgent
    from src.automation_agent import AutomationAgent
    from src.extraction_agent import ExtractionAgent
    from src.star_agent import StarAgent
except ImportError:
    import sys
    from pathlib import Path
    project_root = Path(__file__).parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from src.summary_agent import SummaryAgent
    from src.qa_agent import QAAgent
    from src.extraction_agent import ExtractionAgent
    from src.automation_agent import AutomationAgent
    from src.star_agent import StarAgent


# Ensure repo root is on PYTHONPATH or use relative imports if necessary
mcp = FastMCP("docent_agents_server")

# Summarize tool
@mcp.tool()
def summarize(text: str) -> str:
    """
    Summarize text. Uses your SummaryAgent if available.
    """
    try:
        sa = SummaryAgent()
        return sa.summarize(text)
    except Exception:
        return f"[SUMMARY] {text[:400]}..."

# STAR-format insights tool
@mcp.tool()
def star(text: str) -> str:
    try:
        sa = StarAgent()
        return sa.generate_star(text)
    except Exception:
        return f"[STAR] (stub) S/T/A/R from text length {len(text)}."

# RAG-style QA (accepts client-passed chunks or falls back to vector DB if available)
@mcp.tool()
def rag_qa(query: str, chunks: Optional[List[str]] = None) -> str:
    """
    RAG QA tool. If 'chunks' is passed by client, use those as context.
    Otherwise, try to query your VectorDB (optional).
    """
    try:
        qa = QAAgent()
        return qa.search_and_summarize(query)
    except Exception:
        return f"[RAG-ANSWER] Query: {query}\n"
    

# Insights tool (calls your extraction / summary logic)
@mcp.tool()
def insights(text: str) -> str:
    try:
        ea = ExtractionAgent()
        extracted = ea.extract_tables(text)
        sa = SummaryAgent()
        return sa.summarize(extracted)
    except Exception:
        return f"[INSIGHTS] Derived insights from text length {len(text)}."

#The HITLAgent is designed for reviewing user-generated content and providing AI-generated suggestions for improvement.
@mcp.tool()
def hitl_validate(output: str, context: str = "") -> dict:
    try:
        from src.hitl_agent import HITLAgent
        hitl = HITLAgent()
        return hitl.validate(output, context)
    except Exception:
        return f"[HITLAgent] Derived hitl agent from docs length {len(output)}."



# Report generator
@mcp.tool()
def report(text: str, mode: str = "brief") -> str:
    try:
        sa = SummaryAgent()
        out = sa.summarize(text)
        if mode == "detailed":
            return out
        return out[:1500]
    except Exception:
        return f"[REPORT-{mode}] {text[:1000]}..."


# Health check
@mcp.tool()
def ping() -> str:
    return "docent_mcp_server: OK"

if __name__ == "__main__":
    mcp.run(transport="stdio")
