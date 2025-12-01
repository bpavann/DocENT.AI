import os
from typing import List
from langchain_ollama import ChatOllama
try:
    from src.ingestion import IngestionAgent
    from src.vectordb import FaissVectorStore
except ImportError:
    # Fallback for direct execution
    import sys
    from pathlib import Path
    project_root = Path(__file__).parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from src.vectordb import FaissVectorStore
    from src.ingestion import IngestionAgent


class ReportAgent:
    def __init__(self, persist_dir: str = "faiss_store", embedding_model: str = "all-MiniLM-L6-v2",  llm_model: str ='llama3.1'):
        self.vectorstore = FaissVectorStore(persist_dir, embedding_model)
        faiss_path = os.path.join(persist_dir, "faiss.index")
        meta_path = os.path.join(persist_dir, "metadata.pkl")
        if not (os.path.exists(faiss_path) and os.path.exists(meta_path)):
            ingest=IngestionAgent()
            docs = ingest.load_all_docs("data")
            self.vectorstore.build_from_documents(docs)
        else:
            self.vectorstore.load()
        self.llm = ChatOllama(model=llm_model)
        print(f"Status Ollama LLM initialized: {self.llm}")

    def generate_report(self, texts: List[str], top_k: int = 5) -> str:
        results = self.vectorstore.query(texts, top_k=top_k)
        texts = [r["metadata"].get("text", "") for r in results if r["metadata"]]
        context = "\n\n".join(texts)
        if not context:
            return "No relevant documents found."
        prompt=f"""
        You are an expert report-writing agent. 
        Your job is to generate a professional, polished, multi-section report from provided documents.
        Make the report actionable, deeply insightful, and formatted cleanly.

        The report MUST follow this structure:

        1. Executive Summary  
        2. Key Insights & Findings  
        3. STAR Analysis (Situation, Task, Action, Result)  
        4. Comparative Observations (if multiple docs)  
        5. Extracted Data Tables (if any)  
        6. Risks & Mitigation Strategy  
        7. Recommendations  
        8. Final Conclusion 

        Use only the details from the FAISS-retrieved context. Do not add assumptions.
            Query: '{texts}'
            Context:
            {context} 

        Be thorough, accurate, and ensure the report reads like a high-quality consulting deliverable.
        """
        response = self.llm.invoke([prompt])
        return response.content if hasattr(response, "content") else str(response)

# Example usage
if __name__ == "__main__":
    sa = ReportAgent()
    text = "I want to develop and end to end project on Agentic AI "
    print("\n\n")
    print("STAR :\n", sa.generate_report(text))