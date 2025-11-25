import os
from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage

# Handle imports for both module and direct execution
try:
    from src.ingestion import IngestionAgent
except ImportError:
    # Fallback for direct execution
    import sys
    from pathlib import Path
    project_root = Path(__file__).parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from src.vectordb import FaissVectorStore
    from src.ingestion import IngestionAgent

QAPROMPT=SystemMessage("""
            * You are a highly capable question-answering agent. 
            * Use the document context to answer the user query accurately. 
            * If the information is not directly available, provide reasoned speculation based on document patterns and domain knowledge. 
            * Always ensure answers are actionable, precise, and reflect the dominant information in the text.
"""    
)

class QAAgent:
    def __init__(self, persist_dir: str = "faiss_store", embedding_model: str = "all-MiniLM-L6-v2", llm_model: str = "gemma2-9b-it"):
        self.vectorstore = FaissVectorStore(persist_dir, embedding_model)
        faiss_path = os.path.join(persist_dir, "faiss.index")
        meta_path = os.path.join(persist_dir, "metadata.pkl")
        if not (os.path.exists(faiss_path) and os.path.exists(meta_path)):
            ingest=IngestionAgent()
            docs = ingest.load_all_docs("data")
            self.vectorstore.build_from_documents(docs)
        else:
            self.vectorstore.load()
        self.llm = ChatOllama(model="llama3.1")
        print(f"Status Ollama LLM initialized: {self.llm}")

    def search_and_summarize(self, query: str, top_k: int = 5) -> str:
        results = self.vectorstore.query(query, top_k=top_k)
        texts = [r["metadata"].get("text", "") for r in results if r["metadata"]]
        context = "\n\n".join(texts)
        if not context:
            return "No relevant documents found."
        prompt = f""" {QAPROMPT}\n\n Summarize the following context for the query: '{query}'\n\nContext:\n{context}\n\nSummary:"""
        response = self.llm.invoke([prompt])
        return response.content

# Example usage
if __name__ == "__main__":

    rag_search = QAAgent()
    query = "Explain about model context protocol in 5 points ?"
    summary = rag_search.search_and_summarize(query, top_k=3)
    print("Summary:", summary)
    