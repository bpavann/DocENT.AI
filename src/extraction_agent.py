import os
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

class ExtractionAgent:
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

    def extract_tables(self, query: str, top_k: int = 5) -> str:
        results = self.vectorstore.query(query, top_k=top_k)
        texts = [r["metadata"].get("text", "") for r in results if r["metadata"]]
        context = "\n\n".join(texts)
        if not context:
            return "No relevant documents found."

        prompt = f"""You are a data extraction specialist. 
            Extract all tables, numeric data, and structured information from the document context.
            Ensure the extraction is complete, accurate, and NO assumptions.

            Also identify:
            - Anomalies
            - Dominant trends
            - Repeated patterns
    
            Use only the details from the FAISS-retrieved context. Do not add assumptions.
            Query: '{query}'
            Context:
            {context}

        Return the results in a clean, well-structured format suitable for analysis or export.
        """
        response = self.llm.invoke([prompt])
        return response.content if hasattr(response, "content") else str(response)
       

# Example usage
if __name__ == "__main__":
    extractor = ExtractionAgent()
    sample_text = """
    Sales report Q3:
    | Product | Units Sold | Revenue |
    |---------|------------|---------|
    | A       | 150        | 3000    |
    | B       | 200        | 5000    |
    """
    extracted = extractor.extract_tables(sample_text)
    print("Extracted Tables / Data:\n", extracted)
