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

class StarAgent:
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

    def generate_star(self, query: str, top_k: int = 5) -> str:
        results = self.vectorstore.query(query, top_k=top_k)
        texts = [r["metadata"].get("text", "") for r in results if r["metadata"]]
        context = "\n\n".join(texts)
        if not context:
            return "No relevant documents found."
        prompt = f"""
            You are an expert business analyst generating STAR-format insights (Situation, Task, Action, Result) from documents. 
            Analyze the content deeply, identify key events or decisions, and convert them into STAR points.
            Provide actionable outcomes and emphasize the dominant achievements or decisions reflected in the text.

            Use the following FAISS-retrieved context to analyze the situation deeply, identify the key events/decisions, and convert them into clear STAR points.  
            Your goal:
            - Extract the Situation  
            - Identify the Task  
            - Describe all Actions taken  
            - Summarize the Result (impact, improvement, or decision made)  
            - Return the final answer ONLY in STAR format.
            Use only the details from the FAISS-retrieved context. Do not add assumptions.
            Query: '{query}'
            Context:
            {context}

            Provide the final answer strictly in STAR format.
            """
        response = self.llm.invoke([prompt])
        return response.content if hasattr(response, "content") else str(response)

# Example usage
if __name__ == "__main__":
    sa = StarAgent()
    text = "I want to develop and end to end project on Agentic AI "
    print("\n\n")
    print("STAR :\n", sa.generate_star(text))

        
        