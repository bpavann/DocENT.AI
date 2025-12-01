import os
from langchain_ollama import ChatOllama

# Handle imports for both module and direct execution
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

class QAAgent:
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

    def search_and_summarize(self, query: str, top_k: int = 5) -> str:
        results = self.vectorstore.query(query, top_k=top_k)
        texts = [r["metadata"].get("text", "") for r in results if r["metadata"]]
        context = "\n\n".join(texts)
        if not context:
            return "No relevant documents found."
        prompt = f"""
            You are an advanced question-answering agent.
            Use the provided document context to answer the user query with accuracy and clarity.

            Rules:
            - Base your answer strictly on the information available in the context.
            - If the answer is not explicitly stated, provide a reasonable inference only if it aligns with patterns or facts in the document.
            - Keep the answer precise, actionable, and directly relevant to the query and No Assumptions

            User Query: '{query}'

            Context:
            {context}

            Provide the final answer below:
            """
        response = self.llm.invoke([prompt])
        return response.content

# Example usage
if __name__ == "__main__":

    rag_search = QAAgent()
    query = "Explain about model context protocol?"
    summary = rag_search.search_and_summarize(query, top_k=3)
    print("Summary:", summary)
    