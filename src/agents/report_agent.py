import os
from langchain_ollama import ChatOllama
from langchain_core.messages import AIMessage
from src.deep_agent_docent.state import DeepAgentState
from src.ingestion_vector.ingestion import IngestionAgent
from src.ingestion_vector.vectordb import FaissVectorStore

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

    def run(self, state: DeepAgentState) -> dict:
        query = state["user_query"]
        results = self.vectorstore.query(query, top_k=5)
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
        retrieved_documents = []
        for i, text in enumerate(texts, start=1):
            retrieved_documents.append(
                f"### Chunk {i}\n\n{text}\n\n"
            )
        return {
            "agent_result": {"agent": "qa","result": response.content},
            "documents": retrieved_documents,
            "messages": [AIMessage(content=response.content)]
        }