import os
from langchain_ollama import ChatOllama
from langchain_core.messages import AIMessage
from src.deep_agent_docent.state import DeepAgentState
from src.ingestion_vector.ingestion import IngestionAgent
from src.ingestion_vector.vectordb import FaissVectorStore

class QAAgent:
    def __init__(self, persist_dir: str = "faiss_store", embedding_model: str = "all-MiniLM-L6-v2", llm_model: str ='llama3.1'):
        self.vectorstore = FaissVectorStore(persist_dir, embedding_model)
        faiss_path = os.path.join(persist_dir, "faiss.index")
        meta_path = os.path.join(persist_dir, "metadata.pkl")
        if not (os.path.exists(faiss_path) and os.path.exists(meta_path)):
            ingest = IngestionAgent()
            docs = ingest.load_all_docs("data")
            self.vectorstore.build_from_documents(docs)
        else:
            self.vectorstore.load()
        self.llm = ChatOllama(model=llm_model)
        print(f"Status Ollama LLM initialized: {self.llm}")

    def run(self, state: DeepAgentState) -> dict:
        query = state["user_query"]
        results = self.vectorstore.query(query, top_k=5)
        texts = [
            r["metadata"].get("text", "")
            for r in results
            if r["metadata"]
        ]

        context = "\n\n".join(texts)
        if not context:
            answer = "No relevant documents found."

            return {
                "agent_result": {
                    "agent": "qa",
                    "result": answer
                },
                "documents": [],
                "messages": [
                    AIMessage(content=answer)
                ]
            }
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