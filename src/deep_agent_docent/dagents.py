import uuid
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from src.deep_agent_docent.graph import build_graph
        
class DeepAgent:
    def __init__(self):
        print("🚀 Initializing DocentAI DeepAgent...")
        self.main_llm = ChatOllama(model="llama3.1")
        self.checkpointer = InMemorySaver()
        self.store = InMemoryStore()
        workflow = build_graph()
        self.graph = workflow.compile(checkpointer=self.checkpointer,store=self.store)
        print("✅ DocentAI DeepAgent initialized.")

    def invoke(self,user_query: str,thread_id: str | None = None,documents: str = ""):
        if thread_id is None:
            thread_id = str(uuid.uuid4())

        initial_state = {
            "messages": [HumanMessage(content=user_query)],
            "user_query": user_query,
            "documents": documents,
            "route": "",
            "agent_result": {}
        }

        config = {"configurable": {"thread_id": thread_id}}
        result = self.graph.invoke(initial_state,config)
        route = result.get("route", "")
        agent_result = result.get("agent_result", {})
        agent_response = agent_result.get("result", "")
        retrieved_documents = result.get("documents", [])
        print(f"\n✅ Analysis complete!")
        print(f"Selected agent: {route}")

        if route == "conversational":
            return result

        if not agent_response:
            return result

        print("🧠 Main LLM enhancing agent response...")
        document_context = "\n\n".join(retrieved_documents)
        prompt = f"""
            You are the final response generator for DocENTAI.
            The user asked:
            {user_query}
            The specialized agent used was:
            {route}
            The specialized agent produced this answer:
            {agent_response}
            The following information was retrieved from the
            user's uploaded academic documents:
            {document_context}

            IMPORTANT RULES:
            1. Answer the user's question directly.
            2. Use ONLY information supported by the retrieved
            documents and the specialized agent response.
            3. Do NOT add outside knowledge.
            4. Do NOT invent facts, datasets, assignments,
            lecture content, statistics, examples, page numbers,
            URLs, or other information.
            5. Do NOT change the factual meaning of the agent response.
            6. Remove irrelevant extraction metadata, anomalies,
            debugging information, or unrelated content.
            7. Make the answer clear, natural, concise, and easy
            to understand.
            8. If the retrieved documents do not contain enough
            information to answer the question, say so clearly.
            Return ONLY the final answer.
            """

        response = self.main_llm.invoke(prompt)
        final_response = (
            response.content
            if hasattr(response, "content")
            else str(response)
        )
        result["agent_result"]["result"] = final_response
        return result
    
    # THREAD STATE
    def get_thread_state(self,thread_id: str):
        config = {"configurable": {"thread_id": thread_id}}
        return self.graph.get_state(config)