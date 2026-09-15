import uuid
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore
from src.deep_agent_docent.graph import build_graph
        
class DeepAgent:
    def __init__(self):
        print("🚀 Initializing DocentAI DeepAgent...")
        self.checkpointer = InMemorySaver()
        self.store = InMemoryStore()
        workflow = build_graph()
        self.graph = workflow.compile(checkpointer=self.checkpointer,store=self.store)
        print("✅ DocentAI DeepAgent initialized.")

    

    def invoke(self,user_query: str,thread_id: str | None = None,documents: list[str] | None = None):
        if thread_id is None:
            thread_id = str(uuid.uuid4())

        initial_state = {
            "messages": [HumanMessage(content=user_query)],
            "user_query": user_query,
            "retrieval_required": False,
            "documents": documents or [],
            "plan": [],
            "current_step": 0,
            "agent_results": []
        }
        run_thread_id = str(uuid.uuid4())
        config = {"configurable": {"thread_id": run_thread_id}}
        result = self.graph.invoke(initial_state,config)
        plan = result.get("plan",[])
        agent_results = result.get("agent_results",[])
        print("\n✅ Analysis complete!")
        print(f"Selected plan: {plan}")
        print(f"Agent results: {len(agent_results)}")
        return result
    
    # THREAD STATE
    def get_thread_state(self,thread_id: str):
        config = {"configurable": {"thread_id": thread_id}}
        return self.graph.get_state(config)
