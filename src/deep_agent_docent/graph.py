import os
from src.ingestion_vector.ingestion import IngestionAgent
from src.ingestion_vector.vectordb import FaissVectorStore
from src.deep_agent_docent.state import DeepAgentState
from src.agents.qa_agent import QAAgent
from src.agents.planner_agent import PlannerAgent
from src.agents.summary_agent import SummaryAgent
from src.agents.star_agent import StarAgent
from src.agents.present import PresentationAgent
from src.agents.report_agent import ReportAgent
from src.agents.conv_agent import ConversationalAgent
from langgraph.graph import END, START, StateGraph

# Retrieval
def retrieve_documents(state: DeepAgentState) -> dict:
    if not state["retrieval_required"]:
        return {"documents": []}
    vectorstore = FaissVectorStore("faiss_store","all-MiniLM-L6-v2")
    faiss_path = os.path.join("faiss_store","faiss.index")
    meta_path = os.path.join("faiss_store","metadata.pkl")
    if not (os.path.exists(faiss_path) and os.path.exists(meta_path)):
        ingest = IngestionAgent()
        docs = ingest.load_all_docs("data")
        vectorstore.build_from_documents(docs)
    else:
        vectorstore.load()
    results = vectorstore.query(state["user_query"],top_k=5)
    documents = [
        r["metadata"].get("text", "")
        for r in results
        if r.get("metadata")
    ]
    print(f"📚 Retrieved documents: {len(documents)}")
    return {"documents": documents}

def route_from_plan(state: DeepAgentState) -> str:
    plan = state["plan"]
    current_step = state["current_step"]
    if current_step >= len(plan):
        print("🏁 Plan completed.")
        return END
    route = plan[current_step]
    print(
        f"🔀 LangGraph routing to: {route} "
        f"(step {current_step + 1}/{len(plan)})")
    return route


def route_after_planner(state: DeepAgentState) -> str:
    if state["retrieval_required"]:
        print("📚 Retrieval required → routing to retrieval")
        return "retrieval"
    return route_from_plan(state)


def next_agent(state: DeepAgentState) -> dict:
    next_step = state["current_step"] + 1
    print(f"➡️ Moving to next step: {next_step}")
    return {"current_step": next_step}


# BUILD GRAPH
def build_graph():
    # AGENTS
    planner_agent = PlannerAgent()
    qa_agent = QAAgent()
    summary_agent = SummaryAgent()
    star_agent = StarAgent()
    presentation_agent = PresentationAgent()
    report_agent = ReportAgent()
    conversational_agent = ConversationalAgent()

    # BUILD GRAPH
    workflow = StateGraph(DeepAgentState)

    # Nodes
    workflow.add_node("planner",planner_agent.run)
    workflow.add_node("retrieval",retrieve_documents)
    workflow.add_node("conversational",conversational_agent.run)
    workflow.add_node("qa",qa_agent.run)
    workflow.add_node("summary",summary_agent.run)
    workflow.add_node("star",star_agent.run)
    workflow.add_node("present",presentation_agent.run)
    workflow.add_node("report",report_agent.run)
    workflow.add_node("next_agent",next_agent)


    # START → PLANNER
    workflow.add_edge(START,"planner")

    # PLANNER → RETRIEVAL OR AGENT
    workflow.add_conditional_edges(
        "planner",
        route_after_planner,
        {
            "retrieval": "retrieval",
            "conversational": "conversational",
            "qa": "qa",
            "summary": "summary",
            "star": "star",
            "present": "present",
            "report": "report",
            END: END,
        }
    )


    # RETRIEVAL → FIRST AGENT
    workflow.add_conditional_edges(
        "retrieval",
        route_from_plan,
        {
            "conversational": "conversational",
            "qa": "qa",
            "summary": "summary",
            "star": "star",
            "present": "present",
            "report": "report",
            END: END,
        }
    )
    # AGENTS → NEXT AGENT
    workflow.add_edge("conversational","next_agent")
    workflow.add_edge("qa","next_agent")
    workflow.add_edge("summary","next_agent")
    workflow.add_edge("star","next_agent")
    workflow.add_edge("present","next_agent")
    workflow.add_edge("report","next_agent")

    # NEXT AGENT → NEXT ROUTE OR END
    workflow.add_conditional_edges(
        "next_agent",
        route_from_plan,
        {
            "conversational": "conversational",
            "qa": "qa",
            "summary": "summary",
            "star": "star",
            "present": "present",
            "report": "report",
            END: END,
        }
    )
    return workflow