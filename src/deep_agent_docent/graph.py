from src.deep_agent_docent.state import DeepAgentState
from src.agents.qa_agent import QAAgent
from src.agents.planner_agent import PlannerAgent
from src.agents.summary_agent import SummaryAgent
from src.agents.star_agent import StarAgent
from src.agents.extraction_agent import ExtractionAgent
from src.agents.report_agent import ReportAgent
from src.agents.conv_agent import ConversationalAgent
from langgraph.graph import END, START, StateGraph

def route_from_planner(state: DeepAgentState) -> str:
    route = state["route"]
    print(f"🔀 LangGraph routing to: {route}")
    return route

# BUILD GRAPH
def build_graph():
    # AGENTS
    planner_agent=PlannerAgent()
    qa_agent=QAAgent()
    summary_agent=SummaryAgent()
    star_agent=StarAgent()
    extraction_agent=ExtractionAgent()
    report_agent=ReportAgent()
    conversational_agent=ConversationalAgent()

    # BUILD GRAPH
    workflow = StateGraph(DeepAgentState)
    
    # Nodes
    workflow.add_node("planner", planner_agent.run)
    workflow.add_node("conversational", conversational_agent.run)
    workflow.add_node("qa", qa_agent.run)
    workflow.add_node("summary", summary_agent.run)
    workflow.add_node("star", star_agent.run)
    workflow.add_node("extraction", extraction_agent.run)
    workflow.add_node("report", report_agent.run)

    # EDGES
    workflow.add_edge(START,"planner")
    workflow.add_conditional_edges(
        "planner",
        route_from_planner,
        {
            "conversational": "conversational",
            "qa": "qa",
            "summary": "summary",
            "star": "star",
            "extraction": "extraction",
            "report": "report",
        }
    )
    workflow.add_edge("conversational", END)
    workflow.add_edge("qa", END)
    workflow.add_edge("summary", END)
    workflow.add_edge("star", END)
    workflow.add_edge("extraction", END)
    workflow.add_edge("report", END)

    return workflow