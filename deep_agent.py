
import uuid
import os
from dotenv import load_dotenv
load_dotenv()
from typing import Annotated, Sequence, TypedDict, Optional
from enum import Enum
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_groq import ChatGroq
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.store.memory import InMemoryStore

try:
    from src.summary_agent import SummaryAgent
    from src.qa_agent import QAAgent
    from src.extraction_agent import ExtractionAgent
    from src.star_agent import StarAgent
    from src.report_agent import ReportAgent
except ImportError:
    import sys
    from pathlib import Path
    project_root = Path(__file__).parent.parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from src.summary_agent import SummaryAgent
    from src.qa_agent import QAAgent
    from src.report_agent import ReportAgent
    from src.extraction_agent import ExtractionAgent
    from src.star_agent import StarAgent

os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_PROJECT"]="DocENT.AI"

class SubAgentType(str, Enum):
    QA = "qa"
    SUMMARY = "summary"
    STAR = "star"
    EXTRACTION = "extraction"
    REPORT = "report"


class DeepAgentState(TypedDict):
    """Shared state across all agents and subagents."""
    messages: Annotated[Sequence[BaseMessage], add_messages]
    documents: str  # Retrieved from FAISS
    user_query: str
    current_subagent: Optional[SubAgentType]
    subagent_results: dict
    approval_status: dict
    final_report: str
    analysis_complete: bool
    required_agents: list


class SubAgentRunner:
    """Runs individual agents and returns results."""
    
    def __init__(self):
        self.qa = QAAgent()
        self.summary = SummaryAgent()
        self.star = StarAgent()
        self.extraction = ExtractionAgent()
        self.report = ReportAgent()
    
    def run_qa(self, query: str) -> str:
        """Run QA agent with vector search."""
        return self.qa.search_and_summarize(query, top_k=5)
    
    def run_summary(self, text: str) -> str:
        """Run Summary agent."""
        return self.summary.summarize(text)
    
    def run_star(self, text: str) -> str:
        """Run STAR analysis."""
        return self.star.generate_star(text)
    
    def run_extraction(self, text: str) -> str:
        """Run data extraction."""
        return self.extraction.extract_tables(text)
    
    def run_report(self, texts: list) -> str:
        """Generate final report from all results."""
        return self.report.generate_report(texts)


class DeepAgent:
    """
    Intelligent DeepAgent that routes to specific subagents based on user query.
    Documents are automatically retrieved from FAISS vector store.
    
    Usage:
        agent = DeepAgent()
        result = agent.invoke("Your query here")
    """
    
    def __init__(self, model_name: str = "qwen/qwen3-32b"):
        self.model_name = model_name
        self.llm = ChatGroq(model=model_name,temperature=0,max_retries=2)
        self.runner = SubAgentRunner()
        self.checkpointer = InMemorySaver()
        self.store = InMemoryStore()

        self.graph = self._build_graph()
        self.compiled_graph = self.graph.compile(
            checkpointer=self.checkpointer,
            store=self.store
        )
        print(f"Status Groq LLM initialized: {self.llm}")

    
    def analyze_query(self, user_query: str) -> list:
        """
        Analyze user query and determine which subagents to run.
        
        Args:
            user_query (str): The user's query
        
        Returns:
            List of SubAgentType enums to execute
        """
        query_lower = user_query.lower()
        required_agents = []
        
        # QA Keywords - Direct question answering
        qa_keywords = [
            "what", "who", "where", "when", "why", "how",
            "find", "search", "look for", "question", "ask",
            "specific", "detail", "information about", "tell me",
            "?", "answer", "explain", "define", "describe"
        ]
        
        # SUMMARY Keywords - Overview/summary requests
        summary_keywords = [
            "summary", "summarize", "overview", "brief", "short",
            "key points", "main", "highlights", "outline", "condensed",
            "gist", "essence", "recap", "digest"
        ]
        
        # STAR Keywords - Business/event analysis
        star_keywords = [
            "star", "situation", "task", "action", "result",
            "achieved", "accomplished", "success", "events",
            "decision", "problem", "solve", "overcome", "challenge",
            "situation and task"
        ]
        
        # EXTRACTION Keywords - Data/table extraction
        extraction_keywords = [
            "extract", "table", "data", "structured", "numbers",
            "metrics", "statistics", "values", "figures", "numeric",
            "list", "enumerate", "items", "columns", "rows"
        ]
        
        # REPORT Keywords - Comprehensive analysis
        report_keywords = [
            "report", "analysis", "comprehensive", "detailed", "full",
            "complete", "overall", "everything", "all", "total",
            "combine", "merge", "aggregate", "holistic"
        ]
        
        # Check for each agent type
        if any(keyword in query_lower for keyword in qa_keywords):
            required_agents.append(SubAgentType.QA)
        
        if any(keyword in query_lower for keyword in summary_keywords):
            required_agents.append(SubAgentType.SUMMARY)
        
        if any(keyword in query_lower for keyword in star_keywords):
            required_agents.append(SubAgentType.STAR)
        
        if any(keyword in query_lower for keyword in extraction_keywords):
            required_agents.append(SubAgentType.EXTRACTION)
        
        if any(keyword in query_lower for keyword in report_keywords):
            required_agents.append(SubAgentType.REPORT)
        
        # If no agents matched, default to QA + SUMMARY
        if not required_agents:
            required_agents = [SubAgentType.QA, SubAgentType.SUMMARY]
        
        return required_agents
    
    def _build_graph(self) -> StateGraph:
        workflow = StateGraph(DeepAgentState)

        # --- Node: Retrieve documents from FAISS and analyze query ---
        def retrieve_and_analyze(state: DeepAgentState) -> dict:
            user_query = state["user_query"]
            
            # Retrieve documents from FAISS using QA agent
            print(f"📚 Retrieving documents from FAISS for query: {user_query[:50]}...")
            retrieved_docs = self.runner.qa.vectorstore.query(user_query, top_k=10)
            
            # Extract text from retrieved documents
            documents = "\n\n".join([
                doc.get("metadata", {}).get("text", "")
                for doc in retrieved_docs if doc.get("metadata")
            ])
            
            if not documents:
                documents = "No relevant documents found in FAISS."
            
            # Analyze query to determine required agents
            required_agents = self.analyze_query(user_query)
            agent_names = ", ".join([a.value for a in required_agents])
            
            msg = AIMessage(
                content=f"📚 Retrieved {len(retrieved_docs)} documents from FAISS ({len(documents)} chars)\n"
                f"🤖 Running agents: {agent_names}"
            )
            
            return {
                "documents": documents,
                "messages": state["messages"] + [msg],
                "subagent_results": {},
                "approval_status": {},
                "required_agents": required_agents
            }

        workflow.add_node("retrieve_and_analyze", retrieve_and_analyze)

        # --- Node generator for subagents ---
        def make_subagent_node(subagent_name: SubAgentType, runner_func):
            def node(state: DeepAgentState) -> dict:
                # Skip if this agent is not required
                if subagent_name not in state.get("required_agents", []):
                    print(f"⏭️  Skipping {subagent_name.value} (not required)")
                    return {
                        "subagent_results": {**state["subagent_results"], subagent_name.value: None},
                        "approval_status": {**state["approval_status"], subagent_name.value: "skipped"}
                    }
                
                if not state["documents"]:
                    return {
                        "subagent_results": {**state["subagent_results"], subagent_name.value: None},
                        "approval_status": {**state["approval_status"], subagent_name.value: "skipped"}
                    }

                print(f"🤖 Running {subagent_name.value}...")
                
                # Run the appropriate agent
                if subagent_name == SubAgentType.QA:
                    result = runner_func(state["user_query"])
                else:
                    result = runner_func(state["documents"])

                return {
                    "current_subagent": subagent_name,
                    "subagent_results": {**state["subagent_results"], subagent_name.value: result},
                    "approval_status": {**state["approval_status"], subagent_name.value: "completed"},
                    "messages": state["messages"] + [AIMessage(content=f"✅ {subagent_name.value} completed")]
                }

            return node

        # Add QA, Summary, STAR, Extraction nodes
        workflow.add_node("qa_subagent", make_subagent_node(SubAgentType.QA, self.runner.run_qa))
        workflow.add_node("summary_subagent", make_subagent_node(SubAgentType.SUMMARY, self.runner.run_summary))
        workflow.add_node("star_subagent", make_subagent_node(SubAgentType.STAR, self.runner.run_star))
        workflow.add_node("extraction_subagent", make_subagent_node(SubAgentType.EXTRACTION, self.runner.run_extraction))

        # --- Node: report ---
        def report_node(state: DeepAgentState) -> dict:
            # Skip if report not required
            if SubAgentType.REPORT not in state.get("required_agents", []):
                print(f"⏭️  Skipping report (not required)")
                return {
                    "final_report": "",
                    "analysis_complete": True,
                    "approval_status": {**state["approval_status"], "report": "skipped"}
                }
            
            print("📋 Generating report...")
            
            approved_texts = [r for r in state["subagent_results"].values() if r]
            final_report = self.runner.run_report(approved_texts) if approved_texts else "No results to generate report."

            return {
                "current_subagent": SubAgentType.REPORT,
                "final_report": final_report,
                "analysis_complete": True,
                "approval_status": {**state["approval_status"], "report": "completed"},
                "messages": state["messages"] + [AIMessage(content="📋 Report generated")]
            }

        workflow.add_node("report_subagent", report_node)

        # --- EDGES ---
        workflow.add_edge(START, "retrieve_and_analyze")
        workflow.add_edge("retrieve_and_analyze", "qa_subagent")
        workflow.add_edge("qa_subagent", "summary_subagent")
        workflow.add_edge("summary_subagent", "star_subagent")
        workflow.add_edge("star_subagent", "extraction_subagent")
        workflow.add_edge("extraction_subagent", "report_subagent")
        workflow.add_edge("report_subagent", END)

        return workflow
    
    def invoke(self, user_query: str, thread_id: str = None) -> dict:
        """
        Run deep agent analysis with intelligent agent routing.
        Documents are automatically retrieved from FAISS.
        
        Args:
            user_query (str): The user's query/question
            thread_id (str): Optional thread ID for persistence
        
        Returns:
            dict: Analysis results with all subagent outputs
        """
        if thread_id is None:
            thread_id = str(uuid.uuid4())

        initial_state = {
            "messages": [HumanMessage(content=f"Query: {user_query}")],
            "documents": "",
            "user_query": user_query,
            "current_subagent": None,
            "subagent_results": {},
            "approval_status": {},
            "final_report": "",
            "analysis_complete": False,
            "required_agents": []
        }

        config = {"configurable": {"thread_id": thread_id}}

        try:
            result = self.compiled_graph.invoke(initial_state, config)
            print("\n✅ Analysis complete!")
            return result
        except Exception as e:
            print("❌ Error:", str(e))
            raise

    def get_thread_state(self, thread_id: str) -> dict:
        """Get the current state of a thread."""
        config = {"configurable": {"thread_id": thread_id}}
        return self.compiled_graph.get_state(config)

if __name__ == "__main__":
    agent = DeepAgent()
    result1 = agent.invoke("Explain about MCP?")
