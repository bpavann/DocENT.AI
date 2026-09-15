from langchain_ollama import ChatOllama
from src.deep_agent_docent.state import DeepAgentState

class PlannerAgent:
    def __init__(self,llm_model: str = "llama3.1"):
        self.llm = ChatOllama(model=llm_model)
        print(f"Status Ollama LLM initialized: {self.llm}")

    def run(self, state: DeepAgentState) -> dict:
            user_query = state["user_query"]
            print("🧠 Running Main LLM Planner...")
            prompt = f"""
                You are the routing controller for DocENTAI.
    
                DocENTAI is a personal document intelligence application.
                The user's uploaded knowledge base contains academic materials
                primarily related to:
    
                - Social Network
                - Social Network Analysis
                - Big Data
                - Big Data Analytics
                - Big Data Analysis
    
                The knowledge base may contain PDF, DOC, DOCX, HTML, TXT, CSV,
                and other supported document formats.
    
                Your job is to understand the user's intent and select exactly ONE agent.
    
                AVAILABLE AGENTS:
    
                conversational
                Use this ONLY for normal conversation that does not require
                the uploaded knowledge base.
    
                Examples:
                - hi
                - hello
                - how are you?
                - how are you doing?
                - who are you?
                - what can you do?
                - what are you here for?
                - what are you specialized for?
                - casual conversation
    
                qa
                Use this for questions that require knowledge from the uploaded
                documents or questions about the user's academic subject.
    
                Examples:
                - explain Social Network
                - explain Big Data Analytics
                - what is Social Network Analysis?
                - what are the characteristics of Big Data?
                - explain the different types of social networks
                - what does my document say about PageRank?
                - explain this topic from my documents
                - compare two concepts from my documents
    
                IMPORTANT:
                If the user asks about Social Network, Social Network Analysis,
                Big Data, Big Data Analytics, or Big Data Analysis, prefer QA
                unless the user explicitly requests a summary, extraction,
                STAR response, or report.
    
                summary
                Use this when the user explicitly asks to summarize information
                from the documents.
    
                Examples:
                - summarize this document
                - summarize Big Data Analytics
                - give me the key points
                - provide an overview of this document
    
                star
                Use this for STAR interview responses.
    
                Examples:
                - create a STAR answer
                - give me a Situation Task Action Result response
                - convert this experience into STAR format
    
                extraction
                Use this when the user wants specific structured information
                extracted from documents.
    
                Examples:
                - extract the table
                - extract all numbers
                - extract the metrics
                - extract the names
                - extract the structured fields
    
                report
                Use this when the user explicitly asks for a formal,
                detailed, or professional report.
    
                Examples:
                - create a report
                - generate a detailed report
                - prepare a report on Big Data Analytics
    
                IMPORTANT RULES:
    
                1. Understand the meaning of the complete user request.
    
                2. Do not choose conversational simply because the question
                is written casually.
    
                3. Questions about the user's academic subject should normally
                go to QA.
    
                4. "Explain", "what is", "how does", "why", and "describe"
                questions about the academic subject should normally go to QA.
    
                5. Only use summary when summarization is explicitly requested.
    
                6. Only use extraction when extraction is explicitly requested.
    
                7. Only use report when a report is explicitly requested.
    
                8. Return ONLY ONE route name.
    
                VALID ROUTES:
    
                conversational
                qa
                summary
                star
                extraction
                report
    
                USER REQUEST:
    
                {user_query}
    
                ROUTE:
            """
            response = self.llm.invoke(prompt)
            raw_route = response.content.strip().lower()
            print(f"🧠 Planner raw output: {raw_route}")
            allowed_routes = {"conversational","qa","summary","star","extraction","report"}
            if raw_route in allowed_routes:
                route = raw_route
            else:
                route = None
                for candidate in ["conversational","extraction","summary","report","star","qa"]:
                    if candidate in raw_route:
                        route = candidate
                        break
    
                if route is None:
                    print(f"⚠️ Invalid planner route: {raw_route}")
                    route = "conversational"
    
                else:
                    print(
                        f"⚠️ Planner returned explanation. "
                        f"Extracted route: {route}"
                    )
    
            print(f"🎯 Planner selected: {route}")
            return {
                "route": route
            }