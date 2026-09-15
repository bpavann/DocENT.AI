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
                You are the planning controller for DocENTAI.
                DocENTAI is a document intelligence application that works with
                uploaded academic documents.

                The main knowledge areas are:

                - Social Network
                - Social Network Analysis
                - Big Data
                - Big Data Analytics
                - Big Data Analysis

                Your job is to determine the sequence of specialized agents required
                to satisfy the COMPLETE user request.

                You are NOT answering the user.
                You are ONLY creating an execution plan.
                You must also decide whether document retrieval is required.

                RETRIEVAL RULES:
                - Set retrieval to YES when the request requires information
                from the uploaded academic documents or knowledge base.

                - Set retrieval to NO for simple conversation and basic questions
                about DocENTAI itself.

                - If any selected agent needs document knowledge, retrieval is YES.

                - Do NOT retrieve documents for simple greetings, farewells,
                or basic DocENTAI capability questions.

                AVAILABLE AGENTS:

                conversational
                Use for:
                - greetings
                - farewells
                - simple conversation
                - questions about DocENTAI itself
                - basic questions about what DocENTAI can do

                qa
                Use when the user wants an explanation or answer based on
                the uploaded academic documents.

                Examples:
                - What is PageRank?
                - Explain PySpark.
                - What is Social Network Analysis?
                - Explain Hadoop.
                - What does my document say about PageRank?

                summary
                Use when the user explicitly asks for:
                - a summary
                - key points
                - an overview
                - a concise explanation of retrieved material

                star
                Use when the user explicitly asks for:
                - STAR format
                - Situation, Task, Action, Result
                - an interview-style STAR response

                present
                Use when the user wants information to be presented
                in a structured or easy-to-understand format.

                Examples:
                - Show the differences in a table.
                - Compare PageRank and HITS in a table.
                - Put the key differences into a table.
                - Organize the information in a structured format.
                - Show the results in a table for better understanding.

                report
                Use when the user explicitly asks for:
                - a report
                - detailed report
                - formal report
                - professional report

                IMPORTANT PLANNING RULES:

                1. Understand the COMPLETE user request.

                2. You may select MORE THAN ONE agent.

                3. Preserve the order required by the user's request.

                4. If the user asks a knowledge question and then asks for a report,
                use QA first and Report second.

                5. If the user asks for an explanation, then a summary, then a report,
                use QA, Summary, and Report in that order.

                6. If the user explicitly requests multiple tasks in a specific order,
                follow that order exactly.
                
                7. Do NOT add Summary unless the user explicitly requests a summary
                or the requested task clearly requires summarization.

                8. Do NOT add Report unless the user explicitly requests a report.

                9. Do NOT add Extraction unless the user explicitly requests present.

                10. Do NOT add STAR unless the user explicitly requests STAR format.

                11. For a simple greeting, use only conversational.

                12. For a simple farewell, use only conversational.

                13. Do not use conversational together with knowledge agents.

                14. Do not duplicate agents.

                15. The plan must contain ONLY valid agent names.

                VALID AGENTS:

                conversational
                qa
                summary
                star
                present
                report

                EXAMPLES:
                User:
                Hi

                retrieval: no
                agents:
                conversational

                User:
                What is PageRank?

                retrieval: yes
                agents:
                qa
                
                User:
                Summarize PageRank, make a report, and show the differences
                between PageRank and HITS in a table.

                retrieval: yes
                agents:
                summary
                report
                present

                IMPORTANT OUTPUT FORMAT:
                Return exactly two sections.

                First line:
                retrieval: yes
                or
                retrieval: no

                Then:
                agents:

                Return exactly one agent name per line.

                Example:

                retrieval: yes
                agents:
                summary
                report

                Do NOT provide explanations.
                Do NOT provide bullet points.
                Do NOT provide markdown.
                Do NOT write anything before or after the required output.

                USER REQUEST:

                {user_query}
            """
            response = self.llm.invoke(prompt)
            raw_output = response.content.strip().lower()
            print(f"🧠 Planner raw output: {raw_output}")
            allowed_agents = {"conversational","qa","summary","star","present","report"}
            plan = []
            retrieval_required = False
            for line in raw_output.splitlines():
                line = line.strip()
                if line == "retrieval: yes":
                    retrieval_required = True
                elif line == "retrieval: no":
                    retrieval_required = False
                elif line in allowed_agents and line not in plan:
                    plan.append(line)

            if not plan:
                print("⚠️ Planner returned an invalid plan.")
                plan = ["conversational"]
                retrieval_required = False

            if plan == ["conversational"]:
                retrieval_required = False
            print(f"🎯 Planner selected plan: {plan}")
            return {
                "plan": plan,
                "retrieval_required": retrieval_required,
                "current_step": 0
            }
