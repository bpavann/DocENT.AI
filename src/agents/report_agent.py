from langchain_ollama import ChatOllama
from langchain_core.messages import AIMessage
from src.deep_agent_docent.state import DeepAgentState

class ReportAgent:
    def __init__(self, llm_model: str = "llama3.1"):
        self.llm = ChatOllama(model=llm_model)
        print(f"Status Ollama LLM initialized: {self.llm}")

    def run(self, state: DeepAgentState) -> dict:
        query = state["user_query"]
        documents = state["documents"]
        agent_results = state["agent_results"]

        if not documents:
            answer = "No relevant document context is available."
            return {
                "agent_results": [{"agent": "report","result": answer}],
                "messages": [AIMessage(content=answer)]
            }

        context = "\n\n".join(documents)
        previous_results = "\n\n".join(
            f"{result['agent'].upper()}:\n{result['result']}"
            for result in agent_results
        )
        prompt = f"""
            You are an expert report-writing agent.

            Your job is to generate a professional, polished,
            multi-section report from the provided document context
            and previous agent outputs.

            The report MUST follow this structure:

            1. Executive Summary
            2. Key Insights & Findings
            3. STAR Analysis (Situation, Task, Action, Result)
            4. Comparative Observations (if relevant)
            5. Extracted Data Tables (if relevant)
            6. Risks & Mitigation Strategy
            7. Recommendations
            8. Final Conclusion

            Rules:
            - Use only information from the provided document context
            and previous agent outputs.
            - Do not add outside knowledge.
            - Do not make assumptions.
            - Do not invent facts, statistics, outcomes, or examples.
            - If a section is not supported by the available information,
            omit it or clearly state that the information is not available.
            - Keep the report accurate and relevant to the user's request.

            User Query:
            {query}

            Document Context:
            {context}

            Previous Agent Outputs:
            {previous_results}

            Generate the final professional report.
        """

        response = self.llm.invoke([prompt])
        answer = (
            response.content
            if hasattr(response, "content")
            else str(response)
        )

        return {
            "agent_results": [{"agent": "report","result": answer}],
            "messages": [AIMessage(content=answer)]
        }