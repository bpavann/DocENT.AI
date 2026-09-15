from langchain_ollama import ChatOllama
from langchain_core.messages import AIMessage
from src.deep_agent_docent.state import DeepAgentState

class PresentationAgent:
    def __init__(self, llm_model: str = "llama3.1"):
        self.llm = ChatOllama(model=llm_model)
        print(f"Status Ollama LLM initialized: {self.llm}")

    def run(self, state: DeepAgentState) -> dict:
        query = state["user_query"]
        agent_results = state["agent_results"]
        documents = state["documents"]
        if not documents:
            answer = "No relevant document context is available."
            return {
                "agent_results": [{"agent": "present","result": answer}],
                "messages": [AIMessage(content=answer)]
            }
        context = "\n\n".join(documents)
        previous_results = "\n\n".join(
            f"{result['agent'].upper()}:\n{result['result']}"
            for result in agent_results
        )
        prompt = f"""
            You are a presentation and structuring specialist.

            Your job is to present information from the provided
            document context in a clear and easy-to-understand format.

            Use this agent when the user asks for:
            - Tables
            - Comparisons
            - Differences
            - Structured information
            - Organized results
            - Easy-to-understand presentation

            Rules:
            - Use only information from the provided document context.
            - Do not add outside knowledge.
            - Do not make assumptions.
            - Do not invent information.
            - Preserve the factual meaning of the source information.
            - Use a clean format appropriate for the user's request.

            User Query:
            {query}

            Document Context:
            {context}

            Previous Agent Outputs:
            {previous_results}

            Return only the requested structured presentation.
        """

        response = self.llm.invoke([prompt])
        answer = (
            response.content
            if hasattr(response, "content")
            else str(response)
        )

        return {
            "agent_results": [{"agent": "present","result": answer}],
            "messages": [AIMessage(content=answer)]
        }