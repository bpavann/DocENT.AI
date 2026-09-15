from langchain_ollama import ChatOllama
from langchain_core.messages import AIMessage
from src.deep_agent_docent.state import DeepAgentState

class QAAgent:
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
                "agent_results": [{"agent": "qa","result": answer}],
                "messages": [
                    AIMessage(content=answer)
                ]
            }

        context = "\n\n".join(documents)
        previous_results = "\n\n".join(
            f"{result['agent'].upper()}:\n{result['result']}"
            for result in agent_results
        )
        prompt = f"""
            You are an advanced question-answering agent.

            Use the provided document context to answer the user query
            with accuracy and clarity.

            You may also use previous agent outputs when they are relevant
            to the user's request.

            Rules:
            - Base your answer strictly on the provided document context
            and relevant previous agent outputs.
            - Do not add outside knowledge.
            - Do not make assumptions.
            - Do not invent facts or information.
            - Keep the answer precise and directly relevant to the query.

            User Query:
            {query}

            Document Context:
            {context}

            Previous Agent Outputs:
            {previous_results}

            Provide the final answer below:
        """
        response = self.llm.invoke([prompt])
        answer = (
            response.content
            if hasattr(response, "content")
            else str(response)
        )
        return {
            "agent_results": [{"agent": "qa","result": answer}],
            "messages": [AIMessage(content=answer)]
        }