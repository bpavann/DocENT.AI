from langchain_ollama import ChatOllama
from langchain_core.messages import AIMessage
from src.deep_agent_docent.state import DeepAgentState

class StarAgent:
    def __init__(self, llm_model: str = "llama3.1"):
        self.llm = ChatOllama(model=llm_model)
        print(f"Status Ollama LLM initialized: {self.llm}")

    def run(self, state: DeepAgentState) -> dict:
        query = state["user_query"]
        documents = state["documents"]
        if not documents:
            answer = "No relevant document context is available."
            return {
                "agent_results": [{"agent": "star","result": answer}],
                "messages": [AIMessage(content=answer)]
            }
        context = "\n\n".join(documents)
        prompt = f"""
            You are an expert analyst generating STAR-format insights
            (Situation, Task, Action, Result) from the provided documents.

            Analyze the content and identify relevant events, decisions,
            activities, or outcomes that can be expressed in STAR format.

            Your goal:
            - Identify the Situation
            - Identify the Task
            - Describe the Actions taken
            - Summarize the Result or outcome

            Rules:
            - Use only information from the provided document context.
            - Do not add outside knowledge.
            - Do not make assumptions.
            - Do not invent achievements, outcomes, or statistics.
            - If a STAR element is not supported by the documents,
            do not invent it.
            - Return the final answer only in STAR format.

            User Query:
            {query}

            Document Context:
            {context}

            Provide the final answer strictly in STAR format.
        """
        response = self.llm.invoke([prompt])
        answer = (
            response.content
            if hasattr(response, "content")
            else str(response)
        )
        return {
            "agent_results": [{"agent": "star","result": answer}],
            "messages": [AIMessage(content=answer)]
        }
            
        