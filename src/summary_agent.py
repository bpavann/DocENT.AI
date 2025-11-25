from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage

SUMMARY_PROMPT = SystemMessage("""
You are a highly intelligent document analyst. 
Read the document carefully and generate a detailed, context-aware executive summary. 
Focus on key insights, trends, anomalies, and actionable recommendations. 
Make sure your summary dominates the critical points and is concise yet thorough.
""")

class SummaryAgent:
    def __init__(self, model_name="llama3.1"):
        self.model = ChatOllama(model=model_name)
        print(f"Status: Ollama LLM initialized for SummaryAgent: {model_name}")

    def summarize(self, text: str) -> str:
        prompt = f"{SUMMARY_PROMPT}\n\n{text}"
        response = self.model.invoke([prompt])
        return response.content if hasattr(response, "content") else str(response)

# Example usage
if __name__ == "__main__":
    sa = SummaryAgent()
    text = "I want to develop and end to end project on Agentic AI "
    print("\n\n")
    print("Summary:\n", sa.summarize(text))