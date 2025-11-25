from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage

STAR_PROMPT = SystemMessage("""
You are an expert business analyst generating STAR-format insights (Situation, Task, Action, Result) from documents. 
Analyze the content deeply, identify key events or decisions, and convert them into STAR points.
Provide actionable outcomes and emphasize the dominant achievements or decisions reflected in the text.
""")

class StarAgent:
    def __init__(self, model_name="llama3.1"):
        self.model = ChatOllama(model=model_name)
        print(f"Status: Ollama LLM initialized for SummaryAgent: {model_name}")

    def generate_star(self, text: str) -> str:
        prompt = f"{STAR_PROMPT}\n\n{text}"
        response = self.model.invoke([prompt])
        return response.content if hasattr(response, "content") else str(response)

# Example usage
if __name__ == "__main__":
    sa = StarAgent()
    text = "I want to develop and end to end project on Agentic AI "
    print("\n\n")
    print("STAR :\n", sa.generate_star(text))