from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage

SYNTHESIS_PROMPT = SystemMessage("""
        You are an AI assistant helping to review and improve user-generated content. 
        Your task is to provide constructive suggestions that enhance clarity, correctness, and quality.

        User Output:
        {output}

        Context (if any):
        {context}

        Instructions:
        - Identify any errors, inconsistencies, or unclear parts.
        - Suggest improvements in a concise and actionable way.
        - Keep suggestions professional and easy to implement.
        - Provide examples if necessary.

        Return only the suggestion text without additional commentary."""
        )

class HITLAgent:
    def __init__(self, model_name="llama3.1"):
        # Use ChatOllama directly
        self.model = ChatOllama(model=model_name)
        print(f"Status: Ollama LLM initialized for HITLAgent: {model_name}")

    def _generate_ai_suggestion(self, user_output: str, context_text: str = "") -> str:
        # Format the prompt with actual text
        prompt = f""" {SYNTHESIS_PROMPT}\n\n Summarize the following context for the query: '{context_text}'\n\nContext:\n{user_output}\n\nSummary:"""
        # Call ChatOllama directly
        response = self.model.invoke([prompt])
        return response.content if hasattr(response, "content") else str(response)

    def validate(self, output: str, context_text: str = "") -> dict:
        suggestion = self._generate_ai_suggestion(output, context_text)
        final = output + "\n\n---\n### AI Suggestion:\n" + suggestion
        return {
            "validated": output,
            "ai_suggestion": suggestion,
            "final_output": final
        }

if __name__ == "__main__":
    hitl = HITLAgent()
    user_text = "I dont know how to explain this but the app just stop working suddenly."
    result = hitl.validate(user_text)
    print("Validated you query:", result["validated"])
    print("AI Suggestion:", result["ai_suggestion"])
    print("Final Output:", result["final_output"])
