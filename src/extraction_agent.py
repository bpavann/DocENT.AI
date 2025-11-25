from langchain_ollama import ChatOllama
from langchain_core.messages import SystemMessage

TABLE_EXTRACTION_PROMPT = SystemMessage("""
You are a data extraction specialist. 
Extract all tables, numeric data, and structured information from the document.
Ensure completeness and highlight any anomalies, dominant trends, or patterns in the extracted data.
Return results in a structured format ready for analysis or export.
""")

class ExtractionAgent:
    def __init__(self, model_name: str = "llama3.1"):
        # Use ChatOllama directly
        self.model = ChatOllama(model=model_name)
        print(f"[INFO] Ollama LLM initialized for ExtractionAgent: {model_name}")

    def extract_tables(self, text: str) -> str:
        # Prepare the prompt for the LLM
        prompt = f"{TABLE_EXTRACTION_PROMPT}\n\nDocument:\n{text}"
        # Call the LLM
        response = self.model.invoke([prompt])
        return response.content if hasattr(response, "content") else str(response)

# Example usage
if __name__ == "__main__":
    extractor = ExtractionAgent()
    sample_text = """
    Sales report Q3:
    | Product | Units Sold | Revenue |
    |---------|------------|---------|
    | A       | 150        | 3000    |
    | B       | 200        | 5000    |
    """
    extracted = extractor.extract_tables(sample_text)
    print("Extracted Tables / Data:\n", extracted)
