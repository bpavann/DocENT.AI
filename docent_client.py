import asyncio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from langchain_mcp_adapters.tools import load_mcp_tools
from langchain.agents import create_agent
from langchain_ollama import ChatOllama

server_params = StdioServerParameters(
    command="python",
    # Make sure to update to the full absolute path to your math_server.py file
    args=["/Users/pavankumarb/Documents/My Learning/DocENTmcp/docent_server.py"],
)
async def main():
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            # Initialize the connection
            await session.initialize()
            prompt = """
                    You are DocENT.AI — a professional ReAct agent for document processing, summarization, RAG QA, and structured analysis.

                    Responsibilities:
                    - Use MCP tools when they improve accuracy.
                    - Never call the `export_docent` tool; respond in text only.
                    - For summarization → `summarize`.
                    - For document QA → `rag_qa`.
                    - For structured insights → `insights` or `star`.
                    - For human-in-the-loop review → `hitl_validate`.
                    - For report generation → `report`.
                    - If no tool fits, answer directly using your reasoning.

                    Behavior:
                    - Think step-by-step internally (Reason → Act → Observe) but do not expose reasoning.
                    - Return valid tool JSON only when required; otherwise, plain text.
                    - Always provide factual, concise, professional answers.
                    - Never call the export_docent tool; respond in text only.
                    """

            tools = await load_mcp_tools(session)
            model=ChatOllama(model="llama3.1")
            agent=create_agent(model,tools,system_prompt=prompt)        

            response = await agent.ainvoke(
                {"messages": [{"role": "user", "content": "Develop and end-to-end ML project and summarize it"}]}
            )

            print("Docent response:", response['messages'][-1].content)


if __name__ == "__main__":
    asyncio.run(main())