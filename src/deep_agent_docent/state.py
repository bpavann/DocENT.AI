from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

def add_agent_results(existing: list[dict],new: list[dict]) -> list[dict]:
    return existing + new

class DeepAgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    user_query: str
    retrieval_required: bool
    documents: list[str]
    plan: list[str]
    current_step: int
    agent_results: Annotated[list[dict], add_agent_results]