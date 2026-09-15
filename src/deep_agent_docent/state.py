from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class DeepAgentState(TypedDict):
    """
    Shared state that travels through the DeepAgent graph.
    """
    messages: Annotated[Sequence[BaseMessage], add_messages]
    user_query: str
    documents: str
    route: str
    agent_result: dict