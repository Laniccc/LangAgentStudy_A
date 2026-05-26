"""LangGraph 状态定义。"""

from typing import Annotated

from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

from langchain_core.messages import BaseMessage


class AgentState(TypedDict):
    """ReAct Agent 的消息状态。"""

    messages: Annotated[list[BaseMessage], add_messages]
