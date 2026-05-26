"""语音鉴伪研究工作流状态。"""

from typing import Annotated

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict


def merge_dicts(left: dict, right: dict) -> dict:
    """并行子 Agent 结果合并。"""
    merged = dict(left or {})
    merged.update(right or {})
    return merged


class ResearchState(TypedDict, total=False):
    """主控 / 子 Agent / 检查 Agent 共享状态。"""

    messages: Annotated[list[BaseMessage], add_messages]
    target_model: str
    user_brief: str
    innovation_directions: list[str]
    current_direction: str
    sub_task_results: Annotated[dict[str, str], merge_dicts]
    draft_plan: str
    review_feedback: str
    final_plan: str
    revision_round: int
    phase: str
    paper_context: str
    loaded_papers: list[str]
