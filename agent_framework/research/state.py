"""语音鉴伪研究工作流状态。"""

from typing import Annotated, Literal

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from typing_extensions import TypedDict

PhaseStatus = Literal[
    "pending",
    "running",
    "done",
    "pending_review",
    "skipped",
]

# 工作流阶段键（与 phase_status / phase_artifacts 对齐）
WORKFLOW_PHASES = (
    "analyze",
    "dispatch",
    "synthesize",
    "review",
    "supplement",
    "human_review",
    "revise",
    "follow_up_plan",
    "follow_up_dispatch",
    "follow_up",
)


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
    # --- 借鉴架构：阶段状态与产物 ---
    phase_status: Annotated[dict[str, PhaseStatus], merge_dicts]
    phase_artifacts: Annotated[dict[str, str], merge_dicts]
    plan_announcement: str
    # --- 会话 / Run ---
    session_id: str
    run_id: str
    # --- Human-in-the-loop ---
    human_review_enabled: bool
    human_approved: bool | None
    human_review_notes: str
    # --- 多轮续问（phase=done 后：plan_followup -> sub_agent -> follow_up_revise）---
    user_followup: str
    follow_up_query: str
    # --- 控制：跳过补充调研 ---
    skip_supplement: bool


def phase_update(
    phase: str,
    status: PhaseStatus,
    artifact: str | None = None,
) -> dict:
    """构造单阶段状态/产物补丁（供节点 return 合并）。"""
    patch: dict = {"phase_status": {phase: status}}
    if artifact is not None:
        # 避免单次写入过大
        patch["phase_artifacts"] = {phase: artifact[:12000]}
    return patch
