"""语音鉴伪多 Agent 研究工作流图。"""

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from agent_framework.research.nodes import (
    follow_up_revise,
    human_review_gate,
    orchestrator_analyze,
    orchestrator_revise,
    orchestrator_synthesize,
    plan_followup_research,
    plan_supplement_research,
    reviewer_critique,
    sub_agent_research,
    _sub_agent_send_payload,
)
from agent_framework.research.session import get_checkpointer
from agent_framework.research.state import ResearchState


def route_entry(state: ResearchState) -> str:
    """入口路由：新任务 / 多轮续问 / 人工审批恢复。"""
    followup = (state.get("user_followup") or "").strip()
    if state.get("phase") == "done" and followup:
        return "plan_followup"

    if state.get("human_review_enabled") and state.get("human_approved") is not None:
        if state.get("phase") in ("pending_review", "revise", "supplement"):
            return "human_review_gate"

    return "orchestrator_analyze"


def _dispatch_sub_agents(state: ResearchState):
    """主控分发：并行启动多个子 Agent。"""
    directions = state.get("innovation_directions") or []
    if not directions:
        return "orchestrator_synthesize"

    return [
        Send("sub_agent", _sub_agent_send_payload(state, direction))
        for direction in directions
    ]


def _route_after_sub_agent(state: ResearchState) -> str:
    """首轮调研后汇总；补充调研后进审批；续问查证后定稿。"""
    if state.get("phase") == "follow_up_dispatch":
        return "follow_up_revise"
    if state.get("phase") == "supplement":
        return "human_review_gate"
    return "orchestrator_synthesize"


def _dispatch_followup_research(state: ResearchState):
    """续问：并行子 Agent 工具查证。"""
    directions = state.get("innovation_directions") or []
    if not directions:
        return "follow_up_revise"
    return [Send("sub_agent", _sub_agent_send_payload(state, d)) for d in directions]


def _dispatch_supplement(state: ResearchState):
    """修订前按需补充子 Agent 调研。"""
    directions = state.get("innovation_directions") or []
    if not directions:
        return "human_review_gate"

    return [Send("sub_agent", _sub_agent_send_payload(state, d)) for d in directions]


def build_research_graph():
    """
    语音鉴伪研究工作流：

        START -> [route] orchestrator_analyze | plan_followup
              -> [并行] sub_agent x N
              -> orchestrator_synthesize -> reviewer
              -> plan_supplement -> [可选] sub_agent
              -> human_review_gate -> orchestrator_revise -> END
              plan_followup -> [并行] sub_agent -> follow_up_revise -> END
    """
    graph = StateGraph(ResearchState)

    graph.add_node("orchestrator_analyze", orchestrator_analyze)
    graph.add_node("sub_agent", sub_agent_research)
    graph.add_node("orchestrator_synthesize", orchestrator_synthesize)
    graph.add_node("reviewer", reviewer_critique)
    graph.add_node("plan_supplement", plan_supplement_research)
    graph.add_node("human_review_gate", human_review_gate)
    graph.add_node("orchestrator_revise", orchestrator_revise)
    graph.add_node("plan_followup", plan_followup_research)
    graph.add_node("follow_up_revise", follow_up_revise)

    graph.add_conditional_edges(
        START,
        route_entry,
        {
            "orchestrator_analyze": "orchestrator_analyze",
            "plan_followup": "plan_followup",
            "human_review_gate": "human_review_gate",
        },
    )
    graph.add_conditional_edges(
        "plan_followup",
        _dispatch_followup_research,
        ["sub_agent", "follow_up_revise"],
    )
    graph.add_edge("follow_up_revise", END)

    graph.add_conditional_edges(
        "orchestrator_analyze",
        _dispatch_sub_agents,
        ["sub_agent", "orchestrator_synthesize"],
    )
    graph.add_conditional_edges(
        "sub_agent",
        _route_after_sub_agent,
        ["orchestrator_synthesize", "human_review_gate", "follow_up_revise"],
    )
    graph.add_edge("orchestrator_synthesize", "reviewer")
    graph.add_edge("reviewer", "plan_supplement")
    graph.add_conditional_edges(
        "plan_supplement",
        _dispatch_supplement,
        ["sub_agent", "human_review_gate"],
    )
    graph.add_edge("human_review_gate", "orchestrator_revise")
    graph.add_edge("orchestrator_revise", END)

    return graph


def create_research_agent(
    *,
    enable_human_review: bool = False,
    checkpointer=None,
):
    """编译研究工作流；支持 Checkpointer 多轮续跑与可选人工审批。"""
    cp = checkpointer if checkpointer is not None else get_checkpointer()
    interrupt_before = ["human_review_gate"] if enable_human_review else []
    return build_research_graph().compile(
        checkpointer=cp,
        interrupt_before=interrupt_before,
    )
