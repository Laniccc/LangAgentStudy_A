"""语音鉴伪多 Agent 研究工作流图。"""

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from agent_framework.research.nodes import (
    build_review_context,
    followup_prepare,
    human_review_gate,
    orchestrator_analyze,
    orchestrator_finalize_followup,
    orchestrator_revise,
    orchestrator_synthesize,
    plan_supplement_research,
    prompt_input_enhance,
    reviewer_critique,
    sub_agent_research,
    _sub_agent_send_payload,
)
from agent_framework.research.session import get_checkpointer
from agent_framework.research.state import ResearchState

_REVIEW_RESUME_PHASES = ("pending_review", "revise", "supplement")


def route_entry(state: ResearchState) -> str:
    """入口路由：新任务 / 多轮续问（全图重研）/ 人工审批恢复。"""
    followup = (state.get("user_followup") or "").strip()
    if state.get("phase") == "done" and followup:
        return "followup_prepare"

    if state.get("human_review_enabled") and state.get("human_approved") is not None:
        if state.get("phase") in _REVIEW_RESUME_PHASES:
            return "human_review_gate"

    return "prompt_input_enhance"


def _dispatch_to_sub_agents(state: ResearchState, fallback_node: str):
    """并行分发子 Agent；无方向时回退到指定节点。"""
    directions = state.get("innovation_directions") or []
    if not directions:
        return fallback_node

    return [Send("sub_agent", _sub_agent_send_payload(state, direction)) for direction in directions]


def _dispatch_sub_agents(state: ResearchState):
    """主控分发：并行启动多个子 Agent。"""
    return _dispatch_to_sub_agents(state, "orchestrator_synthesize")


def _route_after_sub_agent(state: ResearchState) -> str:
    """首轮/续问调研后汇总；补充调研后进审批。"""
    if state.get("phase") == "supplement":
        return "human_review_gate"
    return "orchestrator_synthesize"


def _route_after_synthesize(state: ResearchState) -> str:
    """续问轮默认跳过检查 Agent；input.md 可开启续问审阅。"""
    if state.get("is_followup_round"):
        if state.get("followup_review_enabled"):
            return "review_context"
        return "orchestrator_finalize_followup"
    return "review_context"


def _dispatch_supplement(state: ResearchState):
    """修订前按需补充子 Agent 调研。"""
    return _dispatch_to_sub_agents(state, "human_review_gate")


def build_research_graph():
    """
    语音鉴伪研究工作流：

        START -> [route] prompt_input_enhance | followup_prepare
              -> orchestrator_analyze -> [并行] sub_agent x N
              -> orchestrator_synthesize -> review_context -> reviewer
              -> plan_supplement -> [可选] sub_agent
              -> human_review_gate -> orchestrator_revise -> END

        续问（phase=done + user_followup）：
              followup_prepare -> prompt_input_enhance -> orchestrator_analyze -> sub_agent x N
              -> orchestrator_synthesize
              -> [默认] orchestrator_finalize_followup -> END
              -> [审阅开启] review_context -> reviewer -> plan_supplement
                 -> human_review_gate -> orchestrator_revise -> END
    """
    graph = StateGraph(ResearchState)

    graph.add_node("orchestrator_analyze", orchestrator_analyze)
    graph.add_node("prompt_input_enhance", prompt_input_enhance)
    graph.add_node("followup_prepare", followup_prepare)
    graph.add_node("sub_agent", sub_agent_research)
    graph.add_node("orchestrator_synthesize", orchestrator_synthesize)
    graph.add_node("review_context", build_review_context)
    graph.add_node("reviewer", reviewer_critique)
    graph.add_node("plan_supplement", plan_supplement_research)
    graph.add_node("human_review_gate", human_review_gate)
    graph.add_node("orchestrator_revise", orchestrator_revise)
    graph.add_node("orchestrator_finalize_followup", orchestrator_finalize_followup)

    graph.add_conditional_edges(
        START,
        route_entry,
        {
            "prompt_input_enhance": "prompt_input_enhance",
            "followup_prepare": "followup_prepare",
            "human_review_gate": "human_review_gate",
        },
    )
    graph.add_edge("followup_prepare", "prompt_input_enhance")
    graph.add_edge("prompt_input_enhance", "orchestrator_analyze")

    graph.add_conditional_edges(
        "orchestrator_analyze",
        _dispatch_sub_agents,
        ["sub_agent", "orchestrator_synthesize"],
    )
    graph.add_conditional_edges(
        "sub_agent",
        _route_after_sub_agent,
        ["orchestrator_synthesize", "human_review_gate"],
    )
    graph.add_conditional_edges(
        "orchestrator_synthesize",
        _route_after_synthesize,
        ["review_context", "orchestrator_finalize_followup"],
    )
    graph.add_edge("orchestrator_finalize_followup", END)

    graph.add_edge("review_context", "reviewer")
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
