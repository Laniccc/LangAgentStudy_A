"""语音鉴伪多 Agent 研究工作流图。"""

from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from agent_framework.research.nodes import (
    orchestrator_analyze,
    orchestrator_revise,
    orchestrator_synthesize,
    plan_supplement_research,
    reviewer_critique,
    sub_agent_research,
)
from agent_framework.research.state import ResearchState


def _dispatch_sub_agents(state: ResearchState):
    """主控分发：并行启动多个子 Agent。"""
    directions = state.get("innovation_directions") or []
    if not directions:
        return "orchestrator_synthesize"

    target_model = state.get("target_model", "")
    return [
        Send(
            "sub_agent",
            {
                "current_direction": direction,
                "target_model": target_model,
            },
        )
        for direction in directions
    ]


def _route_after_sub_agent(state: ResearchState) -> str:
    """首轮调研后汇总；补充调研后直接进入修订。"""
    if state.get("phase") == "supplement":
        return "orchestrator_revise"
    return "orchestrator_synthesize"


def _dispatch_supplement(state: ResearchState):
    """修订前按需补充子 Agent 调研。"""
    directions = state.get("innovation_directions") or []
    if not directions:
        return "orchestrator_revise"

    target_model = state.get("target_model", "")
    return [
        Send(
            "sub_agent",
            {"current_direction": d, "target_model": target_model},
        )
        for d in directions
    ]


def build_research_graph():
    """
    语音鉴伪研究工作流：

        START -> orchestrator_analyze
              -> [并行] sub_agent (ReAct + tools) x N
              -> orchestrator_synthesize -> reviewer
              -> plan_supplement -> [可选并行] sub_agent
              -> orchestrator_revise -> END
    """
    graph = StateGraph(ResearchState)

    graph.add_node("orchestrator_analyze", orchestrator_analyze)
    graph.add_node("sub_agent", sub_agent_research)
    graph.add_node("orchestrator_synthesize", orchestrator_synthesize)
    graph.add_node("reviewer", reviewer_critique)
    graph.add_node("plan_supplement", plan_supplement_research)
    graph.add_node("orchestrator_revise", orchestrator_revise)

    graph.add_edge(START, "orchestrator_analyze")
    graph.add_conditional_edges(
        "orchestrator_analyze",
        _dispatch_sub_agents,
        ["sub_agent", "orchestrator_synthesize"],
    )
    graph.add_conditional_edges(
        "sub_agent",
        _route_after_sub_agent,
        ["orchestrator_synthesize", "orchestrator_revise"],
    )
    graph.add_edge("orchestrator_synthesize", "reviewer")
    graph.add_edge("reviewer", "plan_supplement")
    graph.add_conditional_edges(
        "plan_supplement",
        _dispatch_supplement,
        ["sub_agent", "orchestrator_revise"],
    )
    graph.add_edge("orchestrator_revise", END)

    return graph.compile()


def create_research_agent():
    """编译并返回语音鉴伪研究工作流应用。"""
    return build_research_graph()
