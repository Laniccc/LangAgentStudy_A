"""主控 / 子 Agent / 检查 Agent 节点实现。"""

import json
import re

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from agent_framework.llm import create_llm
from agent_framework.research.prompts import (
    ORCHESTRATOR_ANALYZE_PROMPT,
    ORCHESTRATOR_REVISE_PROMPT,
    ORCHESTRATOR_SYNTHESIZE_PROMPT,
    REVIEWER_PROMPT,
    SUB_AGENT_PROMPT,
)
from agent_framework.research.state import ResearchState
from agent_framework.research.tools import get_research_tools
from agent_framework.state import AgentState

_SUB_AGENT_APP = None


def reset_sub_agent_app() -> None:
    """论文库更新后重建子 Agent ReAct 图。"""
    global _SUB_AGENT_APP
    _SUB_AGENT_APP = None


def _paper_context_block(state: ResearchState) -> str:
    ctx = state.get("paper_context") or ""
    return f"\n\n{ctx}" if ctx else ""


def _papers_tool_note(state: ResearchState) -> str:
    names = state.get("loaded_papers") or []
    if not names:
        return ""
    return (
        "\n\n## 用户提供的论文 PDF（已加载）\n"
        f"文件：{', '.join(names)}\n"
        "请使用 `read_loaded_paper` 工具按文件名读取原文（可用 section_hint 定位方法/实验等）。"
    )


def _extract_json(text: str) -> dict:
    """从 LLM 输出中解析 JSON。"""
    text = text.strip()
    fence = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if fence:
        text = fence.group(1)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{[\s\S]*\}", text)
        if match:
            return json.loads(match.group())
        raise


def _build_sub_agent_react():
    """子 Agent 内部 ReAct 子图（可调用研究工具）。"""
    tools = get_research_tools()
    llm = create_llm()
    llm_with_tools = llm.bind_tools(tools)

    def agent_node(state: dict) -> dict:
        messages = state["messages"]
        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=SUB_AGENT_PROMPT), *messages]
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(tools))
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", tools_condition)
    graph.add_edge("tools", "agent")
    return graph.compile()


def get_sub_agent_app():
    global _SUB_AGENT_APP
    if _SUB_AGENT_APP is None:
        _SUB_AGENT_APP = _build_sub_agent_react()
    return _SUB_AGENT_APP


def orchestrator_analyze(state: ResearchState) -> dict:
    """分析目标模型，产出创新方向列表。"""
    llm = create_llm()
    user_brief = state.get("user_brief") or ""
    if not user_brief and state.get("messages"):
        last = state["messages"][-1]
        user_brief = getattr(last, "content", str(last))

    prompt = (
        f"{ORCHESTRATOR_ANALYZE_PROMPT}\n\n"
        f"## 用户研究任务\n{user_brief}"
        f"{_paper_context_block(state)}"
    )
    response = llm.invoke([HumanMessage(content=prompt)])
    parsed = _extract_json(response.content)

    directions = parsed.get("innovation_directions") or []
    target_model = parsed.get("target_model") or "未指定模型"

    return {
        "target_model": target_model,
        "user_brief": user_brief,
        "innovation_directions": directions,
        "phase": "dispatch",
        "messages": [
            AIMessage(
                content=(
                    f"【主控】已解析目标模型：{target_model}\n"
                    f"创新方向（{len(directions)} 项）：\n"
                    + "\n".join(f"- {d}" for d in directions)
                )
            )
        ],
    }


def sub_agent_research(state: ResearchState) -> dict:
    """对单一方向执行 ReAct 调研。"""
    direction = state.get("current_direction", "")
    target_model = state.get("target_model", "")
    app = get_sub_agent_app()

    task = (
        f"## 主控分发任务\n"
        f"- **目标模型**：{target_model}\n"
        f"- **本方向**：{direction}\n"
        f"{_papers_tool_note(state)}\n\n"
        f"请系统性调研该方向：优先对照用户提供的论文原文（若有），"
        f"并结合工具检索其他开源成果，输出完整 Markdown 调研报告。"
    )
    result = app.invoke({"messages": [HumanMessage(content=task)]})
    report = result["messages"][-1].content

    return {
        "sub_task_results": {direction: report},
        "messages": [
            AIMessage(content=f"【子 Agent · {direction}】调研完成（已写入 sub_task_results）")
        ],
    }


def orchestrator_synthesize(state: ResearchState) -> dict:
    """汇总子 Agent 结果，撰写初稿方案。"""
    llm = create_llm()
    results = state.get("sub_task_results") or {}
    blocks = "\n\n---\n\n".join(
        f"### 方向：{direction}\n{content}" for direction, content in results.items()
    )
    prompt = (
        f"{ORCHESTRATOR_SYNTHESIZE_PROMPT}\n\n"
        f"## 目标模型\n{state.get('target_model', '')}\n\n"
        f"## 用户诉求\n{state.get('user_brief', '')}\n"
        f"{_paper_context_block(state)}\n\n"
        f"## 子 Agent 调研报告\n{blocks or '（无子任务结果）'}"
    )
    response = llm.invoke([HumanMessage(content=prompt)])
    return {
        "draft_plan": response.content,
        "phase": "review",
        "messages": [AIMessage(content="【主控】已完成方案初稿 synthesis")],
    }


def plan_supplement_research(state: ResearchState) -> dict:
    """根据检查意见判断是否需要子 Agent 补充调研。"""
    llm = create_llm()
    review = state.get("review_feedback", "")
    existing = set((state.get("sub_task_results") or {}).keys())

    prompt = (
        "你是主控 Agent 的调度模块。根据检查 Agent 的审查意见，判断是否需要**额外**调研方向。\n"
        "若审查中要求补充证据、补充某技术路线调研，则输出新方向（勿与已有方向重复）。\n"
        "若初稿已足够修订，则输出空列表。\n\n"
        f"## 已有调研方向\n{list(existing)}\n\n"
        f"## 审查意见\n{review}\n\n"
        '只输出 JSON：{"supplement_directions": ["方向A", ...]}'
    )
    response = llm.invoke([HumanMessage(content=prompt)])
    try:
        parsed = _extract_json(response.content)
        raw = parsed.get("supplement_directions") or []
    except (json.JSONDecodeError, TypeError):
        raw = []

    supplement = [d for d in raw if d and d not in existing]
    return {"innovation_directions": supplement, "phase": "supplement"}


def reviewer_critique(state: ResearchState) -> dict:
    """批判性审查初稿方案。"""
    llm = create_llm()
    draft = state.get("draft_plan", "")
    prompt = (
        f"{REVIEWER_PROMPT}\n\n"
        f"## 待审查方案\n{draft}"
        f"{_paper_context_block(state)}"
    )
    response = llm.invoke([HumanMessage(content=prompt)])
    return {
        "review_feedback": response.content,
        "revision_round": (state.get("revision_round") or 0) + 1,
        "phase": "revise",
        "messages": [AIMessage(content="【检查 Agent】已完成批判性审查")],
    }


def orchestrator_revise(state: ResearchState) -> dict:
    """根据审查意见修订，输出最终方案。"""
    llm = create_llm()
    results = state.get("sub_task_results") or {}
    blocks = "\n\n".join(
        f"### {direction}\n{content}" for direction, content in results.items()
    )
    prompt = (
        f"{ORCHESTRATOR_REVISE_PROMPT}\n\n"
        f"## 目标模型\n{state.get('target_model', '')}\n\n"
        f"## 初稿方案\n{state.get('draft_plan', '')}\n\n"
        f"## 检查 Agent 意见\n{state.get('review_feedback', '')}\n\n"
        f"## 子 Agent 调研（备查）\n{blocks}"
        f"{_paper_context_block(state)}"
    )
    response = llm.invoke([HumanMessage(content=prompt)])
    return {
        "final_plan": response.content,
        "phase": "done",
        "messages": [
            AIMessage(content="【主控】已输出最终模型改进方案"),
            AIMessage(content=response.content),
        ],
    }
