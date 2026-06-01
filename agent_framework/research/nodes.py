"""主控 / 子 Agent / 检查 Agent 节点实现。"""

import json
import re

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from agent_framework.llm import create_llm
from agent_framework.research.prompts import (
    FOLLOW_UP_REVISE_PROMPT,
    ORCHESTRATOR_ANALYZE_PROMPT,
    ORCHESTRATOR_REVISE_PROMPT,
    ORCHESTRATOR_SYNTHESIZE_PROMPT,
    PLAN_FOLLOWUP_PROMPT,
    REVIEWER_PROMPT,
    SUB_AGENT_PROMPT,
)
from agent_framework.research.state import ResearchState, phase_update
from agent_framework.research.store import append_memory, sync_from_state_patch
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


def _session_id(state: ResearchState) -> str:
    return state.get("session_id") or state.get("run_id") or ""


def _sub_agent_send_payload(state: ResearchState, direction: str) -> dict:
    """并行子 Agent 需携带的上下文字段。"""
    return {
        "current_direction": direction,
        "target_model": state.get("target_model", ""),
        "session_id": _session_id(state),
        "user_brief": state.get("user_brief", ""),
        "paper_context": state.get("paper_context", ""),
        "loaded_papers": state.get("loaded_papers") or [],
        "follow_up_query": state.get("follow_up_query", ""),
        "phase": state.get("phase", ""),
    }


def _finalize(state: ResearchState, patch: dict) -> dict:
    """将节点补丁写入文件 Store。"""
    sid = _session_id(state) or patch.get("session_id", "")
    if sid:
        sync_from_state_patch(sid, patch)
    return patch


def _build_sub_agent_react():
    """子 Agent 内部 ReAct 子图（可调用研究工具）。"""
    tools = get_research_tools()
    llm = create_llm(role="sub_agent")
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
    llm = create_llm(role="orchestrator")
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
    n = len(directions)

    plan_announcement = (
        "## 执行计划（Plan Announcement）\n"
        f"1. **分析完成**：目标模型 `{target_model}`，拟定 **{n}** 个创新方向\n"
        f"2. **并行调研**：启动 {n} 个子 Agent（ReAct + 领域工具）\n"
        "3. **汇总初稿**：主控整合调研结果\n"
        "4. **批判审查**：检查 Agent 审查初稿\n"
        "5. **可选补充**：按审查意见追加调研\n"
        + (
            "6. **人工审批**：定稿前等待用户确认（已启用）\n"
            if state.get("human_review_enabled")
            else "6. **修订定稿**：主控输出最终方案\n"
        )
        + "7. **多轮续问**：追问将先调度子 Agent 工具查证，再修订方案（禁止主控直接改稿）\n"
    )

    analyze_artifact = json.dumps(
        {"target_model": target_model, "innovation_directions": directions},
        ensure_ascii=False,
        indent=2,
    )

    patch = {
        "target_model": target_model,
        "user_brief": user_brief,
        "innovation_directions": directions,
        "phase": "dispatch",
        "plan_announcement": plan_announcement,
        "messages": [
            AIMessage(
                content=(
                    f"【主控】已解析目标模型：{target_model}\n"
                    f"创新方向（{n} 项）：\n"
                    + "\n".join(f"- {d}" for d in directions)
                    + f"\n\n{plan_announcement}"
                )
            )
        ],
        **phase_update("analyze", "done", analyze_artifact),
        **phase_update("dispatch", "running" if n else "skipped"),
    }
    return _finalize(state, patch)


def sub_agent_research(state: ResearchState) -> dict:
    """对单一方向执行 ReAct 调研。"""
    direction = state.get("current_direction", "")
    target_model = state.get("target_model", "")
    follow_q = (state.get("follow_up_query") or "").strip()
    is_followup = state.get("phase") == "follow_up_dispatch" or bool(follow_q)
    app = get_sub_agent_app()

    if is_followup:
        task = (
            f"## 续问查证任务（必须先工具检索，禁止凭记忆编造 EER/SOTA）\n"
            f"- **用户追问**：{follow_q or '（见本方向）'}\n"
            f"- **目标模型**：{target_model}\n"
            f"- **查证方向**：{direction}\n"
            f"{_papers_tool_note(state)}\n\n"
            f"**必须**至少 2 次联网检索（含 2024/2025/2026 或 ASVspoof 5 关键词），"
            f"并调用 search_asvspoof2021_eer_research 与 search_deepfake_cross_domain。\n"
            f"输出 Markdown：针对用户追问的新证据、近年工作、可修订建议（注明来源，勿写无出处 EER）。"
        )
    else:
        task = (
            f"## 主控分发任务\n"
            f"- **目标模型**：{target_model}\n"
            f"- **本方向**：{direction}\n"
            f"{_papers_tool_note(state)}\n\n"
            f"请系统性调研该方向：优先对照用户提供的论文原文（若有）。\n"
            f"**必须**调用 search_asvspoof2021_eer_research（含近年关键词）与 search_deepfake_cross_domain。\n"
            f"禁止将 AASIST/RawNet 等经典基线当作当前 SOTA；EER 须有检索出处。\n"
            f"输出完整 Markdown 调研报告（含 LA/DF 线索与跨域迁移分析）。"
        )
    result = app.invoke({"messages": [HumanMessage(content=task)]})
    report = result["messages"][-1].content

    patch = {
        "sub_task_results": {direction: report},
        "messages": [
            AIMessage(content=f"【子 Agent · {direction}】调研完成（已写入 sub_task_results）")
        ],
        **phase_update("dispatch", "done", f"方向「{direction}」调研报告"),
    }
    return _finalize(state, patch)


def orchestrator_synthesize(state: ResearchState) -> dict:
    """汇总子 Agent 结果，撰写初稿方案。"""
    llm = create_llm(role="orchestrator")
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
    draft = response.content
    patch = {
        "draft_plan": draft,
        "phase": "review",
        "messages": [AIMessage(content="【主控】已完成方案初稿 synthesis")],
        **phase_update("synthesize", "done", draft),
        **phase_update("review", "running"),
    }
    return _finalize(state, patch)


def plan_supplement_research(state: ResearchState) -> dict:
    """根据检查意见判断是否需要子 Agent 补充调研。"""
    if state.get("skip_supplement"):
        patch = {
            "innovation_directions": [],
            "phase": "supplement",
            **phase_update("supplement", "skipped"),
        }
        return _finalize(state, patch)

    llm = create_llm(role="orchestrator")
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
    status = "running" if supplement else "skipped"
    patch = {
        "innovation_directions": supplement,
        "phase": "supplement",
        **phase_update("supplement", status),
    }
    return _finalize(state, patch)


def reviewer_critique(state: ResearchState) -> dict:
    """批判性审查初稿方案。"""
    llm = create_llm(role="reviewer")
    draft = state.get("draft_plan", "")
    prompt = (
        f"{REVIEWER_PROMPT}\n\n"
        f"## 待审查方案\n{draft}"
        f"{_paper_context_block(state)}"
    )
    response = llm.invoke([HumanMessage(content=prompt)])
    feedback = response.content
    patch = {
        "review_feedback": feedback,
        "revision_round": (state.get("revision_round") or 0) + 1,
        "phase": "pending_review" if state.get("human_review_enabled") else "revise",
        "messages": [AIMessage(content="【检查 Agent】已完成批判性审查")],
        **phase_update("review", "done", feedback),
    }
    if state.get("human_review_enabled"):
        patch.update(phase_update("human_review", "pending_review"))
    return _finalize(state, patch)


def human_review_gate(state: ResearchState) -> dict:
    """人工审批闸门：未启用时直通；启用时由 interrupt 注入 human_approved。"""
    if not state.get("human_review_enabled"):
        return _finalize(state, {**phase_update("human_review", "skipped")})

    approved = state.get("human_approved")
    if approved is True:
        sid = _session_id(state)
        if sid:
            append_memory(
                sid,
                {"type": "human_review", "decision": "approved"},
            )
        return _finalize(
            state,
            {
                "phase": "revise",
                **phase_update("human_review", "done", "用户已批准进入定稿"),
            },
        )

    if approved is False:
        notes = (state.get("human_review_notes") or "").strip()
        merged_review = state.get("review_feedback", "")
        if notes:
            merged_review = f"{merged_review}\n\n## 用户审查意见\n{notes}"
        sid = _session_id(state)
        if sid:
            append_memory(
                sid,
                {"type": "human_review", "decision": "rejected", "notes": notes},
            )
        return _finalize(
            state,
            {
                "review_feedback": merged_review,
                "phase": "revise",
                "human_approved": None,
                **phase_update("human_review", "done", notes or "用户要求修订"),
            },
        )

    # interrupt 恢复前不应执行到此处；兜底直通
    return _finalize(state, {**phase_update("human_review", "running")})


def orchestrator_revise(state: ResearchState) -> dict:
    """根据审查意见修订，输出最终方案。"""
    llm = create_llm(role="orchestrator")
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
    final = response.content
    patch = {
        "final_plan": final,
        "phase": "done",
        "user_followup": "",
        "messages": [
            AIMessage(content="【主控】已输出最终模型改进方案"),
            AIMessage(content=final),
        ],
        **phase_update("revise", "done", final),
    }
    return _finalize(state, patch)


def plan_followup_research(state: ResearchState) -> dict:
    """续问调度：拆解查证方向，后续由子 Agent 带工具检索。"""
    followup = (state.get("user_followup") or "").strip()
    if not followup and state.get("messages"):
        last = state["messages"][-1]
        followup = getattr(last, "content", str(last)).strip()

    llm = create_llm(role="orchestrator")
    prompt = (
        f"{PLAN_FOLLOWUP_PROMPT}\n\n"
        f"## 目标模型\n{state.get('target_model', '')}\n\n"
        f"## 当前最终方案（节选前 6000 字）\n{(state.get('final_plan') or '')[:6000]}\n\n"
        f"## 用户本轮追问/修订要求\n{followup}"
    )
    response = llm.invoke([HumanMessage(content=prompt)])
    try:
        parsed = _extract_json(response.content)
        directions = parsed.get("followup_directions") or []
        rationale = parsed.get("rationale", "")
    except (json.JSONDecodeError, TypeError):
        directions = []
        rationale = ""

    if not directions:
        directions = [f"针对用户追问的查证：{followup[:180]}"]
    directions = directions[:3]

    announce = (
        f"【主控·续问】已拆解 **{len(directions)}** 个查证方向，"
        f"将启动子 Agent（ReAct + 工具）检索后再修订方案。\n"
        f"说明：{rationale or '需工具查证后定稿'}\n"
        + "\n".join(f"- {d}" for d in directions)
    )

    patch = {
        "follow_up_query": followup,
        "innovation_directions": directions,
        "phase": "follow_up_dispatch",
        "user_followup": "",
        "messages": [AIMessage(content=announce)],
        **phase_update("follow_up_plan", "done", announce),
        **phase_update("follow_up_dispatch", "running"),
    }
    return _finalize(state, patch)


def follow_up_revise(state: ResearchState) -> dict:
    """续问定稿：仅在本轮子 Agent 查证完成后，汇总修订最终方案。"""
    followup = (state.get("follow_up_query") or "").strip()
    llm = create_llm(role="orchestrator")
    results = state.get("sub_task_results") or {}
    # 续问方向通常为新 key；全文传入供主控引用
    blocks = "\n\n---\n\n".join(
        f"### {direction}\n{content}" for direction, content in results.items()
    )
    prompt = (
        f"{FOLLOW_UP_REVISE_PROMPT}\n\n"
        f"## 目标模型\n{state.get('target_model', '')}\n\n"
        f"## 用户本轮追问\n{followup}\n\n"
        f"## 当前最终方案\n{state.get('final_plan', '')}\n\n"
        f"## 本轮续问子 Agent 调研（必须使用）\n{blocks or '（无，禁止定稿：应重新跑续问）'}\n\n"
        f"## 历史初稿/审查（备查）\n"
        f"初稿节选：{(state.get('draft_plan') or '')[:3000]}\n\n"
        f"审查节选：{(state.get('review_feedback') or '')[:2000]}"
    )
    response = llm.invoke([HumanMessage(content=prompt)])
    updated = response.content
    sid = _session_id(state)
    if sid:
        append_memory(
            sid,
            {"type": "follow_up", "user": followup, "revised": True},
        )

    patch = {
        "final_plan": updated,
        "phase": "done",
        "follow_up_query": "",
        "messages": [
            HumanMessage(content=followup),
            AIMessage(content=updated),
        ],
        **phase_update("follow_up_dispatch", "done"),
        **phase_update("follow_up", "done", updated),
    }
    return _finalize(state, patch)
