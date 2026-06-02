"""主控 / 子 Agent / 检查 Agent 节点实现。"""

import json
import re

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

from agent_framework.llm import create_llm
from agent_framework.research.memory import (
    append_structured_memory,
    extract_plan_modules,
    retrieve_memory_view,
)
from agent_framework.research.prompts import (
    ORCHESTRATOR_ANALYZE_PROMPT,
    ORCHESTRATOR_FOLLOWUP_ANALYZE_NOTE,
    ORCHESTRATOR_FOLLOWUP_FINALIZE_PROMPT,
    ORCHESTRATOR_FOLLOWUP_SYNTHESIZE_PROMPT,
    ORCHESTRATOR_REVISE_PROMPT,
    ORCHESTRATOR_SYNTHESIZE_PROMPT,
    PROMPT_AGENT_PROMPT,
    REVIEWER_REACT_PROMPT,
    SUB_AGENT_PROMPT,
)
from agent_framework.research.output_utils import clean_user_markdown
from agent_framework.research.schemas import (
    build_review_context_packet,
    dumps_compact,
    extract_json,
    parse_sub_agent_json,
    render_draft_markdown,
    render_review_markdown,
    sub_agent_json_to_brief,
)
from agent_framework.research.state import ResearchState, phase_update
from agent_framework.research.store import append_memory, sync_from_state_patch
from agent_framework.research.tools import get_research_tools, get_reviewer_tools, search_loaded_paper_vectors
from agent_framework.state import AgentState

_SUB_AGENT_APP = None
_REVIEWER_APP = None
_MAX_ANALYZE_DIRECTIONS = 3
_MAX_SUPPLEMENT_DIRECTIONS = 2
_MAX_FOLLOWUP_PLAN_CHARS = 3600
_MAX_PAPER_EXCERPT_CHARS = 2600


def _clip_text(text: str, max_len: int) -> str:
    raw = (text or "").strip()
    if len(raw) <= max_len:
        return raw
    return raw[:max_len] + "\n…(truncated)"


def _extract_keywords(text: str, *, max_terms: int = 8) -> list[str]:
    tokens = re.findall(r"[A-Za-z0-9_\-+/]{3,}|[\u4e00-\u9fff]{2,}", (text or "").lower())
    # 去重并保序
    out: list[str] = []
    seen: set[str] = set()
    for t in tokens:
        if t in seen:
            continue
        seen.add(t)
        out.append(t)
        if len(out) >= max_terms:
            break
    return out


def _select_relevant_excerpt(
    text: str,
    *,
    query: str,
    max_chars: int,
    chunk_chars: int = 520,
    max_chunks: int = 5,
) -> str:
    raw = (text or "").strip()
    if not raw:
        return ""
    if len(raw) <= max_chars:
        return raw

    keywords = _extract_keywords(query)
    chunks = [c.strip() for c in re.split(r"\n\s*\n", raw) if c.strip()]
    scored: list[tuple[int, int, str]] = []
    for idx, chunk in enumerate(chunks):
        snippet = chunk[:chunk_chars]
        low = snippet.lower()
        score = sum(low.count(k) for k in keywords) if keywords else 0
        scored.append((score, -idx, snippet))
    scored.sort(reverse=True)

    picked: list[str] = []
    total = 0
    for score, _, snippet in scored:
        if score == 0 and picked:
            continue
        if not snippet:
            continue
        if total + len(snippet) > max_chars:
            remain = max_chars - total
            if remain > 120:
                picked.append(snippet[:remain] + "…")
            break
        picked.append(snippet)
        total += len(snippet)
        if len(picked) >= max_chunks:
            break

    if not picked:
        return _clip_text(raw, max_chars)
    return "\n\n".join(picked)


def _compact_plan_for_followup(plan_text: str, *, follow_up_query: str) -> str:
    plan = (plan_text or "").strip()
    if not plan:
        return ""
    headings = [ln.strip() for ln in plan.splitlines() if ln.strip().startswith("#")][:8]
    outline = "\n".join(f"- {h}" for h in headings) if headings else ""
    excerpt = _select_relevant_excerpt(
        plan,
        query=follow_up_query,
        max_chars=_MAX_FOLLOWUP_PLAN_CHARS,
    )
    if outline:
        return f"## 首轮方案目录\n{outline}\n\n## 与本轮追问最相关片段\n{excerpt}"
    return excerpt


def _infer_followup_intent(raw_input: str) -> dict:
    """用启发式规则兜底识别简略续问背后的操作意图。"""
    text = (raw_input or "").lower()
    critical_terms = (
        "为什么不",
        "是否合理",
        "合不合理",
        "论证",
        "评估",
        "可行吗",
        "有必要",
        "不要直接",
        "不是直接",
        "反方",
        "质疑",
        "缺点",
        "风险",
    )
    structural_terms = (
        "重新",
        "大幅",
        "推翻",
        "不满意",
        "重排",
        "调整方案",
        "结构性",
    )
    if any(term in text for term in structural_terms):
        return {
            "intent": "structural_revision",
            "change_strength": "high",
            "required_actions": [
                "重新评估上一版核心结论",
                "重排 P0/P1/P2 优先级",
                "明确旧方案中哪些内容被保留、降级、替换或删除",
                "输出相比上一版的结构性变化摘要",
            ],
        }
    if any(term in text for term in critical_terms):
        return {
            "intent": "critical_reevaluation",
            "change_strength": "medium",
            "required_actions": [
                "不要默认采纳或保留被讨论方案",
                "先以反方评审方式列出风险和反证",
                "比较收益、计算成本、训练复杂度、特征冗余、过拟合和跨域泛化风险",
                "根据证据明确给出保留、降级、替换或删除结论",
            ],
        }
    return {
        "intent": "general_followup",
        "change_strength": "low",
        "required_actions": [],
    }


def _append_intent_guardrails(enhanced: str, intent: dict, *, is_followup: bool) -> str:
    if not is_followup:
        return enhanced
    actions = [str(a).strip() for a in intent.get("required_actions") or [] if str(a).strip()]
    if not actions:
        return enhanced
    block = "\n".join(f"- {action}" for action in actions)
    return (
        f"{enhanced}\n\n"
        "## 提示词 Agent 推断的隐含意图与强制执行要求\n"
        f"- 意图类别：{intent.get('intent', 'general_followup')}\n"
        f"- 修改强度：{intent.get('change_strength', 'low')}\n"
        "- 主控必须执行：\n"
        f"{block}\n"
        "- 若最终仍保留上一版方案，必须说明保留依据；若证据不足，必须降级为备选而不是继续作为 P0。"
    )


def _build_react_app(*, role: str, tools: list, system_prompt: str):
    """构建通用 ReAct 子图（agent + tools 循环）。"""
    llm = create_llm(role=role)
    llm_with_tools = llm.bind_tools(tools)

    def agent_node(state: dict) -> dict:
        messages = state["messages"]
        if not messages or not isinstance(messages[0], SystemMessage):
            messages = [SystemMessage(content=system_prompt), *messages]
        response = llm_with_tools.invoke(messages)
        return {"messages": [response]}

    graph = StateGraph(AgentState)
    graph.add_node("agent", agent_node)
    graph.add_node("tools", ToolNode(tools))
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", tools_condition)
    graph.add_edge("tools", "agent")
    return graph.compile()


def _invoke_json_with_prompt(*, role: str, prompt: str, payload: dict) -> dict:
    """统一调用 LLM 并解析 JSON。"""
    llm = create_llm(role=role)
    body = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    response = llm.invoke([HumanMessage(content=f"{prompt}\n\n{body}")])
    return extract_json(response.content)


def _invoke_json_with_tools(*, role: str, prompt: str, payload: dict, tools: list, max_rounds: int = 2) -> dict:
    """允许少量工具调用后，仍要求模型最终输出 JSON。"""
    tool_map = {tool.name: tool for tool in tools}
    llm = create_llm(role=role).bind_tools(tools)
    body = json.dumps(payload, ensure_ascii=False, sort_keys=True)
    messages = [HumanMessage(content=f"{prompt}\n\n{body}")]

    for _ in range(max_rounds):
        response = llm.invoke(messages)
        messages.append(response)
        tool_calls = getattr(response, "tool_calls", None) or []
        if not tool_calls:
            return extract_json(response.content)
        for call in tool_calls:
            name = call.get("name", "")
            args = call.get("args") or {}
            tool = tool_map.get(name)
            if tool is None:
                content = f"未知工具：{name}"
            else:
                try:
                    content = tool.invoke(args)
                except Exception as e:
                    content = f"工具调用失败：{e}"
            messages.append(
                ToolMessage(
                    content=str(content),
                    tool_call_id=call.get("id", name),
                )
            )

    final_prompt = "请基于以上用户输入与工具结果，只输出符合要求的 JSON。"
    response = create_llm(role=role).invoke([*messages, HumanMessage(content=final_prompt)])
    return extract_json(response.content)


def reset_sub_agent_app() -> None:
    global _SUB_AGENT_APP
    _SUB_AGENT_APP = None


def reset_reviewer_app() -> None:
    global _REVIEWER_APP
    _REVIEWER_APP = None


def reset_research_react_apps() -> None:
    reset_sub_agent_app()
    reset_reviewer_app()


def _session_id(state: ResearchState) -> str:
    return state.get("session_id") or state.get("run_id") or ""


def _memory_view(state: ResearchState, *, query: str, role: str, top_k: int = 5) -> str:
    sid = _session_id(state)
    if not sid:
        return ""
    return retrieve_memory_view(sid, query=query, role=role, top_k=top_k)


def _remember(
    state: ResearchState,
    *,
    memory_type: str,
    text: str,
    source: str,
    tags: list[str] | tuple[str, ...] | None = None,
    metadata: dict | None = None,
) -> None:
    sid = _session_id(state)
    if not sid:
        return
    append_structured_memory(
        sid,
        memory_type=memory_type,
        text=text,
        source=source,
        tags=tags,
        metadata=metadata,
    )


def _finalize(state: ResearchState, patch: dict) -> dict:
    sid = _session_id(state) or patch.get("session_id", "")
    if sid:
        sync_from_state_patch(sid, patch)
    return patch


def _followup_round_result_key(direction: str) -> str:
    return f"续问轮|{direction}"


def _collect_sub_briefs(state: ResearchState, *, round: str = "auto") -> dict:
    """收集子 Agent 摘要。round: auto | first | followup"""
    results = state.get("sub_task_results") or {}
    if round == "auto":
        round = "followup" if state.get("is_followup_round") else "first"
    briefs: dict = {}
    for k, v in results.items():
        is_fu_key = k.startswith("续问轮|") or k.startswith("续问|")
        if round == "followup" and not k.startswith("续问轮|"):
            continue
        if round == "first" and is_fu_key:
            continue
        briefs[k] = sub_agent_json_to_brief(parse_sub_agent_json(v))
    return briefs


def _first_round_context_packet(state: ResearchState) -> dict:
    """续问主控/子 Agent 共用的首轮上下文包。"""
    follow_up_query = state.get("follow_up_query") or ""
    compact_prior = _compact_plan_for_followup(
        state.get("prior_final_plan") or "",
        follow_up_query=follow_up_query,
    )
    return {
        "first_round_user_brief": _clip_text(state.get("user_brief") or "", 1800),
        "first_round_final_plan": compact_prior,
        "first_round_sub_briefs": _collect_sub_briefs(state, round="first"),
        "first_round_draft_excerpt": dumps_compact(state.get("draft_plan_json") or {}, 4000),
    }


def _sub_agent_send_payload(state: ResearchState, direction: str) -> dict:
    return {
        "current_direction": direction,
        "target_model": state.get("target_model", ""),
        "session_id": _session_id(state),
        "user_brief": state.get("user_brief", ""),
        "loaded_papers": state.get("loaded_papers") or [],
        "follow_up_query": state.get("follow_up_query", ""),
        "is_followup_round": state.get("is_followup_round", False),
        "prior_final_plan": state.get("prior_final_plan", ""),
        "phase": state.get("phase", ""),
    }


def _build_sub_agent_react():
    return _build_react_app(
        role="sub_agent",
        tools=get_research_tools(),
        system_prompt=SUB_AGENT_PROMPT,
    )


def get_sub_agent_app():
    global _SUB_AGENT_APP
    if _SUB_AGENT_APP is None:
        _SUB_AGENT_APP = _build_sub_agent_react()
    return _SUB_AGENT_APP


def _build_reviewer_react():
    return _build_react_app(
        role="reviewer",
        tools=get_reviewer_tools(),
        system_prompt=REVIEWER_REACT_PROMPT,
    )


def get_reviewer_app():
    global _REVIEWER_APP
    if _REVIEWER_APP is None:
        _REVIEWER_APP = _build_reviewer_react()
    return _REVIEWER_APP


def _reviewer_finalize_json(react_result: dict, packet: dict) -> dict:
    """从 ReAct 末条消息解析审查 JSON；失败则用无工具兜底。"""
    messages = react_result.get("messages") or []
    for msg in reversed(messages):
        content = getattr(msg, "content", None) or ""
        if not content or not str(content).strip():
            continue
        try:
            return extract_json(str(content))
        except (json.JSONDecodeError, TypeError, ValueError):
            continue

    llm = create_llm(role="reviewer")
    fallback_prompt = (
        f"{REVIEWER_REACT_PROMPT}\n\n"
        "上一步工具核查已完成但未能解析 JSON。请仅根据对话中的工具结果输出审查 JSON。\n\n"
        + json.dumps({"review_context": packet}, ensure_ascii=False)
    )
    summary = "\n".join(
        str(getattr(m, "content", ""))[:500]
        for m in messages[-6:]
        if getattr(m, "content", None)
    )
    response = llm.invoke(
        [
            HumanMessage(content=fallback_prompt),
            HumanMessage(content=f"对话摘要：\n{summary}"),
        ]
    )
    return extract_json(response.content)


def prompt_input_enhance(state: ResearchState) -> dict:
    """提示词 Agent：整理初次输入或续问，不读取 PDF 正文。"""
    is_followup = bool(state.get("is_followup_round"))
    raw_input = (
        state.get("follow_up_query")
        if is_followup
        else (state.get("user_brief") or "")
    )
    if not raw_input and state.get("messages"):
        last = state["messages"][-1]
        raw_input = getattr(last, "content", str(last))

    inferred_intent = _infer_followup_intent(str(raw_input)) if is_followup else {
        "intent": "initial_task",
        "change_strength": "medium",
        "required_actions": [],
    }
    memory_view = _memory_view(
        state,
        query=str(raw_input),
        role="prompt_agent",
        top_k=6,
    )
    payload = {
        "inferred_intent_hint": inferred_intent,
        "is_followup_round": is_followup,
        "loaded_papers": state.get("loaded_papers") or [],
        "memory_view": memory_view,
        "paper_loaded": bool(state.get("loaded_papers")),
        "prior_final_plan_excerpt": (
            _compact_plan_for_followup(
                state.get("prior_final_plan") or "",
                follow_up_query=str(raw_input),
            )
            if is_followup
            else ""
        ),
        "raw_follow_up_query": state.get("follow_up_query", "") if is_followup else "",
        "raw_user_input": raw_input,
        "rule": "不要读取或总结 PDF 正文；只提及 PDF 已加载，并要求主控/后续 Agent 阅读 PDF 核实模型与方法。",
    }
    try:
        parsed = _invoke_json_with_prompt(
            role="orchestrator",
            prompt=PROMPT_AGENT_PROMPT,
            payload=payload,
        )
    except (json.JSONDecodeError, TypeError, ValueError):
        pdf_note = ""
        papers = state.get("loaded_papers") or []
        if papers:
            pdf_note = f"\n\n已加载用户 PDF：{', '.join(papers)}。主控必须阅读 PDF 后识别论文模型与方法细节。"
        parsed = {
            "enhanced_prompt": f"{raw_input}{pdf_note}",
            "summary": "提示词整理失败，已保留原始输入。",
            "pdf_note": pdf_note.strip(),
        }
    if not isinstance(parsed, dict):
        parsed = {
            "enhanced_prompt": str(raw_input),
            "summary": "提示词整理输出不是 JSON 对象，已保留原始输入。",
        }

    parsed_actions = parsed.get("required_actions") if isinstance(parsed, dict) else None
    if not parsed_actions:
        parsed["required_actions"] = inferred_intent.get("required_actions", [])
    if not parsed.get("intent"):
        parsed["intent"] = inferred_intent.get("intent", "")
    if not parsed.get("change_strength"):
        parsed["change_strength"] = inferred_intent.get("change_strength", "low")

    enhanced = str(parsed.get("enhanced_prompt") or raw_input).strip()
    if not enhanced:
        enhanced = str(raw_input or "").strip()
    enhanced = _append_intent_guardrails(enhanced, parsed, is_followup=is_followup)

    patch = {
        "memory_view": memory_view,
        "prompt_agent_output": parsed,
        "phase": "prompt_input",
        "messages": [AIMessage(content=f"[prompt_input]{'[续问]' if is_followup else ''} 已整理用户需求")],
        **phase_update("prompt_input", "done", enhanced[:12000]),
    }
    if is_followup:
        patch.update(
            {
                "raw_follow_up_query": state.get("raw_follow_up_query") or state.get("follow_up_query", ""),
                "follow_up_query": enhanced,
                "follow_up_combined_brief": (
                    f"## 首轮用户任务\n{state.get('user_brief') or ''}\n\n"
                    f"## 首轮已输出方案（主控续问时必须充分结合、继承与修订，禁止忽视）\n{state.get('prior_final_plan') or ''}\n\n"
                    f"## 用户本轮追问（已由提示词 Agent 整理）\n{enhanced}"
                ),
            }
        )
        _remember(
            state,
            memory_type="user_intent",
            source="prompt_input.followup",
            text=(
                f"原始追问：{state.get('raw_follow_up_query') or raw_input}\n"
                f"识别意图：{parsed.get('intent', '')} / 修改强度：{parsed.get('change_strength', '')}\n"
                f"增强提示：{enhanced}"
            ),
            tags=["followup", parsed.get("intent", ""), parsed.get("change_strength", "")],
            metadata={"roles": ["prompt_agent", "orchestrator"]},
        )
    else:
        patch.update(
            {
                "raw_user_brief": state.get("raw_user_brief") or state.get("user_brief", ""),
                "user_brief": enhanced,
            }
        )
        _remember(
            state,
            memory_type="user_intent",
            source="prompt_input.initial",
            text=f"首轮任务（提示词化后）：{enhanced}",
            tags=["initial_task", parsed.get("intent", "")],
            metadata={"roles": ["prompt_agent", "orchestrator"]},
        )
    return _finalize(state, patch)


def followup_prepare(state: ResearchState) -> dict:
    """续问入口：拼接上一轮终稿与本轮追问，进入与首轮相同的主控分析→子 Agent ReAct 流水线。"""
    followup = (state.get("user_followup") or "").strip()
    if not followup and state.get("messages"):
        last = state["messages"][-1]
        followup = getattr(last, "content", str(last)).strip()

    prior = (state.get("final_plan") or "").strip()
    first_task = (state.get("user_brief") or "").strip()
    combined = (
        f"## 首轮用户任务\n{first_task}\n\n"
        f"## 首轮已输出方案（主控续问时必须充分结合、继承与修订，禁止忽视）\n{prior}\n\n"
        f"## 用户本轮追问\n{followup}"
    )

    sid = _session_id(state)
    if sid:
        append_memory(sid, {"type": "follow_up", "user": followup})
        _remember(
            state,
            memory_type="user_intent",
            source="followup_prepare.raw",
            text=f"用户原始续问：{followup}",
            tags=["followup", "raw_user_input"],
            metadata={"roles": ["prompt_agent"]},
        )

    return _finalize(
        state,
        {
            "is_followup_round": True,
            "raw_follow_up_query": followup,
            "follow_up_query": followup,
            "prior_final_plan": prior,
            "follow_up_combined_brief": combined,
            "user_followup": "",
            "innovation_directions": [],
            "phase": "follow_up_prepare",
            "messages": [AIMessage(content=f"[followup_prepare] 续问全图重研 · {len(followup)} 字")],
            **phase_update("follow_up_prepare", "done", followup[:2000]),
        },
    )


def orchestrator_analyze(state: ResearchState) -> dict:
    is_followup = bool(state.get("is_followup_round"))
    user_brief = state.get("follow_up_combined_brief") if is_followup else (state.get("user_brief") or "")
    if not user_brief and state.get("messages"):
        last = state["messages"][-1]
        user_brief = getattr(last, "content", str(last))

    analyze_prompt = ORCHESTRATOR_ANALYZE_PROMPT
    if is_followup:
        analyze_prompt = f"{ORCHESTRATOR_ANALYZE_PROMPT}\n{ORCHESTRATOR_FOLLOWUP_ANALYZE_NOTE}"

    memory_view = _memory_view(
        state,
        query=user_brief,
        role="orchestrator",
        top_k=6,
    )
    analyze_payload: dict = {
        "first_round_context": {},
        "follow_up_query": state.get("follow_up_query", "") if is_followup else "",
        "is_followup_round": is_followup,
        "loaded_papers": state.get("loaded_papers") or [],
        "memory_view": memory_view,
        "paper_loaded": bool(state.get("loaded_papers")),
        "paper_retrieval": "如需识别论文模型、方法或实验设置，请调用 search_loaded_paper_vectors；不要凭常识猜测。",
        "user_brief": user_brief,
    }
    if is_followup:
        analyze_payload.update(
            {
                "first_round_context": _first_round_context_packet(state),
            }
        )

    if state.get("loaded_papers"):
        parsed = _invoke_json_with_tools(
            role="orchestrator",
            prompt=analyze_prompt,
            payload=analyze_payload,
            tools=[search_loaded_paper_vectors],
        )
    else:
        parsed = _invoke_json_with_prompt(
            role="orchestrator",
            prompt=analyze_prompt,
            payload=analyze_payload,
        )

    directions = parsed.get("innovation_directions") or []
    if isinstance(directions, list):
        directions = [str(d).strip() for d in directions if str(d).strip()][:_MAX_ANALYZE_DIRECTIONS]
    else:
        directions = []
    target_model = parsed.get("target_model") or "未指定模型"
    _remember(
        state,
        memory_type="decision",
        source="orchestrator_analyze",
        text=(
            f"目标模型：{target_model}\n"
            f"调研方向：{json.dumps(directions, ensure_ascii=False)}\n"
            f"分析输入摘要：{_clip_text(user_brief, 1000)}"
        ),
        tags=["target_model", target_model, "directions"],
        metadata={"roles": ["orchestrator", "prompt_agent", "sub_agent"]},
    )

    phase_label = "follow_up_analyze" if is_followup else "analyze"
    patch = {
        "memory_view": memory_view,
        "target_model": target_model,
        "innovation_directions": directions,
        "phase": "dispatch",
        "plan_announcement": dumps_compact(parsed),
        "messages": [
            AIMessage(
                content=f"[analyze]{'[续问]' if is_followup else ''} {target_model} · {len(directions)} 方向"
            )
        ],
        **phase_update(phase_label, "done", dumps_compact(parsed)),
        **phase_update("dispatch", "running" if directions else "skipped"),
    }
    if not is_followup:
        patch["user_brief"] = user_brief
    return _finalize(state, patch)


def sub_agent_research(state: ResearchState) -> dict:
    direction = state.get("current_direction", "")
    target_model = state.get("target_model", "")
    is_followup = bool(state.get("is_followup_round"))
    follow_q = (state.get("follow_up_query") or "").strip()
    app = get_sub_agent_app()
    memory_view = _memory_view(
        state,
        query=f"{direction}\n{follow_q or state.get('user_brief', '')}",
        role="sub_agent",
        top_k=5,
    )

    task_payload = {
        "target_model": target_model,
        "direction": direction,
        "is_followup_round": is_followup,
        "user_followup": follow_q if is_followup else "",
        "loaded_papers": state.get("loaded_papers") or [],
        "memory_view": memory_view,
    }
    if is_followup:
        task_payload.update(
            {
                "first_round_user_brief": _clip_text(state.get("user_brief") or "", 1200),
                "first_round_final_plan_excerpt": _compact_plan_for_followup(
                    state.get("prior_final_plan") or "",
                    follow_up_query=follow_q,
                ),
            }
        )
    task = f"{SUB_AGENT_PROMPT}\n\n{json.dumps(task_payload, ensure_ascii=False, sort_keys=True)}"
    if state.get("paper_context"):
        paper_hint = _select_relevant_excerpt(
            state.get("paper_context") or "",
            query=f"{direction} {follow_q or state.get('user_brief', '')}",
            max_chars=1600,
        )
        if paper_hint:
            task += f"\n\npaper_hint:{paper_hint}"

    result = app.invoke({"messages": [HumanMessage(content=task)]})
    raw = result["messages"][-1].content
    parsed = parse_sub_agent_json(raw)
    if not parsed.get("direction"):
        parsed["direction"] = direction
    stored = dumps_compact(parsed, max_len=8000)
    _remember(
        state,
        memory_type="research_result",
        source=f"sub_agent:{direction}",
        text=stored,
        tags=["sub_agent", direction, target_model],
        metadata={"roles": ["sub_agent", "orchestrator", "reviewer"]},
    )

    result_key = _followup_round_result_key(direction) if is_followup else direction
    phase_key = "dispatch"

    patch = {
        "sub_task_results": {result_key: stored},
        "messages": [AIMessage(content=f"[sub_agent] {direction}")],
        **phase_update(phase_key, "done", stored),
    }
    return _finalize(state, patch)


def orchestrator_synthesize(state: ResearchState) -> dict:
    is_followup = bool(state.get("is_followup_round"))
    briefs = _collect_sub_briefs(state)
    synth_query = state.get("follow_up_query") or state.get("user_brief") or state.get("target_model", "")
    memory_view = _memory_view(
        state,
        query=synth_query,
        role="orchestrator",
        top_k=6,
    )

    if is_followup:
        payload = {
            "target_model": state.get("target_model", ""),
            "follow_up_query": state.get("follow_up_query", ""),
            "followup_sub_briefs": briefs,
            "memory_view": memory_view,
            **_first_round_context_packet(state),
        }
        draft_json = _invoke_json_with_prompt(
            role="orchestrator",
            prompt=ORCHESTRATOR_FOLLOWUP_SYNTHESIZE_PROMPT,
            payload=payload,
        )
    else:
        payload = {
            "target_model": state.get("target_model", ""),
            "user_brief": state.get("user_brief", ""),
            "sub_agent_briefs": briefs,
            "memory_view": memory_view,
        }
        draft_json = _invoke_json_with_prompt(
            role="orchestrator",
            prompt=ORCHESTRATOR_SYNTHESIZE_PROMPT,
            payload=payload,
        )
    draft_md = render_draft_markdown(draft_json)
    _remember(
        state,
        memory_type="plan_summary",
        source="orchestrator_synthesize",
        text=draft_md,
        tags=["draft_plan", state.get("target_model", "")],
        metadata={"roles": ["orchestrator", "prompt_agent", "reviewer"]},
    )

    next_phase = "follow_up_finalize" if is_followup else "review_context"
    patch = {
        "memory_view": memory_view,
        "draft_plan_json": draft_json,
        "draft_plan": draft_md,
        "phase": next_phase,
        "messages": [
            AIMessage(
                content=f"[synthesize]{'[续问]' if is_followup else ''} 初稿 JSON 已生成"
            )
        ],
        **phase_update("synthesize", "done", dumps_compact(draft_json, 12000)),
    }
    if is_followup:
        patch.update(phase_update("follow_up_finalize", "running"))
    else:
        patch.update(phase_update("review_context", "running"))
    return _finalize(state, patch)


def build_review_context(state: ResearchState) -> dict:
    """图节点：组装检查 Agent 输入（含用户论文），无 LLM 调用。"""
    packet = build_review_context_packet(state)
    patch = {
        "review_context": packet,
        "phase": "review",
        "messages": [AIMessage(content="[review_context] 已注入论文与子 Agent 摘要")],
        **phase_update("review_context", "done", dumps_compact(packet, 12000)),
        **phase_update("review", "running"),
    }
    return _finalize(state, patch)


def reviewer_critique(state: ResearchState) -> dict:
    packet = state.get("review_context") or build_review_context_packet(state)
    memory_view = _memory_view(
        state,
        query=dumps_compact(packet, 2000),
        role="reviewer",
        top_k=5,
    )
    app = get_reviewer_app()
    task = json.dumps(
        {
            "task": "critique_draft",
            "review_context": packet,
            "loaded_papers": state.get("loaded_papers") or [],
            "memory_view": memory_view,
        },
        ensure_ascii=False,
        sort_keys=True,
    )
    react_result = app.invoke({"messages": [HumanMessage(content=task)]})
    review_json = _reviewer_finalize_json(react_result, packet)
    review_md = render_review_markdown(review_json)
    _remember(
        state,
        memory_type="review_issue",
        source="reviewer_critique",
        text=review_md,
        tags=["review", state.get("target_model", "")],
        metadata={"roles": ["reviewer", "orchestrator", "prompt_agent"]},
    )

    patch = {
        "memory_view": memory_view,
        "review_feedback_json": review_json,
        "review_feedback": review_md,
        "revision_round": (state.get("revision_round") or 0) + 1,
        "phase": "pending_review" if state.get("human_review_enabled") else "revise",
        "messages": [AIMessage(content="[review] 审查 JSON 已生成（含工具核查）")],
        **phase_update("review", "done", dumps_compact(review_json, 12000)),
    }
    if state.get("human_review_enabled"):
        patch.update(phase_update("human_review", "pending_review"))
    return _finalize(state, patch)


def plan_supplement_research(state: ResearchState) -> dict:
    if state.get("skip_supplement"):
        return _finalize(
            state,
            {
                "innovation_directions": [],
                "phase": "supplement",
                **phase_update("supplement", "skipped"),
            },
        )

    llm = create_llm(role="orchestrator")
    existing = set((state.get("sub_task_results") or {}).keys())
    review = state.get("review_feedback_json") or {}
    prompt = (
        '{"task":"supplement_dispatch","rule":"只输出JSON",'
        '"schema":{"supplement_directions":["str"]}}\n\n'
        f'{{"existing":{json.dumps(list(existing), ensure_ascii=False)},'
        f'"review":{dumps_compact(review, 4000)}}}'
    )
    response = llm.invoke([HumanMessage(content=prompt)])
    try:
        parsed = extract_json(response.content)
        raw = parsed.get("supplement_directions") or []
    except (json.JSONDecodeError, TypeError):
        raw = []

    supplement = [d for d in raw if d and d not in existing][:_MAX_SUPPLEMENT_DIRECTIONS]
    return _finalize(
        state,
        {
            "innovation_directions": supplement,
            "phase": "supplement",
            **phase_update("supplement", "running" if supplement else "skipped"),
        },
    )


def human_review_gate(state: ResearchState) -> dict:
    if not state.get("human_review_enabled"):
        return _finalize(state, {**phase_update("human_review", "skipped")})

    approved = state.get("human_approved")
    if approved is True:
        sid = _session_id(state)
        if sid:
            append_memory(sid, {"type": "human_review", "decision": "approved"})
        return _finalize(
            state,
            {"phase": "revise", **phase_update("human_review", "done", "approved")},
        )

    if approved is False:
        notes = (state.get("human_review_notes") or "").strip()
        merged = dict(state.get("review_feedback_json") or {})
        if notes:
            merged["user_reject_notes"] = notes
        sid = _session_id(state)
        if sid:
            append_memory(sid, {"type": "human_review", "decision": "rejected", "notes": notes})
            _remember(
                state,
                memory_type="user_intent",
                source="human_review.rejected",
                text=f"用户在人工审批中拒绝/要求修改：{notes}",
                tags=["human_review", "rejected"],
                metadata={"roles": ["prompt_agent", "orchestrator", "reviewer"]},
            )
        return _finalize(
            state,
            {
                "review_feedback_json": merged,
                "review_feedback": render_review_markdown(merged),
                "phase": "revise",
                "human_approved": None,
                **phase_update("human_review", "done", notes or "rejected"),
            },
        )

    return _finalize(state, {**phase_update("human_review", "running")})


def _parse_final_markdown(llm_content: str) -> str:
    """定稿/续问：优先 JSON 中的 full_plan_markdown，否则清洗原文。"""
    try:
        parsed = extract_json(llm_content)
        body = (parsed.get("full_plan_markdown") or "").strip()
        if body:
            return clean_user_markdown(body)
    except (json.JSONDecodeError, TypeError, ValueError):
        pass
    return clean_user_markdown(llm_content)


def orchestrator_revise(state: ResearchState) -> dict:
    llm = create_llm(role="orchestrator")
    memory_view = _memory_view(
        state,
        query=state.get("user_brief") or state.get("target_model") or "",
        role="orchestrator",
        top_k=6,
    )
    paper_excerpt = _select_relevant_excerpt(
        state.get("paper_context") or "",
        query=state.get("user_brief") or state.get("target_model") or "",
        max_chars=_MAX_PAPER_EXCERPT_CHARS,
    )
    packet = {
        "target_model": state.get("target_model", ""),
        "user_brief": state.get("user_brief", ""),
        "draft_plan": state.get("draft_plan_json") or {},
        "review": state.get("review_feedback_json") or {},
        "paper_excerpt": paper_excerpt,
        "sub_agent_briefs": _collect_sub_briefs(state),
        "memory_view": memory_view,
    }
    prompt = f"{ORCHESTRATOR_REVISE_PROMPT}\n\n{json.dumps(packet, ensure_ascii=False, sort_keys=True)}"
    response = llm.invoke([HumanMessage(content=prompt)])
    final = _parse_final_markdown(response.content)
    modules = extract_plan_modules(final)
    _remember(
        state,
        memory_type="plan_summary",
        source="orchestrator_revise.final",
        text=final,
        tags=["final_plan", state.get("target_model", ""), *modules[:6]],
        metadata={"roles": ["prompt_agent", "orchestrator", "reviewer"], "modules": modules},
    )

    return _finalize(
        state,
        {
            "memory_view": memory_view,
            "final_plan": final,
            "phase": "done",
            "user_followup": "",
            "messages": [AIMessage(content="[revise] 最终方案已生成")],
            **phase_update("revise", "done", final[:12000]),
        },
    )


def orchestrator_finalize_followup(state: ResearchState) -> dict:
    """续问轮：汇总后直接定稿，跳过检查 Agent。"""
    llm = create_llm(role="orchestrator")
    memory_view = _memory_view(
        state,
        query=state.get("follow_up_query") or state.get("user_brief") or "",
        role="orchestrator",
        top_k=6,
    )
    paper_excerpt = _select_relevant_excerpt(
        state.get("paper_context") or "",
        query=state.get("follow_up_query") or state.get("user_brief") or "",
        max_chars=_MAX_PAPER_EXCERPT_CHARS,
    )
    packet = {
        "target_model": state.get("target_model", ""),
        "follow_up_query": state.get("follow_up_query", ""),
        "draft_plan": state.get("draft_plan_json") or {},
        "paper_excerpt": paper_excerpt,
        "followup_sub_briefs": _collect_sub_briefs(state, round="followup"),
        "memory_view": memory_view,
        **_first_round_context_packet(state),
    }
    prompt = f"{ORCHESTRATOR_FOLLOWUP_FINALIZE_PROMPT}\n\n{json.dumps(packet, ensure_ascii=False, sort_keys=True)}"
    response = llm.invoke([HumanMessage(content=prompt)])
    final = _parse_final_markdown(response.content)
    modules = extract_plan_modules(final)
    _remember(
        state,
        memory_type="plan_summary",
        source="orchestrator_finalize_followup.final",
        text=final,
        tags=["final_plan", "followup", state.get("target_model", ""), *modules[:6]],
        metadata={"roles": ["prompt_agent", "orchestrator", "reviewer"], "modules": modules},
    )

    return _finalize(
        state,
        {
            "memory_view": memory_view,
            "final_plan": final,
            "phase": "done",
            "is_followup_round": False,
            "follow_up_query": "",
            "follow_up_combined_brief": "",
            "prior_final_plan": "",
            "messages": [AIMessage(content="[follow_up_finalize] 续问全图终稿已生成")],
            **phase_update("follow_up_finalize", "done", final[:12000]),
        },
    )
