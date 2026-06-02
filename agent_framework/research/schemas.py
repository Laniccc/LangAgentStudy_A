"""Agent 间 JSON 契约与 Markdown 渲染（用户可见层）。"""

from __future__ import annotations

import json
import re
from typing import Any


def extract_json(text: str) -> dict:
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


def dumps_compact(obj: Any, max_len: int | None = None) -> str:
    text = json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
    if max_len and len(text) > max_len:
        return text[: max_len - 20] + "…(truncated)"
    return text


def _extract_keywords(text: str, *, max_terms: int = 8) -> list[str]:
    tokens = re.findall(r"[A-Za-z0-9_\-+/]{3,}|[\u4e00-\u9fff]{2,}", (text or "").lower())
    out: list[str] = []
    seen: set[str] = set()
    for token in tokens:
        if token in seen:
            continue
        seen.add(token)
        out.append(token)
        if len(out) >= max_terms:
            break
    return out


def _select_relevant_excerpt(
    text: str,
    *,
    query: str,
    max_chars: int = 2800,
    chunk_chars: int = 520,
    max_chunks: int = 5,
) -> str:
    raw = (text or "").strip()
    if not raw:
        return ""
    if len(raw) <= max_chars:
        return raw

    keys = _extract_keywords(query)
    chunks = [c.strip() for c in re.split(r"\n\s*\n", raw) if c.strip()]
    scored: list[tuple[int, int, str]] = []
    for idx, chunk in enumerate(chunks):
        snippet = chunk[:chunk_chars]
        low = snippet.lower()
        score = sum(low.count(k) for k in keys) if keys else 0
        scored.append((score, -idx, snippet))
    scored.sort(reverse=True)

    selected: list[str] = []
    total = 0
    for score, _, snippet in scored:
        if score == 0 and selected:
            continue
        if total + len(snippet) > max_chars:
            remain = max_chars - total
            if remain > 120:
                selected.append(snippet[:remain] + "…")
            break
        selected.append(snippet)
        total += len(snippet)
        if len(selected) >= max_chunks:
            break

    return "\n\n".join(selected) if selected else (raw[:max_chars] + "\n…(论文摘录已截断)")


# --- 子 Agent 输出 ---

SUB_AGENT_OUTPUT_SCHEMA = """{
  "direction": "方向名",
  "findings": [{"claim": "结论", "source": "出处", "year": 2024}],
  "la_eer_notes": "LA 相关线索或待验证",
  "df_eer_notes": "DF 相关线索或待验证",
  "cross_domain": "图像域可迁移要点或空字符串",
  "model_changes": ["对目标模型的具体改动建议"]
}"""


def parse_sub_agent_json(text: str) -> dict:
    try:
        return extract_json(text)
    except (json.JSONDecodeError, TypeError, ValueError):
        return {"direction": "", "findings": [], "raw_markdown_fallback": text[:4000]}


def sub_agent_json_to_brief(data: dict, max_findings: int = 2) -> dict:
    """压缩子 Agent 结果供主控/检查 Agent 消费。"""
    findings = data.get("findings") or []
    if isinstance(findings, list):
        findings = findings[:max_findings]
    return {
        "direction": data.get("direction", ""),
        "findings": findings,
        "la_eer_notes": (data.get("la_eer_notes") or "")[:240],
        "df_eer_notes": (data.get("df_eer_notes") or "")[:240],
        "cross_domain": (data.get("cross_domain") or "")[:180],
        "model_changes": (data.get("model_changes") or [])[:3],
    }


# --- 主控初稿 ---

DRAFT_PLAN_SCHEMA = """{
  "title": "方案标题（须同时体现 LA 与 DF）",
  "target_model": "模型名",
  "background": "背景与现状，无寒暄",
  "direction_summaries": [
    {"direction": "方向", "summary": "摘要", "la_impact": "对 LA EER", "df_impact": "对 DF EER"}
  ],
  "prioritized_changes": [
    {"priority": "P0", "change": "改动", "benefit": "LA|DF|both", "evidence": "依据出处"}
  ],
  "experiment_design": "实验协议要点",
  "risks": "风险与依赖",
  "cross_domain_notes": "跨域说明"
}"""


def render_draft_markdown(data: dict) -> str:
    """将初稿 JSON 渲染为用户可读的 Markdown。"""
    if not data:
        return ""
    lines = [f"# {data.get('title', '模型改进方案')}", ""]
    if data.get("target_model"):
        lines += [f"**目标模型**：{data['target_model']}", ""]
    if data.get("background"):
        lines += ["## 1. 背景与现状", data["background"], ""]
    summaries = data.get("direction_summaries") or []
    if summaries:
        lines += ["## 2. 各方向调研摘要"]
        for s in summaries:
            if isinstance(s, dict):
                lines.append(
                    f"### {s.get('direction', '方向')}\n"
                    f"{s.get('summary', '')}\n"
                    f"- LA：{s.get('la_impact', '—')} | DF：{s.get('df_impact', '—')}"
                )
        lines.append("")
    changes = data.get("prioritized_changes") or []
    if changes:
        lines += ["## 3. 拟采用的改进点", "| 优先级 | 改动 | 主要利好 | 依据 |", "|---|---|---|---|"]
        for c in changes:
            if isinstance(c, dict):
                lines.append(
                    f"| {c.get('priority', '')} | {c.get('change', '')} | "
                    f"{c.get('benefit', '')} | {c.get('evidence', '')} |"
                )
        lines.append("")
    if data.get("experiment_design"):
        lines += ["## 4. 实验设计", data["experiment_design"], ""]
    if data.get("cross_domain_notes"):
        lines += ["## 5. 跨域迁移", data["cross_domain_notes"], ""]
    if data.get("risks"):
        lines += ["## 6. 风险与依赖", data["risks"], ""]
    return "\n".join(lines)


# --- 检查 Agent 输出 ---

REVIEW_OUTPUT_SCHEMA = """{
  "overall": "总体评价一句",
  "verification": [{"tool": "工具名", "purpose": "核查目的", "finding": "工具返回要点"}],
  "issues": [
    {
      "id": "1",
      "severity": "high|medium|low",
      "topic": "问题主题",
      "detail": "具体问题",
      "draft_ref": "对应初稿字段",
      "paper_ref": "论文摘录或工具检索出处；无则写未验证",
      "evidence_status": "verified|unverified|contradicted"
    }
  ],
  "suggestions": ["可执行修改建议"],
  "must_fix_before_finalize": ["定稿前必须解决项"]
}"""


REVISE_OUTPUT_SCHEMA = """{
  "full_plan_markdown": "以单个 # 标题开头的完整方案正文，无寒暄、无审查对话体"
}"""

FOLLOW_UP_REVISE_SCHEMA = """{
  "answers": [
    {"question": "从用户追问拆出的子问题", "answer": "直接回答", "sources": ["续问子Agent/用户实测/论文/待验证"]}
  ],
  "changes": ["相对上一版必须写明的具体变更，至少2条"],
  "full_plan_markdown": "合并变更后的完整方案，以 # 开头；须体现本轮追问带来的修改，禁止与上一版逐字相同"
}"""


def render_followup_markdown(data: dict) -> str:
    """将续问 JSON 渲染为用户可读 Markdown。"""
    parts: list[str] = ["## 针对用户本轮追问的回答", ""]
    for item in data.get("answers") or []:
        if isinstance(item, dict):
            q = item.get("question", "")
            a = item.get("answer", "")
            src = ", ".join(item.get("sources") or [])
            parts.append(f"### {q}\n{a}\n\n*依据：{src or '待补充'}*\n")
    changes = data.get("changes") or []
    if changes:
        parts += ["## 相对上一版变更", ""]
        parts += [f"- {c}" for c in changes]
        parts.append("")
    body = (data.get("full_plan_markdown") or "").strip()
    if body:
        parts.append(body)
    return "\n".join(parts).strip()


def render_review_markdown(data: dict) -> str:
    if not data:
        return ""
    lines = ["## 审查报告", "", f"**总体评价**：{data.get('overall', '')}", "", "### 主要问题"]
    for issue in data.get("issues") or []:
        if isinstance(issue, dict):
            ev = issue.get("evidence_status", "")
            lines.append(
                f"- **[{issue.get('severity', '')}]** [{ev}] {issue.get('topic', '')}："
                f"{issue.get('detail', '')}（初稿：{issue.get('draft_ref', '—')}；"
                f"出处：{issue.get('paper_ref', '—')}）"
            )
    ver = data.get("verification") or []
    if ver:
        lines += ["", "### 工具核查记录"]
        for v in ver:
            if isinstance(v, dict):
                lines.append(
                    f"- `{v.get('tool', '')}`：{v.get('purpose', '')} → {v.get('finding', '')}"
                )
    lines += ["", "### 修改建议"]
    for s in data.get("suggestions") or []:
        lines.append(f"- {s}")
    must = data.get("must_fix_before_finalize") or []
    if must:
        lines += ["", "### 定稿前必须解决"]
        for m in must:
            lines.append(f"- {m}")
    if data.get("user_reject_notes"):
        lines += ["", "### 用户审批意见", data["user_reject_notes"]]
    return "\n".join(lines)


def build_review_context_packet(state: dict) -> dict:
    """组装检查 Agent 专用上下文（含用户论文）。"""
    sub_briefs = {}
    for key, raw in (state.get("sub_task_results") or {}).items():
        if key.startswith("续问|") or key.startswith("续问轮|"):
            continue
        if isinstance(raw, str) and raw.strip().startswith("{"):
            try:
                sub_briefs[key] = sub_agent_json_to_brief(parse_sub_agent_json(raw))
            except Exception:
                sub_briefs[key] = {"raw_excerpt": raw[:1200]}
        else:
            sub_briefs[key] = {"raw_excerpt": (raw or "")[:1200]}

    query = (
        state.get("follow_up_query")
        or state.get("user_brief")
        or state.get("target_model")
        or ""
    )
    paper = _select_relevant_excerpt(
        state.get("paper_context") or "",
        query=query,
        max_chars=2800,
    )

    return {
        "user_brief": (state.get("user_brief") or "")[:2000],
        "target_model": state.get("target_model", ""),
        "loaded_papers": state.get("loaded_papers") or [],
        "paper_excerpt": paper,
        "draft_plan": state.get("draft_plan_json") or {},
        "sub_agent_briefs": sub_briefs,
    }
