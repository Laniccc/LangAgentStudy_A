"""子 Agent 上下文轻度压缩（工具输出 + 结构化结果入库）。"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

from langchain_core.messages import ToolMessage

from agent_framework.research.session import get_runs_dir

# 研究域高价值 token（保留含这些信号的段落）
_SIGNAL_RE = re.compile(
    r"(eer|asvspoof|arxiv|github|deepfake|spoof|anti-spoof|"
    r"wavlm|xlsr|mamba|rawnet|aasist|min t-dcf|"
    r"\d+\.?\d*\s*%|202[3-9]|la\b|df\b)",
    re.IGNORECASE,
)


def get_sub_agent_tool_output_max_chars() -> int:
    return int(os.getenv("SUB_AGENT_TOOL_OUTPUT_MAX_CHARS", "2400"))


def get_sub_agent_stored_max_chars() -> int:
    return int(os.getenv("SUB_AGENT_STORED_MAX_CHARS", "4200"))


def get_sub_agent_findings_max() -> int:
    return int(os.getenv("SUB_AGENT_FINDINGS_MAX", "5"))


def save_sub_agent_artifacts() -> bool:
    return os.getenv("SUB_AGENT_SAVE_ARTIFACT", "1").strip().lower() not in (
        "0",
        "false",
        "no",
    )


def _clip(text: str, limit: int, *, note: str = "truncated") -> str:
    raw = (text or "").strip()
    if len(raw) <= limit:
        return raw
    return raw[: max(limit - 24, 0)] + f"\n…({note})"


def _score_chunk(chunk: str) -> int:
    low = chunk.lower()
    score = len(_SIGNAL_RE.findall(chunk)) * 3
    if "http" in low:
        score += 2
    if re.search(r"\d+\.?\d*\s*%", chunk):
        score += 4
    return score


def _compress_plain_text(text: str, *, max_chars: int) -> str:
    raw = (text or "").strip()
    if len(raw) <= max_chars:
        return raw

    blocks = [b.strip() for b in re.split(r"\n\s*\n", raw) if b.strip()]
    if len(blocks) <= 1:
        numbered = re.split(r"(?=\n\d+\.\s)", raw)
        blocks = [b.strip() for b in numbered if b.strip()] or [raw]

    ranked = sorted(
        ((_score_chunk(b), -i, b[:900]) for i, b in enumerate(blocks)),
        reverse=True,
    )
    picked: list[str] = []
    total = 0
    header = f"<!-- compressed from {len(raw)} chars -->\n"
    budget = max_chars - len(header)
    for score, _, snippet in ranked:
        if score == 0 and picked:
            continue
        if total + len(snippet) + 2 > budget:
            remain = budget - total
            if remain > 160:
                picked.append(snippet[:remain] + "…")
            break
        picked.append(snippet)
        total += len(snippet) + 2
        if len(picked) >= 8:
            break

    if not picked:
        return _clip(raw, max_chars, note="tool_output")
    return header + "\n\n".join(picked)


def _compress_json_payload(obj: Any, *, max_chars: int) -> str:
    if isinstance(obj, list):
        trimmed = []
        for item in obj[:12]:
            if isinstance(item, dict):
                trimmed.append(
                    {
                        k: _clip(str(v), 280) if isinstance(v, str) else v
                        for k, v in list(item.items())[:10]
                    }
                )
            else:
                trimmed.append(_clip(str(item), 320))
        text = json.dumps(trimmed, ensure_ascii=False, separators=(",", ":"))
        return _clip(text, max_chars, note="json_list")

    if isinstance(obj, dict):
        text = json.dumps(obj, ensure_ascii=False, separators=(",", ":"))
        if len(text) <= max_chars:
            return text
        slim: dict[str, Any] = {}
        for k, v in obj.items():
            if isinstance(v, str):
                slim[k] = _clip(v, 500)
            elif isinstance(v, (list, dict)):
                slim[k] = v
            else:
                slim[k] = v
        return _clip(
            json.dumps(slim, ensure_ascii=False, separators=(",", ":")),
            max_chars,
            note="json_dict",
        )

    return _compress_plain_text(str(obj), max_chars=max_chars)


def compress_tool_output(content: str, *, tool_name: str | None = None) -> str:
    """压缩进入子 Agent ReAct 循环的工具返回（按类型路由 + 信号保留）。"""
    max_chars = get_sub_agent_tool_output_max_chars()
    raw = (content or "").strip()
    if not raw or len(raw) <= max_chars:
        return raw

    name = (tool_name or "").lower()
    if name in ("query_antispoofing_knowledge",):
        if raw.startswith("{") or raw.startswith("["):
            try:
                return _compress_json_payload(json.loads(raw), max_chars=max_chars)
            except json.JSONDecodeError:
                pass
        return _compress_plain_text(raw, max_chars=max_chars)

    if raw.startswith("{") or raw.startswith("["):
        try:
            return _compress_json_payload(json.loads(raw), max_chars=max_chars)
        except json.JSONDecodeError:
            pass

    return _compress_plain_text(raw, max_chars=max_chars)


def compress_react_tool_messages(result: dict) -> dict:
    """压缩 ToolNode 输出中的 ToolMessage。"""
    out: list = []
    for msg in result.get("messages") or []:
        if isinstance(msg, ToolMessage):
            name = getattr(msg, "name", None) or ""
            compressed = compress_tool_output(str(msg.content or ""), tool_name=name or None)
            out.append(
                ToolMessage(
                    content=compressed,
                    tool_call_id=msg.tool_call_id,
                    name=name or None,
                )
            )
        else:
            out.append(msg)
    return {"messages": out}


def _artifact_path(session_id: str, direction: str) -> Path:
    safe = re.sub(r"[^\w\-|]+", "_", (direction or "unknown"))[:96]
    path = get_runs_dir() / "store" / session_id / "artifacts"
    path.mkdir(parents=True, exist_ok=True)
    return path / f"sub_agent_{safe}.json"


def _normalize_sub_agent_parsed(parsed: dict, *, raw_fallback: str = "") -> dict:
    max_findings = get_sub_agent_findings_max()
    findings = parsed.get("findings") or []
    if isinstance(findings, list):
        norm_findings = []
        for item in findings[:max_findings]:
            if not isinstance(item, dict):
                continue
            norm_findings.append(
                {
                    "claim": _clip(str(item.get("claim") or ""), 420),
                    "source": _clip(str(item.get("source") or ""), 200),
                    "year": item.get("year"),
                }
            )
        parsed["findings"] = norm_findings

    parsed["la_eer_notes"] = _clip(str(parsed.get("la_eer_notes") or ""), 360)
    parsed["df_eer_notes"] = _clip(str(parsed.get("df_eer_notes") or ""), 360)
    parsed["cross_domain"] = _clip(str(parsed.get("cross_domain") or ""), 280)
    changes = parsed.get("model_changes") or []
    if isinstance(changes, list):
        parsed["model_changes"] = [_clip(str(c), 200) for c in changes[:6]]

    if parsed.get("raw_markdown_fallback"):
        parsed["raw_markdown_fallback"] = _clip(str(parsed["raw_markdown_fallback"]), 900)
    elif raw_fallback and not parsed.get("findings"):
        parsed["raw_markdown_fallback"] = _clip(raw_fallback, 900)

    return parsed


def prepare_sub_agent_storage(
    session_id: str,
    *,
    direction: str,
    parsed: dict,
    raw: str,
) -> str:
    """
    规范化子 Agent JSON 并限制入库体积。
    超长时可选将完整结果写入 artifacts（CCR 式指针）。
    """
    from agent_framework.research.schemas import dumps_compact

    normalized = _normalize_sub_agent_parsed(dict(parsed), raw_fallback=raw)
    max_len = get_sub_agent_stored_max_chars()
    probe = json.dumps(normalized, ensure_ascii=False, separators=(",", ":"))

    if len(probe) <= max_len:
        return probe

    if session_id and save_sub_agent_artifacts():
        full_record = {
            "direction": direction,
            "parsed": parsed,
            "raw_excerpt": _clip(raw, 12000, note="raw"),
        }
        artifact = _artifact_path(session_id, direction)
        artifact.write_text(
            json.dumps(full_record, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        normalized["_artifact_ref"] = str(
            artifact.relative_to(get_runs_dir())
        ).replace("\\", "/")

    return dumps_compact(normalized, max_len=max_len)
