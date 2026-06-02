"""研究工作流进度事件（轻量 Streaming Observability）。"""

from __future__ import annotations

from typing import Any


def format_node_update(node_name: str, update: dict[str, Any]) -> str | None:
    if not update:
        return None
    parts = [f"[进度] 节点 «{node_name}» 完成"]
    if update.get("plan_announcement"):
        parts.append("  · 已发布执行计划")
    if update.get("phase_status"):
        for ph, st in update["phase_status"].items():
            parts.append(f"  · 阶段 {ph}: {st}")
    if update.get("prompt_agent_output"):
        parts.append("  · 提示词 Agent 已整理用户需求")
    if update.get("target_model"):
        parts.append(f"  · 目标模型: {update['target_model']}")
    dirs = update.get("innovation_directions")
    if dirs is not None:
        parts.append(f"  · 调研方向数: {len(dirs)}")
    if update.get("draft_plan_json") or update.get("draft_plan"):
        parts.append("  · 已生成方案初稿（JSON）")
    if update.get("review_context"):
        parts.append("  · 已组装审查上下文（含论文）")
    if update.get("review_feedback_json") or update.get("review_feedback"):
        parts.append("  · 检查 Agent 已输出审查 JSON")
    if update.get("is_followup_round") is True:
        parts.append("  · 续问全图重研轮次已启动")
    if update.get("followup_review_enabled") is True:
        parts.append("  · 本轮续问将启用审阅 Agent")
    if update.get("final_plan"):
        parts.append("  · 已输出/更新最终方案")
    return "\n".join(parts)


def print_stream_updates(chunks) -> dict[str, Any] | None:
    """消费 stream_mode='updates' 的迭代器并打印进度，返回最后一次完整 state 补丁。"""
    last: dict[str, Any] | None = None
    for chunk in chunks:
        if not isinstance(chunk, dict):
            continue
        for node_name, update in chunk.items():
            line = format_node_update(node_name, update or {})
            if line:
                print(line, flush=True)
            if update:
                last = update
    return last
