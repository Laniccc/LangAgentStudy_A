"""Agent 框架入口：通用 ReAct 或语音鉴伪研究工作流。"""

import argparse
import re
import uuid
from datetime import datetime, timezone
from pathlib import Path

from langchain_core.messages import HumanMessage

# 默认从项目根目录该文件读取任务 / 续问（避免在终端长文本输入）
DEFAULT_INPUT_FILE = Path("input.md")

from agent_framework import create_agent
from agent_framework.llm import check_llm_connection
from agent_framework.research import (
    create_research_agent,
    list_sessions,
    load_session_meta,
    make_thread_config,
    new_thread_id,
    save_session_meta,
)
from agent_framework.research.events import print_stream_updates
from agent_framework.research.nodes import reset_research_react_apps
from agent_framework.research.output_utils import is_same_as_brief, plans_nearly_identical
from agent_framework.research.pdf_loader import clear_paper_store, load_paper_pdfs
from agent_framework.research.session import get_checkpointer, normalize_thread_id
from agent_framework.research.tools import collect_pdf_paths


def _strip_md_comments(text: str) -> str:
    """去掉 HTML 注释与仅含注释/空白的行。"""
    text = re.sub(r"<!--[\s\S]*?-->", "", text)
    lines = [ln for ln in text.splitlines() if ln.strip()]
    return "\n".join(lines).strip()


def read_input_file(path: Path | str | None = None) -> str | None:
    """读取输入 Markdown；无有效正文时返回 None。"""
    p = Path(path) if path else DEFAULT_INPUT_FILE
    if not p.is_file():
        return None
    body = _strip_md_comments(p.read_text(encoding="utf-8"))
    return body if body else None


def run_chat(thread_id: str | None = None, input_file: Path | str | None = None):
    """通用 ReAct；同一 thread_id 下保留对话历史（Checkpointer）。"""
    tid = normalize_thread_id(thread_id, mode="chat") or new_thread_id().replace(
        "research-", "chat-"
    )
    config = make_thread_config(tid)
    app = create_agent(checkpointer=get_checkpointer())

    snap = app.get_state(config)
    if snap.values.get("messages"):
        print(f"已恢复会话 thread_id={tid}（{len(snap.values['messages'])} 条历史消息）\n")
    else:
        print(f"新会话 thread_id={tid}\n")

    inp = Path(input_file) if input_file else DEFAULT_INPUT_FILE
    print("LangGraph ReAct Agent 已启动\n")
    print(f"  · 首轮/每轮输入：编辑 `{inp}` 后回车（空文件时可在终端输入 quit 退出）\n")

    while True:
        user_input = read_input_file(inp)
        if user_input:
            print(f"（已从 {inp} 读取）\n你: {user_input[:200]}{'…' if len(user_input) > 200 else ''}\n")
        else:
            user_input = input("你: ").strip()
            if not user_input:
                continue
        if user_input.lower() in {"quit", "exit", "q"}:
            print(f"会话已保存，续聊请使用：python main.py --mode chat --thread-id {tid}\n")
            break

        result = app.invoke(
            {"messages": [HumanMessage(content=user_input)]},
            config,
        )
        last = result["messages"][-1]
        print(f"\nAgent: {last.content}\n")


def _load_papers(pdf_paths, papers_dir):
    paper_context = ""
    loaded_papers: list[str] = []
    paths = collect_pdf_paths(pdf_paths, papers_dir)
    if paths:
        clear_paper_store()
        reset_research_react_apps()
        try:
            paper_context, store = load_paper_pdfs(paths)
            loaded_papers = list(store.keys())
            print(f"已加载 {len(loaded_papers)} 篇论文 PDF：{', '.join(loaded_papers)}\n")
        except Exception as e:
            print(f"警告：PDF 加载失败（{e}），将仅使用文本任务描述继续。\n")
    elif pdf_paths:
        print("警告：未找到有效 PDF 文件，请检查路径。\n")
    return paper_context, loaded_papers


def _print_final_result(result: dict, out_path: str = "output_final_plan.md"):
    final = result.get("final_plan") or ""
    if final:
        print("\n" + "=" * 60)
        print("最终模型改进方案")
        print("=" * 60 + "\n")
        print(final)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(final)
        print(f"\n已保存至 {out_path}")
    else:
        msgs = result.get("messages") or []
        if msgs:
            print(f"\n{msgs[-1].content}")


def _graph_state_values(app, config: dict) -> dict:
    snap = app.get_state(config)
    return dict(snap.values or {})


def _is_interrupted(app, config: dict) -> bool:
    snap = app.get_state(config)
    return bool(snap.next)


def _handle_human_review(app, config: dict) -> None:
    """人工审批：查看初稿与审查意见后批准或提交修改意见。"""
    state = _graph_state_values(app, config)
    draft = state.get("draft_plan", "")
    review = state.get("review_feedback", "")

    print("\n" + "=" * 60)
    print("人工审批（Human-in-the-Loop）")
    print("=" * 60)
    print("\n--- 方案初稿（节选前 4000 字）---\n")
    print((draft or "（无）")[:4000])
    if len(draft or "") > 4000:
        print("\n...（已截断，完整初稿见 .runs/store/.../phase_outputs/synthesize.md）")
    print("\n--- 检查 Agent 意见 ---\n")
    print(review or "（无）")
    print("\n操作：输入 approve/a 批准定稿；reject/r 后输入修改意见；skip 跳过审批直接定稿")

    while True:
        choice = input("\n审批> ").strip().lower()
        if choice in {"approve", "a", "yes", "y"}:
            app.update_state(config, {"human_approved": True})
            break
        if choice in {"skip", "s"}:
            app.update_state(
                config,
                {"human_approved": True, "human_review_enabled": False},
            )
            break
        if choice in {"reject", "r", "no", "n"}:
            notes = input("请输入修改意见（将并入审查反馈）：\n").strip()
            app.update_state(
                config,
                {"human_approved": False, "human_review_notes": notes},
            )
            break
        print("无效输入，请重试。")

    print("\n--- 继续执行定稿 ---\n")
    print_stream_updates(app.stream(None, config, stream_mode="updates"))


def _invoke_research_stream(app, payload: dict, config: dict) -> dict:
    """流式执行并合并最终状态。"""
    print_stream_updates(app.stream(payload, config, stream_mode="updates"))
    return _graph_state_values(app, config)


def _wait_followup_enter(inp: Path) -> bool:
    """
    输出方案后等待用户确认，再读取 input.md 续问。
    返回 False 表示用户选择结束续问（输入 q/quit/exit）。
    """
    print(
        f"\n请先在 `{inp}` 中写入/更新本轮追问并保存，"
        "完成后按回车开始读取；输入 q 结束续问。"
    )
    line = input().strip().lower()
    return line not in {"q", "quit", "exit"}


def _research_followup_loop(
    app,
    config: dict,
    thread_id: str,
    input_file: Path | str | None = None,
) -> None:
    """多轮续问：在已有 Checkpointer 状态上继续对话。"""
    state = _graph_state_values(app, config)
    if state.get("phase") != "done":
        print("当前会话尚未完成首轮流水线，无法续问。")
        return

    inp = Path(input_file) if input_file else DEFAULT_INPUT_FILE
    print("\n" + "-" * 60)
    print(
        "多轮续问已启用：追问+上轮方案拼接 → 主控分析 → 子 Agent（PDF/网络 ReAct）"
        " → 主控汇总定稿（跳过检查 Agent，全量重写方案）"
    )
    print(f"会话 thread_id={thread_id}")
    print(
        f"续问内容写在 `{inp}`（须与首问不同）；"
        "每轮方案输出后会先等待你按回车，再读取该文件。\n"
    )

    last_used_followup = ""
    while True:
        if not _wait_followup_enter(inp):
            print("结束续问。")
            break

        followup = read_input_file(inp)
        from_file = bool(followup)
        if from_file and followup == last_used_followup:
            print(f"`{inp}` 与上一轮续问相同，请修改文件；或在下方单独输入新追问（勿与提示语写在同一行）。\n")
            followup = input("续问> ").strip()
        elif from_file:
            print(f"（已从 {inp} 读取续问）\n")
        else:
            followup = input("续问> ").strip()
            if not followup:
                print("结束续问。")
                break
        if not followup:
            continue
        if followup.lower() in {"quit", "exit", "q"}:
            break

        state_before = _graph_state_values(app, config)
        user_brief = state_before.get("user_brief") or ""
        if is_same_as_brief(followup, user_brief):
            print(
                "警告：当前 `input.md` 内容与首轮任务过于相似，已跳过本轮（请只写追问，勿重复首问）。\n"
                "示例：「我已在 2019LA 训练、2021DF EER=9%，请评估 XLSR 与 WavLM 能否共用」\n"
            )
            continue

        last_used_followup = followup
        prev_plan = state_before.get("final_plan") or ""

        payload = {
            "user_followup": followup,
            "messages": [HumanMessage(content=followup)],
        }
        _invoke_research_stream(app, payload, config)
        state = _graph_state_values(app, config)
        new_plan = state.get("final_plan") or ""
        if plans_nearly_identical(new_plan, prev_plan):
            print(
                "\n警告：本轮输出与上一版方案几乎相同，可能未真正吸收续问。"
                "请修改 `input.md` 为更具体的追问后重试。\n"
            )
        _print_final_result(state, out_path=f"output_final_plan_{thread_id}.md")
        save_session_meta(
            thread_id,
            {
                "last_followup": followup,
                "phase": state.get("phase"),
            },
        )


def run_research(
    task: str | None = None,
    pdf_paths: list[str] | None = None,
    papers_dir: str | None = None,
    *,
    thread_id: str | None = None,
    continue_session: bool = False,
    human_review: bool = False,
    skip_supplement: bool = False,
    no_followup: bool = False,
    input_file: Path | str | None = None,
):
    tid = normalize_thread_id(thread_id, mode="research") or new_thread_id()
    if thread_id and tid != thread_id.strip():
        print(f"已规范化 thread_id → {tid}\n")
    config = make_thread_config(tid)
    run_id = f"run-{uuid.uuid4().hex[:8]}"

    app = create_research_agent(enable_human_review=human_review)

    if continue_session:
        snap = app.get_state(config)
        if not snap.values:
            print(f"未找到会话 {tid}，请先完整跑一轮或使用新 thread_id。")
            print("提示：完整 ID 形如 research-24b9f385c515，可用 --list-sessions 查看。")
            return
        print(f"继续会话 thread_id={tid}\n")
        meta = load_session_meta(tid)
        print(f"上次更新：{meta.get('updated_at', '未知')}\n")
        if _is_interrupted(app, config):
            print("检测到未完成的人工审批，进入审批流程…\n")
            _handle_human_review(app, config)
            state = _graph_state_values(app, config)
            _print_final_result(state)
        if not no_followup:
            _research_followup_loop(app, config, tid, input_file=input_file)
        return

    print("语音鉴伪多 Agent 研究工作流已启动\n")
    print(f"会话 thread_id={tid}（续跑/续问请保存此 ID）\n")
    print("流程：主控分析 → 子 Agent 并行调研 → 汇总初稿 → 检查 Agent → 可选补充 → 定稿\n")

    paper_context, loaded_papers = _load_papers(pdf_paths, papers_dir)

    inp = Path(input_file) if input_file else DEFAULT_INPUT_FILE
    task_from_cli = bool(task and str(task).strip())
    if not task_from_cli:
        task = read_input_file(inp)

    if not task:
        print(f"未提供研究任务。请编辑 `{inp.resolve()}` 填写内容后重试，")
        print("或使用：python main.py --mode research --task \"你的任务描述\"")
        return

    if task_from_cli:
        print("任务来源：--task 命令行参数\n")
    else:
        print(f"任务来源：{inp.resolve()}\n")

    print("正在检测 LLM API 连接（DNS + 试调用）...")
    try:
        ping = check_llm_connection()
        print(f"API 连接正常（试调用回复：{ping}）\n")
    except RuntimeError as e:
        print(f"\n错误：{e}\n")
        print("可先运行：python scripts/check_api_connection.py")
        return

    print("--- 开始执行（进度事件将实时打印）---\n")

    initial = {
        "messages": [HumanMessage(content=task)],
        "user_brief": task,
        "revision_round": 0,
        "paper_context": paper_context,
        "loaded_papers": loaded_papers,
        "session_id": tid,
        "run_id": run_id,
        "human_review_enabled": human_review,
        "skip_supplement": skip_supplement,
        "phase_status": {},
        "phase_artifacts": {},
    }

    _invoke_research_stream(app, initial, config)
    state = _graph_state_values(app, config)

    if human_review and _is_interrupted(app, config):
        _handle_human_review(app, config)
        state = _graph_state_values(app, config)

    _print_final_result(state)

    save_session_meta(
        tid,
        {
            "run_id": run_id,
            "user_brief": task[:500],
            "target_model": state.get("target_model"),
            "phase": state.get("phase"),
            "human_review": human_review,
            "started_at": datetime.now(timezone.utc).isoformat(),
        },
    )

    if not no_followup and state.get("phase") == "done":
        _research_followup_loop(app, config, tid, input_file=input_file)
    elif state.get("phase") == "done":
        print(f"\n续问示例：python main.py --continue --thread-id {tid}")


def _cmd_list_sessions():
    sessions = list_sessions()
    if not sessions:
        print("暂无已保存会话（.runs/sessions/）。")
        return
    print("最近会话：\n")
    for s in sessions:
        print(
            f"  thread_id={s.get('thread_id')}\n"
            f"    更新: {s.get('updated_at', '?')}\n"
            f"    模型: {s.get('target_model', '?')}\n"
            f"    阶段: {s.get('phase', '?')}\n"
        )


def main():
    parser = argparse.ArgumentParser(description="LangGraph Agent 框架")
    parser.add_argument(
        "--mode",
        choices=["chat", "research"],
        default="research",
        help="chat=通用 ReAct；research=语音鉴伪多 Agent 研究（默认）",
    )
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help="研究工作流的一次性任务描述（仅 research 模式）",
    )
    parser.add_argument(
        "--pdf",
        action="append",
        dest="pdfs",
        default=None,
        metavar="PATH",
        help="参考论文 PDF 路径，可多次指定：--pdf a.pdf --pdf b.pdf",
    )
    parser.add_argument(
        "--papers-dir",
        type=str,
        default="data/papers",
        help="自动加载目录下全部 PDF（默认 data/papers）",
    )
    parser.add_argument(
        "--no-papers-dir",
        action="store_true",
        help="不扫描 papers 目录，仅使用 --pdf 显式指定的文件",
    )
    parser.add_argument(
        "--thread-id",
        type=str,
        default=None,
        help="会话 ID（Checkpointer + Store）；不指定则自动生成",
    )
    parser.add_argument(
        "--continue",
        dest="continue_session",
        action="store_true",
        help="继续已有会话（续问 / 未完成的人工审批）",
    )
    parser.add_argument(
        "--human-review",
        action="store_true",
        help="定稿前人工审批（借鉴 pending_review / approve_phase）",
    )
    parser.add_argument(
        "--skip-supplement",
        action="store_true",
        help="跳过检查后的补充调研阶段",
    )
    parser.add_argument(
        "--no-followup",
        action="store_true",
        help="首轮完成后不进入交互式续问",
    )
    parser.add_argument(
        "--list-sessions",
        action="store_true",
        help="列出 .runs/sessions 中最近会话",
    )
    parser.add_argument(
        "--input-file",
        type=str,
        default=str(DEFAULT_INPUT_FILE),
        help=f"任务/续问输入文件（默认 {DEFAULT_INPUT_FILE}）",
    )
    args = parser.parse_args()

    if args.list_sessions:
        _cmd_list_sessions()
        return

    if args.mode == "chat":
        run_chat(thread_id=args.thread_id, input_file=args.input_file)
    else:
        papers_dir = None if args.no_papers_dir else args.papers_dir
        run_research(
            args.task,
            pdf_paths=args.pdfs,
            papers_dir=papers_dir,
            thread_id=args.thread_id,
            continue_session=args.continue_session,
            human_review=args.human_review,
            skip_supplement=args.skip_supplement,
            no_followup=args.no_followup,
            input_file=args.input_file,
        )


if __name__ == "__main__":
    main()
