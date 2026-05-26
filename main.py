"""Agent 框架入口：通用 ReAct 或语音鉴伪研究工作流。"""

import argparse
import sys

from langchain_core.messages import HumanMessage

from agent_framework import create_agent
from agent_framework.research import create_research_agent
from agent_framework.research.nodes import reset_sub_agent_app
from agent_framework.research.pdf_loader import clear_paper_store, load_paper_pdfs
from agent_framework.research.tools import collect_pdf_paths


def run_chat():
    app = create_agent()
    print("LangGraph ReAct Agent 已启动（输入 quit 退出）\n")

    while True:
        user_input = input("你: ").strip()
        if not user_input:
            continue
        if user_input.lower() in {"quit", "exit", "q"}:
            print("再见。")
            break

        result = app.invoke({"messages": [HumanMessage(content=user_input)]})
        last = result["messages"][-1]
        print(f"\nAgent: {last.content}\n")


def run_research(
    task: str | None = None,
    pdf_paths: list[str] | None = None,
    papers_dir: str | None = None,
):
    app = create_research_agent()
    print("语音鉴伪多 Agent 研究工作流已启动\n")
    print("流程：主控分析 → 子 Agent 并行调研 → 汇总初稿 → 检查 Agent → 可选补充调研 → 最终方案\n")

    paper_context = ""
    loaded_papers: list[str] = []
    paths = collect_pdf_paths(pdf_paths, papers_dir or "data/papers")
    if paths:
        clear_paper_store()
        reset_sub_agent_app()
        try:
            paper_context, store = load_paper_pdfs(paths)
            loaded_papers = list(store.keys())
            print(f"已加载 {len(loaded_papers)} 篇论文 PDF：{', '.join(loaded_papers)}\n")
        except Exception as e:
            print(f"警告：PDF 加载失败（{e}），将仅使用文本任务描述继续。\n")
    elif pdf_paths:
        print("警告：未找到有效 PDF 文件，请检查路径。\n")

    if not task:
        print("请输入研究任务（模型名称、现状、优化目标等），空行结束：")
        lines = []
        while True:
            line = input()
            if not line.strip() and lines:
                break
            if line.strip():
                lines.append(line)
        task = "\n".join(lines)

    if not task or not task.strip():
        print("未提供研究任务，退出。")
        return

    print("\n--- 开始执行（可能需数分钟，含多轮 LLM 与子 Agent 工具调用）---\n")

    result = app.invoke(
        {
            "messages": [HumanMessage(content=task)],
            "user_brief": task,
            "revision_round": 0,
            "paper_context": paper_context,
            "loaded_papers": loaded_papers,
        }
    )

    final = result.get("final_plan") or ""
    if final:
        print("\n" + "=" * 60)
        print("最终模型改进方案")
        print("=" * 60 + "\n")
        print(final)
    else:
        last = result["messages"][-1]
        print(f"\n{last.content}")

    out_path = "output_final_plan.md"
    if final:
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(final)
        print(f"\n已保存至 {out_path}")


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
    args = parser.parse_args()

    if args.mode == "chat":
        run_chat()
    else:
        papers_dir = None if args.no_papers_dir else args.papers_dir
        run_research(args.task, pdf_paths=args.pdfs, papers_dir=papers_dir)


if __name__ == "__main__":
    main()
