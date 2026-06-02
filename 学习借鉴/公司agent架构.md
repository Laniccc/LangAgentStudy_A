# fb-agentic-ai-da Agent Core 架构分析与借鉴

本文档基于某公司 **fb-agentic-ai-da Agent Core** 架构图，对照本仓库（LangAgent_A / LangGraph Agent 框架）进行优点分析与可借鉴项总结。

---

## 一、架构概览

该系统基于 **LangGraph** 与 **DeepAgents**，采用 **Supervised Agent（监督式根 Agent）** 模式：中央 Root Agent / Supervisor 编排各阶段，配合持久化 Store、异步后台任务与实时流式观测。

### 1.1 分层结构

| 层 | 组件 | 职责 |
|----|------|------|
| **入口** | User & FastAPI Chat | WebSocket/SSE、鉴权与会话、流式 text delta |
| **执行** | Run Worker | 创建 Run、初始化上下文、入队交给 Agent Core |
| **核心** | Root Agent / Supervisor | 编排阶段、调用工具、维护 PipelineState、委托与评估 |
| **持久化** | LangGraph Store | phase_state、phase_outputs、artifact_records、jobs、memory |
| **后台** | Background Poller / Reconcile | 轮询外部服务、对账 Job、发布草稿 |
| **观测** | Streaming Observability | 将进度与 tool I/O 流式回传前端 |

### 1.2 Root Agent 配置栈

- **Runtime Prompt**：运行时动态指令
- **Media Ingress**：媒体/文件输入
- **Input Contract**：输入结构约定
- **Pipeline Gate**：按当前状态从 Store 过滤可见工具
- **Plan Announcement**：向用户宣告计划步骤
- **Error Handler**：专用失败处理逻辑

### 1.3 工具三分法

| 类型 | 职责 | 示例 |
|------|------|------|
| **Phase Tools** | 产出阶段产物 | `analyze_music`、`generate_storyboard`、`merge_video` |
| **Control Tools** | 驱动状态迁移 | `approve_phase`、`rollback_to_phase`、`reconcile_job` |
| **Utility Tools** | 横切支撑能力 | `retrieve_memory`、`check_credits` |

### 1.4 双状态模型

- **PhaseState**：某一阶段的权威数据（源）
- **PipelineState**：从 Store 推导的整体进度（投影）

Supervisor 读投影、写各 phase 明细，便于并行阶段与异步 Job 完成后保持一致。

### 1.5 Human-in-the-Loop（HITL）生命周期

```
Phase Tool 产出 → pending_review
    → 人工 approve_phase → approved
    → 异步工具写入 JobRecord (running)
    → Background Poller 更新
    → succeeded → 发布 draft artifact
```

### 1.6 交互类型（图例）

- **实线黑箭头**：请求 / 控制流
- **实线蓝箭头**：写入 Store
- **虚线紫箭头**：从 Store 读取
- **虚线橙箭头**：异步 / 事件流

---

## 二、核心优点分析

### 2.1 分层清晰：入口 / 执行 / 核心 / 持久化

将 **HTTP 接口**、**Run 调度**、**Agent 推理**、**持久化** 解耦。长任务通过 Run Worker 入队，不阻塞 API；状态落在 Store，支持恢复与审计。

**对比本项目**：当前以 **CLI 一次性 `invoke`** 为主（`main.py`），无 Run 抽象与独立 Worker，长研究任务中断后难以续跑。

### 2.2 工具三分法（Phase / Control / Utility）

- **Phase**：业务产出
- **Control**：流程与状态（审批、回滚、对账）
- **Utility**：记忆、配额等通用能力

避免 LLM 将「审批」「回滚」与「生成内容」混在同一工具面，扩展与排错可按类别进行。

**对比本项目**：`agent_framework/research/tools.py` 以领域 **Phase 类**工具为主（检索、知识库、PDF）；`ResearchState.phase` 在图节点中使用，尚无独立 **Control 工具**（如跳过补充调研、人工确认后再修订）。

### 2.3 双状态：PhaseState vs PipelineState

阶段明细为源，总进度为投影，适合多阶段并行与异步 Job 完成后的状态对齐。

**对比本项目**：`ResearchState` 已含 `sub_task_results`、`draft_plan`、`review_feedback`、`final_plan` 等，但均在 **单次图运行的内存状态** 中，无 LangGraph Store 持久化，也未文档化「投影层」概念。

### 2.4 Pipeline Gate（按状态暴露工具）

根据 Store 中当前阶段决定 **哪些工具对 LLM 可见**，避免错误阶段调用错误能力。

**对比本项目**：用 **固定图边**（`conditional_edges` + `Send`）约束流程，比纯工具门控更硬、更稳；若未来演进为「单 Supervisor + 大量工具」，Pipeline Gate 值得借鉴。

### 2.5 HITL 生命周期完整

生成类任务经 **显式人工审批** 再进入耗时/高成本步骤；异步任务由 **JobRecord + Poller** 处理，主 Agent 不阻塞。

**对比本项目**：`reviewer` 节点为 **自动批判**（Agent-in-the-loop），非 Human-in-the-loop。若研究方案需用户点头再定稿，可参考 `pending_review` / `approve_phase`。

### 2.6 可观测性与配置栈

Streaming Observability 将文本 delta、tool 输入输出、阶段进度推送到前端；Runtime Prompt、Input Contract、Error Handler 使运行时可调、失败可治理。

**对比本项目**：以 `print` 与最终输出文件（如 `output_final_plan.md`）为主，缺少结构化事件流。

### 2.7 异步与韧性

Background Poller 轮询外部渲染等服务，与主图解耦，适合分钟级以上外部 API。

**对比本项目**：子 Agent 并行在图内通过 `Send` 完成；若未来接入「外部训练 / 批量评测」，需要类似 Poller 机制。

---

## 三、与本项目（LangAgent_A）对照

### 3.1 架构示意



### 3.2 对照表

| 维度 | 对方架构 | LangAgent_A |
|------|----------|-----------|
| 编排模式 | Supervisor + 工具 + Store | 固定 6 节点流水线 + 条件边 |
| 多 Agent | 阶段工具 + 委托 | 主控 + `Send` 并行子 Agent ✅ |
| 质量把关 | 人工 `approve_phase` | `reviewer` 自动批判 ✅ |
| 持久化 | LangGraph Store 多命名空间 | 单次运行内存状态 |
| 用户入口 | API + 流式 | CLI |
| 工具分类 | Phase / Control / Utility | 以研究 Phase 工具为主 |
| 长任务 | Job + Poller | 同步 `invoke` 至结束 |

### 3.3 本项目已具备的优势

不必推倒重来，以下设计已与对方部分理念对齐：

- LangGraph **多节点 + 并行 `Send`**（`agent_framework/research/graph.py`）
- **角色化状态** `ResearchState`：创新方向、子任务结果、初稿、审查、终稿
- **子 Agent 内 ReAct + 领域工具**
- 文档 [`docs/需求到实现.md`](../docs/需求到实现.md) 中的模式选型表（HITL、Router 等）与扩展路径清晰

### 3.4 本项目研究工作流（参考）

详见 [`docs/语音鉴伪研究工作流.md`](../docs/语音鉴伪研究工作流.md)：

```
START → orchestrator_analyze
      → [并行] sub_agent × N
      → orchestrator_synthesize → reviewer
      → plan_supplement → [可选] sub_agent
      → orchestrator_revise → END
```

---

## 四、可借鉴项（按优先级）

### P0 — 投入小、与现有研究流最契合

#### 1. 工具语义分层

在 `research/tools.py` 或文档中标注三类（可先文档化，再逐步暴露为 Control 工具）：

| 类型 | 研究场景示例 |
|------|----------------|
| Phase | `search_papers`、`query_antispoofing_kb` |
| Control | `mark_phase_complete`、`request_human_review`、`skip_supplement_round` |
| Utility | `read_paper_snippet`、连接检查 |

Control 可继续以 **图节点** 实现，与固定拓扑并存。

#### 2. 强化「阶段 + 产物」状态

向 PhaseState 靠拢，在 `ResearchState` 中增加例如：

```python
# 概念示例
phase_status: dict[str, Literal["pending", "running", "done", "pending_review"]]
phase_artifacts: dict[str, str]  # 如 analyze -> directions_json, synthesize -> draft_md
```

与现有 `phase`、`draft_plan`、`final_plan` 对齐，便于后续接 Store 或 HITL。

#### 3. 将 reviewer 升级为可选 HITL

- **自动模式**（默认）：`reviewer` → `plan_supplement` → `orchestrator_revise`
- **人工模式**：`reviewer` 后 `interrupt_before` 或 CLI 提示，用户 `approve` 或提交修改意见后再 `revise`

对应对方 `pending_review` → `approve_phase`；[`docs/需求到实现.md`](../docs/需求到实现.md) 已提及 `interrupt_before`。

#### 4. Plan Announcement（计划对用户可见）

主控在 `orchestrator_analyze` 后输出结构化说明：本轮并行方向数、后续汇总与审查步骤，写入日志或 Markdown 章节，提升可解释性。

---

### P1 — 中期：体验与可恢复性

#### 5. 轻量 Streaming / 进度事件

不必先上 FastAPI：在节点或 `main.py` 使用 `astream_events` / callback 输出 `[phase] orchestrator_analyze started`、子 Agent 工具调用摘要，为后续 SSE 打基础。

#### 6. Checkpointer + 可选 Store

对 `ResearchState` 使用 LangGraph `MemorySaver` / `SqliteSaver` 与 `thread_id`，支持长研究断点续跑；再将 `sub_task_results`、`draft_plan` 等逐步迁入 Store 命名空间（对齐 `phase_outputs`）。

#### 7. Run 抽象

封装 `Run(id, mode, input, status, created_at) → invoke/astream`，CLI 仅创建 Run 并展示状态，与 Run Worker 思想一致，便于日后替换为 API。

---

### P2 — 有外部长任务需求时再上

#### 8. JobRecord + Background Poller

接入「提交集群训练 / 等待 ASVspoof 评测」时，引入 `jobs` 与轮询协程；主图在 `running` 时结束或挂起，Poller 写回后触发 `reconcile_job` 类 Control 能力。

#### 9. Pipeline Gate

当工具数量多到「单 Supervisor 绑定全部工具」时，按 `phase` 动态 `bind_tools(subset)`，可与固定图混合使用。

#### 10. FastAPI + SSE

产品化时复用 P1 的事件模型，避免在节点内散落 `print`。

---

## 五、不建议照搬的部分

| 项 | 原因 |
|----|------|
| 全量 Store 五类文件夹 | 研究 CLI 阶段过重，按命名空间按需引入 |
| DeepAgents + 媒体管线工具链 | 领域不同；保留 PDF + 检索工具更合适 |
| Credits / 多租户 Auth | 无商业化需求可跳过 |
| 完整 Run Worker 集群 | 可先 Run 数据结构 + 单进程队列 |

---

## 六、推荐学习路径

1. 精读 [`docs/语音鉴伪研究工作流.md`](../docs/语音鉴伪研究工作流.md)，与对方图的 **Supervisor / Phase / HITL** 三条线做概念映射。
2. 在 `ResearchState` 增加 `phase_status` 并文档化 Phase/Control/Utility（P0）。
3. 实现 `main.py --human-review`：`reviewer` 后中断，人工批准再进入 `orchestrator_revise`（P0）。
4. `graph.compile(checkpointer=...)` + `thread_id`（P1）。
5. 有 Web 需求时再接入 FastAPI 与 `astream_events`（P1→P2）。

---

## 七、总结

| 对方强项 | 本项目强项 | 优先借鉴 |
|----------|------------|----------|
| 产品化运行时（Run/Worker/API/流式） | 领域固定多 Agent 研究流水线 | 工具三分法、阶段状态 |
| 持久化状态机 + Store | 并行 Send + 自动 reviewer | 可选 HITL、Checkpointer |
| HITL + 异步 Job + Poller | ReAct 子 Agent + 领域工具 | 进度事件、Run 抽象 |

**一句话**：对方强在 **产品化运行时、持久化状态机、工具与 HITL/异步的标准化**；本项目强在 **语音鉴伪领域的固定多 Agent 研究流水线**。最值得借鉴的是 **工具三分法、阶段状态 + 可选人工审批、可观测与 Checkpointer**；无需为对齐大厂而引入完整视频生成管线架构。

---

## 附录：相关本仓库文件

| 文件 | 说明 |
|------|------|
| `agent_framework/graph.py` | 通用 ReAct 图 |
| `agent_framework/research/graph.py` | 研究工作流图 |
| `agent_framework/research/state.py` | `ResearchState` |
| `agent_framework/research/tools.py` | 研究工具 |
| `agent_framework/research/nodes.py` | 主控 / 子 Agent / reviewer 节点 |
| `main.py` | CLI 入口 |
| `docs/需求到实现.md` | 需求协作与模式选型 |
| `docs/语音鉴伪研究工作流.md` | 研究工作流说明 |

---

## 八、已落地借鉴项（本项目）

| 借鉴点 | 落地位置 |
|--------|----------|
| 工具 Phase 分类 | `agent_framework/research/tools.py` → `ToolCategory` / `PHASE_TOOL_REGISTRY` |
| `phase_status` / `phase_artifacts` | `agent_framework/research/state.py` |
| Plan Announcement | `orchestrator_analyze` → `plan_announcement` |
| Human-in-the-loop | `--human-review` + `human_review_gate` + `interrupt_before` |
| Checkpointer 多轮 | `.runs/checkpoints.sqlite` + `--thread-id` / `--continue` |
| 文件 Store | `.runs/store/{session_id}/phase_outputs/`、`memory/` |
| 多轮续问 | `follow_up` 节点 + CLI 续问循环 |
| 进度 Streaming | `agent_framework/research/events.py` |

---

*文档生成目的：作为学习目标存档；「已落地」章节随实现更新。*
