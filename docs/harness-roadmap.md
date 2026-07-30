# Coding Agent 最小可用 Harness 改造方案

## 0. 目标、边界与结论

本文基于当前仓库的实际代码制定，不是重新设计一套 Agent 平台。

本次改造的最小目标是：模型产出“没有工具调用的文本”时，只形成一个
**候选结果**；任务只有在显式状态机到达 `VERIFYING`、且
`VerificationGate` 接受当前证据后，才能进入 `COMPLETED`。修改类任务若验证失败，
必须进入有界的“诊断 → 修复 → 再验证”闭环。

本方案明确不做以下事情：

- 不替换 `Session`、`ToolRegistry`、`PermissionPolicy`、
  `RunQualityTracker`、`CompletionReport` 或 `ContextManager`。
- 不在第一阶段引入向量数据库、长期记忆、分布式队列、工作流 DSL、插件式调度器。
- 不把所有任务都强制成多步计划，也不把“验证”等同于“必须运行测试”。
- 不在本任务中实现 `RunOrchestrator`、`Planner` 或 `VerificationGate`。

推荐以 `src/harness/` 小包承载生命周期逻辑，但第一阶段只实现下文列出的核心类型。
当前 `pyproject.toml` 使用 `packages = ["src"]`，不会自动打包 `src.harness` 子包，
因此第一阶段若采用该目录结构，必须同步改成包含 `src.*` 的 package discovery。

---

## 1. 当前项目基线

### 1.1 实际执行链

当前 CLI 到完成报告的调用关系如下：

```text
src.main
  -> src.cli.main
     -> load_config
     -> JsonSessionStore / ProviderRegistry / ToolRegistry / PermissionPolicy
     -> AgentLoop.run_stream
        -> GitRunTracker.capture
        -> AgentLoop._initialise_session
           -> Session load/create
           -> ContextManager.initial_system_prompt
           -> 补齐上次中断的 pending tool call
        -> RunQualityTracker
        -> ContextManager.select
        -> ModelProvider.stream/complete
           -> OpenAICompatibleProvider -> httpx.AsyncClient
        -> 持久化 AssistantMessage 与 Usage
        -> 有 tool_calls:
           -> ToolRegistry.get + validate_tool_arguments
           -> PermissionPolicy.decide
           -> Tool.execute
           -> GitRunTracker.mark_agent_paths
           -> RunQualityTracker.observe
           -> ContextManager.observe_tool_result
           -> 持久化 ToolMessage
           -> 下一模型轮次
        -> 无 tool_calls:
           -> GitRunTracker.finish
           -> RunQualityTracker.build_report
           -> AgentEvent(COMPLETED)
     -> terminal_ui 渲染事件及 CompletionReport
```

这里存在本次改造的直接切入点：`src/agent.py` 当前在模型响应不含工具调用时立即
构建 `CompletionReport` 并发出 `COMPLETED`。`CompletionReport.incomplete` 即使包含
“未记录测试或 diff check”或“验证失败”，也只会被 UI 展示，不会改变完成状态和
CLI 退出码。

### 1.2 各模块现状与可复用边界

| 模块 | 当前真实职责 | Harness 中的处理 |
|---|---|---|
| `src/agent.py` | 896 行；会话初始化、上下文选择、模型流解析、工具调度、权限检查、Git/质量跟踪、完成判定、取消/超时 | 保留模型/工具执行内核；迁出 Run 生命周期和完成判定 |
| `src/context.py` | 初始工作区上下文、token 估算/压缩、文件读取版本和 stale context | 保持不动；后续只接收当前计划/步骤的有界摘要 |
| `src/tools.py` | Tool/ToolResult/ToolContext、静态注册表、参数校验、工作区读写工具 | 保持不动；Orchestrator 只调用既有注册表 |
| `src/permissions.py` | READ/WRITE/EXECUTE/NETWORK、ALLOW/ASK/DENY、交互/非交互策略 | 保持策略不动；在调用边界增加状态通知 |
| `src/sessions.py` | v2 JSON、v1/v2 读取、原子保存、权限收紧、消息/usage/context 恢复 | 扩展可选 Run 数据，不新建第二套 Session Store |
| `src/quality.py` | 观察文件修改、测试和 `git diff --check`，建议测试，构建 CompletionReport | 保留 tracker/report；VerificationGate 消费其结果 |
| `src/git_runtime.py` | 捕获任务起点、区分用户已有变更和 Agent 变更、受控 add/commit/diff check | 保持；RunOrchestrator 管理其生命周期和恢复降级 |
| `src/terminal_ui.py` | 渲染 `AgentEvent`、审批、TurnSummary、Human/JSON 输出 | 保持展示层；通过兼容适配接收新 Run 事件 |
| `src/providers.py` | Provider Protocol、Registry、领域错误 | 保持不动 |
| `src/openai_provider.py` | httpx Chat Completions、流式解析、重试、限长、错误映射 | 保持不动 |
| `src/cli.py` | 参数、依赖组装、session/resume/chat/run、退出码和 renderer | 保持命令面；改为组装 RunOrchestrator |

### 1.3 测试基线

项目文档声明的测试入口是：

```bash
python3 -m unittest discover -s tests -v
```

在当前系统 Python 3.14.2 环境直接运行：

- `httpx` 导入失败；
- 测试运行器报告 109 个已加载测试，106 个通过、3 个导入错误；
- 三处错误是 `tests/test_config.py`、`tests/test_openai_provider.py`，以及
  `tests/test_project_structure.py` 中 `src.openai_provider` 的模块边界导入；
- 因为 `test_openai_provider` 整个模块无法导入，其内部 10 个测试没有被加载，
  所以 109 不是完整测试总数。

在 `/tmp` 创建隔离虚拟环境并执行 `python -m pip install -e .` 后：

- 安装到 `httpx 0.28.1`，符合 `>=0.27,<1`；
- 完整发现并运行 120 个测试；
- 120 个全部通过。

现有测试覆盖了 Agent 边界、CLI 流、配置、Context、文件工具、Git、Provider、
Session、安全端到端、Shell 和只读工作区工具，但没有覆盖显式 Run 状态转换、
结构化计划、验证门或跨进程精确恢复。

### 1.4 `httpx` 原因与正确修复

分类结论：**开发环境未安装项目依赖**。

证据：

1. `pyproject.toml` 已将 `httpx>=0.27,<1` 声明在 `[project].dependencies`，
   所以不是项目依赖声明缺失。
2. `src/openai_provider.py` 在模块顶层直接 `import httpx`，默认 CLI Provider 也使用
   它；当前设计将其视为正式运行依赖，不是 optional dependency。
3. `tests/test_openai_provider.py` 直接使用 `httpx.MockTransport`，安装声明依赖后
   120/120 通过，说明不是测试隔离造成的代码失败。
4. 当前解释器中 `pip show independent-coding-agent httpx` 均为空，说明仓库代码被
   直接运行，但项目本身没有安装到该环境。

正确修复是让开发和 CI 在运行测试前安装项目依赖，例如：

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e .
.venv/bin/python -m unittest discover -s tests -v
```

不应通过跳过 Provider 测试、在测试中条件导入 `httpx`、捕获
`ModuleNotFoundError` 后伪造 Provider，或把失败模块排除出 discovery 来“修复”。
如果未来明确要让 OpenAI Provider 成为可选功能，才应把它迁入 dependency extra，
并同时改成延迟导入和清晰的配置错误；那是另一个产品决策，不是当前基线的正确修复。

---

## 2. 推荐的最小代码结构

```text
src/
  agent.py                    # 兼容门面 + 有界模型/工具执行内核
  quality.py                  # 保留 RunQualityTracker/CompletionReport
  sessions.py                 # 保留 Session Store；增加可选 run 数据
  harness/
    __init__.py               # 只导出公共 Harness 类型
    models.py                 # RunState、任务分类、Plan、VerificationResult、
                              # RunCheckpoint、RunEvent
    orchestrator.py           # RunOrchestrator
    planning.py               # Planner
    verification.py           # VerificationGate，复用 quality.py
    repair.py                 # RepairController
tests/
  test_harness_state.py
  test_planning.py
  test_verification_gate.py
  test_repair_loop.py
  test_run_recovery.py        # 第二阶段增加
  test_run_trace.py           # 第二阶段增加
```

不再增加 Repository、Memory、Tool Executor、Permission Manager 或 Report Builder
等平行抽象。`src/harness/` 只负责编排已有能力。

### 2.1 核心组件职责与位置

| 组件 | 文件 | 最小职责 | 明确不负责 |
|---|---|---|---|
| `RunState` | `harness/models.py` | 枚举九个状态，提供合法转移表和 terminal 判断 | 不执行副作用 |
| `PlanStep` | `harness/models.py` | `id`、目标、验收条件、验证方式、状态、尝试次数；支持一条步骤的微型计划 | 不保存对话或工具输出全文 |
| `ExecutionPlan` | `harness/models.py` | 任务类型、总体目标、全局验收条件、有序步骤、计划版本 | 不成为工作流 DSL |
| `Planner` | `harness/planning.py` | 根据用户意图、仓库证据和风险生成/修订结构化计划；简单任务可返回最小计划或无需计划 | 不调用工具、不绕过权限 |
| `VerificationResult` | `harness/models.py` | `passed`、验证方法、证据引用、失败条件、是否可修复、摘要 | 不替代 CompletionReport |
| `VerificationGate` | `harness/verification.py` | 将任务类型、Plan 验收条件、Quality/CompletionReport、Git 结果和实际验证证据合并为可执行的完成判定 | 不自己执行任意 Shell；不猜测测试通过 |
| `RepairController` | `harness/repair.py` | 对失败证据分类，生成一次有界诊断/修复指令，控制重试次数，决定继续、修订计划或失败 | 不拥有第二个模型循环 |
| `RunCheckpoint` | `harness/models.py` | 描述可恢复边界：状态、返回状态、计划、当前步骤、计数器、pending 操作、验证结果、trace cursor | 第一阶段不要求做到 in-flight 精确恢复 |
| `RunEvent` | `harness/models.py` | 统一 Run/Turn/Tool/Verification 因果标识和单调序号 | 不把大段 shell 输出复制多份 |
| `RunOrchestrator` | `harness/orchestrator.py` | 唯一状态机所有者；调用 Planner、Agent 执行内核、权限状态钩子、Gate、RepairController、Session | 不实现 Provider、Tool、权限策略、Context 或 Git 命令 |

任务分类使用一个很小的 `TaskType` 值类型，放在 `harness/models.py`。它不是新子系统。
建议把 `HIGH_RISK` 作为可叠加风险标记，而不是与
`INFORMATIONAL/INSPECTION/MODIFICATION/EXECUTION` 互斥的主类型；否则一个“发布构建”
会丢失它同时属于 EXECUTION 和 HIGH_RISK 的事实。

### 2.2 现有逻辑迁移

从 `src/agent.py` 迁到 `RunOrchestrator`：

- `GitRunTracker.capture/finish` 的 Run 级生命周期；
- `RunQualityTracker` 的创建与 Run 级持有；
- “无工具调用即完成”的分支；
- 最终 `CompletionReport` 构建时机；
- 顶层状态、计划、步骤、验证和修复尝试计数；
- 取消、总超时、上限错误到 `CANCELLED/FAILED` 的状态落盘；
- 权限 ASK 前后对应的 `WAITING_APPROVAL` 转移；
- 最终 `AgentResult.status` 和 CLI 退出语义。

保留在 `src/agent.py`：

- Provider 请求、流式 accumulator 和协议校验；
- tool call ID/fingerprint/重复调用保护；
- ToolRegistry 查找和参数校验；
- 既有 PermissionPolicy 的实际决策与“批准后重新构造请求”检查；
- ToolContext 构造、Tool 执行、输出流和结果限长；
- AssistantMessage/ToolMessage 的协议顺序；
- ContextManager 的选择和 tool result 观察。

`RunQualityTracker.observe` 可以继续由 Agent 执行内核调用，但 tracker 由
RunOrchestrator 注入并跨 repair attempt 复用。这样不需要复制一套 Tool 观察逻辑，
也不会像简单地多次调用当前 `AgentLoop.run()` 那样，每次都重置质量证据和 Git 基线。

### 2.3 `agent.py` 的逐步瘦身

1. 第一小步不改公共 `AgentLoop.run/run_sync/run_stream` 签名，只让它成为
   `RunOrchestrator` 的兼容门面。
2. 将当前 `_run_loop` 的“执行到模型给出候选文本”部分保留为内部执行方法；候选文本
   只返回给 Orchestrator，不发 `COMPLETED`。
3. 把质量 tracker 和 Git tracker 作为 Run 级依赖传入内部执行方法，不在每次 repair
   中重建。
4. 把完成报告、状态转移和 repair loop 完全移出 `agent.py`。
5. 等兼容测试稳定后，再考虑给内部执行方法改名；不在第一阶段顺手重写 streaming、
   tool validation 或 cancellation。

最终 `agent.py` 仍是 Provider-neutral 的模型/工具执行内核，而不是空壳；目标是消除
生命周期混杂，不追求机械地减少到某个行数。

### 2.4 不变的现有逻辑

- `ToolRegistry` 仍是工具的唯一注册和发现入口。
- `PermissionPolicy` 仍决定 READ/WRITE/EXECUTE/NETWORK 的 ALLOW/ASK/DENY。
- ASK 批准后重新计算 `PermissionRequest` 的防 TOCTOU 检查保持不动。
- Shell 的命令分类、workspace 限制和硬拒绝保持不动。
- `ContextManager` 仍负责 token 预算、压缩和 stale read 屏蔽。
- `Session` 仍保存完整本地消息历史；不把计划偷偷塞成普通 assistant 文本。
- `RunQualityTracker` 仍负责观察测试、diff check、文件修改和工具失败。
- `CompletionReport` 仍是用户可读的最终汇总；Gate 读取它，而不是复制其计算。
- `GitRunTracker` 仍负责用户已有变更与 Agent 变更的归属。
- Provider Protocol 和 OpenAI adapter 不因 Harness 改造而变化。

### 2.5 如何避免过度设计

第一阶段遵守以下约束：

- 一个进程内一个 RunOrchestrator；不做队列、租约或分布式锁。
- Plan 只有有序步骤，不做任意 DAG；若确需依赖，仅允许步骤引用更早的步骤。
- 分类最多一个主类型加 `high_risk` 标记，不做无限标签系统。
- RepairController 只返回“诊断并修复 / 修订计划 / 失败”三种决定。
- VerificationGate 只消费显式验收条件和已有证据，不引入规则 DSL。
- trace payload 使用有界 JSON，不保存第二份消息全文或 shell 全量输出。
- 每个修复尝试计入现有轮次、工具调用和总超时预算，并额外设置很小的 repair 上限；
  不能通过 repair loop 绕过现有限制。

---

## 3. 任务分类与策略

### 3.1 分类不是关键词匹配

初始分类应由以下证据共同决定：

1. 用户要求的交付物：回答、审查报告、文件变更、命令结果、发布/外部副作用。
2. 明示约束：只读、不要实现、必须测试、允许网络、要求提交/发布等。
3. 当前仓库证据：是否存在目标文件、测试入口、构建配置和 Git 状态。
4. 预期所需 operation：READ、WRITE、EXECUTE、NETWORK；这是意图上界，不代替
   `PermissionPolicy`。
5. Planner 的结构化判断：主类型、风险标记、证据、置信度和未决项。
6. 运行时实际工具调用：一旦出现 WRITE/EXECUTE/NETWORK，应只允许分类升级，
   不允许模型把任务降级来规避验证或审批。

当模型分类与硬证据冲突时，以更保守的分类为准。例如，模型称任务为 INSPECTION，
但提出 `apply_patch` 时，Run 必须升级为 MODIFICATION；提出 network/publish/credential
操作时，必须附加 HIGH_RISK。`PermissionManager` 仍逐个操作做最终审批。

### 3.2 分类矩阵

| 类型 | Planner | 是否必须验证 | 可接受验证 | 人工审批 |
|---|---|---|---|---|
| `INFORMATIONAL` | 默认不需要；复杂多约束问答可生成单步内部计划 | 需要 Gate，但可为轻量语义验证 | 回答覆盖用户问题、无虚构工具证据、引用的本地事实已读取；纯常识回答可记录“无需外部验证”的理由 | 通常不需要；若实际工具触发 ASK，按权限策略 |
| `INSPECTION` | 简单单文件查看不需要；跨模块审查或要求结构化报告时需要 | 必须验证证据覆盖，不强制测试 | 目标文件已读取且非 stale、发现与具体位置/证据关联、必要时运行只读静态检查或现有测试复现 | READ 通常不需要；执行型检查按现有策略 |
| `MODIFICATION` | 必须；明确的原子修改可用单步骤微型计划 | 必须 | 验收条件逐项检查、目标测试/构建、`git diff --check`、文件内容/格式/静态检查；没有测试时必须记录适当替代证据 | WRITE 按现有 ASK；后续 EXECUTE/NETWORK 分别判断 |
| `EXECUTION` | 单个明确且安全的命令可不规划；多命令、会产生产物或有依赖关系时需要 | 必须 | 实际 exit code、stdout/stderr 摘要、预期产物、构建/测试结果；不能以模型文字声称成功 | 识别的安全测试可 ALLOW；未知执行和写操作按现有 ASK |
| `HIGH_RISK` | 必须，且必须列出副作用、回滚/停止条件和验收条件 | 必须，且不能用自报成功 | 外部系统回执、目标状态复查、凭据未落盘、发布版本/commit 明确匹配；任何不确定副作用阻止完成 | 必须经过现有 PermissionPolicy；Harness 不缓存或伪造批准 |

`HIGH_RISK` 是叠加标记。例如 HIGH_RISK+MODIFICATION 和 HIGH_RISK+EXECUTION 使用
对应主类型的验证规则，再增加风险验证和审批要求。

### 3.3 结构化计划的最小字段

`ExecutionPlan`：

- `plan_id`、`version`、`task_type`、`high_risk`；
- `goal`；
- `acceptance_criteria: tuple[str, ...]`；
- `steps: tuple[PlanStep, ...]`；
- `created_at`，以及可选的 `revision_reason`。

`PlanStep`：

- 稳定的 `step_id` 和简短 `objective`；
- `acceptance_criteria`；
- `verification_methods`；
- `status`：`pending/active/satisfied/failed/skipped`；
- `attempts`；
- `skip_reason`，仅在确实不适用时允许。

不保存自由组合条件表达式。Gate 逐条返回 criterion 的结果和证据引用即可。

---

## 4. Run 状态机

### 4.1 全局转移规则

唯一允许进入 `COMPLETED` 的边是：

```text
VERIFYING --[VerificationGate passed]--> COMPLETED
```

模型停止、模型产生自然语言答案、Plan 步骤全部自报完成、工具成功、测试命令 exit 0，
都不是单独的完成条件。

推荐主路径：

```text
PREPARED -> PLANNING -> EXECUTING -> VERIFYING -> COMPLETED
    |           |            |           |
    |           |            |           +-> REPAIRING -> EXECUTING
    |           |            +-> WAITING_APPROVAL -> EXECUTING
    |           +-> EXECUTING
    +-> EXECUTING
```

任一非终态都可因明确取消进入 `CANCELLED`，因不可恢复错误、预算耗尽或非修复性验证
失败进入 `FAILED`。`FAILED`、`CANCELLED` 和 `COMPLETED` 都是当前 Run 的终态；用户继续
工作时创建一个带 `predecessor_run_id` 的新 Run，而不是篡改终态历史。

### 4.2 各状态定义

#### PREPARED

- 可进入来源：外部创建新 Run；读取旧 v1/v2 Session 时建立的兼容 Run；不接受其他
  RunState 直接回跳。
- 允许转移：`PLANNING`、`EXECUTING`、`FAILED`、`CANCELLED`。
- 转移条件：完成 Session/workspace/provider 校验、初始任务分类、限制快照和 Git
  baseline 捕获；需要 Planner 则到 PLANNING，否则到 EXECUTING。
- 持久化：`run_id`、`session_id`、请求摘要/哈希、主类型和 high-risk、workspace、
  provider/model、限制、创建时间、Git baseline 标识。
- 中断恢复：若准备尚未完成，重新执行只读准备；不得假定 Git baseline 已捕获。

#### PLANNING

- 可进入来源：`PREPARED`；`REPAIRING` 在验收条件或步骤需要修订时。
- 允许转移：`EXECUTING`、`FAILED`、`CANCELLED`。
- 转移条件：计划结构有效、至少有总体验收条件；无需计划的任务不能为了形式停留在
  此状态；Planner 失败在有预算时可重试，否则 FAILED。
- 持久化：计划版本、步骤、验收条件、分类证据、当前 planner attempt、修订原因。
- 中断恢复：丢弃未完整持久化的 planner 响应，从上一个有效计划版本重试；不保存
  半截 JSON 为有效计划。

#### EXECUTING

- 可进入来源：`PREPARED`（无需 Planner）、`PLANNING`、`REPAIRING`、
  `WAITING_APPROVAL`（批准或拒绝结果返回后）。
- 允许转移：`VERIFYING`、`WAITING_APPROVAL`、`FAILED`、`CANCELLED`。
- 转移条件：工具调用继续留在 EXECUTING；模型无工具调用的文本仅产生 candidate，
  然后转 VERIFYING；协议错误/上限耗尽按可恢复性进入 FAILED。
- 持久化：当前 step、model turn 号、tool call 计数、candidate、已完成 tool call ID、
  fingerprint、累计 usage、质量观察摘要、最后安全边界。
- 中断恢复：provider in-flight 从请求前 checkpoint 重发；完成状态未知的 READ 可重试；
  WRITE/EXECUTE 不盲目重放，记录不确定结果并进入恢复诊断，再由 Gate/Repair 决定。

#### VERIFYING

- 可进入来源：`EXECUTING` 产生 candidate；`WAITING_APPROVAL` 返回验证操作的审批结果。
- 允许转移：`COMPLETED`、`REPAIRING`、`WAITING_APPROVAL`、`FAILED`、`CANCELLED`。
- 转移条件：Gate 逐项评估验收条件；全部 required 条件通过且无 pending/uncertain
  操作才可 COMPLETED；可修复失败到 REPAIRING；不可修复或预算耗尽到 FAILED。
- 持久化：verification attempt、每条 criterion 的方法/证据/结果、CompletionReport
  快照引用、candidate 版本、失败是否可修复。
- 中断恢复：安全、只读或已识别测试的验证可重新运行；外部/写入型验证按未知副作用
  处理，不能用旧 candidate 直接完成。

#### REPAIRING

- 可进入来源：仅 `VERIFYING` 的失败结果。
- 允许转移：`EXECUTING`、`PLANNING`、`WAITING_APPROVAL`、`FAILED`、`CANCELLED`；
  若修复只要求重新收集安全验证证据，可经 EXECUTING 的零修改候选再回 VERIFYING，
  不设置 REPAIRING 直达 COMPLETED。
- 转移条件：RepairController 生成基于实际失败证据的诊断指令；需要改实现则 EXECUTING，
  验收目标变化则 PLANNING；修复动作本身需要审批则 WAITING_APPROVAL；重试上限或总
  预算耗尽则 FAILED。
- 持久化：repair attempt、失败分类、诊断摘要、允许修改的 step、剩余预算、关联的
  verification ID。
- 中断恢复：从最后一个 repair checkpoint 重新进入 EXECUTING/PLANNING；同一失败的
  attempt 计数不能因重启清零。

#### WAITING_APPROVAL

- 可进入来源：`EXECUTING`、`VERIFYING`、`REPAIRING`。
- 允许转移：持久化的 `return_state`（EXECUTING、VERIFYING 或 REPAIRING）、
  `FAILED`、`CANCELLED`。
- 转移条件：批准后重新构造并比对现有 `PermissionRequest`，一致才返回；拒绝也作为
  明确结果返回原状态，让模型/控制器选择替代方案；审批通道错误可 FAILED。
- 持久化：原状态、tool/verification ID、PermissionRequest 的无密钥结构化摘要、
  创建时间；**不持久化“永久批准”或凭据**。
- 中断恢复：重新展示请求并再次询问，绝不把进程中断前的未落盘点击视为批准。

#### COMPLETED

- 可进入来源：仅 `VERIFYING`。
- 允许转移：无。
- 转移条件：Gate passed、Plan required 条件满足、无 pending approval、无未知副作用、
  CompletionReport 已基于当前 Git/质量状态生成。
- 持久化：最终 candidate、VerificationResult、CompletionReport、结束时间、usage、
  plan 最终版本、trace cursor。
- 中断恢复：直接返回已持久化结果；后续用户输入创建新 Run。

#### FAILED

- 可进入来源：所有非终态。
- 允许转移：无。
- 转移条件：不可恢复协议/存储错误、修复或资源预算耗尽、非修复性 Gate 失败、恢复时
  无法确定安全状态。
- 持久化：安全错误类型/摘要、失败状态、最后 checkpoint、未满足条件、报告和 trace
  cursor；不保存异常中的密钥。
- 中断恢复：展示失败；用户要求继续时创建 successor Run，从安全 checkpoint 和当前
  workspace 重新准备。

#### CANCELLED

- 可进入来源：所有非终态。
- 允许转移：无。
- 转移条件：CancellationToken、用户中断或上层明确取消。
- 持久化：取消发生状态、时间、pending operation、最后安全 checkpoint、usage。
- 中断恢复：不自动继续；显式 resume 创建 successor Run。若存在中断 tool call，
  保留当前“补 interrupted ToolMessage”的安全语义，并进入恢复诊断。

### 4.3 完成门的最低判定

`VerificationGate` 至少检查：

1. 当前状态确为 VERIFYING。
2. 当前 candidate 与当前 plan version 对应。
3. 所有 required `PlanStep` 和 acceptance criterion 有证据，不接受模型自报。
4. MODIFICATION 的最终 worktree 仍包含预期修改。
5. 适当的验证已经实际执行；存在可发现测试而未执行时，默认失败而非仅提示。
6. 已执行的测试/diff check 没有失败。
7. 没有 pending approval、未关闭 tool call 或未知结果的写操作。
8. repair 次数、模型轮次、工具调用和总超时均未越界。

“未测试”不是一律失败的同义词。INFORMATIONAL 可用回答覆盖检查，INSPECTION 可用读取
证据覆盖，无法运行测试的 MODIFICATION 可使用明确的静态/文件级/构建替代验证；但是
必须由计划预先声明或由 Gate 记录为什么替代方式足够，不能仅由最终回答说“应该可以”。

---

## 5. Session、Checkpoint 与 Trace

### 5.1 Session JSON 向后兼容

当前 Session schema 是 v2，读取 v1/v2，写入时统一升级为 v2；v1 缺少 context 时使用
空 `ContextState`。建议 Harness 首次持久化变更将 schema 升为 v3：

```json
{
  "schema_version": 3,
  "session_id": "...",
  "workspace": "...",
  "created_at": "...",
  "updated_at": "...",
  "provider": "...",
  "model": "...",
  "messages": [],
  "usage": {},
  "context": {},
  "runs": {
    "active": {},
    "recent": []
  }
}
```

兼容规则：

- v1：沿用现有升级逻辑，补空 context 和空 runs。
- v2：原字段原样解析，补空 runs；收到新 prompt 时创建 PREPARED Run。
- v3：解析 active Run 和有界 recent summaries。
- v3 保存仍使用现有同目录临时文件、`fsync`、`0600` 和原子 replace。
- 未知未来版本继续拒绝，不做静默猜测。
- messages、usage、context 的字段和语义不改。
- `runs` 只存 active checkpoint 和最近若干 Run 摘要；完整 trace 第二阶段使用同一
  Session 目录下的有界 sidecar，Session JSON 只保存 trace cursor，避免 32 MB 文件
  被高频事件快速撑满。
- 不持久化 API key、环境快照、审批凭据或未经限长的工具输出。

现有 `test_session_file_contains_only_versioned_safe_metadata` 对顶层键做精确断言，升级时
必须显式更新为包含 `runs`，并新增 v1/v2/v3 迁移测试，而不是放宽成“任意键都可以”。

### 5.2 RunCheckpoint 最小内容

- Run/Session ID、schema version、当前状态和 `return_state`；
- task type、high-risk、当前 plan version/step；
- turn/tool/repair/verification 计数和剩余预算；
- 已完成与 pending tool call ID；
- pending approval 的安全摘要；
- candidate 版本和 VerificationResult 摘要；
- Quality tracker 可恢复摘要；
- Git baseline head、原始 dirty path/digest、agent paths；
- ContextState 仍由 Session.context 保存，只引用其版本；
- trace 最后序号。

`GitRunTracker` 当前包含内存中的文件快照且没有序列化接口。第二阶段不应仓促把任意
二进制工作区内容复制到 checkpoint。MVP 恢复可持久化 path/digest；若无法证明一个
恢复后的变更是 agent-only，就保守标为 overlapping，并阻止无证据完成。

### 5.3 统一 RunEvent

建议字段：

```text
schema_version, event_id, sequence, timestamp,
run_id, session_id,
turn_id?, step_id?, tool_call_id?, verification_id?,
kind, state_from?, state_to?, payload
```

因果层级：

```text
Run
  -> Turn 1..N
     -> Model request/response
     -> Tool call 0..N
  -> Verification attempt 1..N
     -> evidence/check 1..N
  -> Repair attempt 0..N
  -> terminal result
```

`sequence` 在单 Run 内严格单调；`tool_call_id` 继续使用 Provider 的稳定 ID；
verification 和 repair 使用 Harness 生成的 ID。CompletionReport 从当前状态和事件
证据派生，但仍保存最终快照，不能反过来把 report 当作唯一审计记录。

---

## 6. CLI 与 Terminal UI 兼容

现有命令保持：

- `agent run ...`
- `agent chat`
- `agent resume [SESSION_ID]`
- `agent sessions ...`
- `--json`、`--no-color`、workspace/provider/model 等参数

兼容方式：

1. `AgentLoop` 公共入口先保留，内部委托 RunOrchestrator。
2. 默认 Human 输出继续显示 text/tool/output/result 和最终 CompletionReport。
3. 现有 `AgentEventKind.TEXT_DELTA/TOOL_CALL/TOOL_OUTPUT/TOOL_RESULT/COMPLETED` 保留为
   兼容投影；只有 Gate 通过时才投影 `COMPLETED`。
4. 新状态/计划/验证 RunEvent 初期可由 renderer 选择性显示，不强迫旧 renderer 处理。
5. JSON Lines 现有 `type` 和字段不重命名；可增添 `run_id/state/schema_version`，
   但不能删掉 `session_id/usage/turns/tool_calls/report`。
6. `run` 仅在 RunState.COMPLETED 时退出 0；FAILED 退出 1；CANCELLED 保持 130。
7. chat 中每个用户 prompt 是同一 Session 下的新 Run；历史对话继续复用。
8. `resume` 遇到 non-terminal active Run 时恢复它；没有 active Run 时保持当前
   “恢复会话并等待/接受新 prompt”的行为。
9. `HumanRenderer.approve` 仍是 UI 回调；Harness 通过无 UI 依赖的状态钩子观察
   WAITING_APPROVAL，避免 `harness` 反向导入 `terminal_ui`。

---

## 7. 分阶段迁移

## 阶段一：状态机、结构化 Plan、Verification Gate、Repair Loop

### 修改文件

- `src/agent.py`
- `src/quality.py`
- `src/sessions.py`
- `src/cli.py`
- `src/terminal_ui.py`
- `src/port_manifest.py`
- `pyproject.toml`（让安装包包含 `src.harness`）
- `README.md`
- 相关现有测试，尤其 `test_agent.py`、`test_cli_streaming.py`、
  `test_sessions.py`、`test_git_integration.py`、`test_security_e2e.py`

### 新增文件

- `src/harness/__init__.py`
- `src/harness/models.py`
- `src/harness/orchestrator.py`
- `src/harness/planning.py`
- `src/harness/verification.py`
- `src/harness/repair.py`
- `tests/test_harness_state.py`
- `tests/test_planning.py`
- `tests/test_verification_gate.py`
- `tests/test_repair_loop.py`

### 风险

- 当前测试明确期望纯文本一轮即 `COMPLETED`；必须按 INFORMATIONAL 轻量 Gate 保持该
  行为，而不是把所有文本回答卡死。
- Quality/Git tracker 若仍在每个 AgentLoop segment 初始化，会丢失 repair 前证据。
- 审批是同步 callback，若状态通知插入位置不正确，会出现 WAITING_APPROVAL 未落盘或
  批准后状态错误。
- 额外 Planner 轮次可能消耗 token；简单任务必须绕过或使用微型计划。

### 兼容性

- 保留 AgentLoop、CLI 命令和旧事件投影。
- Session v1/v2 可读，写 v3。
- Provider、Tool 和权限接口不变。
- JSON 输出只做加法式扩展。

### 测试方案

- 表驱动测试覆盖每条合法/非法状态边，特别断言任何非 VERIFYING 状态不能 COMPLETED。
- 纯文本 INFORMATIONAL 经轻量 Gate 一轮完成。
- MODIFICATION 纯文本 candidate 在未验证时不能完成。
- CompletionReport.incomplete 阻止完成。
- 验证失败进入 REPAIRING，修复后再次验证才完成。
- repair 上限、总 turn/tool/timeout 仍有效。
- 审批前进入 WAITING_APPROVAL；批准、拒绝、取消都回到正确状态。
- 运行现有 120 个测试，更新的是语义预期而不是删除/跳过断言。

### 完成标准

- 唯一完成边为 VERIFYING → COMPLETED。
- 结构化 Plan 和 acceptance criteria 可序列化。
- MODIFICATION 未验证不能成功退出。
- 至少一个端到端测试证明“失败验证 → 诊断 → 修复 → 再验证 → 完成”。
- CLI/Human/JSON 现有主路径兼容，完整测试全绿。

## 阶段二：Checkpoint、结构化 Trace、精确恢复

### 修改文件

- `src/harness/models.py`
- `src/harness/orchestrator.py`
- `src/sessions.py`
- `src/agent.py`
- `src/git_runtime.py`
- `src/cli.py`
- `src/terminal_ui.py`

### 新增文件

- `src/harness/checkpoint.py`
- `src/harness/events.py`
- `tests/test_run_recovery.py`
- `tests/test_run_trace.py`

### 风险

- 对已开始 WRITE/EXECUTE 的盲目重放会重复副作用。
- trace 高频写入可能拖慢工具输出或撑大 Session。
- Git 原始 baseline 快照持久化可能扩大敏感数据面。
- Provider 请求没有跨进程幂等键，恢复只能从安全边界重发。

### 兼容性

- Session JSON 继续使用 v3，新增字段必须可选；如字段语义不兼容才升 v4。
- 旧 Session 无 checkpoint 时走现有 pending tool call interrupted 路径。
- 默认终端输出仍走 legacy event adapter；trace 是额外审计面。

### 测试方案

- 在 model 请求前后、tool 开始/完成、审批等待、验证开始/完成注入崩溃。
- 验证恢复后 sequence 单调、ID 关联完整、不重复已确认的写操作。
- pending approval 重启后必须重新询问。
- v1/v2/v3 Session round-trip 和损坏 sidecar 隔离。
- trace 限长、原子写入、secret redaction 和文件权限测试。

### 完成标准

- 可从最后安全 checkpoint 恢复当前 step、计数器、计划和验证状态。
- 不会因恢复盲目重放未知结果的副作用。
- Run/Turn/Tool/Verification 事件可按 ID 和 sequence 重建因果链。
- active Run 完成后写入有界摘要并清除 active checkpoint。

## 阶段三：Context Retrieval 和项目知识

### 修改文件

- `src/context.py`
- `src/harness/planning.py`
- `src/harness/orchestrator.py`
- `src/sessions.py`

### 新增文件

- `src/harness/project_knowledge.py`（只保存带来源和文件版本的项目知识索引）
- `tests/test_context_retrieval.py`

### 风险

- 检索到 stale 文件、越权路径或仓库内恶意指令。
- 项目知识与当前 Git/worktree 不一致。
- 索引和摘要挤占现有 context token 预算。

### 兼容性

- `ContextManager.select` 仍是唯一 Provider-facing context 入口。
- 文件版本/stale read 机制继续作为可信度基础。
- 没有索引时退回当前目录摘要 + 显式 read tools。

### 测试方案

- 版本变化使检索证据失效。
- 检索内容有界、路径在 workspace 内、普通文件内容不被无条件预载。
- 压缩仍保留 active plan、验收条件和未完成 tool group。
- 恶意仓库文本仍只作为 untrusted data。

### 完成标准

- Planner/Gate 能引用带版本的项目证据。
- 检索失败不破坏当前 Agent 流程。
- token 预算和 stale context 测试继续通过。

## 阶段四：Sandbox、长期 Memory 和生产级可观测性

### 修改文件

- `src/shell_tools.py`
- `src/tools.py`
- `src/permissions.py`
- `src/harness/orchestrator.py`
- `src/sessions.py`
- `src/logging_config.py`

### 新增文件

- `src/sandbox.py`
- `src/memory.py`
- `src/observability.py`
- `tests/test_sandbox_security.py`
- `tests/test_memory_isolation.py`
- `tests/test_observability_redaction.py`

这些名称只表示边界，不预先固定某个 Sandbox、存储或 telemetry 供应商接口。

### 风险

- OS sandbox 的平台差异与逃逸面。
- 长期 Memory 污染、隐私、过期和跨项目泄漏。
- telemetry 泄漏 prompt、文件内容、凭据或工具输出。

### 兼容性

- Tool 和 Permission 接口保持稳定，Sandbox 放在 Tool 执行边界下方。
- Memory 只提供候选上下文，由 ContextManager 过滤。
- 可观测性默认只发 ID、状态、计数和时延，不发内容。

### 测试方案

- 平台 sandbox 边界、网络隔离、文件系统逃逸、资源限制。
- Memory workspace/session 隔离、删除、过期和污染测试。
- trace/metrics 脱敏、背压、exporter 故障不影响 Run 正确性。

### 完成标准

- Shell 执行有真实 OS 隔离，而不仅是当前策略分类。
- Memory 有明确生命周期、用户控制和来源。
- 生产 trace 能关联 Run/Turn/Tool/Verification，且通过脱敏与负载测试。

---

## 8. 当前设计对迁移的阻碍

没有不可迁移的架构死结，但有六个需要在第一、二阶段显式处理的阻碍：

1. **完成判定与模型循环耦合。** `src/agent.py` 把“无 tool_calls”直接解释为
   `COMPLETED`，也是本次必须先切断的边。
2. **AgentLoop 职责过多。** Git/quality/session/context/permission/tool/provider 和 UI
   事件均在同一个循环汇合，不能通过在外层简单重复调用 `run()` 获得正确 repair loop，
   否则质量证据和 Git baseline 会重置。
3. **审批边界不可观测。** `PermissionPolicy.decide()` 是同步 callback，当前没有
   entering/leaving WAITING_APPROVAL 的状态钩子；应加编排钩子，不应复制 PermissionManager。
4. **可恢复数据仅驻内存。** fingerprints、tool count、RunQualityTracker 和
   GitRunTracker 的关键状态没有 Session codec。当前 resume 只能给 pending tool call
   添加 interrupted 结果，不能恢复 step/verification/repair attempt。
5. **事件缺少因果 ID。** `AgentEvent` 只有 kind/text/tool/result，没有 run_id、
   turn_id、verification_id、sequence，也没有状态转移事件。
6. **打包配置阻止直接增加子包。** `pyproject.toml` 只列出 `packages = ["src"]`；
   采用 `src/harness/` 时必须改 package discovery 并增加安装后 import 测试。

另有一个次要问题：`GitRunTracker.baseline_prompt()` 已实现但当前 AgentLoop 没有调用。
这不阻止 Harness，但说明 Git baseline 目前主要用于工具约束和最终报告，而不是明确的
Run 计划输入。第一阶段可把 baseline 摘要作为 Planner 的结构化输入，仍由
GitRunTracker 提供，不复制 Git 检测。

总体判断：现有 Tool、权限、Session、Context、Git 和质量模块边界足以支撑最小
Harness；迁移难点集中在 `agent.py` 的生命周期解耦和第二阶段的安全恢复，不需要推倒
现有执行与安全基础。
