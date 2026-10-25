# CustomAgent

## 运行方式

需要 Python 3.11+。在项目根目录安装依赖：

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip --isolated install -e ".[dev]"
Copy-Item config.toml config.local.toml
```

在 `config.local.toml` 中填写 `api_key`，按需修改 `base_url`、`model`、`db_path` 和
`workspace_root`。该私有文件已被 Git 忽略；根目录的 `config.toml` 提供完整默认值和空密钥。
配置直接使用 TOML，不读取 `.env` 或旧环境变量。相对数据库、工作区路径均以配置文件所在目录为基准。

启动 HTTP 服务：

```powershell
.\.venv\Scripts\custom-agent.exe serve --config config.local.toml --port 8000
```

默认监听本机，接口浏览页为 `http://127.0.0.1:8000/docs`。省略 `--config` 时读取当前目录的
`config.toml`。
修改配置后重启进程生效。`--log-level info` 可查看 stderr 上的执行日志。

交互或单次 CLI：

```powershell
.\.venv\Scripts\custom-agent.exe chat --config config.local.toml --db data/cli.sqlite3 --user alice --session work
.\.venv\Scripts\custom-agent.exe chat --config config.local.toml --db data/cli.sqlite3 --user alice --session work --message "用 calculator 计算 (17+25)*3" --json
```

输入 `/exit` 退出。`--db` 覆盖配置中的数据库路径，按命令的当前目录解析。
重启时使用原数据库即可恢复。

未配置认证映射时，使用 `X-User-ID`；配置 `[api_tokens]` 后使用
`"私有令牌" = "alice"` 的映射和 `Authorization: Bearer ...`。

```powershell
$headers = @{"X-User-ID" = "alice"}
$body = @{session_id="work"; input="用 search 查询可恢复执行"; request_id="example-1"} | ConvertTo-Json
$run = Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/v1/runs -Headers $headers -ContentType application/json -Body $body
Invoke-RestMethod "http://127.0.0.1:8000/v1/runs/$($run.run_id)" -Headers $headers
curl.exe -N -H "X-User-ID: alice" "http://127.0.0.1:8000/v1/sessions/work/events"
```

`POST /v1/runs` 返回 202 和 `run_id`；可查询运行、异步任务、trace、会话消息和记忆。
会话接口支持创建、列表、改名、归档、恢复与空闲删除。归档拒绝新输入但继续已接收任务；
忙碌会话删除返回 409。完整接口与参数见运行中的接口浏览页。

本地验证使用 `.\.venv\Scripts\python.exe -m pytest -q`，默认不调用付费模型。
策略比较可运行 `.\.venv\Scripts\python.exe -m custom_agent.evaluate --config config.toml`；
真实模型测试的开启方式见 [tests/AGENTS.md](tests/AGENTS.md)。

## 系统设计

这是手写的模型与工具循环。HTTP 和 CLI 共用 Runtime；每条输入先落盘成为 run，
协调器再按 `(user_id, session_id)` 执行 FIFO。同会话等待异步工具时不允许插队，
不同会话可独立推进。默认每次运行最多 8 个持久化模型决策额度。

```text
CLI / HTTP -> SQLite run queue -> Runtime -> ContextManager -> Model
                                    |                         |
                                    +--- ToolRegistry <-------+
                                             |
                          sync handlers / durable async task
                                             |
                   SQLite checkpoints, results, events, attempts
```

| 模块 | 职责 |
| --- | --- |
| config / llm | TOML 加载校验、模型 HTTP 适配 |
| store / runtime | SQLite 事务、会话队列、恢复、尝试计数与故障处理 |
| tools | calculator、异步 mock search、session Todo、mock weather、read_docs |
| memory | 原子记忆抽取、会话内召回、证据片段和上下文预算 |
| api / cli / tracing | 使用入口、查询、SSE 与脱敏的逐次执行记录 |
| evaluate | 相同历史快照上的上下文策略比较 |

工具通过 Schema、handler 和执行模式注册。Todo 修改与对应结果原子提交，
已经提交的结果直接重放。异步任务先保存检查点再执行，重启后继续未结束任务；
尝试次数跨重启累计。

SQLite 保存会话、运行、消息、工具调用、任务、Todo、事件、记忆和执行尝试。
已有版本 1/2 数据库升级至版本 3，保留原有历史和检查点。
SSE 使用持久化事件 ID，支持 `Last-Event-ID` 重放，客户端按 ID 去重；
`follow=false` 在请求时固定回放上界。

`read_docs` 按模型给出的路径读取 `workspace_root` 内的 UTF-8 文本。
`tests/fixtures/sample.md` 是测试资料。search 和 weather 返回明确标注的本地 mock 数据。

Spec-first + BDD 通过模块 `AGENTS.md` 和可执行场景维护。
[模块契约](src/custom_agent/AGENTS.md) 保存具体事务、恢复、接口与追踪约束，
应用不会把这些文件作为运行时规则加载。

## Memory 的召回时机与放置方式

**召回时机**：每次 agent 模型决策前，Runtime 都调用 `ContextManager.build`，
根据当前 run 的输入召回当前会话记忆。工具返回后的下一次决策和恢复后的决策也会重新构建。
召回始终执行；只有上下文压力达到阈值时才进一步压缩和抽取。

**持久化放置**：原子记忆存入同一个 SQLite 的 `memories` 表，包含类型、稳定 key、
文本、实体、来源消息 ID、观察/适用时间和替代关系；原文保留在 `messages`，
压缩前缀位置记录在 `sessions.compacted_seq`。不另设跨会话记忆库。
Todo 和运行状态仍以各自业务表、精确检查点为准。

**模型上下文放置**：输入消息按以下顺序组装：

```text
system: SYSTEM
system: EXACT_EXECUTION_STATE        当前 run、决策计数、未完成调用
system: SESSION_REFERENCE_DATA       memory 原子条目 + evidence 原文片段
历史对话与近期完整交互
当前交互：用户输入、assistant 工具调用、tool 结果
```

召回采用会话内词项、实体和 key 匹配，有效目标、约束、待办语义事项优先保留；
历史追问可取回已替代决定。证据从当前 run 之前的原文定位，避免当前输入占用历史证据名额。

默认输入预算为 `16384 - 1024 回复预留 - 2048 工具结果预留`，工具 Schema 也计入输入。
接近输入额度的 70% 时先缩减旧工具观察，再对已结束的完整交互做结构化抽取，
最后缩小近期窗口；默认偏好保留 6 次完整交互。摘要请求单独预留输出空间。
