# 模型批量压测自动化工具施工计划

## 交付目标

- 新增可直接部署的单文件工具 `model_eval_script.py`，仅依赖 Python 3.9+ 标准库和已安装的 vLLM CLI。
- 无参数启动进入中文交互模式；`--help` 展示全部 7 个参数、规则、默认值及示例，且不创建文件或执行压测。
- 不再询问 CSV/TXT 格式：固定生成原始 JSONL 明细和结构化 CSV 汇总，两种文件均严格追加写入。
- 遍历输入长度、输出长度、并发数的笛卡尔积，逐组调用 `vllm bench serve`；单组失败不中断，最终按全部历史成功记录输出达标并发和最大并发结论。

## 参数与文件契约

交互顺序固定为：

| 参数 | 规则 | 默认值 |
|---|---|---|
| 模型服务地址 | 必填，支持 `http://host[:port]` 或末尾 `/`；拒绝路径、查询串、认证信息和 HTTPS | 无 |
| 模型文件路径 | 必填，必须以 `/` 开头且真实存在；允许文件或目录；原路径传给 `--tokenizer` | 无 |
| 结果目录 | 非必填；不存在时递归创建；已存在但不是目录则拒绝 | `/data/model_eval_result/<YYYYMMDD>/<模型名>/` |
| 目标速度 | 正整数，单位 tokens/s | `10` |
| 输入长度列表 | 正整数；按英文逗号和任意空白拆分；去重并保持顺序 | `2048,8192,65536,120000,258000` |
| 输出长度列表 | 同上 | `2048` |
| 并发数列表 | 同上 | `1,8,16,32,48,64` |

- 模型名取路径最后一级；将所有不属于 `[A-Za-z0-9_-]` 的连续字符替换为一个 `-`，去除首尾 `-`。结果为空时拒绝执行，避免生成无效路径。
- CSV 路径为 `<结果目录>/<模型名>.csv`，原始明细路径为 `<结果目录>/<模型名>.vllm_raw.jsonl`。
- CSV 列固定为：`run_id, timestamp, service_url, model_name, model_path, target_tokens_per_sec, seed, input_len, output_len, concurrency, completed, failed, output_throughput_tok_s, mean_ttft_ms, mean_tpot_ms, duration_s, status, error`。
- 新 CSV 先写一次表头；已有 CSV 表头不匹配、或已有记录的模型原路径不同则拒绝执行。任何写入均使用追加模式，绝不使用覆盖模式。
- 原始明细每个组合保存为一行紧凑 JSON；含 `--save-detailed` 的逐请求数据。失败组合在 CSV 记录 `FAILED/PARTIAL`、空指标和错误摘要，但不参与达标统计。

## 实现结构

- 在 `model_eval_script.py` 中定义 `EvalConfig`、`BenchmarkCase`、`BenchmarkRecord`、`ResultStore`、`VllmRunner` 和汇总逻辑；关键函数与类全部提供详细中文 docstring，主函数开头写明七参数用法和 help 模式。
- 所有输入在首个压测命令前完成校验；同时检查 `vllm` 可执行文件存在。根据用户选择，不对服务执行预检请求，由第一组压测自然暴露连接错误。
- 使用参数数组和 `subprocess.Popen(..., shell=False)` 执行命令并实时转发输出；支持 Ctrl+C 终止子进程、将当前组合标记为 `INTERRUPTED`、打印已有历史总结并返回 130。
- 每组命令固定包含 `--seed 42`、`--dataset-name random`、`--endpoint /v1/chat/completions`、`--backend openai-chat`，且 `--num-prompts` 与 `--max-concurrency` 均等于当前并发数。
- 为避免 vLLM 辅助文件覆盖历史结果，每组先写入 `<结果目录>/.vllm_tmp/<run_id>/` 下的唯一 JSON；解析成功后追加到规范 JSONL，随后只清理本次临时目录。规范 CSV 和 JSONL 始终采用追加写入并立即刷新。
- 组合完整成功定义为 `completed == concurrency` 且 `failed == 0`；仅有部分请求成功时标为 `PARTIAL`，命令失败、无结果文件或 JSON 无法解析时标为 `FAILED`。
- 从 vLLM JSON 提取 `output_throughput`、`mean_ttft_ms`、`mean_tpot_ms`、`duration`、`completed` 和 `failed`；`output_throughput >= 本次目标速度` 即达标，等于目标也计入。

## 历史总结

- 汇总来源为当前结果目录下该模型的完整 CSV；同一次运行只读取该文件可见的全部历史记录。
- 相同 `(input_len, output_len, concurrency)` 只采用最后追加的 `SUCCESS` 记录；`PARTIAL/FAILED/INTERRUPTED` 不参与判断。
- 所有历史记录统一按本次输入的目标速度重新判断，不使用各行保存的旧目标值。
- 按输入长度分组、再按输出长度分组，打印全部达标并发数及该组合的最大达标并发。
- 对每个输入长度，再计算能覆盖该输入长度下所有历史输出长度均达标的最大公共并发数；若任一输出长度没有成功数据，则明确标记为数据不完整而不是猜测结果。
- 没有任何并发达到目标时，逐个输入/输出组合打印明确结论，并提示可增加并发条件或检查服务状态。
- 退出码约定：全部组合完整执行则为 0（即使未达到目标）；参数或前置条件错误为 1；存在失败或部分成功组合为 2；用户中断为 130。

## 测试与验收

- 在 `tests/test_model_eval_script.py` 中使用标准库 `unittest` 覆盖 URL、模型路径、默认值、正整数、混合分隔符、去重、严格默认值和模型名规范化。
- 验证命令参数完整、组合顺序正确、无 shell 拼接，以及默认 `5 × 1 × 6 = 30` 个组合。
- 使用临时目录和模拟 vLLM 结果验证 JSONL/CSV 连续追加、表头只写一次、原有内容不丢失、失败组合仍记录、单组失败后继续执行。
- 覆盖最新成功记录去重、当前目标速度重判、每个输入/输出对最大并发、跨输出长度公共并发、无达标结论及数据不完整提示。
- 更新 `README.md`，记录安装前提、交互示例、help 用法、文件位置、追加保证、CSV 字段和结论口径。
- 最终运行 `python3 -m unittest discover -s tests -v`、`python3 -m py_compile model_eval_script.py` 和 `python model_eval_script.py --help`；再在真实 vLLM 服务上用短输入、短输出和 `1,2` 并发完成一次端到端验收。

## 明确假设

- 目标服务器使用 POSIX 路径和本地时区；日期目录在脚本启动时计算一次。
- 服务仅提供无需认证的 HTTP vLLM OpenAI 兼容接口，服务发现和 token 管理不在本次范围内。
- vLLM 版本支持需求中给出的 `bench serve` 参数和 JSON 指标字段；不支持时以清晰的失败信息记录，不通过解析自由文本日志猜测指标。
- 同一模型的默认日期目录每天独立；选择“全部历史记录”是指汇总当前解析后的该模型 CSV 内全部记录。
- 同一结果文件假定只由一个脚本进程写入，不额外实现多进程文件锁。
