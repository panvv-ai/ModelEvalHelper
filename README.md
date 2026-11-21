# ModelEvalHelper

面向 vLLM OpenAI 兼容服务的本地批量性能评测工具。脚本按输入长度、输出长度
和并发数生成全部组合，逐组执行 `vllm bench serve`，并将原始明细和结构化
指标追加保存到本地。

## 环境要求

- Python 3.9 或更高版本。
- 已安装 vLLM，终端中可直接执行 `vllm bench serve --help`。
- 模型服务无需 token 或 cookie。
- 服务提供 `POST /v1/chat/completions`，地址格式为 `http://host[:port]`。

脚本只使用 Python 标准库，不需要额外安装 Python 依赖。

## 使用

查看完整参数说明：

```bash
python3 model_eval_script.py --help
```

进入交互式批量压测：

```bash
python3 model_eval_script.py
```

八项交互参数如下：

| 参数 | 规则 | 默认值 |
|---|---|---|
| 模型服务地址 | `http://host[:port]`，不能带路径、认证信息或 HTTPS | 无 |
| 模型文件路径 | 本机绝对路径，必须以 `/` 开头且真实存在 | 无 |
| 服务端模型名 | 必填；原值作为 `--model` 传给 vLLM，规范化后用于文件名和默认目录 | 无 |
| 结果目录 | 不存在时自动递归创建 | `/data/model_eval_result/YYYYMMDD/<模型名>/` |
| 目标速度 | 正整数，单位 tokens/s | `10` |
| 输入长度列表 | 正整数，使用英文逗号或空白分隔 | `2048,8192,65536,120000,258000` |
| 输出长度列表 | 正整数，使用英文逗号或空白分隔 | `2048` |
| 并发数列表 | 正整数，使用英文逗号或空白分隔 | `1,8,16,32,48,64` |

示例输入：

```text
[1/8] 模型服务地址：http://127.0.0.1:8000
[2/8] 模型文件绝对路径：/models/Qwen2.5-7B
[3/8] 服务端模型名：qwen2.5-7b-instruct
[4/8] 结果目录：/data/model_eval_result/qwen
[5/8] 目标速度：20
[6/8] 输入长度列表：2048,8192
[7/8] 输出长度列表：2048
[8/8] 并发数列表：1 8 16
```

脚本随后按输入长度、输出长度、并发数的顺序执行全部组合。每组使用固定
`seed=42`，且 `--num-prompts` 与 `--max-concurrency` 均等于当前并发数。

## 结果文件

模型名取服务端模型名，仅保留 ASCII 字母、数字、下划线和连字符，其余连续
字符替换为一个 `-`。例如 `Qwen/Qwen2.5-7B-Instruct` 会转换为
`Qwen-Qwen2-5-7B-Instruct`，并用于 CSV 文件名和默认结果目录。

结果目录中会生成：

- `<模型名>.csv`：结构化汇总，每次运行只追加新行，不覆盖历史数据。
- `<模型名>.vllm_raw.jsonl`：vLLM 原始 JSON 明细，每组占一行，包含
  `--save-detailed` 产生的逐请求数据。

CSV 的主要指标和元数据：

- `output_throughput_tok_s`：vLLM `output_throughput`，输出速度 tokens/s。
- `mean_ttft_ms`：平均首 token 延迟，毫秒。
- `mean_tpot_ms`：平均每输出 token 延迟，毫秒。
- `input_len`、`output_len`、`concurrency`：当前压测条件。
- `completed`、`failed`、`status`、`error`：完成情况和错误摘要。

状态含义：

- `SUCCESS`：完成请求数等于并发数、失败数为零且输出吞吐可用。TTFT/TPOT
  或时长缺失时留空，不影响状态。
- `PARTIAL`：有部分请求成功，但整组没有完整完成。
- `FAILED`：命令失败、没有成功请求或输出吞吐不可用。
- `INTERRUPTED`：用户中断了当前批次。

只有 `SUCCESS` 记录参与达标分析。单组失败会记录错误并继续执行剩余组合。
历史 CSV 中的空行或无法解析的脏行会被跳过并提示，不会中断本次执行。

## 达标结论

脚本结束时会读取当前 CSV 的全部历史记录：

1. 同一输入、输出、并发组合采用最后追加的 `SUCCESS` 记录。
2. 所有历史记录都按本次输入的目标速度重新判断。
3. 只统计与本次服务地址相同的历史记录；其他地址的记录会提示后排除。
4. 按输入长度、输出长度分组，列出所有达标并发数和最大达标并发。
5. 对每个输入长度，计算所有历史输出长度均达标时的最大公共并发。
6. 若较低并发未达标但更高并发已达标，会提示最大达标并发仅供参考。

输出吞吐大于或等于目标速度即视为达标。目标未达到本身不会导致脚本以错误
状态退出。默认结果目录包含当天日期，因此使用默认目录时历史统计只覆盖
当天；如需跨天累计分析，请显式填写稳定的结果目录。

退出码：

- `0`：本批次所有组合完整成功。
- `1`：参数、路径或运行前置条件错误。
- `2`：本批次存在 `FAILED` 或 `PARTIAL` 组合。
- `130`：用户中断。

## 测试

```bash
python3 -m unittest discover -s tests -v
python3 -m py_compile model_eval_script.py
```

测试使用模拟 vLLM 可执行文件验证命令参数、JSON 解析、连续追加、失败续跑、
中断处理和历史达标汇总，不会访问真实模型服务。

实现计划保存在
[`docs/2026-11-22-model-eval-script.md`](docs/2026-11-22-model-eval-script.md)。
