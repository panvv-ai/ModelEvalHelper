#!/usr/bin/env python3
"""vLLM 模型服务批量压测、结果追加与历史达标分析工具。"""

from __future__ import annotations

import csv
import json
import os
import re
import shutil
import subprocess
import sys
import uuid
from collections import deque
from dataclasses import dataclass
from datetime import date, datetime
from itertools import product
from pathlib import Path
from typing import Optional, Sequence, TextIO, Tuple


DEFAULT_TARGET_SPEED = 10
DEFAULT_INPUT_LENGTHS: Tuple[int, ...] = (2048, 8192, 65536, 120000, 258000)
DEFAULT_OUTPUT_LENGTHS: Tuple[int, ...] = (2048,)
DEFAULT_CONCURRENCIES: Tuple[int, ...] = (1, 8, 16, 32, 48, 64)
SEED = 42

CSV_FIELDS: Tuple[str, ...] = (
    "run_id",
    "timestamp",
    "service_url",
    "model_name",
    "model_path",
    "target_tokens_per_sec",
    "seed",
    "input_len",
    "output_len",
    "concurrency",
    "completed",
    "failed",
    "output_throughput_tok_s",
    "mean_ttft_ms",
    "mean_tpot_ms",
    "duration_s",
    "status",
    "error",
)


class ConfigError(ValueError):
    """表示用户输入或本地运行前置条件不满足。"""


class BenchmarkExecutionError(RuntimeError):
    """表示单个 vLLM 压测子进程执行或结果读取失败。"""


@dataclass(frozen=True)
class ServiceEndpoint:
    """用户提供的模型服务地址，原值传给 vLLM 的 ``--base-url``。"""

    base_url: str


@dataclass(frozen=True)
class BenchmarkCase:
    """一组待执行的输入长度、输出长度和并发数。"""

    input_len: int
    output_len: int
    concurrency: int


@dataclass(frozen=True)
class EvalConfig:
    """一次批量压测运行所需的完整配置。

    ``served_model_name`` 是必填的服务端模型名，原值作为 ``--model`` 传给
    vLLM；``model_name`` 由其规范化得到，仅用于 CSV 文件名和默认结果目录。
    """

    endpoint: ServiceEndpoint
    model_path: str
    model_name: str
    result_dir: Path
    target_speed: int
    input_lengths: Tuple[int, ...]
    output_lengths: Tuple[int, ...]
    concurrencies: Tuple[int, ...]
    served_model_name: str

    @property
    def csv_path(self) -> Path:
        """返回当前模型的结构化汇总文件路径。"""

        return self.result_dir / f"{self.model_name}.csv"

    @property
    def raw_path(self) -> Path:
        """返回当前模型的追加式 vLLM 原始明细文件路径。"""

        return self.result_dir / f"{self.model_name}.vllm_raw.jsonl"


@dataclass(frozen=True)
class BenchmarkRecord:
    """写入 CSV 的单组压测记录。"""

    run_id: str
    timestamp: str
    service_url: str
    model_name: str
    model_path: str
    target_tokens_per_sec: int
    seed: int
    input_len: int
    output_len: int
    concurrency: int
    completed: int
    failed: int
    output_throughput_tok_s: Optional[float]
    mean_ttft_ms: Optional[float]
    mean_tpot_ms: Optional[float]
    duration_s: Optional[float]
    status: str
    error: str

    def as_csv_row(self) -> Tuple[object, ...]:
        """按固定字段顺序把记录转换为 CSV 行。"""

        return tuple(getattr(self, field) for field in CSV_FIELDS)


@dataclass(frozen=True)
class BenchmarkRunOutput:
    """单组 vLLM 执行结果及尚未清理的临时目录。"""

    payload: dict
    temp_dir: Path


@dataclass(frozen=True)
class PairAnalysis:
    """单个输入长度与输出长度组合的达标分析。

    ``non_monotonic_concurrencies`` 记录已测试但未达标、却低于某个已达标
    并发的并发数，用于提示最大达标并发可能不代表稳定上限。
    """

    input_len: int
    output_len: int
    passing_concurrencies: Tuple[int, ...]
    max_concurrency: Optional[int]
    has_success: bool
    non_monotonic_concurrencies: Tuple[int, ...] = ()


@dataclass(frozen=True)
class InputAnalysis:
    """单个输入长度下所有输出长度的达标分析。"""

    input_len: int
    pairs: Tuple[PairAnalysis, ...]
    common_max_concurrency: Optional[int]
    common_reason: str


class ResultStore:
    """负责 CSV 汇总与 JSONL 原始明细的追加式持久化。"""

    def __init__(self, config: EvalConfig) -> None:
        self.config = config

    def _ensure_directory(self) -> None:
        """创建结果目录；已存在但不是目录时拒绝继续。"""

        if self.config.result_dir.exists() and not self.config.result_dir.is_dir():
            raise ConfigError(f"结果保存路径不是目录：{self.config.result_dir}")
        try:
            self.config.result_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise ConfigError(
                f"无法创建结果目录 {self.config.result_dir}：{exc}"
            ) from exc

    @staticmethod
    def _ensure_trailing_newline(path: Path) -> None:
        """确保非空追加文件以换行结束，避免新旧记录粘连。"""

        if not path.exists() or path.stat().st_size == 0:
            return
        with path.open("rb") as handle:
            handle.seek(-1, os.SEEK_END)
            last_byte = handle.read(1)
        if last_byte != b"\n":
            with path.open("a", encoding="utf-8") as handle:
                handle.write("\n")

    def validate_existing(self) -> None:
        """校验已有 CSV 的表头和模型路径，防止不兼容或混合数据追加。

        文件不存在或为空时视为可初始化。表头不匹配时拒绝继续；模型路径
        只对能够完整解析的记录进行比对，无法解析的脏行直接跳过，避免
        历史脏数据阻断本次执行。校验过程只读，不会修改文件。
        """

        self._ensure_directory()
        path = self.config.csv_path
        if not path.exists() or path.stat().st_size == 0:
            return

        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames != list(CSV_FIELDS):
                raise ConfigError(
                    f"已有 CSV 表头与当前版本不兼容，拒绝覆盖或追加：{path}"
                )
            for row in reader:
                try:
                    record = self._record_from_row(row)
                except (KeyError, TypeError, ValueError, csv.Error):
                    continue
                if (
                    record.model_path
                    and record.model_path != self.config.model_path
                ):
                    raise ConfigError(
                        "已有 CSV 包含其他模型路径，模型名规范化后发生冲突；"
                        f"请更换结果目录。文件：{path}，第 {reader.line_num} 行"
                    )

    def append_record(self, record: BenchmarkRecord) -> None:
        """向 CSV 追加一条记录，必要时只写入一次表头。"""

        self.validate_existing()
        path = self.config.csv_path
        needs_header = not path.exists() or path.stat().st_size == 0
        if not needs_header:
            self._ensure_trailing_newline(path)

        with path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            if needs_header:
                writer.writerow(CSV_FIELDS)
            writer.writerow(record.as_csv_row())
            handle.flush()
            os.fsync(handle.fileno())

    def append_raw(self, payload: dict) -> None:
        """向 JSONL 文件追加一个紧凑 JSON 对象。"""

        self._ensure_directory()
        path = self.config.raw_path
        self._ensure_trailing_newline(path)
        line = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())

    def load_records(
        self, output_stream: Optional[TextIO] = None
    ) -> Tuple[BenchmarkRecord, ...]:
        """读取并转换当前 CSV 中的全部历史记录。

        无法解析的历史行（空行、截断行、数值列损坏等）会被跳过并给出
        提示，绝不因为历史脏数据中断本次执行。
        """

        self.validate_existing()
        path = self.config.csv_path
        if not path.exists() or path.stat().st_size == 0:
            return ()

        records = []
        skipped = 0
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                try:
                    records.append(self._record_from_row(row))
                except (KeyError, TypeError, ValueError, csv.Error) as exc:
                    skipped += 1
                    if output_stream is not None and skipped <= 5:
                        output_stream.write(
                            "跳过无法解析的历史 CSV 行"
                            f"（第 {reader.line_num} 行）：{exc}\n"
                        )
        if skipped and output_stream is not None:
            output_stream.write(
                f"已跳过 {skipped} 行无法解析的历史记录，本次执行继续。\n"
            )
        return tuple(records)

    @staticmethod
    def _record_from_row(row: dict) -> BenchmarkRecord:
        """把一行 CSV 字典转换为记录；字段缺失或非法时抛出解析异常。"""

        def text(field: str) -> str:
            value = row.get(field)
            return "" if value is None else str(value)

        return BenchmarkRecord(
            run_id=text("run_id"),
            timestamp=text("timestamp"),
            service_url=text("service_url"),
            model_name=text("model_name"),
            model_path=text("model_path"),
            target_tokens_per_sec=int(text("target_tokens_per_sec")),
            seed=int(text("seed")),
            input_len=int(text("input_len")),
            output_len=int(text("output_len")),
            concurrency=int(text("concurrency")),
            completed=int(text("completed") or 0),
            failed=int(text("failed") or 0),
            output_throughput_tok_s=ResultStore._optional_float(
                text("output_throughput_tok_s")
            ),
            mean_ttft_ms=ResultStore._optional_float(text("mean_ttft_ms")),
            mean_tpot_ms=ResultStore._optional_float(text("mean_tpot_ms")),
            duration_s=ResultStore._optional_float(text("duration_s")),
            status=text("status"),
            error=text("error"),
        )

    @staticmethod
    def _optional_float(value: str) -> Optional[float]:
        """把 CSV 空单元格转换为 None，其余值转换为 float。"""

        return float(value) if value not in ("", None) else None


class VllmRunner:
    """执行单组 vLLM 官方压测命令并读取结果 JSON。"""

    def __init__(
        self,
        config: EvalConfig,
        executable: str = "vllm",
        output_stream: Optional[TextIO] = None,
    ) -> None:
        self.config = config
        self.executable = executable
        self.output_stream = output_stream or sys.stdout

    def run(self, case: BenchmarkCase, run_id: str) -> BenchmarkRunOutput:
        """执行一组压测，实时转发输出并返回解析后的完整 JSON。

        每组结果写入独立的隐藏临时目录，成功读取后由上层追加到规范
        JSONL，再清理临时目录。这样可以避免 vLLM 辅助文件互相覆盖。
        """

        segment = (
            f"{case.input_len}_{case.output_len}_{case.concurrency}_"
            f"{uuid.uuid4().hex[:8]}"
        )
        temp_dir = (
            self.config.result_dir
            / ".vllm_tmp"
            / f"{run_id}-{segment}"
        )
        temp_dir.mkdir(parents=True, exist_ok=False)
        result_path = temp_dir / "result.json"
        command = build_vllm_command(
            self.config,
            case,
            result_path,
            executable=self.executable,
        )

        try:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                shell=False,
            )
        except OSError as exc:
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise BenchmarkExecutionError(f"无法启动 vLLM 压测命令：{exc}") from exc

        output_tail = deque(maxlen=40)
        try:
            assert process.stdout is not None
            for line in process.stdout:
                self.output_stream.write(line)
                self.output_stream.flush()
                output_tail.append(line.rstrip())
            return_code = process.wait()
        except KeyboardInterrupt:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise
        finally:
            if process.stdout is not None:
                process.stdout.close()

        if return_code != 0:
            summary = " | ".join(output_tail)
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise BenchmarkExecutionError(
                f"vLLM 压测命令失败，退出码 {return_code}。输出摘要：{summary}"
            )
        if not result_path.is_file():
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise BenchmarkExecutionError("vLLM 未生成结果 JSON 文件")

        try:
            payload = json.loads(result_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise BenchmarkExecutionError(f"无法解析 vLLM 结果 JSON：{exc}") from exc
        if not isinstance(payload, dict):
            shutil.rmtree(temp_dir, ignore_errors=True)
            raise BenchmarkExecutionError("vLLM 结果 JSON 顶层必须是对象")
        return BenchmarkRunOutput(payload=payload, temp_dir=temp_dir)


def build_vllm_command(
    config: EvalConfig,
    case: BenchmarkCase,
    result_path: Path,
    executable: str = "vllm",
) -> list:
    """构造一组 vLLM 官方压测命令，参数始终以列表传递以避免 shell 注入。"""

    return [
        executable,
        "bench",
        "serve",
        "--base-url",
        config.endpoint.base_url,
        "--model",
        config.served_model_name,
        "--tokenizer",
        config.model_path,
        "--dataset-name",
        "random",
        "--endpoint",
        "/v1/chat/completions",
        "--backend",
        "openai-chat",
        "--seed",
        str(SEED),
        "--random-input-len",
        str(case.input_len),
        "--random-output-len",
        str(case.output_len),
        "--num-prompts",
        str(case.concurrency),
        "--max-concurrency",
        str(case.concurrency),
        "--save-result",
        "--save-detailed",
        "--append-result",
        "--result-filename",
        result_path.name,
        "--result-dir",
        str(result_path.parent),
    ]


def _new_run_id() -> str:
    """生成适合作为临时目录和 CSV 标识的批次运行 ID。"""

    stamp = datetime.now().strftime("%Y%m%dT%H%M%S")
    return f"{stamp}-{uuid.uuid4().hex[:8]}"


def _current_timestamp() -> str:
    """返回带本地时区信息的 ISO 8601 时间戳。"""

    return datetime.now().astimezone().isoformat(timespec="seconds")


def _payload_int(payload: dict, field: str, default: int = 0) -> int:
    """读取 vLLM JSON 中的计数字段，非数字值视为缺失。"""

    value = payload.get(field, default)
    if isinstance(value, bool):
        raise ValueError(field)
    return int(value)


def _payload_float(payload: dict, field: str) -> float:
    """读取 vLLM JSON 中必需的浮点指标字段。"""

    value = payload.get(field)
    if isinstance(value, bool) or value is None:
        raise ValueError(field)
    return float(value)


def _optional_payload_float(payload: dict, field: str) -> Optional[float]:
    """读取 vLLM JSON 中可选的浮点指标字段。

    字段缺失、为 null、布尔值或无法转换为浮点数时返回 ``None``，不影响
    记录状态判定。
    """

    value = payload.get(field)
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def make_record_from_result(
    payload: dict,
    config: EvalConfig,
    case: BenchmarkCase,
    run_id: str,
    timestamp: str,
) -> BenchmarkRecord:
    """把 vLLM 结果 JSON 转换为 CSV 记录并判定完整、部分或失败状态。

    只有完成请求数等于当前并发数、失败数为零且输出吞吐可用时，记录才是
    ``SUCCESS``。``mean_ttft_ms``、``mean_tpot_ms`` 和 ``duration`` 属于可选
    指标，缺失时以空值记录，不影响状态判定。部分完成记录保留可用的统计
    值，但不会进入达标汇总。
    """

    completed = 0
    failed = 0
    try:
        completed = _payload_int(payload, "completed")
        failed = _payload_int(payload, "failed")
        output_throughput = _payload_float(payload, "output_throughput")
    except (TypeError, ValueError) as exc:
        missing_field = str(exc) or "必需指标"
        return make_failed_record(
            config,
            case,
            run_id,
            timestamp,
            f"vLLM 结果缺少或包含非法字段：{missing_field}",
            completed=completed,
            failed=failed,
        )

    mean_ttft_ms = _optional_payload_float(payload, "mean_ttft_ms")
    mean_tpot_ms = _optional_payload_float(payload, "mean_tpot_ms")
    duration_s = _optional_payload_float(payload, "duration")

    if completed == case.concurrency and failed == 0:
        status = "SUCCESS"
        error = ""
    elif completed > 0:
        status = "PARTIAL"
        error = f"仅完成 {completed}/{case.concurrency} 个请求，失败 {failed} 个"
    else:
        return make_failed_record(
            config,
            case,
            run_id,
            timestamp,
            f"没有成功请求，失败 {failed} 个",
            completed=completed,
            failed=failed,
        )

    return BenchmarkRecord(
        run_id=run_id,
        timestamp=timestamp,
        service_url=config.endpoint.base_url,
        model_name=config.model_name,
        model_path=config.model_path,
        target_tokens_per_sec=config.target_speed,
        seed=SEED,
        input_len=case.input_len,
        output_len=case.output_len,
        concurrency=case.concurrency,
        completed=completed,
        failed=failed,
        output_throughput_tok_s=output_throughput,
        mean_ttft_ms=mean_ttft_ms,
        mean_tpot_ms=mean_tpot_ms,
        duration_s=duration_s,
        status=status,
        error=error,
    )


def make_failed_record(
    config: EvalConfig,
    case: BenchmarkCase,
    run_id: str,
    timestamp: str,
    error: str,
    completed: int = 0,
    failed: int = 0,
    status: str = "FAILED",
) -> BenchmarkRecord:
    """构造指标为空的失败或中断记录。"""

    return BenchmarkRecord(
        run_id=run_id,
        timestamp=timestamp,
        service_url=config.endpoint.base_url,
        model_name=config.model_name,
        model_path=config.model_path,
        target_tokens_per_sec=config.target_speed,
        seed=SEED,
        input_len=case.input_len,
        output_len=case.output_len,
        concurrency=case.concurrency,
        completed=completed,
        failed=failed,
        output_throughput_tok_s=None,
        mean_ttft_ms=None,
        mean_tpot_ms=None,
        duration_s=None,
        status=status,
        error=" ".join(error.splitlines())[:1000],
    )


def analyze_history(
    records: Sequence[BenchmarkRecord],
    target_speed: int,
    service_url: Optional[str] = None,
) -> Tuple[InputAnalysis, ...]:
    """按当前目标速度分析全部历史记录。

    指定 ``service_url`` 时只统计该服务地址的记录，避免把不同实例的
    结果混在一起。同一输入/输出/并发组合只保留最后追加的 ``SUCCESS``
    记录。每个输入长度先计算各输出长度的最大达标并发，再计算所有输出
    长度共同达标的最大公共并发。
    """

    if service_url is not None:
        records = [
            record for record in records if record.service_url == service_url
        ]

    input_values = sorted({record.input_len for record in records})
    latest_success = {}
    output_values = {}
    tested_values = {}

    for record in records:
        output_values.setdefault(record.input_len, set()).add(record.output_len)
        tested_values.setdefault(
            (record.input_len, record.output_len), set()
        ).add(record.concurrency)
        if (
            record.status == "SUCCESS"
            and record.output_throughput_tok_s is not None
        ):
            key = (record.input_len, record.output_len, record.concurrency)
            latest_success[key] = record

    analyses = []
    for input_len in input_values:
        pair_analyses = []
        for output_len in sorted(output_values[input_len]):
            success_rows = [
                record
                for (in_len, out_len, _), record in latest_success.items()
                if in_len == input_len and out_len == output_len
            ]
            passing = tuple(
                sorted(
                    {
                        record.concurrency
                        for record in success_rows
                        if record.output_throughput_tok_s >= target_speed
                    }
                )
            )
            max_concurrency = passing[-1] if passing else None
            non_monotonic: Tuple[int, ...] = ()
            if max_concurrency is not None:
                passing_set = set(passing)
                non_monotonic = tuple(
                    sorted(
                        concurrency
                        for concurrency in tested_values.get(
                            (input_len, output_len), set()
                        )
                        if concurrency < max_concurrency
                        and concurrency not in passing_set
                    )
                )
            pair_analyses.append(
                PairAnalysis(
                    input_len=input_len,
                    output_len=output_len,
                    passing_concurrencies=passing,
                    max_concurrency=max_concurrency,
                    has_success=bool(success_rows),
                    non_monotonic_concurrencies=non_monotonic,
                )
            )

        common_values: Optional[set] = None
        common_reason = ""
        if not pair_analyses:
            common_reason = "没有可用输出长度"
        elif any(not pair.has_success for pair in pair_analyses):
            common_reason = "存在没有历史成功记录的输出长度，无法计算公共值"
        else:
            for pair in pair_analyses:
                passing_set = set(pair.passing_concurrencies)
                common_values = (
                    passing_set
                    if common_values is None
                    else common_values.intersection(passing_set)
                )
            if not common_values:
                common_reason = "没有并发数能让所有输出长度同时达到目标速度"

        common_max = max(common_values) if common_values else None
        analyses.append(
            InputAnalysis(
                input_len=input_len,
                pairs=tuple(pair_analyses),
                common_max_concurrency=common_max,
                common_reason=common_reason,
            )
        )
    return tuple(analyses)


def format_analysis(
    analysis: Sequence[InputAnalysis], target_speed: int
) -> str:
    """把结构化历史分析格式化为便于终端阅读的中文结论。"""

    lines = [f"历史达标分析（目标速度：{target_speed} tokens/s）"]
    if not analysis:
        lines.append("没有任何可用历史记录。")
        return "\n".join(lines)

    for input_entry in analysis:
        lines.append("")
        lines.append(f"输入长度={input_entry.input_len}")
        for pair in input_entry.pairs:
            lines.append(f"  输出长度={pair.output_len}")
            if not pair.has_success:
                lines.append("    没有任何历史成功记录")
            elif not pair.passing_concurrencies:
                lines.append("    无并发达到目标速度")
            else:
                concurrencies = ", ".join(
                    str(value) for value in pair.passing_concurrencies
                )
                lines.append(f"    达标并发：{concurrencies}")
                lines.append(f"    最大达标并发：{pair.max_concurrency}")
                if pair.non_monotonic_concurrencies:
                    failed_values = ", ".join(
                        str(value)
                        for value in pair.non_monotonic_concurrencies
                    )
                    lines.append(
                        f"    注意：并发 {failed_values} 未达标但更高并发已达标，"
                        "最大达标并发仅供参考"
                    )

        if input_entry.common_max_concurrency is not None:
            lines.append(
                "  所有输出长度均达标的最大公共并发："
                f"{input_entry.common_max_concurrency}"
            )
        elif input_entry.common_reason:
            lines.append(f"  公共并发结论：{input_entry.common_reason}")
    return "\n".join(lines)


def run_evaluation(
    config: EvalConfig,
    runner: Optional[VllmRunner] = None,
    output_stream: Optional[TextIO] = None,
) -> int:
    """执行全部压测组合、追加结果并打印当前 CSV 的历史分析。

    历史分析只统计与本次服务地址相同的记录。返回值 0 表示本次所有组合
    完整成功；2 表示存在失败或部分成功；用户中断返回 130。目标速度是否
    达到本身不作为进程失败。
    """

    stream = output_stream or sys.stdout
    store = ResultStore(config)
    store.validate_existing()
    active_runner = runner or VllmRunner(config, output_stream=stream)
    cases = build_cases(
        config.input_lengths, config.output_lengths, config.concurrencies
    )
    batch_run_id = _new_run_id()
    current_statuses = []
    interrupted = False

    stream.write(
        f"开始批量压测：共 {len(cases)} 组，结果目录 {config.result_dir}\n"
    )
    stream.flush()

    for index, case in enumerate(cases, start=1):
        stream.write(
            f"\n[{index}/{len(cases)}] 输入={case.input_len} "
            f"输出={case.output_len} 并发={case.concurrency}\n"
        )
        stream.flush()
        timestamp = _current_timestamp()

        try:
            result = active_runner.run(case, run_id=batch_run_id)
        except KeyboardInterrupt:
            record = make_failed_record(
                config,
                case,
                batch_run_id,
                timestamp,
                "用户中断当前压测",
                status="INTERRUPTED",
            )
            store.append_record(record)
            current_statuses.append(record.status)
            interrupted = True
            break
        except Exception as exc:
            record = make_failed_record(
                config,
                case,
                batch_run_id,
                timestamp,
                str(exc),
            )
            store.append_record(record)
            current_statuses.append(record.status)
            stream.write(f"本组失败，继续后续组合：{exc}\n")
            stream.flush()
            continue

        try:
            store.append_raw(result.payload)
        except (OSError, TypeError, ValueError) as exc:
            record = make_failed_record(
                config,
                case,
                batch_run_id,
                timestamp,
                f"原始结果追加失败：{exc}",
            )
        else:
            record = make_record_from_result(
                result.payload,
                config,
                case,
                run_id=batch_run_id,
                timestamp=timestamp,
            )
        finally:
            shutil.rmtree(result.temp_dir, ignore_errors=True)

        store.append_record(record)
        current_statuses.append(record.status)

    stream.write("\n")
    history = store.load_records(output_stream=stream)
    other_service_count = sum(
        1
        for record in history
        if record.service_url != config.endpoint.base_url
    )
    if other_service_count:
        stream.write(
            f"注意：有 {other_service_count} 条历史记录来自其他服务地址，"
            "未纳入本次分析。\n"
        )
    stream.write(
        format_analysis(
            analyze_history(
                history,
                config.target_speed,
                service_url=config.endpoint.base_url,
            ),
            config.target_speed,
        )
    )
    stream.write("\n")
    stream.flush()

    if interrupted:
        return 130
    if any(status != "SUCCESS" for status in current_statuses):
        return 2
    return 0


def parse_positive_int(value: str, field_name: str) -> int:
    """把字符串解析为严格的正整数，并返回解析结果。

    只接受 ASCII 十进制数字，拒绝零、负数、小数、符号、Unicode 数字
    和空字符串。
    """

    raw = value.strip()
    if not re.fullmatch(r"[0-9]+", raw):
        raise ConfigError(f"{field_name}必须是正整数，当前输入为：{value!r}")
    parsed = int(raw)
    if parsed <= 0:
        raise ConfigError(f"{field_name}必须是正整数，当前输入为：{value!r}")
    return parsed


def parse_positive_int_list(
    value: str, defaults: Sequence[int], field_name: str
) -> Tuple[int, ...]:
    """按英文逗号和空白拆分正整数列表，去重并保持原始顺序。

    输入为空或只包含空白时返回调用方给定的默认值。任一元素非法都会
    拒绝整组输入，避免带着不完整条件开始长时间压测。
    """

    raw = value.strip()
    if not raw:
        return tuple(defaults)

    parts = [part for part in re.split(r"[,\s]+", raw) if part]
    if not parts:
        return tuple(defaults)

    parsed = tuple(parse_positive_int(part, field_name) for part in parts)
    return tuple(dict.fromkeys(parsed))


def parse_service_url(service_url: str) -> ServiceEndpoint:
    """校验模型服务地址必填，且以 ``http://`` 或 ``https://`` 开头。

    除必填和前缀外不做任何格式校验；地址原值（去除首尾空白）直接作为
    ``--base-url`` 传给 vLLM，可访问性由用户保证。
    """

    raw = service_url.strip()
    if not raw:
        raise ConfigError("模型服务地址不能为空")
    lowered = raw.lower()
    if not (
        lowered.startswith("http://") or lowered.startswith("https://")
    ):
        raise ConfigError("模型服务地址必须以 http:// 或 https:// 开头")

    return ServiceEndpoint(base_url=raw)


def sanitize_model_name(served_model_name: str) -> str:
    """把服务端模型名规范化为可用于 CSV 文件名和目录名的名称。

    仅保留 ASCII 字母、数字、下划线和连字符；其他连续字符替换为一个
    连字符，并移除首尾连字符。规范化后为空时拒绝执行。
    """

    name = re.sub(r"[^A-Za-z0-9_-]+", "-", served_model_name).strip("-")
    if not name:
        raise ConfigError(
            "服务端模型名无法生成合法文件名；至少需要一个英文字母、数字、"
            "下划线或连字符"
        )
    return name


def parse_served_model_name(value: str) -> str:
    """校验服务端模型名必填，返回去除首尾空白后的原值。"""

    name = value.strip()
    if not name:
        raise ConfigError(
            "服务端模型名不能为空；请填写服务端 --served-model-name 的值"
        )
    return name


def validate_model_path(model_path: str) -> str:
    """校验模型路径是存在的绝对路径，并返回规范化后的绝对路径。"""

    raw = model_path.strip()
    if not raw:
        raise ConfigError("模型文件路径不能为空")
    if not raw.startswith("/"):
        raise ConfigError("模型文件路径必须是使用 / 开头的绝对路径")

    path = Path(raw)
    if not path.exists():
        raise ConfigError(f"模型文件路径不存在：{raw}")
    return str(path.resolve())


def resolve_result_dir(
    user_value: str, model_name: str, today: Optional[date] = None
) -> Path:
    """解析用户结果目录；为空时生成按日期和模型分组的默认目录。"""

    raw = user_value.strip()
    if raw:
        return Path(raw).expanduser().resolve()

    current_date = today or date.today()
    return (
        Path("/data")
        / "model_eval_result"
        / current_date.strftime("%Y%m%d")
        / model_name
    ).resolve()


def build_eval_config(
    service_url: str,
    model_path: str,
    result_dir: str,
    target_speed: str,
    input_lengths: str,
    output_lengths: str,
    concurrencies: str,
    served_model_name: str,
    today: Optional[date] = None,
) -> EvalConfig:
    """校验八项原始输入并构造不可变运行配置。"""

    endpoint = parse_service_url(service_url)
    normalized_model_path = validate_model_path(model_path)
    normalized_served_name = parse_served_model_name(served_model_name)
    model_name = sanitize_model_name(normalized_served_name)
    parsed_target = (
        parse_positive_int(target_speed, "目标速度")
        if target_speed.strip()
        else DEFAULT_TARGET_SPEED
    )
    parsed_inputs = parse_positive_int_list(
        input_lengths, DEFAULT_INPUT_LENGTHS, "输入长度列表"
    )
    parsed_outputs = parse_positive_int_list(
        output_lengths, DEFAULT_OUTPUT_LENGTHS, "输出长度列表"
    )
    parsed_concurrencies = parse_positive_int_list(
        concurrencies, DEFAULT_CONCURRENCIES, "并发数列表"
    )
    resolved_result_dir = resolve_result_dir(
        result_dir, model_name, today=today
    )
    return EvalConfig(
        endpoint=endpoint,
        model_path=normalized_model_path,
        model_name=model_name,
        result_dir=resolved_result_dir,
        target_speed=parsed_target,
        input_lengths=parsed_inputs,
        output_lengths=parsed_outputs,
        concurrencies=parsed_concurrencies,
        served_model_name=normalized_served_name,
    )


def build_cases(
    input_lengths: Sequence[int],
    output_lengths: Sequence[int],
    concurrencies: Sequence[int],
) -> Tuple[BenchmarkCase, ...]:
    """按输入、输出、并发的顺序生成全部压测组合。"""

    return tuple(
        BenchmarkCase(input_len=input_len, output_len=output_len, concurrency=conc)
        for input_len, output_len, conc in product(
            input_lengths, output_lengths, concurrencies
        )
    )


def _read_with_retry(input_func, prompt: str, parser, output_stream: TextIO):
    """重复提示直到 parser 接受输入，避免非法参数进入执行阶段。"""

    while True:
        try:
            raw = input_func(prompt)
            return parser(raw)
        except (EOFError, StopIteration) as exc:
            raise ConfigError("输入已结束，未执行压测") from exc
        except ConfigError as exc:
            output_stream.write(f"输入无效：{exc}。请重新输入。\n")
            output_stream.flush()


def prompt_for_config(
    input_func=None,
    output_stream: Optional[TextIO] = None,
    today: Optional[date] = None,
) -> EvalConfig:
    """按固定顺序交互读取八项参数，并逐项校验后构造配置。"""

    reader = input_func or input
    stream = output_stream or sys.stdout
    service_url = _read_with_retry(
        reader,
        "[1/8] 模型服务地址（必填，如 http://127.0.0.1:8000）：",
        parse_service_url,
        stream,
    )
    model_path = _read_with_retry(
        reader,
        "[2/8] 模型文件绝对路径（必填，必须以 / 开头且存在）：",
        validate_model_path,
        stream,
    )
    served_model_name = _read_with_retry(
        reader,
        "[3/8] 服务端模型名（必填，如 qwen2.5-7b-instruct）：",
        parse_served_model_name,
        stream,
    )
    result_dir = _read_with_retry(
        reader,
        "[4/8] 结果目录（可留空使用默认目录）：",
        lambda value: value.strip(),
        stream,
    )
    target_speed = _read_with_retry(
        reader,
        f"[5/8] 目标速度 tokens/s（可留空，默认 {DEFAULT_TARGET_SPEED}）：",
        lambda value: (
            parse_positive_int(value, "目标速度")
            if value.strip()
            else DEFAULT_TARGET_SPEED
        ),
        stream,
    )
    input_lengths = _read_with_retry(
        reader,
        "[6/8] 输入长度列表（逗号或空格分隔，可留空，"
        f"默认 {','.join(map(str, DEFAULT_INPUT_LENGTHS))}）：",
        lambda value: parse_positive_int_list(
            value, DEFAULT_INPUT_LENGTHS, "输入长度列表"
        ),
        stream,
    )
    output_lengths = _read_with_retry(
        reader,
        "[7/8] 输出长度列表（逗号或空格分隔，可留空，"
        f"默认 {','.join(map(str, DEFAULT_OUTPUT_LENGTHS))}）：",
        lambda value: parse_positive_int_list(
            value, DEFAULT_OUTPUT_LENGTHS, "输出长度列表"
        ),
        stream,
    )
    concurrencies = _read_with_retry(
        reader,
        "[8/8] 并发数列表（逗号或空格分隔，可留空，"
        f"默认 {','.join(map(str, DEFAULT_CONCURRENCIES))}）：",
        lambda value: parse_positive_int_list(
            value, DEFAULT_CONCURRENCIES, "并发数列表"
        ),
        stream,
    )

    model_name = sanitize_model_name(served_model_name)
    return EvalConfig(
        endpoint=service_url,
        model_path=model_path,
        model_name=model_name,
        result_dir=resolve_result_dir(result_dir, model_name, today=today),
        target_speed=target_speed,
        input_lengths=input_lengths,
        output_lengths=output_lengths,
        concurrencies=concurrencies,
        served_model_name=served_model_name,
    )


def _help_text() -> str:
    """返回脚本 help 模式的中文说明。"""

    return f"""ModelEvalHelper - vLLM 模型服务批量压测工具

用法：
  python3 model_eval_script.py
  python3 model_eval_script.py --help

八项交互参数：
  1. 模型服务地址：必填，以 http:// 或 https:// 开头；原值作为
     --base-url 传给 vLLM，其余格式不做校验。
  2. 模型文件路径：必填，必须是有 / 开头的本机绝对路径且真实存在。
  3. 服务端模型名：必填，原值作为 --model 传给 vLLM；规范化后用于
     CSV 文件名和默认结果目录。
  4. 结果目录：可留空；默认
     /data/model_eval_result/<今日日期YYYYMMDD>/<模型名>/
  5. 目标速度：正整数 tokens/s，默认 {DEFAULT_TARGET_SPEED}。
  6. 输入长度列表：英文逗号或空白分隔的正整数，默认
     {','.join(map(str, DEFAULT_INPUT_LENGTHS))}
  7. 输出长度列表：英文逗号或空白分隔的正整数，默认
     {','.join(map(str, DEFAULT_OUTPUT_LENGTHS))}
  8. 并发数列表：英文逗号或空白分隔的正整数，默认
     {','.join(map(str, DEFAULT_CONCURRENCIES))}

输出契约：
  - <结果目录>/<模型名>.csv：追加式结构化汇总，绝不覆盖已有结果。
  - <结果目录>/<模型名>.vllm_raw.jsonl：追加式 vLLM 原始明细。
  - 模型名取服务端模型名，只保留字母、数字、下划线和连字符。
  - 所有历史 SUCCESS 记录按本次目标速度重新判断达标情况；仅统计与本次
    服务地址相同的历史记录，其他地址的记录会提示后排除。

前置条件：
  - Python 3.9+。
  - 终端环境中已安装且可直接运行 `vllm bench serve`。
  - vLLM 服务无需 token 或 cookie，并实现 OpenAI Chat Completions API。
"""


def main(argv: Optional[Sequence[str]] = None) -> int:
    """脚本入口；支持交互模式和 help 模式。

    用法：
        python3 model_eval_script.py
        python3 model_eval_script.py --help

    八项交互参数及默认值：
        1. 模型服务地址：必填，http:// 或 https:// 开头的可访问地址。
        2. 模型文件路径：必填，存在的本机绝对路径。
        3. 服务端模型名：必填，如 qwen2.5-7b-instruct。
        4. 结果目录：可留空，默认按日期和模型名生成。
        5. 目标速度：可留空，默认 10 tokens/s。
        6. 输入长度列表：可留空，默认 2048,8192,65536,120000,258000。
        7. 输出长度列表：可留空，默认 2048。
        8. 并发数列表：可留空，默认 1,8,16,32,48,64。

    无参数时进入交互模式；``--help`` 只打印帮助，不创建目录、不执行压测。
    """

    arguments = list(sys.argv[1:] if argv is None else argv)
    if arguments in (["--help"], ["-h"]):
        print(_help_text())
        return 0
    if arguments:
        print(
            f"未知参数：{' '.join(arguments)}。如需帮助请使用 --help。",
            file=sys.stderr,
        )
        return 1

    try:
        vllm_executable = shutil.which("vllm")
        if not vllm_executable:
            raise ConfigError("未找到 vllm 命令，请先安装 vLLM 并确认其在 PATH 中")
        config = prompt_for_config()
        runner = VllmRunner(config, executable=vllm_executable)
        return run_evaluation(config, runner=runner)
    except ConfigError as exc:
        print(f"配置错误：{exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("\n用户取消执行。", file=sys.stderr)
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
