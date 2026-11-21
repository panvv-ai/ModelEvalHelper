import os
import io
import json
import stat
import subprocess
import sys
import tempfile
import unittest
from datetime import date
from pathlib import Path

import model_eval_script as app


class InputParsingTests(unittest.TestCase):
    def test_positive_int_rejects_zero_and_float(self):
        self.assertEqual(app.parse_positive_int(" 12 ", "目标速度"), 12)
        for invalid in ("0", "-1", "1.5", "abc", ""):
            with self.subTest(invalid=invalid):
                with self.assertRaises(app.ConfigError):
                    app.parse_positive_int(invalid, "目标速度")

    def test_positive_int_list_splits_mixed_separators_and_deduplicates(self):
        result = app.parse_positive_int_list(
            " 2048, 8192 65536,,2048 ",
            app.DEFAULT_INPUT_LENGTHS,
            "输入长度",
        )
        self.assertEqual(result, (2048, 8192, 65536))

    def test_positive_int_list_uses_defaults_for_blank_input(self):
        self.assertEqual(
            app.parse_positive_int_list("  ", app.DEFAULT_CONCURRENCIES, "并发数"),
            app.DEFAULT_CONCURRENCIES,
        )

    def test_positive_int_list_rejects_any_invalid_item(self):
        for invalid in ("1,zero", "1,0", "1,-2", "1,2.5"):
            with self.subTest(invalid=invalid):
                with self.assertRaises(app.ConfigError):
                    app.parse_positive_int_list(
                        invalid, app.DEFAULT_CONCURRENCIES, "并发数"
                    )

    def test_service_url_normalizes_http_endpoint(self):
        endpoint = app.parse_service_url("http://127.0.0.1:8000/")
        self.assertEqual(endpoint.host, "127.0.0.1")
        self.assertEqual(endpoint.port, 8000)
        self.assertEqual(endpoint.base_url, "http://127.0.0.1:8000")

    def test_service_url_rejects_unsupported_or_ambiguous_forms(self):
        invalid_urls = (
            "127.0.0.1:8000",
            "https://127.0.0.1:8000",
            "http://user:pass@127.0.0.1:8000",
            "http://127.0.0.1:8000/v1",
            "http://127.0.0.1:8000?x=1",
            "http://127.0.0.1:70000",
        )
        for invalid in invalid_urls:
            with self.subTest(invalid=invalid):
                with self.assertRaises(app.ConfigError):
                    app.parse_service_url(invalid)

    def test_model_name_replaces_every_non_allowed_character(self):
        self.assertEqual(
            app.sanitize_model_name("/models/Qwen 2.5-7B_instruct"),
            "Qwen-2-5-7B_instruct",
        )
        with self.assertRaises(app.ConfigError):
            app.sanitize_model_name("/models/模型")

    def test_model_path_must_be_existing_absolute_path(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            self.assertEqual(
                app.validate_model_path(temp_dir), str(Path(temp_dir).resolve())
            )
            with self.assertRaises(app.ConfigError):
                app.validate_model_path("models/Qwen")
            with self.assertRaises(app.ConfigError):
                app.validate_model_path(str(Path(temp_dir) / "missing"))

    def test_default_result_dir_contains_date_and_sanitized_model(self):
        result = app.resolve_result_dir(
            "", "Qwen-7B", today=date(2026, 11, 22)
        )
        self.assertEqual(
            result,
            Path("/data/model_eval_result/20261122/Qwen-7B").resolve(),
        )

    def test_explicit_result_dir_is_resolved(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            result = app.resolve_result_dir(temp_dir, "Qwen-7B")
            self.assertEqual(result, Path(temp_dir).resolve())

    def test_cases_follow_input_output_concurrency_order(self):
        cases = app.build_cases((128, 256), (8, 16), (1, 2))
        self.assertEqual(
            [(case.input_len, case.output_len, case.concurrency) for case in cases],
            [
                (128, 8, 1),
                (128, 8, 2),
                (128, 16, 1),
                (128, 16, 2),
                (256, 8, 1),
                (256, 8, 2),
                (256, 16, 1),
                (256, 16, 2),
            ],
        )


def make_config(temp_dir, model_path="/models/Qwen 7B"):
    return app.EvalConfig(
        endpoint=app.parse_service_url("http://127.0.0.1:8000"),
        model_path=model_path,
        model_name=app.sanitize_model_name(model_path),
        result_dir=Path(temp_dir).resolve(),
        target_speed=10,
        input_lengths=(128,),
        output_lengths=(8,),
        concurrencies=(1, 2),
    )


def make_record(
    *,
    concurrency=1,
    status="SUCCESS",
    throughput=12.5,
    completed=None,
    failed=0,
):
    if completed is None:
        completed = concurrency if status == "SUCCESS" else 0
    return app.BenchmarkRecord(
        run_id="run-1",
        timestamp="2026-11-22T10:00:00+08:00",
        service_url="http://127.0.0.1:8000",
        model_name="Qwen-7B",
        model_path="/models/Qwen 7B",
        target_tokens_per_sec=10,
        seed=42,
        input_len=128,
        output_len=8,
        concurrency=concurrency,
        completed=completed,
        failed=failed,
        output_throughput_tok_s=throughput if status != "FAILED" else None,
        mean_ttft_ms=20.0 if status != "FAILED" else None,
        mean_tpot_ms=4.0 if status != "FAILED" else None,
        duration_s=1.2 if status != "FAILED" else None,
        status=status,
        error="" if status == "SUCCESS" else "simulated",
    )


class ResultStorageTests(unittest.TestCase):
    def test_csv_appends_header_once_and_preserves_previous_rows(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config = make_config(temp_dir)
            first_store = app.ResultStore(config)
            first_store.append_record(make_record(concurrency=1))

            second_store = app.ResultStore(config)
            second_store.append_record(make_record(concurrency=2))

            lines = config.csv_path.read_text(encoding="utf-8").splitlines()
            self.assertEqual(lines[0].split(","), list(app.CSV_FIELDS))
            self.assertEqual(sum(",".join(app.CSV_FIELDS) == line for line in lines), 1)
            self.assertEqual(len(lines), 3)
            self.assertIn("run-1", lines[1])
            self.assertIn(",2,", lines[2])

    def test_raw_jsonl_appends_one_complete_json_object_per_line(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config = make_config(temp_dir)
            store = app.ResultStore(config)
            store.append_raw({"output_throughput": 12.5, "completed": 1})
            store.append_raw({"output_throughput": 20.0, "completed": 2})

            records = [
                json.loads(line)
                for line in config.raw_path.read_text(encoding="utf-8").splitlines()
            ]
            self.assertEqual(
                records,
                [
                    {"output_throughput": 12.5, "completed": 1},
                    {"output_throughput": 20.0, "completed": 2},
                ],
            )

    def test_existing_incompatible_header_is_rejected_without_modification(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config = make_config(temp_dir)
            original = "bad,header\nold,row\n"
            config.csv_path.write_text(original, encoding="utf-8")

            store = app.ResultStore(config)
            with self.assertRaises(app.ConfigError):
                store.validate_existing()
            self.assertEqual(config.csv_path.read_text(encoding="utf-8"), original)

    def test_existing_csv_for_another_model_path_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            first_config = make_config(temp_dir, "/models/Qwen 7B")
            app.ResultStore(first_config).append_record(make_record())

            second_config = app.EvalConfig(
                endpoint=first_config.endpoint,
                model_path="/other/Qwen 7B",
                model_name=first_config.model_name,
                result_dir=first_config.result_dir,
                target_speed=10,
                input_lengths=(128,),
                output_lengths=(8,),
                concurrencies=(1,),
            )
            with self.assertRaises(app.ConfigError):
                app.ResultStore(second_config).validate_existing()

    def test_load_records_preserves_status_and_numeric_metrics(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config = make_config(temp_dir)
            store = app.ResultStore(config)
            store.append_record(make_record(status="SUCCESS", throughput=13.25))
            store.append_record(
                make_record(
                    concurrency=2,
                    status="PARTIAL",
                    throughput=5.5,
                    completed=1,
                    failed=1,
                )
            )

            rows = store.load_records()
            self.assertEqual([row.status for row in rows], ["SUCCESS", "PARTIAL"])
            self.assertEqual(rows[0].output_throughput_tok_s, 13.25)
            self.assertEqual(rows[1].failed, 1)


def make_vllm_payload(completed=2, failed=0):
    return {
        "duration": 3.25,
        "completed": completed,
        "failed": failed,
        "total_input_tokens": 256,
        "total_output_tokens": 16,
        "request_throughput": 0.615,
        "request_goodput": None,
        "output_throughput": 4.923,
        "total_token_throughput": 83.692,
        "mean_ttft_ms": 20.0,
        "median_ttft_ms": 20.0,
        "std_ttft_ms": 1.0,
        "p99_ttft_ms": 20.5,
        "mean_tpot_ms": 4.0,
        "median_tpot_ms": 4.0,
        "std_tpot_ms": 0.1,
        "p99_tpot_ms": 4.1,
        "mean_itl_ms": 4.0,
        "median_itl_ms": 4.0,
        "std_itl_ms": 0.1,
        "p99_itl_ms": 4.1,
        "max_output_tokens_per_s": 8.0,
        "max_concurrent_requests": 2,
        "input_lens": [128, 128],
        "output_lens": [8, 8],
        "ttfts": [0.019, 0.021],
        "itls": [[0.004, 0.004] for _ in range(2)],
        "latencies": [0.05, 0.06],
        "start_times": [100.0, 100.1],
        "queue_times": [0.0, 0.0],
        "generated_texts": ["a" * 8, "b" * 8],
        "errors": [],
    }


class VllmCommandAndRecordTests(unittest.TestCase):
    def test_vllm_command_contains_exact_benchmark_arguments(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config = make_config(temp_dir)
            case = app.BenchmarkCase(input_len=128, output_len=8, concurrency=2)
            result_path = Path(temp_dir) / ".vllm_tmp" / "case.json"

            command = app.build_vllm_command(config, case, result_path)

            self.assertEqual(command[0:2], ["vllm", "bench"])
            self.assertEqual(
                command[2:],
                [
                    "serve",
                    "--host",
                    "127.0.0.1",
                    "--port",
                    "8000",
                    "--model",
                    "Qwen-7B",
                    "--tokenizer",
                    "/models/Qwen 7B",
                    "--dataset-name",
                    "random",
                    "--endpoint",
                    "/v1/chat/completions",
                    "--backend",
                    "openai-chat",
                    "--seed",
                    "42",
                    "--random-input-len",
                    "128",
                    "--random-output-len",
                    "8",
                    "--num-prompts",
                    "2",
                    "--max-concurrency",
                    "2",
                    "--save-result",
                    "--save-detailed",
                    "--append-result",
                    "--result-filename",
                    "case.json",
                    "--result-dir",
                    str(result_path.parent),
                ],
            )

    def test_success_record_extracts_summary_metrics(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config = make_config(temp_dir)
            case = app.BenchmarkCase(input_len=128, output_len=8, concurrency=2)
            record = app.make_record_from_result(
                make_vllm_payload(),
                config,
                case,
                run_id="run-1",
                timestamp="2026-11-22T10:00:00+08:00",
            )

            self.assertEqual(record.status, "SUCCESS")
            self.assertEqual(record.completed, 2)
            self.assertEqual(record.failed, 0)
            self.assertEqual(record.output_throughput_tok_s, 4.923)
            self.assertEqual(record.mean_ttft_ms, 20.0)
            self.assertEqual(record.mean_tpot_ms, 4.0)
            self.assertEqual(record.duration_s, 3.25)

    def test_incomplete_requests_are_partial_and_excluded_by_status(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config = make_config(temp_dir)
            case = app.BenchmarkCase(input_len=128, output_len=8, concurrency=2)
            record = app.make_record_from_result(
                make_vllm_payload(completed=1, failed=1),
                config,
                case,
                run_id="run-1",
                timestamp="2026-11-22T10:00:00+08:00",
            )

            self.assertEqual(record.status, "PARTIAL")
            self.assertEqual(record.completed, 1)
            self.assertEqual(record.failed, 1)
            self.assertIn("1/2", record.error)

    def test_success_payload_without_required_metrics_becomes_failed(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config = make_config(temp_dir)
            case = app.BenchmarkCase(input_len=128, output_len=8, concurrency=2)
            payload = make_vllm_payload()
            del payload["output_throughput"]

            record = app.make_record_from_result(
                payload,
                config,
                case,
                run_id="run-1",
                timestamp="2026-11-22T10:00:00+08:00",
            )

            self.assertEqual(record.status, "FAILED")
            self.assertIsNone(record.output_throughput_tok_s)
            self.assertIn("output_throughput", record.error)


def history_record(
    *,
    input_len,
    output_len,
    concurrency,
    throughput,
    status="SUCCESS",
    recorded_target=100,
):
    return app.BenchmarkRecord(
        run_id=f"run-{input_len}-{output_len}-{concurrency}-{throughput}",
        timestamp="2026-11-22T10:00:00+08:00",
        service_url="http://127.0.0.1:8000",
        model_name="Qwen-7B",
        model_path="/models/Qwen 7B",
        target_tokens_per_sec=recorded_target,
        seed=42,
        input_len=input_len,
        output_len=output_len,
        concurrency=concurrency,
        completed=concurrency if status == "SUCCESS" else 0,
        failed=0,
        output_throughput_tok_s=throughput,
        mean_ttft_ms=20.0,
        mean_tpot_ms=4.0,
        duration_s=1.0,
        status=status,
        error="" if status == "SUCCESS" else "simulated",
    )


class HistoricalAnalysisTests(unittest.TestCase):
    def test_analysis_uses_latest_success_current_target_and_two_level_maximum(self):
        history = (
            history_record(
                input_len=128,
                output_len=8,
                concurrency=1,
                throughput=100.0,
            ),
            history_record(
                input_len=128,
                output_len=8,
                concurrency=1,
                throughput=5.0,
            ),
            history_record(
                input_len=128,
                output_len=8,
                concurrency=2,
                throughput=20.0,
            ),
            history_record(
                input_len=128,
                output_len=8,
                concurrency=4,
                throughput=12.0,
            ),
            history_record(
                input_len=128,
                output_len=16,
                concurrency=1,
                throughput=11.0,
            ),
            history_record(
                input_len=128,
                output_len=16,
                concurrency=2,
                throughput=10.0,
            ),
            history_record(
                input_len=128,
                output_len=16,
                concurrency=4,
                throughput=9.0,
            ),
            history_record(
                input_len=128,
                output_len=16,
                concurrency=2,
                throughput=999.0,
                status="PARTIAL",
            ),
            history_record(
                input_len=256,
                output_len=8,
                concurrency=1,
                throughput=999.0,
                status="FAILED",
            ),
        )

        analysis = app.analyze_history(history, target_speed=10)

        self.assertEqual([entry.input_len for entry in analysis], [128, 256])
        input_128 = analysis[0]
        self.assertEqual(
            [(pair.output_len, pair.max_concurrency) for pair in input_128.pairs],
            [(8, 4), (16, 2)],
        )
        self.assertEqual(
            input_128.pairs[0].passing_concurrencies,
            (2, 4),
        )
        self.assertEqual(
            input_128.pairs[1].passing_concurrencies,
            (1, 2),
        )
        self.assertEqual(input_128.common_max_concurrency, 2)

        input_256 = analysis[1]
        self.assertFalse(input_256.pairs[0].has_success)
        self.assertIsNone(input_256.common_max_concurrency)

    def test_summary_text_groups_by_input_and_reports_no_target(self):
        history = (
            history_record(
                input_len=128,
                output_len=8,
                concurrency=1,
                throughput=9.0,
            ),
            history_record(
                input_len=128,
                output_len=8,
                concurrency=2,
                throughput=8.0,
            ),
            history_record(
                input_len=256,
                output_len=8,
                concurrency=1,
                throughput=0.0,
                status="FAILED",
            ),
        )
        text = app.format_analysis(
            app.analyze_history(history, target_speed=10), target_speed=10
        )

        self.assertIn("输入长度=128", text)
        self.assertIn("输出长度=8", text)
        self.assertIn("无并发达到目标速度", text)
        self.assertIn("没有任何历史成功记录", text)


def create_fake_vllm(directory, payload, exit_code=0):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    payload_path = directory / "payload.json"
    payload_path.write_text(json.dumps(payload), encoding="utf-8")
    executable = directory / "vllm"
    executable.write_text(
        """#!/usr/bin/env python3
import json
import pathlib
import sys

args = sys.argv[1:]
if len(args) < 3 or args[:2] != ["bench", "serve"]:
    print("unexpected arguments", file=sys.stderr)
    raise SystemExit(9)

def value(flag):
    return args[args.index(flag) + 1]

payload_path = pathlib.Path(__file__).with_name("payload.json")
result_dir = pathlib.Path(value("--result-dir"))
result_dir.mkdir(parents=True, exist_ok=True)
result_path = result_dir / value("--result-filename")
result_path.write_text(payload_path.read_text(encoding="utf-8"), encoding="utf-8")
print("mock vllm benchmark output")
raise SystemExit(%d)
"""
        % exit_code,
        encoding="utf-8",
    )
    executable.chmod(executable.stat().st_mode | stat.S_IXUSR)
    return executable


class VllmRunnerIntegrationTests(unittest.TestCase):
    def test_runner_executes_subprocess_and_reads_result_json(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config = make_config(temp_dir)
            fake_vllm = create_fake_vllm(temp_dir, make_vllm_payload())
            stream = io.StringIO()
            runner = app.VllmRunner(
                config, executable=str(fake_vllm), output_stream=stream
            )
            case = app.BenchmarkCase(input_len=128, output_len=8, concurrency=2)

            result = runner.run(case, run_id="run-1")

            self.assertEqual(result.payload["output_throughput"], 4.923)
            self.assertTrue(result.temp_dir.is_dir())
            self.assertIn("mock vllm benchmark output", stream.getvalue())

    def test_runner_reports_nonzero_exit_and_live_output(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config = make_config(temp_dir)
            fake_vllm = create_fake_vllm(
                temp_dir, make_vllm_payload(), exit_code=3
            )
            stream = io.StringIO()
            runner = app.VllmRunner(
                config, executable=str(fake_vllm), output_stream=stream
            )
            case = app.BenchmarkCase(input_len=128, output_len=8, concurrency=2)

            with self.assertRaises(app.BenchmarkExecutionError) as raised:
                runner.run(case, run_id="run-1")

            self.assertIn("退出码 3", str(raised.exception))
            self.assertIn("mock vllm benchmark output", stream.getvalue())


class FakeRunner:
    def __init__(self, config, outcomes):
        self.config = config
        self.outcomes = list(outcomes)
        self.calls = []
        self._temp_root = config.result_dir / ".fake_tmp"
        self._counter = 0

    def run(self, case, run_id):
        self.calls.append((case, run_id))
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        self._counter += 1
        temp_dir = self._temp_root / f"case-{self._counter}"
        temp_dir.mkdir(parents=True)
        return app.BenchmarkRunOutput(payload=outcome, temp_dir=temp_dir)


class OrchestrationTests(unittest.TestCase):
    def test_run_evaluation_appends_all_successes_and_prints_history_summary(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config = make_config(temp_dir)
            runner = FakeRunner(
                config,
                [
                    make_vllm_payload(completed=1, failed=0),
                    make_vllm_payload(completed=2, failed=0),
                ],
            )
            output = io.StringIO()

            exit_code = app.run_evaluation(
                config, runner=runner, output_stream=output
            )

            self.assertEqual(exit_code, 0)
            rows = app.ResultStore(config).load_records()
            self.assertEqual([row.status for row in rows], ["SUCCESS", "SUCCESS"])
            self.assertEqual(len(config.raw_path.read_text().splitlines()), 2)
            self.assertIn("历史达标分析", output.getvalue())
            self.assertEqual(len(runner.calls), 2)

    def test_run_evaluation_continues_after_failure_and_returns_2(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config = make_config(temp_dir)
            runner = FakeRunner(
                config,
                [
                    app.BenchmarkExecutionError("mock failure"),
                    make_vllm_payload(completed=2, failed=0),
                ],
            )

            exit_code = app.run_evaluation(
                config, runner=runner, output_stream=io.StringIO()
            )

            self.assertEqual(exit_code, 2)
            rows = app.ResultStore(config).load_records()
            self.assertEqual([row.status for row in rows], ["FAILED", "SUCCESS"])
            self.assertEqual(rows[0].concurrency, 1)
            self.assertEqual(rows[1].concurrency, 2)

    def test_keyboard_interrupt_marks_current_case_and_stops_remaining_runs(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config = make_config(temp_dir)
            runner = FakeRunner(config, [KeyboardInterrupt()])

            exit_code = app.run_evaluation(
                config, runner=runner, output_stream=io.StringIO()
            )

            self.assertEqual(exit_code, 130)
            rows = app.ResultStore(config).load_records()
            self.assertEqual([row.status for row in rows], ["INTERRUPTED"])
            self.assertEqual(len(runner.calls), 1)

    def test_unexpected_runner_error_is_recorded_and_batch_continues(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config = make_config(temp_dir)
            runner = FakeRunner(
                config,
                [
                    OSError("unexpected local failure"),
                    make_vllm_payload(completed=2, failed=0),
                ],
            )

            exit_code = app.run_evaluation(
                config, runner=runner, output_stream=io.StringIO()
            )

            self.assertEqual(exit_code, 2)
            rows = app.ResultStore(config).load_records()
            self.assertEqual([row.status for row in rows], ["FAILED", "SUCCESS"])
            self.assertIn("unexpected local failure", rows[0].error)


class CliAndPromptTests(unittest.TestCase):
    def test_build_eval_config_applies_all_defaults(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir) / "Qwen 7B"
            model_dir.mkdir()

            config = app.build_eval_config(
                service_url="http://127.0.0.1:8000",
                model_path=os.fspath(model_dir),
                result_dir="",
                target_speed="",
                input_lengths="",
                output_lengths="",
                concurrencies="",
                today=date(2026, 11, 22),
            )

            self.assertEqual(config.target_speed, 10)
            self.assertEqual(config.input_lengths, app.DEFAULT_INPUT_LENGTHS)
            self.assertEqual(config.output_lengths, app.DEFAULT_OUTPUT_LENGTHS)
            self.assertEqual(config.concurrencies, app.DEFAULT_CONCURRENCIES)
            self.assertEqual(config.model_name, "Qwen-7B")
            self.assertEqual(
                config.result_dir,
                Path("/data/model_eval_result/20261122/Qwen-7B").resolve(),
            )

    def test_prompt_reprompts_after_invalid_value_and_collects_seven_fields(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir) / "Qwen 7B"
            model_dir.mkdir()
            answers = iter(
                [
                    "http://127.0.0.1:8000",
                    os.fspath(model_dir),
                    temp_dir,
                    "0",
                    "10",
                    "128 256",
                    "8,16",
                    "1 2",
                ]
            )
            output = io.StringIO()

            config = app.prompt_for_config(
                input_func=lambda prompt: next(answers),
                output_stream=output,
                today=date(2026, 11, 22),
            )

            self.assertEqual(config.target_speed, 10)
            self.assertEqual(config.input_lengths, (128, 256))
            self.assertEqual(config.output_lengths, (8, 16))
            self.assertEqual(config.concurrencies, (1, 2))
            self.assertIn("必须是正整数", output.getvalue())

    def test_prompt_eof_raises_config_error_instead_of_looping(self):
        calls = []

        def exhausted_reader(prompt):
            calls.append(prompt)
            if len(calls) < 3:
                raise EOFError()
            raise AssertionError("EOF 后不应再次调用输入函数")

        with self.assertRaises(app.ConfigError):
            app._read_with_retry(
                exhausted_reader,
                "输入：",
                lambda value: value,
                io.StringIO(),
            )

    def test_help_runs_without_starting_prompts_and_lists_contract(self):
        script = Path(app.__file__).resolve()
        completed = subprocess.run(
            [sys.executable, os.fspath(script), "--help"],
            cwd=script.parent,
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertEqual(completed.returncode, 0)
        self.assertIn("模型服务地址", completed.stdout)
        self.assertIn("目标速度", completed.stdout)
        self.assertIn("2048,8192,65536,120000,258000", completed.stdout)
        self.assertIn("1,8,16,32,48,64", completed.stdout)
        self.assertNotIn("请输入模型服务地址", completed.stdout)

    def test_unknown_cli_argument_returns_1(self):
        script = Path(app.__file__).resolve()
        completed = subprocess.run(
            [sys.executable, os.fspath(script), "--unknown"],
            cwd=script.parent,
            text=True,
            capture_output=True,
            check=False,
        )

        self.assertEqual(completed.returncode, 1)
        self.assertIn("未知参数", completed.stderr)


if __name__ == "__main__":
    unittest.main()
