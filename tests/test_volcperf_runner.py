import argparse
import asyncio
import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest

from volcperf.cases import CASES
from volcperf.config import ApiConfig
from volcperf.connectivity import connectivity_payload
from volcperf.prompts import Workload
from volcperf.protocols import payload
from volcperf.runner import (
    ResultWriter,
    build_closed_request_plan,
    build_open_request_plan,
    effective_reasoning_effort,
    load_case_prompts,
    metadata,
    request_config,
    run_closed,
)


class TrackingRecord:
    def __init__(self, record_id: str, built: list[str], *, broken: bool = False):
        self.id = record_id
        self.built = built
        self.broken = broken

    def workload(self, fixed_output: bool = False) -> Workload:
        self.built.append(self.id)
        if self.broken:
            raise ValueError("broken workload")
        return Workload(
            "fixed_report" if fixed_output else "qa",
            [{"role": "user", "content": self.id}],
            16,
        )


def test_closed_load_prebuilds_all_requests_and_continues_after_case_errors():
    built: list[str] = []
    records = [
        TrackingRecord("I1-1", built),
        TrackingRecord("I1-2", built, broken=True),
        TrackingRecord("I1-3", built),
    ]
    config = ApiConfig(
        api_key="secret",
        base_url="https://example.test/v1",
        model="test-model",
        request_timeout_seconds=1,
    )
    executed: list[str] = []

    async def fake_stream_chat(client, config, **kwargs):
        # All workload definitions, including the broken one, must have been
        # materialized before the first HTTP execution starts.
        assert built == ["I1-1", "I1-2", "I1-3"]
        executed.append(kwargs["dataset_id"])
        if kwargs["dataset_id"] == "I1-1":
            raise RuntimeError("one request failed")
        return {
            "case": kwargs["case_name"],
            "counted": kwargs["counted"],
            "success": True,
            "error_type": None,
            "dataset_id": kwargs["dataset_id"],
        }

    async def run(path: Path):
        writer = ResultWriter(path)
        try:
            with (
                patch("volcperf.runner.ApiConfig.from_env", return_value=config),
                patch("volcperf.runner.stream_chat", side_effect=fake_stream_chat),
            ):
                return await run_closed(
                    CASES["T0"],
                    records,
                    writer,
                    samples=3,
                    warmup=0,
                    duration=0,
                )
        finally:
            writer.close()

    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "requests.jsonl"
        rows = asyncio.run(run(path))
        persisted = [json.loads(line) for line in path.read_text().splitlines()]

    assert [row["sequence"] for row in rows] == [1, 2, 3]
    assert [row["error_type"] for row in rows] == [
        "internal_RuntimeError",
        "plan_ValueError",
        None,
    ]
    assert executed == ["I1-1", "I1-3"]
    assert len(persisted) == 3


def test_open_load_prebuilds_the_complete_relative_schedule():
    built: list[str] = []
    records = [TrackingRecord(f"C1-{index}", built) for index in range(5)]
    plans = build_open_request_plan(
        CASES["C0"],
        records,
        (("ramp", 5.0, 1, True),),
    )

    assert built == [f"C1-{index}" for index in range(5)]
    assert [plan["sequence"] for plan in plans] == [1, 2, 3, 4, 5]
    assert [plan["planned_offset_seconds"] for plan in plans] == pytest.approx([0.0, 0.2, 0.4, 0.6, 0.8])
    assert all(plan["phase"] == "ramp" and plan["counted"] for plan in plans)


def test_random_load_generates_exact_bounded_request_set_with_4096_output_limit():
    case = CASES["RANDOM_LOAD"]
    prompts = load_case_prompts(case, samples=37)
    plans = build_closed_request_plan(case, prompts, 37)

    assert case.max_tokens == 4096
    assert len(prompts) == len(plans) == 37
    assert all(1000 <= record.target_prompt_tokens <= 20000 for record in prompts)
    assert len({record.target_prompt_tokens for record in prompts}) > 1
    assert [plan["target_prompt_tokens"] for plan in plans] == [record.target_prompt_tokens for record in prompts]
    assert all(plan["max_tokens"] == 4096 for plan in plans)


def _load_config(protocol: str = "qianfan", effort: str | None = None) -> ApiConfig:
    return ApiConfig(
        api_key="secret",
        base_url="https://qianfan.baidubce.com",
        model="deepseek-v4.1-flash",
        protocol=protocol,
        reasoning_effort=effort,
    )


def test_load_cases_align_thinking_with_the_connectivity_probe():
    """正式请求的思考档位必须与前置探针同口径。

    历史事故（批次 06707685f5044526，deepseek-v4.1-flash）：探针固定发
    reasoning_effort=low，正式请求却走供应商默认思考；思考 token 计入输出上限，
    L1–L5 的 1024 预算被整段吃进推理，模型还没开始写正文就 finish_reason=length
    ⇒ 105/300 请求零可见正文，六档全灭（全部满足
    reasoning_tokens == completion_tokens == max_tokens）。
    """
    probe = connectivity_payload(_load_config())
    assert probe["reasoning_effort"] == "low"
    for name in ("T0", "B1", "RANDOM_LOAD", "L1", "L2", "L3", "L4", "L5", "O1", "C0", "C1"):
        case = CASES[name]
        with patch("volcperf.runner.ApiConfig.from_env", return_value=_load_config()):
            config = request_config(case)
        assert config.reasoning_effort == probe["reasoning_effort"], name
        body = payload(
            config.protocol,
            config.model,
            [{"role": "user", "content": "hi"}],
            case.max_tokens,
            config.include_usage,
            config.reasoning_effort,
        )
        assert body["reasoning_effort"] == "low", name


def test_load_cases_reserve_thinking_headroom_in_output_budget():
    """输出上限必须留出思考余量，否则思考吃满预算 ⇒ 零可见正文。

    2026-10-14 两次真实批次都栽在这里：批次 06707685f5044526（L1–L5=1024 / O1=2048，
    供应商默认思考）六档全灭；修成 effort=low 后复跑（批次 bdb8e94cc40c4fcf）flash
    仍 22–26%/档 吃满 1024。故上限按「正文目标 + 思考余量」给：
    L1–L5 / B1 / RANDOM_LOAD = 4096，O1 = 6144（其正文目标 400–700 另需余量）。
    """
    assert {name: CASES[name].max_tokens for name in ("L1", "L2", "L3", "L4", "L5")} == {
        "L1": 4096, "L2": 4096, "L3": 4096, "L4": 4096, "L5": 4096,
    }
    assert CASES["O1"].max_tokens == 6144 and CASES["O1"].max_tokens > CASES["O1"].max_visible_tokens
    # 同族档位（正文长度相同的 L1/L2/L3 与 B1 基线）必须同预算，否则 TTFC 不可比。
    assert CASES["B1"].max_tokens == CASES["L1"].max_tokens
    assert CASES["RANDOM_LOAD"].max_tokens == 4096


def test_full_load_covers_every_tier_sample_requirement():
    """「完整压测」的样本数必须 ≥ 各档位要求（L1–L3=200、L4=150、L5/O1=100）。"""
    from web.server import LOAD_PRESETS

    preset = LOAD_PRESETS["FULL_LOAD"]
    for name in ("L1", "L2", "L3", "L4", "L5"):
        assert preset["samples"] >= CASES[name].samples, name
    assert CASES["O1"].samples >= 100


def test_thinking_effort_env_override_wins_over_case_default():
    with patch("volcperf.runner.ApiConfig.from_env", return_value=_load_config(effort="high")):
        assert request_config(CASES["L1"]).reasoning_effort == "high"


def test_protocol_without_reasoning_control_reports_provider_default():
    """anthropic / gemini / responses 分支不发该字段，就不许把它记成已生效。"""
    for protocol in ("anthropic", "gemini", "openai_responses"):
        with patch("volcperf.runner.ApiConfig.from_env", return_value=_load_config(protocol=protocol)):
            assert effective_reasoning_effort(CASES["L1"]) is None, protocol


def test_metadata_records_the_effective_thinking_effort():
    """报告里的「Thinking：…」直接读 metadata/summary，别写成「供应商默认」。"""
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "metadata.json"
        with patch("volcperf.runner.ApiConfig.from_env", return_value=_load_config()):
            metadata(path, CASES["L1"], argparse.Namespace(case="L1"))
        recorded = json.loads(path.read_text(encoding="utf-8"))
    assert recorded["thinking"] == {"provider_default": False, "reasoning_effort": "low"}
