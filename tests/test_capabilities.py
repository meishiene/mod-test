"""Radar is a separate projection of real verdicts, never a rewrite of 120-point scoring."""
from copy import deepcopy

from common.score_registry import inventory
from common.scoring import score_batch, score_rows


def rows(kind, model=None):
    return [{"case_id": cid, "case_status": "passed"} for cid in inventory(kind, model)]


def axes(report):
    return {axis["key"]: axis for axis in report["capabilities"]["axes"]}


def test_glm_six_scored_axes_and_no_double_count_or_total_change():
    evidence = rows("glm") + [{"case_id": "GLM-53-V01", "case_status": "passed"}]
    before = deepcopy(evidence)
    report = score_rows(evidence, test_kind="glm", model="glm-5.3-flash")
    dimensions = axes(report)
    assert len(dimensions) == 6
    assert all(a["score"] == 100 for a in dimensions.values())
    assert {c["case_id"] for c in dimensions["quality"]["cases"]} == {"GLM-53-06", "GLM-53-F04", "GLM-53-Q02", "GLM-53-Q03", "GLM-53-R03"}
    assert dimensions["vision"]["cases"][0]["case_id"] == "GLM-53-V01"
    ids = [c["case_id"] for a in dimensions.values() for c in a["cases"]]
    assert len(ids) == len(set(ids))
    assert "GLM-53-F01" not in ids and "GLM-53-F08" not in ids
    assert report["score"] == 120
    assert report["case_scores"]["GLM-53-V01"]["weight"] == 0
    assert evidence == before


def test_missing_quality_case_does_not_shrink_denominator():
    report = score_rows([{"case_id": "GLM-53-F04", "case_status": "passed"}], test_kind="glm")
    quality = axes(report)["quality"]
    assert quality["earned"] == 14.2857
    assert quality["coverage"] == .142857
    assert quality["score"] is None
    assert quality["state"] == "证据不足"
    assert axes(report)["vision"]["state"] == "暂无用例"


def test_failure_warning_and_continuous_credit_are_preserved():
    data = rows("glm")
    data += [{"case_id": "GLM-53-06", "case_status": "failed"},
             {"case_id": "GLM-53-F04", "case_status": "warning"},
             {"case_id": "GLM-53-V01", "case_status": "passed", "reward": .6}]
    dimensions = axes(score_rows(data, test_kind="glm"))
    assert dimensions["quality"]["score"] == 50.0
    assert dimensions["vision"]["score"] == 60
    assert dimensions["quality"]["coverage"] == 1
    data += [{"case_id": "GLM-53-V01", "case_status": "failed"}]
    assert axes(score_rows(data, test_kind="glm"))["vision"]["score"] == 0


def test_unsupported_vision_and_unselected_faces_never_get_fake_scores():
    model = "qwen3.6-flash"
    excluded = {cid for cid, meta in inventory("qwen", model).items() if meta.face == "vision"}
    report = score_rows(rows("qwen", model), test_kind="qwen", model=model,
                        not_applicable_ids=excluded, executed_faces=["model"])
    assert axes(report)["vision"]["state"] == "不适用"
    assert axes(report)["vision"]["score"] is None
    assert any(c["status"] == "not_selected" for a in axes(report).values() for c in a["cases"])


def test_suite_quality_has_objective_cases():
    for kind, model in [("glm", "glm-5.3"), ("deepseek", "deepseek-v4-flash-0731"),
                        ("qwen", "qwen3.6-flash"), ("kvv", "kimi-k3"),
                        ("fable", "claude-opus-5"), ("gpt", "gpt-5.6"), ("gemini", "gemini-3.1-pro")]:
        quality = axes(score_rows(rows(kind, model), test_kind=kind, model=model))["quality"]
        assert quality["score"] == 100, kind
        assert quality["cases"], kind


def test_batch_retakes_use_latest_and_keep_models_separate():
    items = []
    for model, status in [("a", "passed"), ("b", "failed"), ("a", "failed")]:
        data = rows("glm") + [{"case_id": "GLM-53-V01", "case_status": status}]
        items.append({"model": model, "summary": {"test_kind": "glm", "results": data}})
    reports = score_batch(items)
    assert [axes(r)["vision"]["score"] for r in reports] == [0, 0]
    assert all(len(axes(r)["vision"]["cases"]) == 1 for r in reports)
