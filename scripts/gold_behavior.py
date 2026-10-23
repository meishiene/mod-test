#!/usr/bin/env python3
"""采集**行为面**样本（拒绝话术 / 约束题 / 推理陷阱 / 工具形态）。

两种模式：

    # A. 有一把 key（官方或中转都行），脚本采样
    GOLD_PROBE_KEY=sk-xxx .venv/bin/python scripts/gold_behavior.py \
        --base-url https://api.deepseek.com/v1 --model deepseek-chat \
        --label deepseek-official --repeats 5

    # B. 端点无法被脚本调用（例如某个 Agent 平台里的模型），手工录入
    .venv/bin/python scripts/gold_behavior.py --record answers.json \
        --label workbuddy-gateway --model deepseek-v4.1-flash

关键口径（写进产物，不允许被下游忽略）：

- 本脚本**永远不产出 "gold" 结论**。它只产出 ``evidence="L2-single"`` 的**样本**，
  ``usable_as_gold`` 恒为 ``false``。理由：是不是"原厂"，脚本无法自证——它只能证明
  "某个端点在这 12 条题面上这样表现过"。
- 同一会话连续提问会互相污染，所以脚本**每条探针一次独立请求**，不复用 messages 历史；
  手工录入的模式做不到这点，必须在 ``caveats`` 里如实写明。
- ``self_report`` 组只记录、不打分。上游注入人格就能让它说谎，历史已踩过。
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from gold.behavior_probes import (BEHAVIOR_CORPUS_VERSION, BEHAVIOR_PROBES,  # noqa: E402
                                  SCORING_STRUCTURAL, autoscore as _autoscore)

DATA_ROOT = ROOT / "gold" / "data" / "behavior"


def _sample_skeleton(label: str, model: str, role: str, repeats: int) -> dict:
    return {
        "schema": "gold/behavior_sample@1",
        "evidence": "L2-single",
        "usable_as_gold": False,
        "why_not_gold": "单个端点的行为样本无法自证其上游身份；只能作为跨端点分歧检测的参照系",
        "endpoint": {"label": label, "claimed_role": role, "role_is_claim_not_evidence": True},
        "model": model,
        "corpus_version": BEHAVIOR_CORPUS_VERSION,
        "repeats": repeats,
        "collected_at": datetime.now().isoformat(timespec="seconds"),
        "caveats": [],
        "probes": {},
    }


def _summarize(sample: dict) -> dict:
    """按探针算「跨重复一致性」——混流检测就靠这个，不看单次对错。"""
    summary = {}
    for pid, rows in sample["probes"].items():
        scores = [r.get("autoscore") for r in rows if r.get("autoscore") is not None]
        oks = [bool(s["ok"]) for s in scores]
        summary[pid] = {
            "n": len(rows),
            "scored": len(oks),
            "pass_rate": (sum(oks) / len(oks)) if oks else None,
            "consistent": (len(set(oks)) == 1) if oks else None,
        }
    return summary


def _post(client, url: str, key: str, model: str, prompt: str, max_tokens: int) -> dict:
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],  # 每条独立，不带历史
        "max_tokens": max_tokens,
        "temperature": 0,
        "top_p": 1,
        "stream": False,
    }
    started = time.time()
    resp = client.post(url, json=payload, headers={"Authorization": f"Bearer {key}"})
    elapsed = round((time.time() - started) * 1000)
    try:
        body = resp.json()
    except Exception:
        body = None
    if not isinstance(body, dict):
        return {"http": resp.status_code, "error": "响应非 JSON", "ms": elapsed, "text": None}
    choice = ((body.get("choices") or [{}])[0] or {})
    message = choice.get("message") or {}
    return {
        "http": resp.status_code,
        "error": None,
        "ms": elapsed,
        "text": message.get("content") or "",
        "reasoning_len": len(message.get("reasoning_content") or ""),
        "finish_reason": choice.get("finish_reason"),
        "usage": body.get("usage"),
    }


def _write(label: str, model: str, sample: dict, out: str | None) -> Path:
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    target = Path(out).resolve() if out else DATA_ROOT / f"{label}" / f"{model.replace('/', '_')}@{stamp}"
    target.mkdir(parents=True, exist_ok=True)
    sample["summary"] = _summarize(sample)
    (target / "behavior_sample.json").write_text(
        json.dumps(sample, ensure_ascii=False, indent=2), encoding="utf-8")
    return target


def _display(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--record", default="", help="手工录入的 answers.json：{probe_id: [答案, ...]}")
    ap.add_argument("--base-url", default="")
    ap.add_argument("--api-key", default="")
    ap.add_argument("--model", default="")
    ap.add_argument("--label", default="", help="端点标签，如 deepseek-official / workbuddy-gateway")
    ap.add_argument("--role", default="unknown", choices=["official", "relay", "gateway", "unknown"],
                    help="端点自述角色——只记录，不作为证据")
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--delay", type=float, default=0.5)
    ap.add_argument("--out", default="")
    ap.add_argument("--no-proxy", action="store_true")
    args = ap.parse_args()

    if not (args.model and args.label):
        ap.error("需要 --model 与 --label（端点标签是产物可读性的前提）")

    sample = _sample_skeleton(args.label, args.model, args.role, args.repeats)

    if args.record:
        raw = json.loads(Path(args.record).read_text(encoding="utf-8"))
        sample["caveats"].append("手工录入：答案由人工/agent 从会话中转录，非独立请求，"
                                 "上下文可能已被前序对话污染")
        sample["caveats"].append("无法记录 usage / finish_reason / SSE，无法核验是否被中转改写")
        if args.role in ("gateway", "unknown"):
            sample["caveats"].append("端点角色未知：宿主只声明了模型名，无法验证上游")
        for probe in BEHAVIOR_PROBES:
            answers = raw.get(probe.id) or []
            if isinstance(answers, str):
                answers = [answers]
            rows = []
            for text in answers:
                rows.append({"text": text, "http": None, "error": None,
                             "autoscore": _autoscore(probe.id, text)})
            sample["probes"][probe.id] = rows
        target = _write(args.label, args.model, sample, args.out)
    else:
        import httpx

        api_key = args.api_key or os.environ.get("GOLD_PROBE_KEY", "")
        if not (args.base_url and api_key):
            ap.error("需要 --base-url 与 --api-key（或 GOLD_PROBE_KEY），或改用 --record")
        url = args.base_url.rstrip("/") + "/chat/completions"
        # 中转前置 system 会污染行为面，所以这一层必须说明"这条产物的上限"
        sample["caveats"].append("每条探针独立请求，但仍在同一 key/同一路由上；"
                                 "中转若按请求随机切档，需靠 pass_rate 与 consistent 两列识别")
        with httpx.Client(timeout=120, trust_env=not args.no_proxy) as client:
            for probe in BEHAVIOR_PROBES:
                rows = []
                for _ in range(max(1, args.repeats)):
                    rows.append(_post(client, url, api_key, args.model,
                                      probe.prompt, args.max_tokens))
                    if args.delay:
                        time.sleep(args.delay)
                for row in rows:
                    if row.get("text") is not None and probe.scoring == SCORING_STRUCTURAL:
                        row["autoscore"] = _autoscore(probe.id, row["text"])
                sample["probes"][probe.id] = rows
        target = _write(args.label, args.model, sample, args.out)

    print(f"写入 {_display(target)}")
    for pid, stat in sample["summary"].items():
        rate = "-" if stat["pass_rate"] is None else f"{stat['pass_rate']:.0%}"
        print(f"  {pid}  n={stat['n']:<2} 通过率={rate:<5} 一致={stat['consistent']}")
    print("\n证据等级：L2-single（usable_as_gold=false）。"
          "要变成真金标，必须换官方直连 + 多轮重复 + 人工签字。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
