"""Seedance 视频生成套件的命令行入口。

参数分两路注入（与平台既有约定一致）：

* **内容型参数**（提示词 / 分辨率 / 画面比例 / 时长 / 图片 / 开关）走环境变量——
  提示词含换行与引号、图片路径是本地文件，命令行不适合承载；
* **执行型参数**（请求数 / 并发 / 轮询与超时 / 是否留存视频）走命令行，Web 派发时按
  界面输入显式追加。

命令行显式给出的值优先于环境变量。
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path

from common.finalizer import finalize_result_dir

from .cases import (
    CASES,
    SEEDANCE_DEFAULT_DURATION,
    SEEDANCE_DEFAULT_GENERATE_AUDIO,
    SEEDANCE_DEFAULT_IMAGE_ROLE,
    SEEDANCE_DEFAULT_RATIO,
    SEEDANCE_DEFAULT_RESOLUTION,
    SEEDANCE_DEFAULT_VIDEO_ROLE,
    SEEDANCE_DEFAULT_WATERMARK,
    SEEDANCE_IMAGE_ROLES,
    SEEDANCE_INPUT_VIDEO_SUFFIXES,
    SEEDANCE_MAX_CONCURRENCY,
    SEEDANCE_MAX_IMAGE_BYTES,
    SEEDANCE_MAX_INPUT_VIDEO_BYTES,
    SEEDANCE_MAX_PROMPT_CHARS,
    SEEDANCE_MAX_REQUESTS,
    SEEDANCE_MODE_IDS,
    SEEDANCE_RATIOS,
    SEEDANCE_RESOLUTIONS,
    SEEDANCE_VIDEO_ROLES,
    duration_range_for,
    mode_allows_image,
    mode_allows_video,
    resolve_mode,
)
from .core import (
    DEFAULT_MAX_VIDEO_GIB,
    DEFAULT_POLL_INTERVAL_SECONDS,
    DEFAULT_REQUEST_TIMEOUT_SECONDS,
    DEFAULT_TASK_TIMEOUT_SECONDS,
    KEEP_VIDEOS_ENV,
    MAX_VIDEO_GIB_ENV,
    VideoConfig,
    keep_videos_default,
    run_videos,
)

_IMAGE_ROLE_IDS = tuple(item[0] for item in SEEDANCE_IMAGE_ROLES)
# 视频角色允许空串（= 不带 role 字段），因此不能交给 argparse 的 choices 校验。
_VIDEO_ROLE_IDS = tuple(item[0] for item in SEEDANCE_VIDEO_ROLES)
_TRUE = {"1", "true", "yes", "on"}


def _env(key: str, default: str = "") -> str:
    return (os.getenv(key) or "").strip() or default


def _env_bool(key: str, default: bool) -> bool:
    raw = (os.getenv(key) or "").strip().lower()
    if not raw:
        return default
    return raw in _TRUE


def _env_float(key: str, default: float) -> float:
    raw = (os.getenv(key) or "").strip()
    if not raw:
        return default
    return float(raw)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Seedance 视频生成（异步任务 + 轮询）")
    parser.add_argument("case", nargs="?", default=next(iter(CASES)), choices=sorted(CASES))
    parser.add_argument("--output-root", type=Path, default=Path("results"))
    parser.add_argument("--requests", type=int, default=None, help=f"提交的视频任务数（1～{SEEDANCE_MAX_REQUESTS}）")
    parser.add_argument("--concurrency", type=int, default=None, help=f"并发任务数（1～{SEEDANCE_MAX_CONCURRENCY}）")
    parser.add_argument("--resolution", default=None)
    parser.add_argument("--ratio", default=None)
    parser.add_argument("--duration", type=int, default=None)
    parser.add_argument("--mode", default=None, choices=SEEDANCE_MODE_IDS,
                        help="生成方式：text 仅文生视频 / image 带参考图 / video 带参考视频 / both 两者都带；"
                             "不传则按实际提供的图片与参考视频推导")
    parser.add_argument("--image-role", default=None, choices=_IMAGE_ROLE_IDS)
    parser.add_argument("--image-url", default=None)
    parser.add_argument("--image-path", default=None)
    parser.add_argument("--video-role", default=None,
                        help="参考视频的 role（默认 reference_video；传空串表示不带 role 字段）")
    parser.add_argument("--video-url", default=None,
                        help="参考视频地址（唯一会真正提交给上游的视频输入）")
    parser.add_argument("--video-path", default=None,
                        help="本地参考视频文件（只归档留痕，不随任务提交）")
    parser.add_argument("--poll-interval", type=float, default=None)
    parser.add_argument("--task-timeout", type=float, default=None)
    parser.add_argument("--request-timeout", type=float, default=None)
    parser.add_argument("--generate-audio", dest="generate_audio",
                        action=argparse.BooleanOptionalAction, default=None,
                        help="是否让模型生成原生音频（generate_audio）")
    parser.add_argument("--watermark", action=argparse.BooleanOptionalAction, default=None,
                        help="生成的视频是否带水印（watermark）")
    parser.add_argument("--keep-videos", action=argparse.BooleanOptionalAction,
                        default=keep_videos_default(),
                        help=f"是否把生成的视频下载到结果目录 videos/（默认开启；可用 {KEEP_VIDEOS_ENV}=0 关闭）")
    parser.add_argument("--max-video-gib", type=float, default=None,
                        help=f"单批保留视频总量上限（GiB，默认 {DEFAULT_MAX_VIDEO_GIB:g}；可用 {MAX_VIDEO_GIB_ENV} 覆盖）")
    return parser.parse_args(argv)


def build_config(args: argparse.Namespace) -> VideoConfig:
    api_key = _env("SEEDANCE_API_KEY") or _env("VOLC_API_KEY")
    base_url = _env("SEEDANCE_BASE_URL") or _env("VOLC_BASE_URL")
    model = _env("SEEDANCE_MODEL") or _env("VOLC_MODEL")
    prompt = _env("SEEDANCE_PROMPT")
    if not api_key:
        raise ValueError("缺少 SEEDANCE_API_KEY")
    if not base_url:
        raise ValueError("缺少 SEEDANCE_BASE_URL")
    if not model:
        raise ValueError("缺少 SEEDANCE_MODEL（视频模型名或接入点 ID）")
    if not prompt:
        raise ValueError("缺少提示词（SEEDANCE_PROMPT）")
    if len(prompt) > SEEDANCE_MAX_PROMPT_CHARS:
        raise ValueError(f"提示词过长（{len(prompt)} 字符，上限 {SEEDANCE_MAX_PROMPT_CHARS}）")

    resolution = args.resolution or _env("SEEDANCE_RESOLUTION", SEEDANCE_DEFAULT_RESOLUTION)
    if resolution not in SEEDANCE_RESOLUTIONS:
        raise ValueError(f"分辨率必须是 {'/'.join(SEEDANCE_RESOLUTIONS)} 之一：{resolution}")
    ratio = args.ratio or _env("SEEDANCE_RATIO", SEEDANCE_DEFAULT_RATIO)
    if ratio not in SEEDANCE_RATIOS:
        raise ValueError(f"画面比例必须是 {'/'.join(SEEDANCE_RATIOS)} 之一：{ratio}")
    duration = args.duration if args.duration is not None else int(_env("SEEDANCE_DURATION") or SEEDANCE_DEFAULT_DURATION)
    duration_min, duration_max = duration_range_for(model)
    if not duration_min <= duration <= duration_max:
        raise ValueError(
            f"时长必须是 {duration_min}～{duration_max} 的整数秒（按模型 {model} 的档位解析）：{duration}"
        )
    image_role = args.image_role or _env("SEEDANCE_IMAGE_ROLE", SEEDANCE_DEFAULT_IMAGE_ROLE)
    if image_role not in _IMAGE_ROLE_IDS:
        raise ValueError(f"图片位置必须是 {'/'.join(_IMAGE_ROLE_IDS)} 之一：{image_role}")

    image_path = args.image_path or _env("SEEDANCE_IMAGE_PATH")
    if image_path:
        source = Path(image_path)
        if not source.is_file():
            raise ValueError(f"上传的图片不存在或不可读：{image_path}")
        size = source.stat().st_size
        if size <= 0:
            raise ValueError("上传的图片为空文件")
        if size > SEEDANCE_MAX_IMAGE_BYTES:
            raise ValueError(f"上传的图片过大（{size} 字节，上限 {SEEDANCE_MAX_IMAGE_BYTES}）")
    image_url = args.image_url or _env("SEEDANCE_IMAGE_URL")
    if image_url and not image_url.startswith(("http://", "https://", "data:")):
        raise ValueError("图片地址必须是 http(s):// 或 data: URI")

    # 角色要区分三种输入：命令行/环境变量显式给值（含空串＝不带 role 字段）、
    # 完全没给（用默认 reference_video）。`_env()` 会把"未设置"和"空串"混成空串，
    # 所以这里直接读 os.getenv，让 None 表示"没给"。
    raw_video_role = args.video_role if args.video_role is not None else os.getenv("SEEDANCE_VIDEO_ROLE")
    video_role = SEEDANCE_DEFAULT_VIDEO_ROLE if raw_video_role is None else str(raw_video_role).strip()
    if video_role not in _VIDEO_ROLE_IDS:
        raise ValueError(
            f"参考视频位置必须是 {'/'.join(item or '(空=不带 role)' for item in _VIDEO_ROLE_IDS)} 之一：{video_role}"
        )
    video_path = args.video_path or _env("SEEDANCE_VIDEO_PATH")
    if video_path:
        source = Path(video_path)
        if not source.is_file():
            raise ValueError(f"上传的参考视频不存在或不可读：{video_path}")
        size = source.stat().st_size
        if size <= 0:
            raise ValueError("上传的参考视频为空文件")
        if size > SEEDANCE_MAX_INPUT_VIDEO_BYTES:
            raise ValueError(
                f"上传的参考视频过大（{size} 字节，上限 {SEEDANCE_MAX_INPUT_VIDEO_BYTES}）"
            )
        suffix = source.suffix.lower().lstrip(".")
        if suffix not in SEEDANCE_INPUT_VIDEO_SUFFIXES:
            raise ValueError(
                f"参考视频格式不支持（.{suffix}）：只接受 {'/'.join('.' + item for item in SEEDANCE_INPUT_VIDEO_SUFFIXES)}"
            )
    video_url = args.video_url or _env("SEEDANCE_VIDEO_URL")
    if video_url and not video_url.startswith(("http://", "https://", "data:")):
        raise ValueError("视频地址必须是 http(s):// 或 data: URI")

    # 生成方式：**显式声明**时与输入严格对齐（多给、少给都直接拒，不悄悄纠偏）；
    # 没声明时按素材推导，沿用"给了就带上"的旧行为——只有本地视频、没有地址那种
    # "仅归档留痕 + 报告 ⚠️"的用法因此保留（见 core.resolve_video）。
    raw_mode = args.mode if args.mode is not None else (os.getenv("SEEDANCE_MODE") or "").strip()
    if raw_mode and raw_mode not in SEEDANCE_MODE_IDS:
        raise ValueError(
            f"生成方式必须是 {'/'.join(SEEDANCE_MODE_IDS)} 之一（text/image/video/both）：{raw_mode}"
        )
    has_image_input = bool(image_path or image_url)
    has_video_input = bool(video_path or video_url)
    mode = resolve_mode(
        raw_mode, has_image=has_image_input, has_video=has_video_input
    )
    if raw_mode:
        if mode_allows_image(mode):
            if not has_image_input:
                raise ValueError(
                    f"生成方式为「{mode}」但没有图片输入：请填参考图（--image-url / --image-path），"
                    "或改用 --mode text / --mode video"
                )
        elif has_image_input:
            raise ValueError(
                f"生成方式为「{mode}」，不会提交图片：请清除图片输入，或改用 --mode image / --mode both"
            )
        if mode_allows_video(mode):
            if not video_url:
                hint = ("（只给了本地视频文件不算：异步任务面只接受 URL，本地素材仅归档留痕）"
                        if video_path else "")
                raise ValueError(
                    f"生成方式为「{mode}」但没有参考视频地址{hint}：请填 --video-url，"
                    "或改用 --mode text / --mode image"
                )
        elif has_video_input:
            raise ValueError(
                f"生成方式为「{mode}」，不会提交参考视频：请清除视频输入，或改用 --mode video / --mode both"
            )

    requested = args.requests if args.requests is not None else int(_env("SEEDANCE_REQUESTS") or 1)
    if not 1 <= requested <= SEEDANCE_MAX_REQUESTS:
        raise ValueError(f"视频任务数必须是 1～{SEEDANCE_MAX_REQUESTS} 的整数")
    concurrency = args.concurrency if args.concurrency is not None else int(_env("SEEDANCE_CONCURRENCY") or 1)
    if not 1 <= concurrency <= SEEDANCE_MAX_CONCURRENCY:
        raise ValueError(f"并发数必须是 1～{SEEDANCE_MAX_CONCURRENCY} 的整数")

    generate_audio = (
        args.generate_audio if args.generate_audio is not None
        else _env_bool("SEEDANCE_GENERATE_AUDIO", SEEDANCE_DEFAULT_GENERATE_AUDIO)
    )
    watermark = (
        args.watermark if args.watermark is not None
        else _env_bool("SEEDANCE_WATERMARK", SEEDANCE_DEFAULT_WATERMARK)
    )
    poll_interval = args.poll_interval if args.poll_interval is not None else _env_float(
        "SEEDANCE_POLL_INTERVAL_SECONDS", DEFAULT_POLL_INTERVAL_SECONDS
    )
    if poll_interval <= 0:
        raise ValueError("轮询间隔必须大于 0 秒")
    task_timeout = args.task_timeout if args.task_timeout is not None else _env_float(
        "SEEDANCE_TASK_TIMEOUT_SECONDS", DEFAULT_TASK_TIMEOUT_SECONDS
    )
    if task_timeout <= 0:
        raise ValueError("单任务观察窗口必须大于 0 秒")
    request_timeout = args.request_timeout if args.request_timeout is not None else _env_float(
        "SEEDANCE_REQUEST_TIMEOUT_SECONDS", DEFAULT_REQUEST_TIMEOUT_SECONDS
    )
    if request_timeout <= 0:
        raise ValueError("单请求超时必须大于 0 秒")

    keep_videos = bool(args.keep_videos)
    raw_gib = args.max_video_gib
    if raw_gib is None:
        raw_gib = os.getenv(MAX_VIDEO_GIB_ENV) or ""
    max_video_gib = DEFAULT_MAX_VIDEO_GIB if raw_gib == "" else float(raw_gib)
    if max_video_gib < 0:
        raise ValueError(f"{MAX_VIDEO_GIB_ENV}/--max-video-gib 不能为负数")
    return VideoConfig(
        api_key=api_key,
        base_url=base_url,
        model=model,
        prompt=prompt,
        resolution=resolution,
        ratio=ratio,
        duration=int(duration),
        generate_audio=bool(generate_audio),
        watermark=bool(watermark),
        mode=mode,
        image_url=image_url or "",
        image_path=image_path or "",
        image_role=image_role,
        video_url=video_url or "",
        video_path=video_path or "",
        video_role=video_role,
        requests=requested,
        concurrency=concurrency,
        request_timeout_seconds=request_timeout,
        poll_interval_seconds=poll_interval,
        task_timeout_seconds=task_timeout,
        keep_videos=keep_videos,
        max_video_bytes=int(max_video_gib * 1024**3),
        trust_env=_env_bool("VOLC_TRUST_ENV", False),
        record_id=_env("VOLC_RECORD_ID"),
    )


async def main_async(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    try:
        config = build_config(args)
    except ValueError as exc:
        print(f"配置错误: {exc}", file=sys.stderr)
        return 2

    def progress(done: int, total: int, passed: int) -> None:
        print(f"\r{config.record_id or 'seedance'}: 完成 {done}/{total} "
              f"(成功 {passed}, 待完成 {total - done})", end="", flush=True)

    started = time.time()
    outcome = await run_videos(config, args.output_root, progress=progress)
    print()
    summary = finalize_result_dir(outcome.result_dir)
    elapsed = time.time() - started
    print(json.dumps(
        {
            "case": summary.get("case"),
            "record_id": summary.get("record_id"),
            "mode": summary.get("mode"),
            "requested": summary.get("requested"),
            "successful": summary.get("successful"),
            "failed": summary.get("failed"),
            "blocked": summary.get("blocked"),
            "admission_passed": summary.get("admission_passed"),
            "videos_saved": summary.get("videos_saved"),
            "video_links": [item.get("url") for item in (summary.get("video_links") or [])],
            "elapsed_seconds": round(elapsed, 3),
        },
        ensure_ascii=False,
        indent=2,
    ))
    print(f"结果目录: {outcome.result_dir.resolve()}")
    for row in outcome.rows:
        if row.get("video_url"):
            print(f"任务 {row['index']} 视频链接: {row['video_url']}")
    return 0 if summary.get("admission_passed") else 2


def cli() -> None:
    try:
        raise SystemExit(asyncio.run(main_async()))
    except KeyboardInterrupt:
        raise SystemExit(130)
    except ValueError as exc:
        print(f"配置错误: {exc}", file=sys.stderr)
        raise SystemExit(2)


__all__ = ["build_config", "cli", "main_async"]
