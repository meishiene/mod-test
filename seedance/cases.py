"""Seedance（视频生成）套件的用例元数据与参数取值域。

协议面是火山方舟线格式的**异步任务**接口（中转网关 ``token-hub.tangrankeji.com``
也按同一形状代理）：

- 提交：``POST {root}/contents/generations/tasks``，返回 ``{"id": "<task id>"}``；
- 查询：``GET  {root}/contents/generations/tasks/{task_id}``，``status`` 取
  ``queued`` / ``running`` / ``succeeded`` / ``failed`` / ``expired``，成功后
  ``content.video_url`` 给出视频下载链接。

参数取值域按官方文档声明（见 ``docs/Seedance视频生成测试.md``）。与图像套件不同，
这里的参数是**界面显式输入**（提示词 / 分辨率 / 画面比例 / 时长 / 图片位置 /
原生音频 / 水印），而不是逐请求随机抽取的矩阵：视频单条成本高、生成慢，
"同一组参数并发 N 次"才是可解释的压测口径。
"""
from __future__ import annotations

import re
from dataclasses import dataclass

SEEDANCE_CASE_ID = "SEEDANCE_VIDEO_GEN"

# 生成方式（界面显式选择）：先声明"这次请求带不带图片 / 参考视频"，再谈各自的位置。
# 异步任务面把参数校验放在执行阶段（见下方分辨率注释），所以"声明文生视频却带着图片提交"
# 这类错配不会在提交期报错、只会烧掉一次完整生成——必须在入队时按模式强校验。
SEEDANCE_MODES: tuple[tuple[str, str], ...] = (
    ("text", "仅文生视频（不带图片 / 参考视频）"),
    ("image", "带参考图（只带图片）"),
    ("video", "带参考视频（只带视频）"),
    ("both", "两者都带（图片 + 参考视频）"),
)
SEEDANCE_MODE_IDS: tuple[str, ...] = tuple(mode for mode, _ in SEEDANCE_MODES)
# 界面默认：纯文生视频（最省、最不容易踩参数坑，也是首跑失败后要复现的第一档）。
SEEDANCE_DEFAULT_MODE = "text"
# 内部语义「未显式声明生成方式」：按**实际给了哪些输入**推导（有图有视频=both、
# 只有图=image、只有视频=video、都没有=text）。CLI 不传 --mode / 直接构造 VideoConfig
# 时走这条，保证既有调用方式不被破坏，也避免"默认 text 悄悄丢掉图片"这种静默行为。
SEEDANCE_MODE_AUTO = ""

# 模型 ID：该面文档给出的可选值就是这两个（**不是** ark 侧的
# doubao-seedance-1-0-pro-250528 一类接入点别名）。模型 ID 写错与档位写错一样，
# 会以「任务级参数校验失败」收场——见下面分辨率注释里的首跑证据。
SEEDANCE_MODEL_IDS = ("Seedance-2.0", "Seedance-2.5")

# 分辨率（resolution）：2.0 / 2.5 的可选集都是 720p / 1080p / 4k（默认 720p）。
# **480p 只属于 1.0 系列，2.0/2.5 的可选集里没有它**：2026-10-13 首跑
# （SEEDANCE_VIDEO_GEN-20261013-134116-c6068e）就是拿 480p 打 2.0——
# 提交 200 拿到 task id，轮询 56 次 ≈4.6 分钟后任务转 failed，文案「请求参数校验失败」。
# 该网关**把参数校验放在异步执行阶段**（提交期不校验），所以档位写错不会立刻 400，
# 而是烧掉一次完整任务才报错。默认档取两档都支持的 720p。
SEEDANCE_RESOLUTIONS = ("720p", "1080p", "4k")
SEEDANCE_DEFAULT_RESOLUTION = "720p"

# 画面比例（ratio）：16:9 / 4:3 / 1:1 / 3:4 / 9:16 / 21:9 / adaptive，
# 文档写明默认值就是 adaptive（且未对文生视频设限），因此默认值跟官方一致。
SEEDANCE_RATIOS = ("16:9", "4:3", "1:1", "3:4", "9:16", "21:9", "adaptive")
SEEDANCE_DEFAULT_RATIO = "adaptive"

# 时长（duration，必需，整数秒）：2.0 为 4–15，2.5 为 4–30。
# 区间按**被测模型档位**解析（duration_range_for）：档位识别不出来时取并集 4–30，
# 不本地收紧——档位相关的非法值由上游拒绝，那也是可观测结论。
SEEDANCE_DURATION_MIN = 4
SEEDANCE_DURATION_MAX = 30
SEEDANCE_DEFAULT_DURATION = 5
# 模型档位 → 时长上限。token 已去点/下划线（`2.5`/`2_5`/`2-5` 都归一成 `2-5`）。
SEEDANCE_DURATION_CAPS: tuple[tuple[str, int], ...] = (("2-5", 30), ("2-0", 15))

# 上传图片的"位置"（content[].role）。文档只定义了**单图**的两种合法用法：
#   * 首帧：传 1 个 image_url，role 为 first_frame 或不填；
#   * 参考图：传 1 个 image_url，role 必填 reference_image。
# 「首尾帧」需要传 **2 个** image_url（first_frame + last_frame），而当前输入面只有
# 一张图——单图配 last_frame 属文档外用法，于是不提供该档（要做首尾帧得先扩成双图输入）。
# 注意：图片是否**整体进入请求体**由生成方式决定（只有 image / both 两种模式会带图），
# 角色只在带图的模式下才有意义。
SEEDANCE_IMAGE_ROLES: tuple[tuple[str, str], ...] = (
    ("first_frame", "首帧（first_frame）"),
    ("reference_image", "参考图（reference_image）"),
)
SEEDANCE_DEFAULT_IMAGE_ROLE = "first_frame"

# 上传视频的"位置"（content[].role，视频块）。
# 视频只能以 **URL** 形式提交（异步任务面不接受本地文件、也没有文件直传接口），
# 因此本地视频文件只做归档留痕（见 core.resolve_video）；生成方式选 video / both 时
# **必须**填视频地址，只上传本地文件不算满足（否则"带参考视频"的批次其实一条视频都没提交）。
# 角色留一个"不指定"：网关/模型对视频块 role 的支持程度未经实测，留出不带该字段的退路。
SEEDANCE_VIDEO_ROLES: tuple[tuple[str, str], ...] = (
    ("reference_video", "参考视频（reference_video）"),
    ("", "不指定角色（不带 role 字段）"),
)
SEEDANCE_DEFAULT_VIDEO_ROLE = "reference_video"

# 一次生成的开关（默认值跟官方）：generate_audio 官方默认 true、watermark 官方默认 false。
SEEDANCE_DEFAULT_GENERATE_AUDIO = True
SEEDANCE_DEFAULT_WATERMARK = False

# 任务数与并发上限。视频是重任务（单条 720p/5s 通常几十秒到数分钟），
# 上限刻意远低于文字压测，避免一次提交把网关与费用打爆；界面仍按显式输入执行。
SEEDANCE_MAX_REQUESTS = 100
SEEDANCE_MAX_CONCURRENCY = 16
SEEDANCE_DEFAULT_REQUESTS = 4
SEEDANCE_DEFAULT_CONCURRENCY = 2

# 提示词与上传素材的体量上限（Web 请求体上限 64 KiB，data URI 会膨胀约 4/3）。
SEEDANCE_MAX_PROMPT_CHARS = 2000
SEEDANCE_MAX_IMAGE_BYTES = 8 * 1024 * 1024
# 上传的**输入**视频（参考视频素材）：只归档留痕、不随任务提交，因此可以放宽到 64 MiB。
# 注意与"生成结果的保留预算"（core.DEFAULT_MAX_VIDEO_GIB / config.max_video_bytes）区分。
SEEDANCE_MAX_INPUT_VIDEO_BYTES = 64 * 1024 * 1024
# 允许上传的输入视频类型（与 Web 端 /api/uploads 的白名单同一份）。
SEEDANCE_INPUT_VIDEO_MIME_EXTENSIONS: dict[str, str] = {
    "video/mp4": "mp4",
    "video/webm": "webm",
    "video/quicktime": "mov",
    "video/x-matroska": "mkv",
    "video/x-m4v": "m4v",
}
SEEDANCE_INPUT_VIDEO_SUFFIXES: tuple[str, ...] = tuple(
    sorted({suffix for suffix in SEEDANCE_INPUT_VIDEO_MIME_EXTENSIONS.values()})
)

# 结果目录里的视频原文件名（`videos/video-0001.mp4`）：名字随任务序号变化，不在固定
# 白名单里，但 Web 的产物通道只放行这一条**受控正则**——白名单同时是取值域，避免把
# 结果目录变成任意文件下载通道。
SEEDANCE_VIDEO_PATTERN = re.compile(r"videos/video-\d{4}\.(?:mp4|webm|mov|mkv|m4v)")


def mode_allows_image(mode: str) -> bool:
    """该生成方式是否**允许**图片输入进入请求体。"""
    return mode in ("image", "both")


def mode_allows_video(mode: str) -> bool:
    """该生成方式是否**允许**参考视频进入请求体。"""
    return mode in ("video", "both")


def resolve_mode(mode: str | None, *, has_image: bool, has_video: bool) -> str:
    """把（可能为空的）生成方式收口成四档之一。

    显式给了四档之一就原样采信（哪怕与输入不符——不符由校验层报错，不在这里悄悄纠偏）；
    空值 = ``SEEDANCE_MODE_AUTO``，按实际输入推导。

    ``has_image`` / ``has_video`` 指"**给了**这类素材"（含只上传本地文件的情形）——
    它只用来推断"这批是关于什么的"，不含"能不能提交给上游"的判断：参考视频只认 URL
    这条约束属于**校验层**（mode=video/both 时必须有视频地址），不在这里混进档位推断，
    否则"只上传本地视频"会被推断成 text，连带把归档留痕也一起丢掉。
    """
    text = str(mode or "").strip().lower()
    if text in SEEDANCE_MODE_IDS:
        return text
    if has_image and has_video:
        return "both"
    if has_image:
        return "image"
    if has_video:
        return "video"
    return "text"


def duration_range_for(model: str) -> tuple[int, int]:
    """按被测模型档位解析时长区间：2.5 → 4–30，2.0 → 4–15。

    模型名里的档位写法不止一种（``Seedance-2.5`` / ``doubao-seedance-2-5-*``），
    因此先把点与下划线归一成短横再按 token 匹配。**识别不出档位时取并集 4–30**：
    平台口径是不做本地白名单收紧，档位相关的非法值交给上游拒绝（可观测结论），
    只有档位明确时才按该档收紧，免得白烧一次完整生成。
    """
    text = (model or "").strip().lower().replace(".", "-").replace("_", "-")
    for token, cap in SEEDANCE_DURATION_CAPS:
        if token in text:
            return SEEDANCE_DURATION_MIN, cap
    return SEEDANCE_DURATION_MIN, SEEDANCE_DURATION_MAX


@dataclass(frozen=True)
class SeedanceCase:
    case_id: str
    title: str
    description: str
    cost: str
    protocols: tuple[str, ...]


SEEDANCE_VIDEO_GEN = SeedanceCase(
    case_id=SEEDANCE_CASE_ID,
    title="Seedance 视频生成",
    description=(
        "先选生成方式（仅文生视频 / 带参考图 / 带参考视频 / 两者都带），再按界面输入提交"
        "视频生成任务（提示词 / 分辨率 / 画面比例 / 时长 / 图片及其位置 / 参考视频及其位置 / "
        "原生音频 / 水印），轮询任务状态直到终态，并把生成的视频链接与本地留存一并归档。"
        "生成方式决定哪些输入会真的进入请求体，入队时按模式强校验，避免「声明文生视频却带着图片提交」。"
    ),
    cost="极高",
    # protocols 是**家族**（不是协议 id）：只有 seedance 家族被放行，
    # 文字面（openai/volcengine/…）与图像面（openai_images/gemini_images）都不会被误启用。
    protocols=("seedance",),
)

CASES: dict[str, SeedanceCase] = {
    SEEDANCE_VIDEO_GEN.case_id: SEEDANCE_VIDEO_GEN,
}
