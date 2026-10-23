"""从 HuggingFace 拉官方发布的 tokenizer / config，并**钉死版本**。

为什么要自己写而不直接用 ``huggingface_hub``：
1. 本机出口走代理，默认 SSL 上下文拿不到根证书（``CERTIFICATE_VERIFY_FAILED``），
   必须显式用 ``certifi``；``huggingface_hub`` 的证书/代理行为不受我们控制；
2. 金标必须可复现：每条记录要带**仓库 commit revision**，而不是 ``main``。
   ``main`` 明天可能指向另一份权重，那样金标会静默漂移——这是本库最不能接受的事；
3. 不装 ``huggingface_hub`` / ``transformers`` 能少一层依赖，回归基线不受影响。

缓存策略：``gold/.cache/<repo>@<rev>/<file>``，文件本身**不入库**（几十 MB），
但 ``lineage.json`` 里记 ``sha256 + revision + 字节数``，任何人可原样重建。
"""
from __future__ import annotations

import hashlib
import json
import os
import ssl
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path

try:  # 沙箱/CI 里 certifi 一定在（requests/httpx 的依赖）
    import certifi
    _CAFILE = certifi.where()
except Exception:  # pragma: no cover - 兜底走系统证书
    _CAFILE = None

CACHE_DIR = Path(__file__).resolve().parent / ".cache"
DEFAULT_ENDPOINT = os.environ.get("GOLD_HF_ENDPOINT", "https://huggingface.co").rstrip("/")
TIMEOUT = 60

_ctx = ssl.create_default_context(cafile=_CAFILE) if _CAFILE else ssl.create_default_context()


class FetchError(RuntimeError):
    """官方文件拉不下来。调用方必须记 ``uncollected``，**不许**用近似值兜底。"""


@dataclass(frozen=True)
class FetchedFile:
    repo: str
    revision: str
    filename: str
    path: Path
    sha256: str
    size: int

    @property
    def source_url(self) -> str:
        return f"{DEFAULT_ENDPOINT}/{self.repo}/blob/{self.revision}/{self.filename}"


def _get(url: str) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "ml-test-platform-gold/1.0"})
    with urllib.request.urlopen(req, timeout=TIMEOUT, context=_ctx) as resp:
        return resp.read()


def repo_revision(repo: str) -> str:
    """取仓库当前 commit（``sha`` 字段），作为金标的版本锚点。"""
    raw = _get(f"{DEFAULT_ENDPOINT}/api/models/{repo}")
    data = json.loads(raw)
    sha = data.get("sha")
    if not sha:
        raise FetchError(f"{repo} 的 API 未返回 commit sha，无法钉版本")
    return str(sha)


def fetch(repo: str, filename: str, revision: str | None = None) -> FetchedFile:
    """下载 ``repo/filename``（默认取当前 commit），带本地缓存与 sha256。"""
    rev = revision or repo_revision(repo)
    target_dir = CACHE_DIR / f"{repo.replace('/', '__')}@{rev}"
    path = target_dir / filename
    if path.exists() and path.stat().st_size > 0:
        data = path.read_bytes()
    else:
        url = f"{DEFAULT_ENDPOINT}/{repo}/resolve/{rev}/{filename}"
        try:
            data = _get(url)
        except urllib.error.HTTPError as exc:
            raise FetchError(f"HTTP {exc.code} 拉取 {repo}/{filename}@{rev[:8]} 失败") from exc
        except Exception as exc:  # 网络/证书/超时
            raise FetchError(f"拉取 {repo}/{filename} 失败：{type(exc).__name__} {exc}") from exc
        target_dir.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    return FetchedFile(
        repo=repo,
        revision=rev,
        filename=filename,
        path=path,
        sha256=hashlib.sha256(data).hexdigest(),
        size=len(data),
    )


def sha256_of(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
