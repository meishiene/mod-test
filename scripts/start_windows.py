#!/usr/bin/env python3
"""Windows 一键启动引导（由项目根目录的 start.bat 调用）。

流程与 start.sh 对齐：
  1. 校验 Python 版本
  2. 创建/修复 .venv
  3. 按 requirements.txt 安装并校验依赖（带缓存标记）
  4. 加载 .env
  5. 释放被占用的端口
  6. 启动 scripts/start_web.py

除 Windows 外也能在 macOS/Linux 上运行，便于本地验证。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
IS_WINDOWS = os.name == "nt"
MIN_PYTHON = (3, 12)
VENV_PYTHON_CHECK = (
    "import sys\n"
    "ok = sys.version_info >= (3, 12)"
    + (" and sys.maxsize > 2**32" if IS_WINDOWS else "")
    + "\nsys.exit(0 if ok else 1)\n"
)
DEPS_MARKER = ".mltp-deps.json"
IMPORT_MODULES = ("httpx", "openai", "jsonschema", "pytest", "xdist", "PIL", "c2pa", "datasets")
IMPORT_CHECK = (
    "import importlib, sys\n"
    f"modules = {IMPORT_MODULES!r}\n"
    "failed = []\n"
    "for name in modules:\n"
    "    try:\n"
    "        importlib.import_module(name)\n"
    "    except Exception as exc:\n"
    "        print(f'{name}: {exc}', file=sys.stderr)\n"
    "        failed.append(name)\n"
    "sys.exit(1 if failed else 0)\n"
)
MIRROR_INDEX_URL = "https://pypi.tuna.tsinghua.edu.cn/simple"


def _emit(message: str, *, error: bool = False) -> None:
    print(message, file=sys.stderr if error else sys.stdout, flush=True)


def info(message: str) -> None:
    _emit(f"[启动] {message}")


def warn(message: str) -> None:
    _emit(f"[警告] {message}")


def fail(message: str, code: int = 1) -> None:
    _emit(f"[错误] {message}", error=True)
    raise SystemExit(code)


def _command_text(command: list[object]) -> str:
    return " ".join(f'"{part}"' if " " in str(part) else str(part) for part in command)


def probe(command: list[object], *, timeout: float = 60, cwd: Path = ROOT) -> int:
    """静默执行命令并返回退出码；命令不存在或超时都视为失败。"""
    try:
        result = subprocess.run(
            [str(part) for part in command],
            cwd=cwd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError):
        return 1
    return result.returncode


def capture(command: list[object], *, timeout: float = 60, cwd: Path = ROOT) -> tuple[int, str]:
    try:
        result = subprocess.run(
            [str(part) for part in command],
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return 1, str(exc)
    return result.returncode, result.stdout or ""


def run_visible(command: list[object], *, cwd: Path = ROOT) -> int:
    info(f"执行：{_command_text(command)}")
    try:
        return subprocess.call([str(part) for part in command], cwd=cwd)
    except OSError as exc:
        fail(f"无法执行 {command[0]}：{exc}")
        return 1


def resolve_venv() -> Path:
    configured = os.environ.get("VENV_DIR", "").strip()
    venv = Path(configured).expanduser() if configured else ROOT / ".venv"
    if not venv.is_absolute():
        venv = (ROOT / venv).resolve()
    return venv


def venv_python(venv: Path) -> Path:
    return venv / ("Scripts/python.exe" if IS_WINDOWS else "bin/python")


def runtime_version(python: Path) -> str:
    code, output = capture([python, "-c", "import platform;print(platform.python_version())"])
    return output.strip() if code == 0 else ""


def ensure_venv() -> tuple[Path, Path]:
    venv = resolve_venv()
    python = venv_python(venv)
    if python.is_file() and probe([python, "-c", VENV_PYTHON_CHECK], timeout=30) == 0:
        info(f"使用已有虚拟环境：{venv}")
        return venv, python

    if venv.exists():
        warn(f"虚拟环境不可用或版本低于 3.12，正在重建：{venv}")
    else:
        info(f"创建虚拟环境：{venv}")
    venv.parent.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "-m", "venv"]
    if venv.exists():
        command.append("--clear")
    command.append(str(venv))
    if run_visible(command) != 0 or not python.is_file():
        fail(
            "虚拟环境创建失败。请确认安装的是完整的 Python 3.12+（包含 venv/pip 组件），"
            "并以普通用户权限重试。"
        )
    return venv, python


def ensure_pip(python: Path) -> None:
    if probe([python, "-m", "pip", "--version"]) != 0:
        info("正在初始化 pip…")
        if run_visible([python, "-m", "ensurepip", "--upgrade"]) != 0:
            fail("pip 初始化失败。请重新安装 Python 3.12+，安装时勾选 pip 组件。")


def requirements_digest(requirements: Path) -> str:
    return hashlib.sha256(requirements.read_bytes()).hexdigest()


def deps_marker(venv: Path) -> Path:
    return venv / DEPS_MARKER


def deps_ready(venv: Path, python: Path, digest: str, runtime: str) -> bool:
    marker = deps_marker(venv)
    if not marker.is_file():
        return False
    try:
        state = json.loads(marker.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    if state.get("requirements_sha256") != digest:
        return False
    if state.get("runtime") != runtime:
        return False
    if state.get("platform") != sys.platform:
        return False
    return probe([python, "-c", IMPORT_CHECK], timeout=120) == 0


def install_deps(venv: Path, python: Path, requirements: Path, digest: str, runtime: str) -> None:
    if deps_ready(venv, python, digest, runtime):
        info("运行依赖已就绪，跳过安装。")
        return

    info("正在检查并安装依赖（首次运行需要下载较多内容，请耐心等待）…")
    base = [
        python, "-m", "pip", "install",
        "--disable-pip-version-check", "--no-input", "--prefer-binary",
        "--timeout", "120", "--retries", "3",
    ]
    code = run_visible([*base, "-r", requirements])
    mirror_disabled = os.environ.get("MLTP_NO_MIRROR", "").lower() in {"1", "true", "yes"}
    if code != 0 and not os.environ.get("PIP_INDEX_URL") and not mirror_disabled:
        warn("官方 PyPI 源安装失败，改用清华镜像重试…")
        code = run_visible([*base, "-i", MIRROR_INDEX_URL, "-r", requirements])
    if code != 0:
        fail(
            "依赖安装失败。请检查网络或代理设置；也可以先设置 PIP_INDEX_URL "
            "指向可用镜像后重新运行 start.bat。"
        )
    if probe([python, "-c", IMPORT_CHECK], timeout=120) != 0:
        fail("依赖安装完成，但关键模块仍无法导入，请查看上方 pip 输出。")

    marker = deps_marker(venv)
    marker.write_text(
        json.dumps(
            {
                "requirements_sha256": digest,
                "runtime": runtime,
                "platform": sys.platform,
                "installed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            },
            ensure_ascii=False,
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    info("依赖安装完成。")


def parse_env_value(raw: str) -> str:
    value = raw.strip()
    if not value:
        return ""
    if value[0] in {'"', "'"}:
        quote = value[0]
        chars: list[str] = []
        index = 1
        while index < len(value):
            char = value[index]
            if char == "\\" and index + 1 < len(value):
                escaped = value[index + 1]
                replacements = {"n": "\n", "r": "\r", "t": "\t", "\\": "\\", '"': '"', "'": "'"}
                replacement = replacements.get(escaped)
                # 未知转义（例如 Windows 路径里的 \U）保留反斜杠。
                chars.append("\\" + escaped if replacement is None else replacement)
                index += 2
                continue
            if char == quote:
                return "".join(chars)
            chars.append(char)
            index += 1
        return "".join(chars)
    # 未加引号时，只有空白后的 # 才是行内注释，避免误伤 URL 片段。
    comment = re.search(r"\s+#", value)
    if comment:
        value = value[: comment.start()]
    return value.strip()


def load_dotenv(path: Path) -> None:
    if not path.is_file():
        info("未找到 .env；启动后可在页面中填写接口配置。")
        return
    loaded = 0
    for lineno, raw in enumerate(path.read_text(encoding="utf-8-sig", errors="replace").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export "):].lstrip()
        key, separator, value = line.partition("=")
        key = key.strip()
        if not separator or not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            warn(f".env 第 {lineno} 行格式无法识别，已跳过。")
            continue
        os.environ[key] = parse_env_value(value)
        loaded += 1
    info(f"已加载配置：.env（{loaded} 项）")


def _netstat_owner_pids(port: int) -> set[int]:
    code, output = capture(["netstat", "-ano", "-p", "tcp"], timeout=20)
    if code != 0:
        return set()
    owners: set[int] = set()
    for line in output.splitlines():
        parts = line.split()
        if len(parts) < 5 or parts[0].upper() != "TCP":
            continue
        local, foreign, state, pid_text = parts[1], parts[2], parts[3].upper(), parts[4]
        if local.rsplit(":", 1)[-1] != str(port):
            continue
        # 中文系统上状态列可能被本地化，用对端地址兜底识别监听行。
        if not (state.startswith("LISTEN") or foreign in {"0.0.0.0:0", "[::]:0", "*:*"}):
            continue
        try:
            owners.add(int(pid_text))
        except ValueError:
            continue
    return owners


def port_owner_pids(port: int) -> set[int]:
    if IS_WINDOWS:
        return _netstat_owner_pids(port)
    if shutil.which("lsof"):
        code, output = capture(["lsof", f"-tiTCP:{port}", "-sTCP:LISTEN"], timeout=20)
        if code == 0:
            return {int(item) for item in output.split() if item.isdigit()}
    return set()


def port_connectable(host: str, port: int) -> bool:
    if host in {"0.0.0.0", ""}:
        target = "127.0.0.1"
    elif host == "::":
        target = "::1"
    else:
        target = host
    try:
        with socket.create_connection((target, port), timeout=0.5):
            return True
    except OSError:
        return False


def kill_owners(pids: set[int]) -> None:
    for pid in sorted(pids):
        if pid == os.getpid():
            continue
        if IS_WINDOWS:
            subprocess.run(
                ["taskkill", "/F", "/PID", str(pid)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            continue
        try:
            os.kill(pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass


def ensure_port_free(host: str, port: int) -> None:
    owners = port_owner_pids(port)
    if not owners and not port_connectable(host, port):
        return
    if owners:
        info(f"端口 {port} 被占用（PID: {', '.join(str(pid) for pid in sorted(owners))}），正在终止…")
        kill_owners(owners)
    else:
        warn(f"端口 {port} 已被占用，但未能定位占用进程。")
    for _ in range(12):
        time.sleep(0.5)
        owners = port_owner_pids(port)
        if not owners and not port_connectable(host, port):
            info(f"端口 {port} 已释放。")
            return
    fail(f"端口 {port} 仍被占用，请手动停止占用进程，或使用 --port 指定其他端口。")


def launch(python: Path, host: str, port: int) -> int:
    script = ROOT / "scripts" / "start_web.py"
    if not script.is_file():
        fail(f"未找到启动入口：{script}")
    environment = os.environ.copy()
    existing_pythonpath = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = str(ROOT) + (os.pathsep + existing_pythonpath if existing_pythonpath else "")
    environment["PYTHONUTF8"] = "1"
    environment["PYTHONIOENCODING"] = "utf-8"
    environment["PYTHONUNBUFFERED"] = "1"
    vendor_python = ROOT / "vendor_verifier" / ".venv" / (
        "Scripts/python.exe" if IS_WINDOWS else "bin/python"
    )
    if vendor_python.is_file() and not environment.get("KVV_PYTHON"):
        environment["KVV_PYTHON"] = str(vendor_python)

    display_host = "127.0.0.1" if host in {"0.0.0.0", "::", ""} else host
    info(
        f"测试平台启动中：http://{display_host}:{port}  "
        "（本机访问请用 127.0.0.1 或 localhost，勿用 0.0.0.0）"
    )
    info("按 Ctrl+C 停止服务。")

    process = subprocess.Popen(
        [str(python), str(script), "--host", host, "--port", str(port)],
        cwd=ROOT,
        env=environment,
    )
    try:
        return process.wait()
    except KeyboardInterrupt:
        info("正在停止服务…")
        try:
            return process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            process.terminate()
            try:
                return process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                return process.wait()


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ML 测试平台 Windows 一键启动引导（通常由 start.bat 调用）。",
    )
    parser.add_argument("--host", default="0.0.0.0", help="监听地址，默认 0.0.0.0")
    parser.add_argument("--port", type=int, default=8085, help="监听端口，默认 8085")
    parser.add_argument("--check", action="store_true", help="只检查虚拟环境与依赖，不安装、不启动")
    parser.add_argument("--bootstrap-only", action="store_true", help="准备好虚拟环境和依赖后退出，不启动服务")
    args, unknown = parser.parse_known_args(argv)
    if unknown:
        parser.error(f"无法识别的参数：{' '.join(unknown)}")
    if not 1 <= args.port <= 65535:
        parser.error("--port 必须在 1-65535 之间")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if sys.version_info < MIN_PYTHON:
        fail(f"需要 Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]}+，当前为 {platform.python_version()}。")
    if IS_WINDOWS and sys.maxsize <= 2**32:
        fail("需要 64 位 Python：部分依赖（如 c2pa-python、pyarrow）没有 32 位 Windows wheel。")
    info(f"引导程序 Python：{platform.python_version()}（{sys.executable}）")

    requirements = ROOT / "requirements.txt"
    if not requirements.is_file():
        fail(f"未找到依赖清单：{requirements}")
    digest = requirements_digest(requirements)

    if args.check:
        venv = resolve_venv()
        python = venv_python(venv)
        if not python.is_file():
            fail(f"未找到虚拟环境：{venv}。请运行 start.bat 完成首次初始化。")
        if probe([python, "-c", VENV_PYTHON_CHECK], timeout=30) != 0:
            fail(f"虚拟环境的 Python 版本低于 3.12 或不可用：{venv}。请运行 start.bat 重建。")
        runtime = runtime_version(python)
        if deps_ready(venv, python, digest, runtime):
            info(f"环境检查通过：{venv}（Python {runtime}）")
            return 0
        fail("依赖未安装，或 requirements.txt 已变化；请运行 start.bat 重新安装。")

    venv, python = ensure_venv()
    ensure_pip(python)
    runtime = runtime_version(python)
    if not runtime:
        fail(f"无法运行虚拟环境中的 Python：{python}")
    install_deps(venv, python, requirements, digest, runtime)

    if args.bootstrap_only:
        info("环境准备完成；运行 start.bat 即可启动服务。")
        return 0

    load_dotenv(ROOT / ".env")
    ensure_port_free(args.host, args.port)
    return launch(python, args.host, args.port)


if __name__ == "__main__":
    raise SystemExit(main())
