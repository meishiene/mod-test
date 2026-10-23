#!/usr/bin/env bash
set -Eeuo pipefail

# 一键启动测试平台：bash start.sh
# 可选：bash start.sh --host 0.0.0.0 --port 8085

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3}"
VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
    echo "错误：未找到 Python 3，请安装 Python 3.12+，或设置 PYTHON_BIN。" >&2
    exit 1
fi

if [[ ! -x "$VENV_DIR/bin/python" ]]; then
    echo "正在创建虚拟环境：$VENV_DIR"
    "$PYTHON_BIN" -m venv "$VENV_DIR"
fi

PYTHON="$VENV_DIR/bin/python"

# 检查项目运行时和内置 KVV 官方测试所需的依赖，缺失时自动安装。
if ! "$PYTHON" -c 'import httpx, openai, jsonschema, pytest, xdist, PIL' >/dev/null 2>&1; then
    echo "正在安装/补齐项目依赖……"
    "$PYTHON" -m pip install -r "$ROOT_DIR/requirements.txt"
fi

if [[ -f "$ROOT_DIR/.env" ]]; then
    echo "正在加载配置：.env"
    set -a
    # shellcheck disable=SC1091
    source "$ROOT_DIR/.env"
    set +a
fi

export PYTHONPATH="$ROOT_DIR${PYTHONPATH:+:$PYTHONPATH}"

HOST="0.0.0.0"
PORT="8085"
SERVER_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --host=*) HOST="${1#*=}"; shift ;;
        --host) [[ $# -ge 2 ]] || { echo "错误：--host 缺少参数。" >&2; exit 1; }; HOST="$2"; shift 2 ;;
        --port=*) PORT="${1#*=}"; shift ;;
        --port) [[ $# -ge 2 ]] || { echo "错误：--port 缺少参数。" >&2; exit 1; }; PORT="$2"; shift 2 ;;
        *) SERVER_ARGS+=("$1"); shift ;;
    esac
done

# 端口占用检测与自动释放：若已被占用，定位持有进程并终止，避免反复手动清理。
release_port() {
  local host="$1" port="$2"
  local pids=""
  if command -v lsof >/dev/null 2>&1; then
    pids="$(lsof -tiTCP:"$port" -sTCP:LISTEN 2>/dev/null)"
  elif command -v fuser >/dev/null 2>&1; then
    pids="$(fuser "${port}/tcp" 2>/dev/null)"
  fi
  if [[ -n "${pids:-}" ]]; then
    echo "端口 ${host}:${port} 被进程占用 (PID: ${pids})，正在终止……"
    # shellcheck disable=SC2086
    kill -9 $pids 2>/dev/null || true
    sleep 1
  fi
}

# 纯 bind 探测（加 SO_REUSEADDR，TIME_WAIT 不会误报）；命中再尝试自动释放。
if "$PYTHON" - "$HOST" "$PORT" <<'PY'
import socket, sys
host, port = sys.argv[1], int(sys.argv[2])
s = socket.socket(); s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
try:
    s.bind((host, port))
except OSError:
    raise SystemExit(0)
else:
    s.close(); raise SystemExit(1)
PY
then
  release_port "$HOST" "$PORT"
  # 释放后再次确认端口可绑定
  if "$PYTHON" - "$HOST" "$PORT" <<'PY'
import socket, sys
host, port = sys.argv[1], int(sys.argv[2])
s = socket.socket(); s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
try:
    s.bind((host, port))
except OSError:
    raise SystemExit(0)
else:
    s.close(); raise SystemExit(1)
PY
  then
    echo "错误：${HOST}:${PORT} 仍被占用，请手动停止相关进程或更换端口。" >&2
    exit 1
  fi
fi

# 展示用的访问地址：绑定 0.0.0.0（所有网卡）时，浏览器应通过 127.0.0.1 访问。
# 0.0.0.0 不是可路由地址，会掉进系统代理（不在 Chrome 默认绕行名单里），
# 代理无法回连 0.0.0.0 就会报 HTTP 502 —— 这不是服务挂了。
DISPLAY_HOST="$HOST"
if [[ "$HOST" == "0.0.0.0" || "$HOST" == "::" || "$HOST" == "" ]]; then
    DISPLAY_HOST="127.0.0.1"
fi

echo "测试平台启动中：http://${DISPLAY_HOST}:${PORT}  （本机访问请用 127.0.0.1 或 localhost，勿用 0.0.0.0）"
# 在 set -u（nounset）下，直接展开空数组可能触发 unbound variable；
# 无额外参数时单独调用，避免 start.sh 在真正启动 Web 服务前退出。
if ((${#SERVER_ARGS[@]})); then
    exec "$PYTHON" "$ROOT_DIR/scripts/start_web.py" --host "$HOST" --port "$PORT" "${SERVER_ARGS[@]}"
else
    exec "$PYTHON" "$ROOT_DIR/scripts/start_web.py" --host "$HOST" --port "$PORT"
fi
