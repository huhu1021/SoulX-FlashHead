#!/usr/bin/env bash
# ============================================================
# FlashHead 推理服务 —— 服务器侧拉取 ACR 镜像并部署
# 用法（在部署目录执行，本脚本与 compose.prod.yaml、.env 同目录）:
#   ./deploy_server.sh                    # 使用默认标签 latest
#   IMAGE_TAG=20261022 ./deploy_server.sh # 指定标签
#   ACR_PASSWORD=xxx ./deploy_server.sh   # 从环境变量读密码
# 密码来源优先级: 环境变量 ACR_PASSWORD > 同目录 .acr.env 文件
# 前置条件:
#   1. 部署目录结构（Ubuntu，需 NVIDIA 驱动 + Docker + nvidia-container-toolkit）:
#      deploy/
#      ├── deploy_server.sh      本脚本
#      ├── compose.prod.yaml     生产编排（拉 ACR 镜像）
#      ├── .env                  生产环境变量（MODEL_TYPE/API_KEY 等）
#      ├── .acr.env              ACR 密码（可选，也可用环境变量）
#      ├── models/               模型权重（见 README 下载命令）
#      │   ├── SoulX-FlashHead-1_3B/
#      │   └── wav2vec2-base-960h/
#      └── data/                 形象库(avatars/) 与产出视频(media/)
#   2. GPU 侧要求: nvidia-smi 可用；5090/4090 均可跑 Lite 档
# ============================================================
set -euo pipefail

# ===================== 配置区（按项目修改） =====================
ACR_REGISTRY="crpi-3mnw2c0qqumul4h9.cn-heyuan.personal.cr.aliyuncs.com"
ACR_NAMESPACE="huhan_9527"
ACR_USERNAME="胡17603097990"

IMAGE_TAG="${IMAGE_TAG:-latest}"
COMPOSE_FILE="compose.prod.yaml"
ENV_FILE="${ENV_FILE:-.env}"
# ===============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ---- 凭据加载：环境变量 > .acr.env ----
if [ -z "${ACR_PASSWORD:-}" ] && [ -f "$SCRIPT_DIR/.acr.env" ]; then
  # shellcheck disable=SC1091
  source "$SCRIPT_DIR/.acr.env"
fi
if [ -z "${ACR_PASSWORD:-}" ]; then
  echo "错误: 未提供 ACR_PASSWORD。请设置环境变量或在 $SCRIPT_DIR/.acr.env 写入 ACR_PASSWORD=xxx" >&2
  exit 1
fi

if [ ! -f "$COMPOSE_FILE" ]; then
  echo "错误: 未在 $SCRIPT_DIR 下找到 $COMPOSE_FILE，请确认它与本脚本在同一目录。" >&2
  exit 1
fi
if [ ! -f "$ENV_FILE" ]; then
  echo "错误: 当前目录缺少生产环境配置 $ENV_FILE。请参照 .env.example 创建。" >&2
  exit 1
fi
if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "警告: 未检测到 nvidia-smi，容器可能无法使用 GPU（检查驱动与 nvidia-container-toolkit）。" >&2
fi

export IMAGE_TAG
export ENV_FILE

# 1. 登录（拉取个人版 ACR 私有镜像也需要登录）
echo "==> 登录阿里云容器镜像服务..."
echo "$ACR_PASSWORD" | docker login --username="$ACR_USERNAME" "$ACR_REGISTRY" --password-stdin

# 2. 拉取镜像
echo "==> 拉取镜像（tag=${IMAGE_TAG}）..."
docker compose --env-file "$ENV_FILE" -f "$COMPOSE_FILE" pull

# 3. 启动服务（后台运行，强制重建替换旧容器）
echo "==> 启动服务..."
docker compose --env-file "$ENV_FILE" -f "$COMPOSE_FILE" up -d --force-recreate

# 4. 展示状态与预热提示
echo "==> 服务状态："
docker compose --env-file "$ENV_FILE" -f "$COMPOSE_FILE" ps

echo ""
echo "==> 部署完成。注意：容器启动后会进行 torch.compile 预热（约 1~2 分钟），"
echo "    等待日志出现「预热完成，开始对外服务」后即可调用："
echo "    docker compose --env-file $ENV_FILE -f $COMPOSE_FILE logs -f flashhead"
echo "    健康检查: curl http://127.0.0.1:8000/health"
