#!/usr/bin/env bash
# ============================================================
# FlashHead 推理服务 —— 本地构建并推送镜像到阿里云 ACR
# 用法（在 flash_head_server/ 目录执行）:
#   ./build_push.sh                    # 使用默认标签 latest
#   IMAGE_TAG=20261022 ./build_push.sh # 指定标签（同时额外推送 latest）
#   ACR_PASSWORD=xxx ./build_push.sh   # 从环境变量读密码
# 密码来源优先级: 环境变量 ACR_PASSWORD > 同目录 .acr.env 文件
#   .acr.env 格式: ACR_PASSWORD=xxx   （该文件已被 .gitignore，不入库）
# 依赖: docker（本地需有 GPU 版 Docker 或普通 Docker 均可，构建无需 GPU）
# 说明: 构建上下文为仓库根目录（../），Dockerfile 为本目录下 Dockerfile
# ============================================================
set -euo pipefail

# ===================== 配置区（按项目修改） =====================
ACR_REGISTRY="crpi-3mnw2c0qqumul4h9.cn-heyuan.personal.cr.aliyuncs.com"
ACR_NAMESPACE="huhan_9527"
ACR_USERNAME="胡17603097990"

# 镜像标签：可用环境变量 IMAGE_TAG 覆盖，默认 latest
IMAGE_TAG="${IMAGE_TAG:-latest}"

# 镜像清单，每行格式: "镜像名|构建上下文目录|可选的Dockerfile相对路径"
# - 本项目：单服务，Dockerfile 在本目录（flash_head_server/Dockerfile），
#   构建上下文为仓库根目录（Dockerfile 需 COPY flash_head/ 与 requirements.txt）
IMAGES=(
  "flashhead-server|..|flash_head_server/Dockerfile"
)
# ===============================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ---- 凭据加载：环境变量 > .acr.env（gitignore，不入库） ----
if [ -z "${ACR_PASSWORD:-}" ] && [ -f "$SCRIPT_DIR/.acr.env" ]; then
  # shellcheck disable=SC1091
  source "$SCRIPT_DIR/.acr.env"
fi
if [ -z "${ACR_PASSWORD:-}" ]; then
  echo "错误: 未提供 ACR_PASSWORD。请设置环境变量或在 $SCRIPT_DIR/.acr.env 写入 ACR_PASSWORD=xxx" >&2
  exit 1
fi

if [ ${#IMAGES[@]} -eq 0 ]; then
  echo "错误: IMAGES 配置为空，请先在脚本顶部配置区添加镜像清单。" >&2
  exit 1
fi

# 1. 登录阿里云 ACR（密码通过 stdin 传入，避免出现在进程列表中）
echo "==> 登录阿里云容器镜像服务..."
echo "$ACR_PASSWORD" | docker login --username="$ACR_USERNAME" "$ACR_REGISTRY" --password-stdin

# 2. 构建并推送镜像
# --provenance=false --sbom=false：禁用 OCI attestation 清单（含 vnd.oci.empty.v1+json），
# 阿里云个人版 ACR 不支持该 manifest class，会导致 push 报 "unknown manifest class"
PUSHED=()
for entry in "${IMAGES[@]}"; do
  IFS='|' read -r NAME CONTEXT DOCKERFILE <<< "$entry"
  FULL="${ACR_REGISTRY}/${ACR_NAMESPACE}/${NAME}:${IMAGE_TAG}"
  echo "==> 构建镜像 ${NAME} (context: ${CONTEXT}, dockerfile: ${DOCKERFILE})..."
  docker build --provenance=false --sbom=false -t "$FULL" -f "$SCRIPT_DIR/$DOCKERFILE" "$SCRIPT_DIR/$CONTEXT"
  echo "==> 推送镜像 ${NAME}（首次约 10GB+，耐心等待）..."
  docker push "$FULL"
  PUSHED+=("$FULL")
  # 若标签不是 latest，额外推送一份 latest 便于回滚/默认拉取
  if [ "$IMAGE_TAG" != "latest" ]; then
    LATEST="${ACR_REGISTRY}/${ACR_NAMESPACE}/${NAME}:latest"
    docker tag "$FULL" "$LATEST"
    docker push "$LATEST"
  fi
done

echo "==> 完成！镜像已推送到阿里云仓库（tag=${IMAGE_TAG}）："
for img in "${PUSHED[@]}"; do
  echo "    $img"
done
