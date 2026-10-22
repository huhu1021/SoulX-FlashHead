# FlashHead 推理服务镜像
# 构建上下文必须是项目根目录：docker build -f flash_head_server/Dockerfile .
# FROM 走 daocloud 加速：本机直连 registry-1.docker.io 遭 DNS 污染（解析到非 DockerHub IP，连接被拒）
FROM docker.m.daocloud.io/pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg libgl1 libglib2.0-0 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 先装依赖（利用层缓存）；pip 走阿里云源：容器内直连 PyPI 仅 ~50KB/s
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
        -i https://mirrors.aliyun.com/pypi/simple/ \
    && pip install --no-cache-dir huggingface_hub \
        -i https://mirrors.aliyun.com/pypi/simple/

# 安装 flash-attn（预编译 wheel 优先；构建机若网络受限可改为挂载 whl）
RUN pip install --no-cache-dir flash_attn==2.8.0.post2 --no-build-isolation \
        -i https://mirrors.aliyun.com/pypi/simple/ \
    || echo "[WARN] flash-attn 安装失败，服务将以 SDPA 兼容模式运行（速度略降）"

# 拷贝代码
COPY flash_head/ ./flash_head/
COPY flash_head_server/server.py ./server.py
COPY examples/ ./examples/

# 模型权重通过 volume 挂载（避免打进镜像）：
#   ./models/SoulX-FlashHead-1_3B  -> /app/models/SoulX-FlashHead-1_3B
#   ./models/wav2vec2-base-960h    -> /app/models/wav2vec2-base-960h
ENV CKPT_DIR=/app/models/SoulX-FlashHead-1_3B \
    WAV2VEC_DIR=/app/models/wav2vec2-base-960h \
    MODEL_TYPE=lite \
    AVATAR_DIR=/data/avatars \
    MEDIA_DIR=/data/media \
    PORT=8000

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=180s --retries=5 \
    CMD curl -sf http://localhost:8000/health | grep -q '"ok"' || exit 1

CMD ["python", "server.py"]
