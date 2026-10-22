# -*- coding: utf-8 -*-
"""
FlashHead 数字人推理服务（句子级 REST 版）
==========================================

将 SoulX-FlashHead 封装为独立 HTTP 服务，供智能客服后端远程调用。

架构：
    智能客服(TTS 出音频) --HTTP--> 本服务(单卡 GPU) --返回--> mp4 视频 URL

接口：
    GET    /health                    健康检查（含预热状态、可用形象列表）
    POST   /session                   创建会话 {avatar: "<名称>"} 或上传自定义形象
    POST   /session/<sid>/generate    句子级生成：上传音频文件，返回 mp4 URL
    DELETE /session/<sid>             释放会话
    GET    /media/<file>              获取生成的视频文件

设计要点：
    - 单 GPU 单 pipeline 实例，全局锁串行推理（Lite 单卡 4090 = 3 路会话共享 96 FPS 算力）
    - 每个会话独立保存"运动帧缓存"(latent_motion_frames)，跨句子口型/动作连续
    - 多形象预加载：启动时扫描 AVATAR_DIR 下所有 *.png，一次 VAE 编码缓存
    - 启动后台预热（torch.compile 编译约 30~90s），期间 /generate 返回 503
    - 会话空闲 30 分钟自动回收；生成的视频默认保留 1 小时
"""
import os
import shutil
import threading
import time
import uuid
import subprocess
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import torch
import librosa
import imageio
from flask import Flask, request, jsonify, send_from_directory
from loguru import logger

from flash_head.inference import (
    get_pipeline, get_base_data, get_infer_params,
    get_audio_embedding, run_pipeline,
)
from flash_head.utils.utils import resize_and_centercrop
from flash_head.utils.facecrop import process_image

# ---------------------------------------------------------------------------
# 配置（环境变量可覆盖）
# ---------------------------------------------------------------------------
CKPT_DIR = os.environ.get("CKPT_DIR", "models/SoulX-FlashHead-1_3B")
WAV2VEC_DIR = os.environ.get("WAV2VEC_DIR", "models/wav2vec2-base-960h")
MODEL_TYPE = os.environ.get("MODEL_TYPE", "lite")           # lite | pro
AVATAR_DIR = os.environ.get("AVATAR_DIR", "avatars")
MEDIA_DIR = os.environ.get("MEDIA_DIR", "media")
PORT = int(os.environ.get("PORT", "8000"))
API_KEY = os.environ.get("API_KEY", "")                      # 非空则要求 X-API-Key 头
USE_FACE_CROP = os.environ.get("USE_FACE_CROP", "0") == "1"
SESSION_TTL_S = int(os.environ.get("SESSION_TTL_S", "1800"))     # 会话空闲回收 30min
MEDIA_TTL_S = int(os.environ.get("MEDIA_TTL_S", "3600"))         # 视频保留 1h

os.makedirs(AVATAR_DIR, exist_ok=True)
os.makedirs(MEDIA_DIR, exist_ok=True)

app = Flask(__name__)

# ---------------------------------------------------------------------------
# 全局状态
# ---------------------------------------------------------------------------
PIPELINE = None          # FlashHeadPipeline 单例
INFER_PARAMS = None      # 推理参数（frame_num / motion_frames_num / sample_rate / tgt_fps ...）
WARM = False             # 预热完成标志
GPU_LOCK = threading.Lock()      # GPU 推理串行锁
AVATAR_LOCK = threading.Lock()   # 形象注册锁
SESSIONS = {}                    # session_id -> SessionState


@dataclass
class SessionState:
    avatar: str
    saved_frames: object = None            # 该会话上一句结束时的 latent_motion_frames
    created_at: float = field(default_factory=time.time)
    last_used: float = field(default_factory=time.time)


# ---------------------------------------------------------------------------
# 启动加载
# ---------------------------------------------------------------------------
def _ensure_default_avatar():
    """形象目录为空时，用项目自带示例兜底，保证 pipeline 可初始化。"""
    if any(f.lower().endswith(".png") for f in os.listdir(AVATAR_DIR)):
        return
    fallback = os.path.join("examples", "girl.png")
    if os.path.exists(fallback):
        shutil.copy(fallback, os.path.join(AVATAR_DIR, "girl.png"))
        logger.info("avatar 目录为空，已复制示例形象 examples/girl.png")


def load_model():
    """加载模型并预编码所有形象（对应 generate_video.py 的 get_pipeline + get_base_data）。"""
    global PIPELINE, INFER_PARAMS
    _ensure_default_avatar()
    PIPELINE = get_pipeline(world_size=1, ckpt_dir=CKPT_DIR,
                            wav2vec_dir=WAV2VEC_DIR, model_type=MODEL_TYPE)
    get_base_data(PIPELINE, cond_image_path_or_dir=AVATAR_DIR,
                  base_seed=42, use_face_crop=USE_FACE_CROP)
    INFER_PARAMS = get_infer_params()
    logger.info(f"模型加载完成 model_type={MODEL_TYPE}, "
                f"形象列表={list(PIPELINE.cond_image_dict.keys())}")


# ---------------------------------------------------------------------------
# 核心生成逻辑
# ---------------------------------------------------------------------------
def generate_sentence_video(audio_np: np.ndarray) -> torch.Tensor:
    """'once' 模式整句生成（复刻 generate_video.py 的 once 分支）。

    输入：16kHz 单声道 numpy 音频（一句话）
    输出：(T, H, W, C) float 张量（0~255）
    依赖调用前已 restore 好会话的 latent_motion_frames。
    """
    ip = INFER_PARAMS
    frame_num = ip["frame_num"]
    mf = ip["motion_frames_num"]
    slice_len = frame_num - mf
    sr = ip["sample_rate"]
    fps = ip["tgt_fps"]

    min_len = (frame_num + slice_len) * sr // fps      # 至少 33+25=58 帧(约2.32s)才能切出1个完整chunk，不足补静音
    if len(audio_np) < min_len:
        audio_np = np.concatenate([audio_np, np.zeros(min_len - len(audio_np))])

    frame_len = frame_num * sr // fps
    remainder = (len(audio_np) - frame_len) % slice_len
    if remainder > 0:
        audio_np = np.concatenate([audio_np, np.zeros(slice_len - remainder)])

    emb_all = get_audio_embedding(PIPELINE, audio_np)
    n_chunks = (emb_all.shape[1] - frame_num) // slice_len
    parts = []
    for i in range(n_chunks):
        chunk = emb_all[:, i * slice_len: i * slice_len + frame_num].contiguous()
        video = run_pipeline(PIPELINE, chunk)
        if i != 0:
            video = video[mf:]                          # 后续段去掉运动帧重叠部分
        parts.append(video.cpu())
    return torch.cat(parts, dim=0)


def save_video_mp4(frames: torch.Tensor, path: str, audio_np: np.ndarray, sr: int):
    """帧序列写 mp4 并 ffmpeg 合流音频（与 generate_video.py 的 save_video 一致）。"""
    tmp = path.replace(".mp4", "_tmp.mp4")
    with imageio.get_writer(tmp, format="mp4", mode="I", fps=INFER_PARAMS["tgt_fps"],
                            codec="h264", ffmpeg_params=["-bf", "0"]) as w:
        for f in frames.numpy().astype(np.uint8):
            w.append_data(f)
    # 音频重采样到标准 16k 再合流，-shortest 对齐长度
    tmp_wav = path.replace(".mp4", "_src.wav")
    librosa.output.write_wav(tmp_wav, audio_np, sr) if hasattr(librosa, "output") \
        else _write_wav(tmp_wav, audio_np, sr)
    cmd = ["ffmpeg", "-y", "-i", tmp, "-i", tmp_wav,
           "-c:v", "copy", "-c:a", "aac", "-shortest", path]
    subprocess.run(cmd, check=True, capture_output=True)
    os.remove(tmp)
    os.remove(tmp_wav)


def _write_wav(path: str, audio: np.ndarray, sr: int):
    import wave
    audio16 = (np.clip(audio, -1, 1) * 32767).astype(np.int16)
    with wave.open(path, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(audio16.tobytes())


def register_avatar(name: str, image_path: str):
    """运行时注册新形象：复刻 prepare_params 对单张图的处理，仅追加不重建。"""
    from PIL import Image
    img = Image.open(image_path).convert("RGB")
    tensor = resize_and_centercrop(
        img, (PIPELINE.target_h, PIPELINE.target_w)
    ).to(PIPELINE.device, dtype=PIPELINE.param_dtype)
    tensor = (tensor / 255 - 0.5) * 2
    video_frames = tensor.repeat(1, 1, PIPELINE.frame_num, 1, 1)
    latent = PIPELINE.vae.encode(video_frames)
    with AVATAR_LOCK:
        PIPELINE.cond_image_dict[name] = img
        PIPELINE.cond_image_tensor_dict[name] = tensor
        PIPELINE.ref_img_latent_dict[name] = latent
    logger.info(f"注册新形象: {name}")


def warmup():
    """后台预热：跑一次 4 秒静音生成，触发 CUDA kernel 缓存。

    注意：once 模式需要至少 frame_num + slice_len 个音频帧才能切出完整 chunk
    （约 2.32s），2 秒会因 n_chunks=0 触发 torch.cat 空列表错误，故用 4 秒。
    """
    global WARM
    t0 = time.time()
    try:
        silence = np.zeros(INFER_PARAMS["sample_rate"] * 4, dtype=np.float32)
        first_avatar = next(iter(PIPELINE.cond_image_dict))
        with GPU_LOCK:
            PIPELINE.reset_person_name(first_avatar)
            generate_sentence_video(silence)
        WARM = True
        logger.info(f"预热完成，耗时 {time.time() - t0:.1f}s，开始对外服务")
    except Exception as e:
        logger.error(f"预热失败: {e}")


# ---------------------------------------------------------------------------
# 后台清理线程
# ---------------------------------------------------------------------------
def janitor():
    while True:
        time.sleep(60)
        now = time.time()
        for sid in [s for s, st in SESSIONS.items() if now - st.last_used > SESSION_TTL_S]:
            SESSIONS.pop(sid, None)
            logger.info(f"回收空闲会话 {sid}")
        for f in os.listdir(MEDIA_DIR):
            p = os.path.join(MEDIA_DIR, f)
            if os.path.isfile(p) and now - os.path.getmtime(p) > MEDIA_TTL_S:
                os.remove(p)


# ---------------------------------------------------------------------------
# 鉴权
# ---------------------------------------------------------------------------
@app.before_request
def check_api_key():
    if API_KEY and request.path != "/health" \
            and request.headers.get("X-API-Key") != API_KEY:
        return jsonify(error="invalid api key"), 401


# ---------------------------------------------------------------------------
# 路由
# ---------------------------------------------------------------------------
@app.get("/health")
def health():
    ok = PIPELINE is not None
    return jsonify(
        status="ok" if ok else "loading",
        warm=WARM,
        model_type=MODEL_TYPE,
        device=str(PIPELINE.device) if ok else None,
        gpu=torch.cuda.get_device_name(0) if ok and torch.cuda.is_available() else None,
        avatars=list(PIPELINE.cond_image_dict.keys()) if ok else [],
        active_sessions=len(SESSIONS),
    )


@app.post("/session")
def create_session():
    """创建会话。两种方式：
    1. JSON  {"avatar": "girl"}                     —— 使用启动时预加载的形象
    2. multipart 上传 file=<图片>（可带 use_face_crop=1）—— 动态注册自定义形象
    """
    if not WARM:
        return jsonify(error="service warming up, retry later"), 503

    # 方式2：上传自定义形象
    file = request.files.get("file")
    if file is not None:
        name = f"upload_{uuid.uuid4().hex[:8]}"
        tmp = os.path.join(MEDIA_DIR, f"{name}.png")
        file.save(tmp)
        if request.form.get("use_face_crop") == "1":
            try:
                cropped = process_image(tmp)
                cropped.save(tmp)
            except Exception as e:
                logger.warning(f"人脸裁剪失败，使用原图: {e}")
        try:
            register_avatar(name, tmp)
        except Exception as e:
            return jsonify(error=f"形象注册失败: {e}"), 400
        avatar = name
    else:
        avatar = (request.get_json(silent=True) or {}).get("avatar", "")
        with AVATAR_LOCK:
            if avatar not in PIPELINE.cond_image_dict:
                return jsonify(error=f"unknown avatar: {avatar}",
                               available=list(PIPELINE.cond_image_dict.keys())), 404

    sid = uuid.uuid4().hex[:12]
    SESSIONS[sid] = SessionState(avatar=avatar)
    logger.info(f"创建会话 {sid}, avatar={avatar}")
    return jsonify(session_id=sid, avatar=avatar)


@app.post("/session/<sid>/generate")
def generate(sid):
    """句子级生成：multipart 上传音频（wav/mp3/flac），返回 mp4 URL。

    同一会话跨句子调用，口型与动作连续（运动帧缓存跨句子保持）。
    """
    st = SESSIONS.get(sid)
    if st is None:
        return jsonify(error="session not found or expired"), 404
    if not WARM:
        return jsonify(error="service warming up, retry later"), 503

    file = request.files.get("audio")
    if file is None:
        return jsonify(error="missing audio file (form field: audio)"), 400

    # 音频解码在锁外做（CPU 密集，不占 GPU）
    tmp_audio = os.path.join(MEDIA_DIR, f"_{uuid.uuid4().hex[:8]}.wav")
    file.save(tmp_audio)
    try:
        audio_np, _sr = librosa.load(tmp_audio, sr=INFER_PARAMS["sample_rate"], mono=True)
    finally:
        os.remove(tmp_audio)
    audio_duration = len(audio_np) / INFER_PARAMS["sample_rate"]

    t0 = time.time()
    with GPU_LOCK:
        # 1) 切到该会话的形象；2) 恢复该会话上一句结束的运动帧（保证跨句连续）
        PIPELINE.reset_person_name(st.avatar)
        if st.saved_frames is not None:
            PIPELINE.latent_motion_frames = st.saved_frames.clone()
        frames = generate_sentence_video(audio_np)
        st.saved_frames = PIPELINE.latent_motion_frames.clone()   # 保存状态供下一句续接
    gen_time = time.time() - t0

    fname = f"{datetime.now().strftime('%Y%m%d%H%M%S')}_{sid}_{uuid.uuid4().hex[:6]}.mp4"
    out_path = os.path.join(MEDIA_DIR, fname)
    save_video_mp4(frames, out_path, audio_np, INFER_PARAMS["sample_rate"])
    st.last_used = time.time()

    return jsonify(
        video_url=f"/media/{fname}",
        avatar=st.avatar,
        audio_duration_s=round(audio_duration, 2),
        generation_time_s=round(gen_time, 2),
        realtime_factor=round(gen_time / max(audio_duration, 0.01), 3),
    )


@app.delete("/session/<sid>")
def delete_session(sid):
    if SESSIONS.pop(sid, None) is None:
        return jsonify(error="session not found"), 404
    return jsonify(deleted=sid)


@app.get("/media/<path:fname>")
def media(fname):
    return send_from_directory(MEDIA_DIR, fname)


# ---------------------------------------------------------------------------
# 入口
# ---------------------------------------------------------------------------
load_model()
threading.Thread(target=warmup, daemon=True).start()
threading.Thread(target=janitor, daemon=True).start()

if __name__ == "__main__":
    # 生产环境用 gunicorn：gunicorn -w 1 --threads 8 -t 300 -b 0.0.0.0:8000 server:app
    app.run(host="0.0.0.0", port=PORT, threaded=True)
