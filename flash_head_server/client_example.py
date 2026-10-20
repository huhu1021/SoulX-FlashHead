# -*- coding: utf-8 -*-
"""
FlashHead 服务客户端示例 —— 智能客服后端集成用

依赖：pip install requests
用法：python client_example.py [wav文件]（缺省用项目自带示例音频）
"""
import sys
import time
import requests

BASE = "http://127.0.0.1:8000"      # GPU 服务器地址
API_KEY = ""                        # 服务端设置了 API_KEY 时填写
HEADERS = {"X-API-Key": API_KEY} if API_KEY else {}


def health():
    r = requests.get(f"{BASE}/health", timeout=5).json()
    print("health:", r)
    return r


def create_session(avatar="girl"):
    """方式1：用预加载形象创建会话"""
    r = requests.post(f"{BASE}/session", json={"avatar": avatar},
                      headers=HEADERS, timeout=10).json()
    print("session:", r)
    return r["session_id"]


def create_session_with_image(image_path):
    """方式2：上传自定义形象（一寸照/证件照均可，可选人脸裁剪）"""
    with open(image_path, "rb") as f:
        r = requests.post(f"{BASE}/session",
                          files={"file": f},
                          data={"use_face_crop": "1"},
                          headers=HEADERS, timeout=60).json()
    print("session(自定义形象):", r)
    return r["session_id"]


def generate(sid, wav_path):
    """句子级生成：传一句话的音频，拿回 mp4 地址"""
    t0 = time.time()
    with open(wav_path, "rb") as f:
        r = requests.post(f"{BASE}/session/{sid}/generate",
                          files={"audio": f},
                          headers=HEADERS, timeout=300).json()
    print(f"generate: {r}  (端到端 {time.time()-t0:.1f}s)")
    return r


def download(video_url, save_as):
    r = requests.get(f"{BASE}{video_url}" if video_url.startswith("/") else video_url,
                     timeout=60)
    with open(save_as, "wb") as f:
        f.write(r.content)
    print("saved:", save_as)


if __name__ == "__main__":
    wav = sys.argv[1] if len(sys.argv) > 1 else "examples/podcast_sichuan_16k.wav"

    health()
    sid = create_session("girl")            # 或 create_session_with_image("my_face.png")

    # 模拟客服对话：同一会话连续多句，口型/动作跨句连续
    for i in range(1):
        result = generate(sid, wav)
        download(result["video_url"], f"reply_{i}.mp4")

    # 会话结束可主动释放（不调用也会被 30 分钟空闲回收）
    requests.delete(f"{BASE}/session/{sid}", headers=HEADERS, timeout=10)
