# FlashHead 推理服务（flash_head_server/）

把 SoulX-FlashHead 封装成独立 HTTP 服务，供智能客服后端远程调用。
**部署形态：单卡 GPU 机器独立部署，客服系统走内网 HTTP 调用，零侵入。**

## 架构

```
智能客服（117.163.35.12 容器栈）              GPU 工作站（本服务）
┌──────────────────────────┐            ┌─────────────────────────┐
│ LLM → TTS 出一句音频 ────┼───HTTP───→ │ /session/<sid>/generate │
│                          │            │  → FlashHead Lite 推理   │
│ 前端 <video> 播放 ←──────┼←──mp4 URL──│  → ffmpeg 合流           │
└──────────────────────────┘            └─────────────────────────┘
```

## 接口一览

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/health` | 健康检查（warm=true 才可生成） |
| POST | `/session` | 创建会话：`{"avatar":"girl"}` 或 multipart 上传自定义形象图 |
| POST | `/session/<sid>/generate` | 句子级生成：multipart `audio=<wav>`，返回 `video_url` |
| DELETE | `/session/<sid>` | 释放会话（不调也会 30 分钟空闲自动回收） |
| GET | `/media/<file>` | 获取生成的 mp4 |

设置环境变量 `API_KEY` 后，除 `/health` 外所有接口需带 `X-API-Key` 头。

## 服务器部署步骤（Ubuntu + 单卡 4090）

```bash
# 1. 下载权重（国内先 export HF_ENDPOINT=https://hf-mirror.com）
pip install "huggingface_hub[cli]"
huggingface-cli download Soul-AILab/SoulX-FlashHead-1_3B --local-dir models/SoulX-FlashHead-1_3B
huggingface-cli download facebook/wav2vec2-base-960h --local-dir models/wav2vec2-base-960h

# 2. 放入品牌形象（可选，不放则用项目自带示例）
cp your_avatar.png flash_head_server/data/avatars/

# 3. 构建并启动（首次构建约 20~40 分钟，主要耗在 flash-attn）
cd flash_head_server
docker compose up -d --build

# 4. 等待预热完成（torch.compile 编译，约 1~2 分钟，看日志）
docker compose logs -f flashhead
# 出现 "预热完成，开始对外服务" 即就绪

# 5. 冒烟测试
curl http://127.0.0.1:8000/health
python client_example.py
```

## 客服后端接入（伪代码）

```python
# TTS 每合成完一句后：
resp = requests.post(
    f"http://<gpu机内网IP>:8000/session/{sid}/generate",
    files={"audio": wav_bytes}, timeout=120,
).json()
# 把 resp["video_url"] 随回复下发前端，前端 <video src="..."> 播放
```

跨句连续性：同一 `session_id` 连续调用，口型与头部动作自动衔接
（服务端为每个会话保持运动帧缓存）。

## 性能预期（Lite 档 / 单卡 4090）

| 指标 | 数值 |
|---|---|
| 单句 3s 音频生成耗时 | ≈ 0.8~1.2s（realtime_factor ≈ 0.3） |
| 算力上限 | 96 FPS ≈ 3.8 倍实时 ≈ 3 路并发会话 |
| 首句延迟 | 服务预热后无编译开销，直接推理 |
| 分辨率/帧率 | 512×512 @ 25FPS |

并发超过 3 路：同机加卡横向扩（每卡再跑一个容器实例，端口错开），
或上 Pro 档双卡（见 docker-compose.yml 内注释）。

## 运维要点

- **预热**：服务启动后台自动跑一次静音生成完成 torch.compile 编译，
  `/health` 的 `warm=false` 期间 generate 返回 503，编排侧做重试即可
- **降级**：本服务不可用时客服主链路（纯语音）不受影响，前端隐藏视频区即可
- **磁盘**：生成的 mp4 默认保留 1 小时自动清理（`MEDIA_TTL_S` 可调）；
  若需永久存档，客服侧拿到 video_url 后转存自己的 OSS
- **监控**：`/health` 返回 active_sessions / gpu 名称；建议叠加
  `nvidia-smi` 显存告警（>22GB 说明并发超限）
