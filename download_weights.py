"""
下载 SoulX-FlashHead 权重（走 hf-mirror）

只拉 Lite 模式实际需要的文件，避开 Model_Pro / VAE_Wan（可省约 6.1GB）：
  Model_Lite/  -> pipeline 里 model_type=lite 时的 DiT 权重
  VAE_LTX/     -> pipeline 里 model_type=lite 时的 VAE
外加 wav2vec2-base-960h（音频编码器）
"""
import os
import sys
import time

# 必须在 import huggingface_hub 之前设置
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
# 关闭 Xet 存储后端：hf-mirror 未代理 cas-server.xethub.hf.co，
# 大文件走 Xet 会报 CAS 401 Unauthorized，改走传统 LFS 直链
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

from huggingface_hub import snapshot_download

REPO = "Soul-AILab/SoulX-FlashHead-1_3B"
ALLOW = ["Model_Lite/*", "VAE_LTX/*"]

WAV2VEC_REPO = "facebook/wav2vec2-base-960h"

# 本脚本位于项目根目录，父目录即项目根
ROOT = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(ROOT, "models")


def grab(repo, local_dir, allow=None, tries=5):
    last = None
    for attempt in range(1, tries + 1):
        try:
            print(f"[download] {repo} -> {local_dir} (attempt {attempt})", flush=True)
            snapshot_download(
                repo_id=repo,
                local_dir=local_dir,
                allow_patterns=allow,
                max_workers=4,
            )
            return True
        except Exception as e:  # noqa: BLE001
            last = e
            print(f"[warn] attempt {attempt} failed: {type(e).__name__}: {e}", flush=True)
            time.sleep(5 * attempt)
    print(f"[error] give up {repo}: {last}", flush=True)
    return False


def main():
    os.makedirs(MODELS, exist_ok=True)
    ok1 = grab(REPO, os.path.join(MODELS, "SoulX-FlashHead-1_3B"), ALLOW)
    ok2 = grab(WAV2VEC_REPO, os.path.join(MODELS, "wav2vec2-base-960h"))
    print("RESULT", "OK" if (ok1 and ok2) else "PARTIAL", flush=True)
    return 0 if (ok1 and ok2) else 1


if __name__ == "__main__":
    sys.exit(main())
