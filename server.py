import io
import math
import re
import asyncio
import uuid
import threading
import numpy as np
import av
import zhconv
from pathlib import Path
from fastapi import FastAPI, WebSocket, UploadFile, File
from fastapi.responses import FileResponse
from fastapi.requests import Request
from pywhispercpp.model import Model
import requests

def _to_simplified(text: str) -> str:
    """Convert Traditional Chinese to Simplified Chinese."""
    return zhconv.convert(text, 'zh-cn')

app = FastAPI()

BASE_DIR = Path(__file__).parent

@app.get("/")
async def index():
    return FileResponse(BASE_DIR / "index.html")

# whisper.cpp medium model on Metal GPU (Apple M1/M2/M3)
# ~10x faster than CPU faster-whisper; better Chinese accuracy than small
asr_model = Model('small', n_threads=8, print_progress=False, print_realtime=False)
_asr_lock = threading.Lock()  # whisper.cpp context is not thread-safe

# whisper.cpp special-token markers to filter out
_WHISPER_SPECIAL = {
    "[BLANK_AUDIO]", "[MUSIC]", "[NOISE]", "[APPLAUSE]",
    "(music)", "(noise)", "(applause)",
}

# Common Whisper hallucination substring seeds — if ANY seed is found in the text, reject it.
_HALLUCINATION_SEEDS = (
    # Subscribe / outro hallucinations
    "請訂閱", "请订阅",
    "點贊", "点赞", "轉發", "转发",
    "字幕由", "字幕製作", "字幕作者",
    "本片由", "本节目由", "本節目由",
    "♪", "♫", "♬",
    "MBC", "KBS", "CCTV",
)

# Short exact-match phrases that are complete hallucinations on their own
_HALLUCINATION_EXACT = frozenset({
    # single-char fillers only
    "嗯", "啊", "哦", "哎", "唉", "呢", "噢", "哟", "哼", "嘿",
    "嗯嗯", "啊啊", "哦哦", "哎哎",
    "OK", "ok", "嗯哼",
})


def _is_hallucination(text: str) -> bool:
    """Return True if text looks like a Whisper hallucination, not real speech."""
    stripped = text.strip("!！。，,? ？\t\n~～。…·、")
    # Exact match for very short standalone hallucinations
    if stripped in _HALLUCINATION_EXACT:
        return True
    # Substring match: any known hallucination seed present in the text
    for seed in _HALLUCINATION_SEEDS:
        if seed in text:
            return True
    return False

# Hotwords: domain-specific terms injected into ASR as initial_prompt.
# Only static vocabulary — do NOT put previous transcriptions here.
# Update at runtime via POST /api/hotwords
HOTWORDS: list = []

def build_initial_prompt() -> str | None:
    if not HOTWORDS:
        return None
    return ", ".join(HOTWORDS)

async def get_translation(text, history=None):
    try:
        messages = [
            {"role": "system", "content": (
                "你是专业同声传译员，将口语中文（可能含中英混杂）同时译成自然口语英文和越南语。\n"
                "输出格式严格如下（两行，不要多余内容）：\n"
                "EN: <英文译文>\n"
                "VI: <越南语译文>\n"
                "规则：①只输出上述格式；②专有名词/人名/品牌保留原文；"
                "③语言口语化自然，贴近日常对话用语，避免书面腔；④不要解释，不要补充。"
            )},
        ]
        # Inject last 2 exchanges only — more context = more tokens = slower
        if history:
            for item in history[-2:]:
                messages.append({"role": "user", "content": item["cn"]})
                messages.append({"role": "assistant", "content": f"EN: {item['en']}\nVI: {item.get('vi', '')}"})
        messages.append({"role": "user", "content": text})
        loop = asyncio.get_event_loop()
        raw = await loop.run_in_executor(
            None,
            lambda: requests.post(
                "http://127.0.0.1:11434/api/chat",
                json={
                    "model": "qwen2.5:3b",
                    "messages": messages,
                    "stream": False,
                    "options": {"num_predict": 200}
                },
                timeout=30
            ).json()["message"]["content"].strip()
        )
        en, vi = "", ""
        for line in raw.splitlines():
            if line.startswith("EN:"):
                en = line[3:].strip()
            elif line.startswith("VI:"):
                vi = line[3:].strip()
        if not en:
            en = raw  # fallback
        return {"en": en, "vi": vi}
    except Exception as e:
        return {"en": f"Translation Error: {e}", "vi": ""}

@app.get("/api/hotwords")
async def get_hotwords():
    """Get current hotwords list."""
    return {"hotwords": HOTWORDS}


@app.post("/api/hotwords")
async def set_hotwords(request: Request):
    """Update hotwords list. Body: {\"hotwords\": [\"term1\", \"term2\"]}"""
    global HOTWORDS
    data = await request.json()
    HOTWORDS = [w.strip() for w in data.get("hotwords", []) if w.strip()]
    print(f"Hotwords updated: {HOTWORDS}")
    return {"hotwords": HOTWORDS}


@app.get("/test-translate")
async def test_translate(text: str = "你好世界"):
    """Test translation endpoint: /test-translate?text=你好"""
    result = await get_translation(text)
    return {"original": text, "translated": result}


@app.post("/test-asr")
async def test_asr(file: UploadFile = File(...)):
    """Test ASR endpoint: POST an audio file to /test-asr"""
    data = await file.read()
    audio_array = decode_audio(data)
    loop = asyncio.get_event_loop()
    segments = await loop.run_in_executor(
        None,
        lambda: _run_asr_sync(audio_array, language=None)
    )
    text = "".join([s.text for s in segments]).strip()
    translation = await get_translation(text) if text else ""
    return {"original": text, "translated": translation}


def _run_asr_sync(audio: np.ndarray, language: str = "zh", segment_callback=None) -> list:
    """Thread-safe whisper.cpp transcription via Metal GPU."""
    # temperature=0 for deterministic decoding — reduces hallucinations
    # Do NOT use instruction-style initial_prompt: Whisper sometimes transcribes it verbatim
    params = {"temperature": 0.0}
    if language:
        params["language"] = language
    extra = build_initial_prompt()
    if extra:
        params["initial_prompt"] = extra
    with _asr_lock:
        return asr_model.transcribe(audio, new_segment_callback=segment_callback, **params)


def _is_repetitive(text: str) -> bool:
    """Detect hallucinated repetitive text (word-level and character-level)."""
    # Word-level: exact half-split repeat (e.g. "hello world hello world")
    words = text.split()
    half = len(words) // 2
    if half >= 2 and words[:half] == words[half:half*2]:
        return True
    # Character-level: any single char repeated 5+ consecutive times
    for i in range(len(text) - 4):
        if len(set(text[i:i+5])) == 1 and text[i].strip():
            return True
    # N-gram: a 2–4 char sequence appearing 6+ times in the text
    stripped = text.replace(' ', '').replace(',', '').replace('，', '')
    for n in range(2, 5):
        for i in range(len(stripped) - n + 1):
            gram = stripped[i:i+n]
            if stripped.count(gram) >= 6:
                return True
    return False


def decode_audio(data: bytes) -> np.ndarray:
    """Decode any audio format (webm, opus, wav, etc.) to float32 numpy array at 16kHz."""
    container = av.open(io.BytesIO(data))
    resampler = av.AudioResampler(format="fltp", layout="mono", rate=16000)
    samples = []
    for frame in container.decode(audio=0):
        for rf in resampler.resample(frame):
            samples.append(rf.to_ndarray()[0])
    audio = np.concatenate(samples) if samples else np.zeros(0, dtype=np.float32)
    return _trim_silence(audio)


def _trim_silence(audio: np.ndarray, threshold: float = 0.005, frame_ms: int = 50) -> np.ndarray:
    """Trim leading/trailing silence to reduce audio length sent to Whisper."""
    frame_size = int(16000 * frame_ms / 1000)  # samples per frame at 16kHz
    if len(audio) < frame_size * 2:
        return audio
    n_frames = len(audio) // frame_size
    energies = [
        np.sqrt(np.mean(audio[i*frame_size:(i+1)*frame_size] ** 2))
        for i in range(n_frames)
    ]
    # Find first and last active frames
    start = next((i for i, e in enumerate(energies) if e > threshold), 0)
    end = next((i for i, e in reversed(list(enumerate(energies))) if e > threshold), n_frames - 1)
    # Keep 1-frame padding on each side to avoid clipping
    start = max(0, start - 1)
    end = min(n_frames - 1, end + 1)
    return audio[start*frame_size:(end+1)*frame_size]


@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("Client connected", flush=True)

    # Per-connection state
    trans_history: list = []
    last_text: str = ""
    last_text_time: float = 0.0
    translate_sem = asyncio.Semaphore(1)  # per-connection: avoid Ollama queue buildup
    loop = asyncio.get_event_loop()

    async def _translate_and_send(text: str, history_snap: list, lang: str, prob: int, msg_id: str):
        """Background task: translate and push result to client."""
        try:
            async with translate_sem:
                result = await get_translation(text, history_snap)
            trans_history.append({"cn": text, "en": result["en"], "vi": result["vi"]})
            if len(trans_history) > 5:
                trans_history.pop(0)
            await websocket.send_json({
                "type": "translation",
                "id": msg_id,
                "en": result["en"],
                "vi": result["vi"],
            })
        except Exception as e:
            print(f"Translation task error: {e}")

    try:
        while True:
            audio_data = await websocket.receive_bytes()
            print(f"[RECV] raw_bytes={len(audio_data)}", flush=True)
            try:
                audio_array = decode_audio(audio_data)
            except Exception as e:
                print(f"[RECV] decode_audio failed: {e}", flush=True)
                continue
            rms_raw = float(np.sqrt(np.mean(audio_array**2))) if len(audio_array) else 0
            print(f"[RECV] decoded={len(audio_array)} samples, rms={rms_raw:.4f}", flush=True)

            # Skip clips that are too short (< 0.5s)
            if len(audio_array) < 8000:
                print(f"[SKIP] too short: {len(audio_array)} samples", flush=True)
                continue

            # Skip low-energy audio — must be clearly above background noise
            rms = rms_raw
            if rms < 0.012:
                print(f"[SKIP] low energy: rms={rms:.4f}", flush=True)
                continue

            # Generate msg_id before ASR so partial segments can reference it
            msg_id = str(uuid.uuid4())
            partial_sent = [False]  # mutable flag: did we stream any asr_partial to client?

            def _on_segment(segment, _id=msg_id):
                """Fires from transcribe() thread for each completed segment (streaming ASR)."""
                t = segment.text.strip()
                # Strip bracketed descriptors e.g. (開啟音樂)
                t = re.sub(r'^[\uff08(][^\uff09)]{0,10}[\uff09)]$', '', t).strip()
                if not t or t in _WHISPER_SPECIAL or _is_hallucination(t):
                    return
                if not any('\u4e00' <= c <= '\u9fff' or '\u3400' <= c <= '\u4dbf' for c in t):
                    return
                t = _to_simplified(t)  # convert to Simplified Chinese
                partial_sent[0] = True
                asyncio.run_coroutine_threadsafe(
                    websocket.send_json({"type": "asr_partial", "id": _id, "text": t}),
                    loop,
                )

            # Run ASR in thread pool — streams segments via callback as they complete
            segments = await loop.run_in_executor(
                None,
                lambda: _run_asr_sync(audio_array, language="zh", segment_callback=_on_segment),
            )
            detected_lang = "zh"
            lang_prob = 100

            # Log raw ASR output for diagnostics
            print(f"[ASR] {len(segments)} segments: {[s.text for s in segments]}", flush=True)

            # Gate 3: filter special-token segments and hallucinations
            def _seg_ok(s) -> bool:
                t = s.text.strip()
                if t in _WHISPER_SPECIAL:
                    return False
                if _is_hallucination(t):
                    return False
                return True

            valid = [s for s in segments if _seg_ok(s)]
            # Join, strip bracketed descriptors (全括号短句), and convert to Simplified Chinese
            raw_joined = "".join([s.text for s in valid]).strip()
            text = re.sub(r'^[\uff08(][^\uff09)]{0,20}[\uff09)]$', '', raw_joined).strip()
            text = _to_simplified(text)

            # Helper: cancel streaming entry on client when we reject this audio
            async def _cancel_partial():
                if partial_sent[0]:
                    await websocket.send_json({"type": "asr_cancel", "id": msg_id})

            # Also reject if the combined text is itself a hallucination
            if _is_hallucination(text):
                print(f"ASR: rejected — hallucination: {text!r}")
                await _cancel_partial()
                continue

            # Gate 3b: must contain at least one CJK character (real Chinese speech)
            if not any('\u4e00' <= c <= '\u9fff' or '\u3400' <= c <= '\u4dbf' for c in text):
                if text:
                    print(f"ASR: rejected — no CJK: {text!r}")
                await _cancel_partial()
                continue

            # Gate 4: detect repetitive/hallucinated text
            if text and _is_repetitive(text):
                print(f"ASR: rejected — repetitive: {text!r}")
                await _cancel_partial()
                continue

            print(f"ASR: {text!r} (filtered {len(segments)-len(valid)}/{len(segments)} segments)")

            # Gate 5: skip short text — require at least 2 meaningful characters
            meaningful_chars = len(text.replace(' ', '').replace(',', '').replace('，', '').replace('。', ''))
            if meaningful_chars < 2:
                print(f"ASR: rejected — too short: {text!r}", flush=True)
                await _cancel_partial()
                continue

            if not text:
                await _cancel_partial()
                continue

            if text == last_text and (asyncio.get_event_loop().time() - last_text_time) < 5.0:
                print(f"ASR: rejected — duplicate within 5s: {text!r}", flush=True)
                await _cancel_partial()
                continue

            last_text = text
            last_text_time = asyncio.get_event_loop().time()
            print(f"ASR: sending to client: {text!r}", flush=True)
            # Send final ASR confirmation — client upgrades streaming entry to confirmed state
            await websocket.send_json({
                "type": "asr",
                "id": msg_id,
                "original": text,
                "lang": detected_lang,
                "lang_prob": lang_prob,
            })
            # Fire translation as background task — doesn't block audio loop
            asyncio.create_task(_translate_and_send(
                text, list(trans_history), detected_lang, lang_prob, msg_id
            ))
    except Exception as e:
        import traceback
        print(f"Connection closed: {e}", flush=True)
        traceback.print_exc()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
