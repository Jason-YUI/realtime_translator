# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a real-time **trilingual** simultaneous interpretation system (Chinese → English + Vietnamese). It uses:
- **whisper.cpp** (small model, Metal GPU) for automatic speech recognition (ASR) via pywhispercpp
- **Ollama** (qwen2.5:3b) for translation to English AND Vietnamese via local API
- **FastAPI** with WebSocket for real-time bidirectional communication
- **VAD (Voice Activity Detection)** on both client and server sides

## Running the Application

```bash
# Start the server (requires Ollama running with qwen2.5:3b model)
python server.py

# Server runs on http://127.0.0.1:8000
```

**Prerequisites:**
1. Ollama must be running locally with `qwen2.5:3b` model available (`ollama run qwen2.5:3b` to download)
2. Python dependencies: `fastapi`, `uvicorn`, `pywhispercpp`, `numpy`, `av`, `zhconv`, `requests`

## Architecture

### Frontend (index.html)
- Single-page web UI with WebSocket connection
- Client-side VAD using Web Audio API (RMS energy threshold: 0.015)
- Records audio when speech is detected, sends to server on silence
- Displays ASR results and **trilingual** translations (Chinese / English / Vietnamese) in real-time

### Backend (server.py)
- `GET /` - Serves index.html
- `WebSocket /ws` - Main real-time audio processing endpoint
  - Per-connection rolling context (last 5 translation pairs) for better translation
  - 6-stage filtering: language confidence, special-token filter, hallucination detection, CJK character check, repetition detection, duplicate detection (5s window)
- `POST /api/hotwords` - Update hotwords (domain-specific terms injected into ASR prompt)
- `GET /test-translate?text=...` - Test translation endpoint
- `POST /test-asr` - Test ASR with uploaded audio file

### Audio Processing Pipeline
1. Client records audio chunks on speech detection (VAD: RMS > 0.015, silence timeout 1200ms)
2. Server receives audio via WebSocket, decodes to 16kHz float32, trims silence
3. whisper.cpp transcribes with streaming segment callback (temperature=0)
4. Qwen2.5:3b translates with conversation history context (last 2 exchanges) to BOTH English and Vietnamese
5. Results sent back to client via WebSocket

### Translation Output Format
```
EN: <English translation>
VI: <Vietnamese translation>
```

### Filters (6-stage)
1. Special-token filter: removes `[BLANK_AUDIO]`, `[MUSIC]`, `[NOISE]`, `[APPLAUSE]`, etc.
2. Hallucination detection: rejects "请订阅", "点赞", "♪", "MBC", "CCTV", etc.
3. CJK character check: must contain at least one Chinese character
4. Repetition detection: rejects word-level/char-level repetition and N-gram patterns
5. Short text filter: requires at least 2 meaningful characters
6. Duplicate filter: rejects same text within 5 seconds

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Serve index.html |
| WS | `/ws` | Real-time audio translation (trilingual output) |
| GET | `/api/hotwords` | Get current hotwords |
| POST | `/api/hotwords` | Update hotwords |
| GET | `/test-translate?text=` | Test translation |
| POST | `/test-asr` | Test ASR with audio file |