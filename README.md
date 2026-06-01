# Real-Time Trilingual Simultaneous Translation

A real-time simultaneous interpretation system that translates Chinese speech into English and Vietnamese simultaneously.

## Tech Stack

- **ASR**: whisper.cpp (small model) via pywhispercpp, running on Apple Metal GPU
- **Translation**: Ollama (qwen2.5:3b) via local API
- **Backend**: FastAPI + WebSocket
- **Frontend**: Single-page web UI with client-side VAD

## Architecture

```
┌──────────────┐     WebSocket      ┌──────────────┐
│   Browser    │ ──────────────────▶│   FastAPI    │
│  (VAD + mic) │◀────────────────────│   Server     │
└──────────────┘   ASR + Trilingual │              │
                     Translation    └──────┬───────┘
                                          │
                     ┌────────────────────┼────────────────────┐
                     ▼                    ▼                    ▼
              ┌─────────────┐      ┌────────────────┐     ┌─────────────┐
              │ whisper.cpp│      │  Ollama        │     │   Audio    │
              │ (Metal GPU)│      │ qwen2.5:3b     │     │  Pipeline  │
              └─────────────┘      └────────────────┘     └─────────────┘
```

## Quick Start

```bash
# 1. Ensure Ollama is running with qwen2.5:3b
ollama run qwen2.5:3b

# 2. Start the server
python server.py

# 3. Open http://127.0.0.1:8000 in your browser
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Web UI |
| WS | `/ws` | Real-time audio translation |
| POST | `/api/hotwords` | Update ASR hotwords |
| GET | `/test-translate?text=` | Test translation |
| POST | `/test-asr` | Test ASR with audio file |

## How It Works

1. **VAD**: Client detects speech via RMS energy threshold (0.015), records on speaking, sends on silence (1200ms timeout)
2. **ASR**: Server decodes audio to 16kHz, runs whisper.cpp with streaming segments
3. **Filters**: 6-stage filtering removes hallucinations, repetitions, and noise
4. **Translation**: qwen2.5:3b translates to English + Vietnamese with conversation history (last 2 exchanges)
5. **Output**: Results stream to client via WebSocket in real-time

## Requirements

- Python 3.10+
- Ollama with `qwen2.5:3b` model
- pywhispercpp, fastapi, uvicorn, av, zhconv, requests