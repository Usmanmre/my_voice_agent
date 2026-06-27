# Educational Voice Agent (FastAPI + LiveKit AgentSession)

A voice agent built on **LiveKit AgentSession** with **LiveKit Inference** for STT, LLM, and TTS. Observability via stdout and a debug WebSocket.

## What you get

| Module | Role |
|--------|------|
| `main.py` | FastAPI server, LiveKit room, AgentSession lifecycle, event WebSocket |
| `agent_session.py` | AgentSession / Agent / RoomOptions factory |
| `config.py` | Environment-driven settings (Inference model strings) |

## Quick start

### 1. Start LiveKit (local) or use LiveKit Cloud

Local:

```bash
docker run --rm -p 7880:7880 \
  -e LIVEKIT_KEYS="devkey: secret" \
  livekit/livekit-server --dev
```

Or point `LIVEKIT_URL` at your LiveKit Cloud project (recommended for Inference).

### 2. Install dependencies

**Requires Python 3.10+** (`livekit-agents` does not support 3.9).

```bash
python3.11 -m venv .venv   # or python3.12
source .venv/bin/activate
pip install -r requirements.txt
```

First run downloads the turn-detector model (~100 MB) and Silero VAD weights.

### 3. Configure environment

```bash
cp .env.example .env
# Set LIVEKIT_API_KEY and LIVEKIT_API_SECRET (LiveKit Cloud or local dev keys)
```

LiveKit Inference routes STT/LLM/TTS through `agent-gateway.livekit.cloud` — you do **not** need separate Deepgram or OpenAI API keys when using Inference model strings.

### 4. Run the agent

```bash
python main.py
```

### 5. Join the room

Open `GET /token` in a browser or curl, then open the `meet_url` from the response. Speak into the microphone.

### 6. Watch events (optional)

```bash
wscat -c ws://localhost:8000/ws/events
```

## Architecture

```
┌─────────────┐     WebRTC      ┌──────────────────┐
│ LiveKit     │ ◄──────────────►│  Browser client  │
│ Server      │                 └──────────────────┘
└──────┬──────┘
       │
       ▼
┌──────────────────────────────────────────────────┐
│  main.py — FastAPI + AgentSession                │
│                                                  │
│  AgentSession (preemptive_generation=True)        │
│    STT  → Deepgram nova-3 via provider key      │
│    LLM  → OpenAI gpt-4o-mini via provider key   │
│    TTS  → Deepgram Aura via provider key        │
│    VAD  → Silero plugin                         │
│    Turn → Multilingual turn-detector plugin     │
└──────────────────────────────────────────────────┘
       │
       ▼
┌─────────────┐
│ FastAPI     │  GET /health, /token
│             │  WS  /ws/events
└─────────────┘
```

### Overlapping pipeline (`preemptive_generation`)

When `PREEMPTIVE_GENERATION=true` (default), AgentSession starts LLM and TTS inference **before** the user finishes speaking — the same latency win as a custom speculative LLM, but built into the framework.

Partial transcripts arrive via streaming STT (`interim_results`). At end-of-turn, the session commits or reuses the preemptive response.

### Configuration

| Variable | Default | Purpose |
|----------|---------|---------|
| `DEEPGRAM_API_KEY` | required | Deepgram STT/TTS provider key |
| `OPENAI_API_KEY` | required | OpenAI LLM provider key |
| `STT_MODEL` | `nova-3` | Deepgram STT model |
| `STT_LANGUAGE` | `en` | Deepgram STT language |
| `LLM_MODEL` | `gpt-4o-mini` | OpenAI LLM model |
| `TTS_MODEL` | `aura-2-thalia-en` | Deepgram TTS model |
| `PREEMPTIVE_GENERATION` | `true` | Overlap LLM/TTS with user speech |
| `MIN_ENDPOINTING_DELAY` | `0.5` | Min silence before end-of-turn (seconds) |
| `MAX_ENDPOINTING_DELAY` | `3.0` | Max wait before forcing end-of-turn |

## Observability

AgentSession events are bridged to `EventLogger`:

| Event | When |
|-------|------|
| `speech_start` / `speech_end` | User state changes |
| `turn_detected` | User stopped speaking |
| `transcription_partial` | Interim STT while user speaks |
| `transcription_completed` | Final STT at end-of-turn |
| `llm_speculative_started` | Agent enters `thinking` state |
| `llm_hypothesis_updated` | New partial transcript fed to preemptive path |
| `llm_committed` | User message committed / LLM metrics |
| `llm_completed` | Assistant reply added to conversation |
| `tts_started` / `tts_completed` | TTS synthesis lifecycle |

## API endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/health` | Server + AgentSession status |
| GET | `/token` | LiveKit join token for browser client |
| WS | `/ws/events` | Real-time pipeline event stream |

## Project philosophy

- **LiveKit-native** — AgentSession handles STT/LLM/TTS orchestration
- **Observable** — events bridged to stdout and WebSocket for learning
- **Low custom code** — no manual VAD/STT/WebSocket plumbing

## License

MIT — use freely for learning.
