"""
Central configuration loaded from environment variables.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


def _bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _int(value: str | None, default: int) -> int:
    return int(value) if value is not None else default


def _float(value: str | None, default: float) -> float:
    return float(value) if value is not None else default


@dataclass(frozen=True)
class Settings:
    # --- LiveKit (credentials for room + Inference gateway) ---
    livekit_url: str = os.getenv("LIVEKIT_URL", "ws://localhost:7880")
    livekit_api_key: str = os.getenv("LIVEKIT_API_KEY", "devkey")
    livekit_api_secret: str = os.getenv("LIVEKIT_API_SECRET", "secret")
    livekit_room: str = os.getenv("LIVEKIT_ROOM", "voice-agent")
    livekit_identity: str = os.getenv("LIVEKIT_IDENTITY", "voice-agent-bot")

    # --- LiveKit Inference models ---
    stt_model: str = os.getenv("STT_MODEL", "deepgram/nova-3:en")
    llm_model: str = os.getenv("LLM_MODEL", "openai/gpt-4o-mini")
    tts_model: str = os.getenv(
        "TTS_MODEL",
        "cartesia/sonic-3:9626c31c-bec5-4cca-baa8-f8ba9e84c8bc",
    )
    preemptive_generation: bool = _bool(os.getenv("PREEMPTIVE_GENERATION"), True)

    # Turn detection: stt (default, works embedded), vad, multilingual (needs job worker)
    turn_detection: str = os.getenv("TURN_DETECTION", "stt")

    # Endpointing delays (seconds) passed to AgentSession
    min_endpointing_delay: float = _float(os.getenv("MIN_ENDPOINTING_DELAY"), 0.5)
    max_endpointing_delay: float = _float(os.getenv("MAX_ENDPOINTING_DELAY"), 3.0)

    # --- Server ---
    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = _int(os.getenv("PORT"), 8000)
    debug_mode: bool = _bool(os.getenv("DEBUG_MODE"), False)

    llm_system_prompt: str = os.getenv(
        "LLM_SYSTEM_PROMPT",
        "You are a helpful voice assistant. Keep replies short and conversational.",
    )


settings = Settings()
