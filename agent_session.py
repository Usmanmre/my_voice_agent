"""
LiveKit AgentSession construction and room I/O options.

Uses provider plugins with API keys from environment variables.
"""

from __future__ import annotations

import logging

from livekit.agents import Agent, AgentSession, room_io
from livekit.plugins import deepgram, openai, silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

from config import settings

logger = logging.getLogger(__name__)


def _resolve_turn_detection() -> str | MultilingualModel:
    mode = settings.turn_detection.strip().lower()
    if mode == "multilingual":
        return MultilingualModel()
    if mode in ("stt", "vad", "manual"):
        return mode  # type: ignore[return-value]
    logger.warning("Unknown TURN_DETECTION=%r — using stt", settings.turn_detection)
    return "stt"


def build_agent_session() -> AgentSession:
    return AgentSession(
        stt=deepgram.STT(
            model=settings.stt_model,
            language=settings.stt_language,
            api_key=settings.deepgram_api_key or None,
            interim_results=True,
        ),
        llm=openai.LLM(
            model=settings.llm_model,
            api_key=settings.openai_api_key or None,
        ),
        tts=deepgram.TTS(
            model=settings.tts_model,
            api_key=settings.deepgram_api_key or None,
        ),
        vad=silero.VAD.load(),
        turn_handling={
            "turn_detection": _resolve_turn_detection(),
            "endpointing": {
                "min_delay": settings.min_endpointing_delay,
                "max_delay": settings.max_endpointing_delay,
            },
            "interruption": {"enabled": True},
            "preemptive_generation": {"enabled": settings.preemptive_generation},
        },
    )


def build_agent() -> Agent:
    return Agent(instructions=settings.llm_system_prompt)


def build_room_options() -> room_io.RoomOptions:
    # Keep audio I/O simple — BVC can fail on some embedded setups.
    return room_io.RoomOptions(
        audio_input=room_io.AudioInputOptions(),
        audio_output=room_io.AudioOutputOptions(),
    )
