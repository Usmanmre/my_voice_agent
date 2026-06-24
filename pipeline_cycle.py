"""
End-to-end turn cycle tracking: STT → LLM → TTS with timing and summary logs.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

logger = logging.getLogger("voice-agent.cycle")


@dataclass
class CycleMetrics:
    stt_audio_duration_s: float | None = None
    stt_streamed: bool | None = None
    llm_duration_s: float | None = None
    llm_ttft_s: float | None = None
    llm_tokens: int | None = None
    tts_duration_s: float | None = None
    tts_ttfb_s: float | None = None
    tts_audio_duration_s: float | None = None
    eou_delay_s: float | None = None
    transcription_delay_s: float | None = None


@dataclass
class TurnCycle:
    cycle_id: int
    started_at: float = field(default_factory=time.perf_counter)
    first_partial_at: float | None = None
    final_stt_at: float | None = None
    turn_detected_at: float | None = None
    llm_started_at: float | None = None
    llm_completed_at: float | None = None
    tts_started_at: float | None = None
    tts_completed_at: float | None = None
    partial_count: int = 0
    final_transcript: str = ""
    last_partial: str = ""
    user_text: str = ""
    assistant_text: str = ""
    metrics: CycleMetrics = field(default_factory=CycleMetrics)
    tts_pending: bool = False
    completed: bool = False

    def elapsed_ms(self, at: float | None = None) -> int:
        ref = at if at is not None else time.perf_counter()
        return int((ref - self.started_at) * 1000)


class PipelineCycleTracker:
    """Tracks one user turn through STT → LLM → TTS and prints a full cycle summary."""

    def __init__(self) -> None:
        self._cycle_id = 0
        self._active: TurnCycle | None = None

    @property
    def active(self) -> TurnCycle | None:
        return self._active

    def _ensure_cycle(self) -> TurnCycle:
        if self._active is None:
            self._cycle_id += 1
            self._active = TurnCycle(cycle_id=self._cycle_id)
            logger.info(
                "========== PIPELINE CYCLE #%s START (STT → LLM → TTS) ==========",
                self._active.cycle_id,
            )
        return self._active

    def on_speech_start(self) -> dict[str, Any]:
        cycle = self._ensure_cycle()
        return {"cycle_id": cycle.cycle_id, "elapsed_ms": cycle.elapsed_ms()}

    def on_speech_end(self) -> dict[str, Any]:
        cycle = self._ensure_cycle()
        return {"cycle_id": cycle.cycle_id, "elapsed_ms": cycle.elapsed_ms()}

    def on_turn_detected(self) -> dict[str, Any]:
        cycle = self._ensure_cycle()
        cycle.turn_detected_at = time.perf_counter()
        logger.info(
            "[CYCLE #%s | t+%sms] TURN END — committing STT → LLM → TTS",
            cycle.cycle_id,
            cycle.elapsed_ms(cycle.turn_detected_at),
        )
        return {"cycle_id": cycle.cycle_id, "elapsed_ms": cycle.elapsed_ms(cycle.turn_detected_at)}

    def on_partial_stt(self, text: str, revision: int) -> dict[str, Any]:
        cycle = self._ensure_cycle()
        cycle.partial_count += 1
        cycle.last_partial = text
        if cycle.first_partial_at is None:
            cycle.first_partial_at = time.perf_counter()
        logger.info(
            '[CYCLE #%s | t+%sms] STT partial #%s: "%s"',
            cycle.cycle_id,
            cycle.elapsed_ms(),
            revision,
            text,
        )
        return {
            "cycle_id": cycle.cycle_id,
            "elapsed_ms": cycle.elapsed_ms(),
            "revision": revision,
            "text": text,
        }

    def on_final_stt(self, text: str) -> dict[str, Any]:
        cycle = self._ensure_cycle()
        cycle.final_transcript = text
        cycle.user_text = text
        cycle.final_stt_at = time.perf_counter()
        logger.info(
            '[CYCLE #%s | t+%sms] STT final: "%s"',
            cycle.cycle_id,
            cycle.elapsed_ms(cycle.final_stt_at),
            text,
        )
        return {
            "cycle_id": cycle.cycle_id,
            "elapsed_ms": cycle.elapsed_ms(cycle.final_stt_at),
            "text": text,
        }

    def on_stt_metrics(
        self,
        *,
        audio_duration: float,
        streamed: bool,
        duration: float,
    ) -> None:
        cycle = self._ensure_cycle()
        cycle.metrics.stt_audio_duration_s = round(audio_duration, 3)
        cycle.metrics.stt_streamed = streamed
        logger.info(
            "[CYCLE #%s | t+%sms] STT metrics — audio=%.2fs streamed=%s request=%.2fs",
            cycle.cycle_id,
            cycle.elapsed_ms(),
            audio_duration,
            streamed,
            duration,
        )

    def on_llm_thinking(self) -> dict[str, Any]:
        cycle = self._ensure_cycle()
        if cycle.llm_started_at is None:
            cycle.llm_started_at = time.perf_counter()
            logger.info(
                "[CYCLE #%s | t+%sms] LLM started (preemptive=%s)",
                cycle.cycle_id,
                cycle.elapsed_ms(cycle.llm_started_at),
                cycle.final_stt_at is None,
            )
        return {"cycle_id": cycle.cycle_id, "elapsed_ms": cycle.elapsed_ms()}

    def on_llm_hypothesis(self, hypothesis: str, revision: int) -> dict[str, Any]:
        cycle = self._ensure_cycle()
        logger.info(
            '[CYCLE #%s | t+%sms] LLM hypothesis #%s: "%s"',
            cycle.cycle_id,
            cycle.elapsed_ms(),
            revision,
            hypothesis,
        )
        return {"cycle_id": cycle.cycle_id, "hypothesis": hypothesis, "revision": revision}

    def on_llm_committed(self, user_text: str) -> dict[str, Any]:
        cycle = self._ensure_cycle()
        cycle.user_text = user_text
        logger.info(
            '[CYCLE #%s | t+%sms] LLM committed user text: "%s"',
            cycle.cycle_id,
            cycle.elapsed_ms(),
            user_text,
        )
        return {"cycle_id": cycle.cycle_id, "user_text": user_text}

    def on_llm_metrics(
        self,
        *,
        duration: float,
        ttft: float,
        tokens: int,
    ) -> None:
        cycle = self._ensure_cycle()
        cycle.metrics.llm_duration_s = round(duration, 3)
        cycle.metrics.llm_ttft_s = round(ttft, 3)
        cycle.metrics.llm_tokens = tokens
        logger.info(
            "[CYCLE #%s | t+%sms] LLM metrics — duration=%.2fs ttft=%.2fs tokens=%s",
            cycle.cycle_id,
            cycle.elapsed_ms(),
            duration,
            ttft,
            tokens,
        )

    def on_llm_completed(self, response: str) -> dict[str, Any]:
        cycle = self._ensure_cycle()
        cycle.assistant_text = response
        cycle.llm_completed_at = time.perf_counter()
        logger.info(
            '[CYCLE #%s | t+%sms] LLM reply: "%s"',
            cycle.cycle_id,
            cycle.elapsed_ms(cycle.llm_completed_at),
            response,
        )
        return {"cycle_id": cycle.cycle_id, "response": response}

    def on_tts_started(self, source: str) -> dict[str, Any]:
        cycle = self._ensure_cycle()
        cycle.tts_started_at = time.perf_counter()
        cycle.tts_pending = True
        logger.info(
            "[CYCLE #%s | t+%sms] TTS started (source=%s)",
            cycle.cycle_id,
            cycle.elapsed_ms(cycle.tts_started_at),
            source,
        )
        return {"cycle_id": cycle.cycle_id, "source": source}

    def on_tts_metrics(
        self,
        *,
        duration: float,
        ttfb: float,
        audio_duration: float,
    ) -> dict[str, Any] | None:
        cycle = self._ensure_cycle()
        cycle.tts_completed_at = time.perf_counter()
        cycle.metrics.tts_duration_s = round(duration, 3)
        cycle.metrics.tts_ttfb_s = round(ttfb, 3)
        cycle.metrics.tts_audio_duration_s = round(audio_duration, 3)
        cycle.tts_pending = False
        logger.info(
            "[CYCLE #%s | t+%sms] TTS complete — duration=%.2fs ttfb=%.2fs audio=%.2fs",
            cycle.cycle_id,
            cycle.elapsed_ms(cycle.tts_completed_at),
            duration,
            ttfb,
            audio_duration,
        )
        return self._finalize_cycle()

    def on_eou_metrics(
        self,
        *,
        end_of_utterance_delay: float,
        transcription_delay: float,
    ) -> None:
        cycle = self._ensure_cycle()
        cycle.metrics.eou_delay_s = round(end_of_utterance_delay, 3)
        cycle.metrics.transcription_delay_s = round(transcription_delay, 3)
        logger.info(
            "[CYCLE #%s | t+%sms] End-of-utterance — silence=%.2fs transcription_delay=%.2fs",
            cycle.cycle_id,
            cycle.elapsed_ms(),
            end_of_utterance_delay,
            transcription_delay,
        )

    def _finalize_cycle(self) -> dict[str, Any] | None:
        cycle = self._active
        if cycle is None or cycle.completed:
            return None

        cycle.completed = True
        total_ms = cycle.elapsed_ms(
            cycle.tts_completed_at or cycle.llm_completed_at or time.perf_counter()
        )
        m = cycle.metrics

        summary_lines = [
            f"========== PIPELINE CYCLE #{cycle.cycle_id} COMPLETE (total={total_ms}ms) ==========",
            f"  User said:     \"{cycle.user_text or cycle.final_transcript or cycle.last_partial}\"",
            f"  Agent replied: \"{cycle.assistant_text}\"",
            "  --- timings ---",
        ]

        if cycle.first_partial_at is not None:
            summary_lines.append(
                f"  STT first partial:  t+{cycle.elapsed_ms(cycle.first_partial_at)}ms ({cycle.partial_count} partials)"
            )
        if cycle.final_stt_at is not None:
            summary_lines.append(f"  STT final:          t+{cycle.elapsed_ms(cycle.final_stt_at)}ms")
        if cycle.turn_detected_at is not None:
            summary_lines.append(f"  Turn detected:      t+{cycle.elapsed_ms(cycle.turn_detected_at)}ms")
        if m.stt_audio_duration_s is not None:
            summary_lines.append(f"  STT audio:          {m.stt_audio_duration_s}s (streamed={m.stt_streamed})")
        if cycle.llm_started_at is not None:
            summary_lines.append(f"  LLM started:        t+{cycle.elapsed_ms(cycle.llm_started_at)}ms")
        if m.llm_duration_s is not None:
            summary_lines.append(
                f"  LLM inference:      {m.llm_duration_s}s (ttft={m.llm_ttft_s}s, tokens={m.llm_tokens})"
            )
        if cycle.tts_started_at is not None:
            summary_lines.append(f"  TTS started:        t+{cycle.elapsed_ms(cycle.tts_started_at)}ms")
        if m.tts_duration_s is not None:
            summary_lines.append(
                f"  TTS synthesis:      {m.tts_duration_s}s (ttfb={m.tts_ttfb_s}s, audio={m.tts_audio_duration_s}s)"
            )
        if m.eou_delay_s is not None:
            summary_lines.append(
                f"  Endpointing:        silence={m.eou_delay_s}s transcript_delay={m.transcription_delay_s}s"
            )
        summary_lines.append("=" * 62)

        summary = "\n".join(summary_lines)
        logger.info("\n%s", summary)

        payload = {
            "cycle_id": cycle.cycle_id,
            "total_ms": total_ms,
            "user_text": cycle.user_text or cycle.final_transcript,
            "assistant_text": cycle.assistant_text,
            "partial_count": cycle.partial_count,
            "stt_audio_duration_s": m.stt_audio_duration_s,
            "llm_duration_s": m.llm_duration_s,
            "llm_ttft_s": m.llm_ttft_s,
            "llm_tokens": m.llm_tokens,
            "tts_duration_s": m.tts_duration_s,
            "tts_ttfb_s": m.tts_ttfb_s,
            "summary": summary,
        }

        self._active = None
        return payload

    def force_complete_if_stale(self) -> dict[str, Any] | None:
        """Complete cycle without TTS if agent finished without metrics (edge case)."""
        cycle = self._active
        if cycle is None or cycle.completed:
            return None
        if cycle.assistant_text and not cycle.tts_pending:
            return self._finalize_cycle()
        return None
