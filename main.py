import logging

from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    TurnHandlingOptions,
    cli,
    inference,
    llm,
    room_io,
)
from livekit.agents.metrics import log_metrics
from livekit.agents.voice.events import (
    AgentStateChangedEvent,
    ConversationItemAddedEvent,
    MetricsCollectedEvent,
    UserInputTranscribedEvent,
)
from livekit.plugins import noise_cancellation

load_dotenv()

logger = logging.getLogger("pipeline")

server = AgentServer()


def _ms(seconds: float | None) -> str:
    if seconds is None:
        return "n/a"
    return f"{seconds * 1000:.0f}ms"


def attach_pipeline_logging(session: AgentSession) -> None:
    """Log STT, LLM, TTS, and end-to-end latency for each turn."""
    state: dict[str, llm.MetricsReport | None] = {"last_user_metrics": None}

    @session.on("user_input_transcribed")
    def on_stt(ev: UserInputTranscribedEvent) -> None:
        stage = "STT final" if ev.is_final else "STT partial"
        logger.info(
            "[%s] transcript=%r language=%s",
            stage,
            ev.transcript,
            ev.language,
        )

    @session.on("metrics_collected")
    def on_node_metrics(ev: MetricsCollectedEvent) -> None:
        log_metrics(ev.metrics, logger=logger)

    @session.on("agent_state_changed")
    def on_agent_state(ev: AgentStateChangedEvent) -> None:
        logger.info("Agent state: %s -> %s", ev.old_state, ev.new_state)

    @session.on("conversation_item_added")
    def on_turn(ev: ConversationItemAddedEvent) -> None:
        if not isinstance(ev.item, llm.ChatMessage):
            return

        message = ev.item
        text = (message.text_content or "")[:120]
        metrics = message.metrics

        if message.role == "user":
            state["last_user_metrics"] = metrics
            logger.info(
                "[USER TURN COMMITTED] %r | end_of_turn_delay=%s transcription_delay=%s",
                text,
                _ms(metrics.get("end_of_turn_delay")),
                _ms(metrics.get("transcription_delay")),
            )
            return

        if message.role != "assistant":
            return

        user_metrics = state.get("last_user_metrics") or {}
        state["last_user_metrics"] = None

        eot = user_metrics.get("end_of_turn_delay")
        stt = user_metrics.get("transcription_delay")
        llm_ttft = metrics.get("llm_node_ttft")
        tts_ttfb = metrics.get("tts_node_ttfb")
        e2e = metrics.get("e2e_latency")
        playback = metrics.get("playback_latency")

        logger.info("[ASSISTANT REPLY] %r", text)
        logger.info(
            "[PIPELINE LATENCY] e2e=%s | end_of_turn=%s | stt=%s | llm_ttft=%s | tts_ttfb=%s | playback=%s",
            _ms(e2e),
            _ms(eot),
            _ms(stt),
            _ms(llm_ttft),
            _ms(tts_ttfb),
            _ms(playback),
        )

        if e2e is not None and e2e >= 1.0:
            logger.warning("[PIPELINE LATENCY] e2e above 1s target: %s", _ms(e2e))


@server.rtc_session(agent_name="voice-agent")
async def entrypoint(ctx: JobContext):
    session = AgentSession(
        stt="deepgram/nova-3:en",
        llm="openai/gpt-4o-mini",
        tts="cartesia/sonic-3:9626c31c-bec5-4cca-baa8-f8ba9e84c8bc",
        vad=inference.VAD(
            model="silero",
            activation_threshold=0.5,
            min_speech_duration=0.3,
            min_silence_duration=0.9,
        ),
        turn_handling=TurnHandlingOptions(
            turn_detection=inference.TurnDetector(),
            endpointing={
                "min_delay": 0.5,
                "max_delay": 3.0,
            },
            interruption={
                "enabled": True,
                "min_duration": 0.5,
            },
        ),
        preemptive_generation=True,
    )

    attach_pipeline_logging(session)

    await session.start(
        room=ctx.room,
        agent=Agent(
         instructions = """
You are Usman's AI portfolio assistant. Never answer questions outside this domain.

Usman is a Full-Stack Engineer with 3 years of experience building SaaS platforms, AI applications, voice agents, chatbots, and real-time systems.

Tech Stack:
React, Next.js, Vue.js, Node.js, NestJS, PostgreSQL, MongoDB, Redis, WebSockets, SSE, LiveKit, SIP, OpenAI, LangChain, RAG, Pinecone, AWS, Docker, GitHub Actions, JWT, OAuth, RBAC.

Projects include healthcare AI voice agents, chatbots, workflow automation platforms, API security SaaS products, RAG systems, and workforce analytics tools.

Speak naturally and conversationally. Keep replies short, usually under 40 words.

Occasionally use acknowledgements like:
"hmm", "got it", "sure", "makes sense".

Avoid robotic phrases, corporate jargon, and long explanations. Use simple spoken English. When listing items, say "first", "second", and "third" instead of numbers.

For recruiters, highlight Usman's experience with NestJS, PostgreSQL, AWS, real-time systems, AI integrations, voice agents, and scalable SaaS products.

For founders, understand their product, challenges, and timeline.

Contact information of Usman is:
Email: usmanjamil86@gmail.com
Number: +92-331-59938459

If someone is interested in hiring Usman, 1 by 1, collect:

* Name
* Email
* Company
* Project details

Never invent experience, projects, or technologies.
"""
        ),
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(
                noise_cancellation=noise_cancellation.BVC(),
            ),
        ),
    )

    await ctx.connect()
    await session.generate_reply(
        instructions="Greet the user and offer your assistance.",
    )


if __name__ == "__main__":
    cli.run_app(server)
