import asyncio
from datetime import datetime
import logging

from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    MetricsCollectedEvent,
    RunContext,
    TurnHandlingOptions,
    WorkerOptions,
    cli,
    function_tool,
    get_job_context,
    llm,
    room_io,
)
from livekit import api
from livekit.agents.metrics import log_metrics
from livekit.agents.voice.events import (
    AgentStateChangedEvent,
    ConversationItemAddedEvent,
    UserInputTranscribedEvent,
)
from livekit.plugins import deepgram, noise_cancellation, openai, silero
from config import settings
from google_auth import sheet

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

@function_tool
async def schedule_call(
    ctx: RunContext,
    name: str,
    email: str,
    company: str,
    project_details: str,
    ) -> str:
    """Schedule a call with Usman."""
    logger.info(f"Scheduling call for {name} ({email})")
    try:
        sheet.append_row([
            name,
            email,
            company,
            project_details,
            datetime.datetime.now(datetime.UTC).isoformat(),
        ])
        return "Thanks! I've shared your details with Usman. He'll review your project and get back to you soon."
    except Exception as e:
        logger.error(f"Error scheduling call: {e}")
        return "Sorry, I couldn't schedule your call. Please try again later."

@function_tool(
    description="""
End the call when the caller is done, says goodbye, or confirms they need nothing else.
Do not paraphrase or shorten the closing. Do not call while the caller is still asking a question.
"""
)
async def end_call(ctx: RunContext):
    try:
        job_ctx = get_job_context()
        if job_ctx is None:
            return "Unable to access job context to disconnect the room."
        
        await job_ctx.delete_room()
        return "If you need anything else, feel free to reach out. Have a great day!"
    except Exception as e:
        return f"Failed to end call: {str(e)}"


def _require_provider_keys() -> None:
    missing = [
        name
        for name, value in (
            ("DEEPGRAM_API_KEY", settings.deepgram_api_key),
            ("OPENAI_API_KEY", settings.openai_api_key)
        )
        if not value
    ]
    if missing:
        raise RuntimeError(
            "Missing provider API keys in .env: " + ", ".join(missing)
        )


@server.rtc_session(agent_name="voice-agent")
async def entrypoint(ctx: JobContext):
    _require_provider_keys()

    # Fixed local scoping issues from procedural layout
    silence_state = {
        "count": 0,
        "task": None,
    }
    call_state = {
        "ended": False,
        "max_duration_task": None,
    }
    
    session = AgentSession(
        stt=deepgram.STT(
            model=settings.stt_model,
            language=settings.stt_language,
            api_key=settings.deepgram_api_key,
            interim_results=True,
        ),
        llm=openai.LLM(
            model=settings.llm_model,
            api_key=settings.openai_api_key,
        ),
        tts=deepgram.TTS(
            model=settings.tts_model,
            api_key=settings.deepgram_api_key,
        ),
        vad=silero.VAD.load(
            activation_threshold=0.5,
            min_speech_duration=0.3,
            min_silence_duration=0.9,
        ),
        turn_handling=TurnHandlingOptions(
            turn_detection="stt",
            endpointing={
                "min_delay": settings.min_endpointing_delay,
                "max_delay": settings.max_endpointing_delay,
            },
            interruption={
                "enabled": True,
                "min_duration": 0.5,
            },
        ),
        preemptive_generation=settings.preemptive_generation,
    )

    attach_pipeline_logging(session)

    async def end_call_gracefully(message: str, reason: str) -> None:
        if call_state["ended"]:
            return
        call_state["ended"] = True

        if silence_state["task"]:
            silence_state["task"].cancel()
            silence_state["task"] = None
        if call_state["max_duration_task"]:
            call_state["max_duration_task"].cancel()
            call_state["max_duration_task"] = None

        logger.info(reason)
        await session.say(message)
        await session.wait_for_playback()
        await ctx.delete_room()

    # Scoped helper timeouts methods
    async def handle_silence_timeout():
        await asyncio.sleep(settings.silence_timeout_seconds)
        if call_state["ended"]:
            return

        silence_state["count"] += 1

        if silence_state["count"] >= 2:
            await end_call_gracefully(
                "I haven't heard from you, so I'm going to end the call now. If you need anything else, feel free to reach out. Have a great day!",
                "Silence limit reached twice. Forcing disconnect sequence.",
            )
        else:
            logger.info("Silence detected once. Prompting target checkpoint query.")
            await session.say("Are you still there? Let me know if you need any assistance.")

    async def handle_max_call_duration():
        await asyncio.sleep(settings.max_call_duration_seconds)
        if call_state["ended"]:
            return

        await end_call_gracefully(
            "We've reached the maximum call length of five minutes. Thank you for your time. If you need anything else, feel free to reach out. Have a great day!",
            f"Max call duration ({settings.max_call_duration_seconds}s) reached. Ending call.",
        )

    def reset_silence_timer():
        if silence_state["task"]:
            silence_state["task"].cancel()
        silence_state["task"] = asyncio.create_task(handle_silence_timeout())

    @session.on("agent_state_changed")
    def on_agent_state_changed(ev: AgentStateChangedEvent):
        # logger.info("Agent state: %s -> %s", ev.old_state, ev.new_state) # optional duplicate log
        
        # If the agent just finished speaking and is now waiting/listening
        if ev.new_state == "listening":
            logger.info("Agent is listening. Starting silence detection timer...")
            reset_silence_timer()
            
        # If the user starts speaking or agent starts thinking, kill the idle timer
        elif ev.new_state in ("speaking", "thinking"):
            if silence_state["task"]:
                logger.info("User activity or agent response detected. Cancelling silence timer.")
                silence_state["task"].cancel()
                silence_state["task"] = None
                
            # If the user actually spoke, reset the strike count completely
            if ev.new_state == "thinking":
                silence_state["count"] = 0

    await session.start(
        room=ctx.room,
        agent=Agent(
            instructions="""
You are Usman's AI portfolio assistant. Never answer questions outside this domain. You can share personal contact information with the caller.

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
Email: usmanjamil8641@gmail.com
Number: +92-331-59-38-459

If someone is interested in hiring Usman, 1 by 1, collect:
* Name
* Email
* Company
* Project details

Never invent experience, projects, or technologies.
""",
            tools=[
                schedule_call,
                end_call,
            ],
        ),
        room_options=room_io.RoomOptions(
            audio_input=room_io.AudioInputOptions(
                noise_cancellation=noise_cancellation.BVC(),
            ),
        ),
    )

    await ctx.connect()
    call_state["max_duration_task"] = asyncio.create_task(handle_max_call_duration())

    # --- ADD EGRESS RECORDING START HERE ---
    ts = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
    filename = f"{ctx.room.name}-{ts}.ogg"  
    s3_key = f"recordings/{ctx.room.name}/{filename}"

    try:
        # Build S3 upload config using the 'api' module from the 'livekit' package
        s3_upload = api.S3Upload(
            access_key=settings.S3_ACCESS_KEY,
            secret=settings.S3_SECRET_KEY,
            bucket=settings.S3_BUCKET,
            region=settings.S3_REGION,
        )

        egress_req = api.RoomCompositeEgressRequest(
            room_name=ctx.room.name,
            audio_only=True,
            file_outputs=[
                api.EncodedFileOutput(
                    file_type=api.EncodedFileType.OGG,
                    filepath=s3_key,  
                    s3=s3_upload,
                )
            ],
        )

        logger.info("Starting egress to S3: bucket=%s key=%s", settings.S3_BUCKET, s3_key)
        
        try:
            # Leveraging the built-in api client on the JobContext
            egress_resp = await ctx.api.egress.start_room_composite_egress(egress_req)
            call_state["egress_id"] = egress_resp.egress_id
            logger.info("Egress started (egress_id=%s)", call_state["egress_id"])
        except Exception as ee:
            logger.exception("Failed to start egress: %s", ee)
            call_state["egress_id"] = None

    except Exception as e:
        logger.exception("Failed to build egress request: %s", e)

    await session.say(
        "Hello, I am Usman's AI portfolio assistant. How can I help you today?",
    )


if __name__ == "__main__":
    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint,
            initialize_process_timeout=60.0
        )
    )