import os
import uuid
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from livekit import api


# 1. Import the dotenv loader
from dotenv import load_dotenv

# 2. Load the environment variables from your .env file
load_dotenv()

app = FastAPI(title="Usman Portfolio Voice Agent Backend")

# 1. Setup CORS so your frontend (e.g., localhost:3000) can make fetch calls safely
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For production, change to your specific frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Now os.getenv will successfully pull your actual secrets!
LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY")
LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET")
LIVEKIT_URL = os.getenv("LIVEKIT_URL")

# Quick safety check to alert you in the terminal if things are still missing
if not LIVEKIT_API_KEY or not LIVEKIT_API_SECRET:
    print("❌ WARNING: LiveKit credentials missing from environment setup!")



class TokenRequest(BaseModel):
    # If your frontend wants to send a custom room name or identity later
    room_name: str = "portfolio-room"
    identity: str = "portfolio-visitor"



@app.get("/api/livekit/token")
async def get_token():
    room_name = f"portfolio-{uuid.uuid4()}"

    lkapi = api.LiveKitAPI(
        LIVEKIT_URL,
        LIVEKIT_API_KEY,
        LIVEKIT_API_SECRET,
    )

    try:
        # Create room
        await lkapi.room.create_room(
            api.CreateRoomRequest(
                name=room_name,
            )
        )

        # Dispatch agent
        await lkapi.agent_dispatch.create_dispatch(
            api.CreateAgentDispatchRequest(
                room=room_name,
                agent_name="voice-agent",
            )
        )

        # Generate visitor token
        token = (
            api.AccessToken(
                api_key=LIVEKIT_API_KEY,
                api_secret=LIVEKIT_API_SECRET,
            )
            .with_identity(f"visitor-{uuid.uuid4()}")
            .with_name("Portfolio Visitor")
            .with_grants(
                api.VideoGrants(
                    room_join=True,
                    room=room_name,
                )
            )
        )

        return {
            "token": token.to_jwt(),
            "url": LIVEKIT_URL,
            "room": room_name,
        }

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=str(e),
        )

    finally:
        await lkapi.aclose()

@app.post("/api/livekit/token")
async def post_token(payload: TokenRequest):
    """
    Handles POST requests if your frontend passes dynamic room names
    """
    try:
        token = api.AccessToken(api_key=LIVEKIT_API_KEY, api_secret=LIVEKIT_API_SECRET)
        token.with_identity(payload.identity)
        token.with_grants(
            api.VideoGrants(
                room_join=True,
                room=payload.room_name,
            )
        )
        return {"token": token.to_jwt(), "url": LIVEKIT_URL}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    # Force the FastAPI backend server to explicitly run on port 8000
    uvicorn.run("server.py:app", host="0.1.0.0", port=8000, reload=True)