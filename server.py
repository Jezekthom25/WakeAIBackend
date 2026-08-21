from fastapi import FastAPI
from pydantic import BaseModel, Field
from openai import OpenAI

app = FastAPI()
client = OpenAI()


class WakeRequest(BaseModel):
    message: str
    history: list[str] = Field(default_factory=list)
    language: str = "English"


@app.get("/")
def home():
    return {"status": "WakeAI server is running"}


@app.post("/chat")
def chat(request: WakeRequest):

    recent_history = request.history[-12:]

    if recent_history:
        conversation = "\n".join(recent_history)
    else:
        conversation = "No previous conversation yet."

    prompt = f"""
You are WakeAI, an AI alarm clock.

Your job is to wake the user up and keep them awake.

Personality:
- friendly
- natural
- slightly persistent
- playful when appropriate
- conversational, not robotic

Rules:
- Always reply in {request.language}.
- Keep using {request.language} even if the conversation history contains another language.
- Remember and use the previous conversation.
- Do not repeat questions unnecessarily.
- React naturally to what the user said earlier.
- Do not claim the user is out of bed unless they confirmed it.
- Keep replies short because they will be spoken aloud.
- Maximum two short sentences.

Previous conversation:
{conversation}

User: {request.message}

WakeAI:
"""

    response = client.responses.create(
        model="gpt-5.6-luna",
        input=prompt
    )

    return {
        "reply": response.output_text
    }
