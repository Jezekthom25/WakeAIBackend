from fastapi import FastAPI
from pydantic import BaseModel
from openai import OpenAI

app = FastAPI()
client = OpenAI()


class WakeRequest(BaseModel):
    message: str


@app.get("/")
def home():
    return {"status": "WakeAI server is running"}


@app.post("/chat")
def chat(request: WakeRequest):

    response = client.responses.create(
        model="gpt-5.6-luna",
        input=(
            "You are WakeAI, an AI alarm clock. "
            "Your job is to wake the user up and keep them awake. "
            "Be friendly, natural, slightly persistent and sometimes playful. "
            "Keep your reply short because it will be spoken aloud. "
            "Reply with no more than two short sentences.\n\n"
            f"User: {request.message}"
        )
    )

    return {
        "reply": response.output_text
    }