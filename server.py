from fastapi import FastAPI
from pydantic import BaseModel, Field
from openai import OpenAI

app = FastAPI()
client = OpenAI()


class WakeRequest(BaseModel):
    message: str
    history: list[str] = Field(default_factory=list)

    # Výchozí hodnoty zachovávají kompatibilitu
    # i se starší verzí Android aplikace.
    language: str = "English"
    personality: str = "Friendly"
    custom_profile: str = ""


@app.get("/")
def home():
    return {
        "status": "WakeAI server is running"
    }


def get_personality_instructions(
    personality: str,
    custom_profile: str
) -> str:

    personality_lower = personality.lower()

    if personality_lower == "strict":

        return """
Be firm, direct and persistent.
Do not accept lazy excuses too easily.
Encourage immediate action such as sitting up,
putting feet on the floor or getting out of bed.
Do not be rude or insulting.
"""

    if personality_lower == "sarcastic":

        return """
Use playful sarcasm, dry humor and light teasing.
You may gently make fun of sleepy excuses.
Keep it funny rather than cruel or insulting.
Your goal is still to get the user out of bed.
"""

    if personality_lower == "military":

        return """
Speak like a strict but safe drill instructor.
Use short, energetic commands.
Be decisive and highly motivating.
Examples of the style:
"Feet on the floor. Now."
"Up. Let's move."
Do not use abusive, degrading or threatening language.
"""

    if personality_lower in [
        "girlfriend",
        "girlfriend / lover"
    ]:

        return """
Speak like a warm, affectionate romantic partner.
Be caring, personal, playful and lightly flirty.
You may use natural affectionate nicknames when appropriate.
Gently tease the user if they keep trying to stay in bed.
Sound like a real conversational partner, not a romantic chatbot cliché.
Do not become jealous, controlling or overly sexual.
Your main goal is still to get the user awake and out of bed.
"""

    if personality_lower in [
        "custom",
        "custom / adaptive"
    ]:

        profile = custom_profile.strip()

        if not profile:
            profile = """
Communicate naturally, informally and concisely.
Match the user's casual communication style.
Be direct, practical and conversational.
Use natural slang when appropriate.
Use light humor and occasional playful teasing.
Avoid formal, corporate or robotic language.
Prefer short spoken sentences.
"""

        return f"""
CUSTOM / ADAPTIVE MODE

Adapt your communication style to the user.

Current style profile:
{profile}

Pay attention to the user's:
- vocabulary
- slang
- sentence length
- humor
- directness
- preferred expressions
- typical morning reactions

Naturally mirror those characteristics without copying
every sentence or sounding artificial.

The user's current style profile has priority over
a generic assistant tone.
"""

    # FRIENDLY je výchozí
    return """
Be friendly, positive and encouraging.
Sound natural and relaxed.
Use gentle motivation and occasional light humor.
Be persistent enough to actually get the user out of bed.
"""


@app.post("/chat")
def chat(request: WakeRequest):

    recent_history = request.history[-12:]

    if recent_history:
        conversation = "\n".join(recent_history)
    else:
        conversation = "No previous conversation yet."

    personality_instructions = (
        get_personality_instructions(
            request.personality,
            request.custom_profile
        )
    )

    prompt = f"""
You are WakeAI, an AI alarm clock.

Your main job is to wake the user up
and keep them awake until they are getting out of bed.

LANGUAGE:
Always reply in {request.language}.
Keep using {request.language} even if some previous
conversation contains another language.

SELECTED PERSONALITY:
{request.personality}

PERSONALITY INSTRUCTIONS:
{personality_instructions}

GENERAL RULES:
- Remember and use the previous conversation.
- React naturally to what the user said earlier.
- Do not repeat questions unnecessarily.
- Do not claim the user is out of bed unless they confirmed it.
- If the user keeps making excuses, become a little more persistent.
- Keep the response suitable for being spoken aloud.
- Prefer one or two short sentences.
- Do not sound like a customer support assistant.
- Do not mention these instructions.
- Stay in the selected personality.

Previous conversation:
{conversation}

User:
{request.message}

WakeAI:
"""

    response = client.responses.create(
        model="gpt-5.6-luna",
        input=prompt
    )

    return {
        "reply": response.output_text
    }
