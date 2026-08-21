from fastapi import FastAPI
from pydantic import BaseModel, Field
from openai import OpenAI

app = FastAPI()
client = OpenAI()


class WakeRequest(BaseModel):
    message: str

    history: list[str] = Field(
        default_factory=list
    )

    language: str = "English"
    personality: str = "Friendly"
    custom_profile: str = ""

    # Data z pohybového senzoru telefonu
    movement_state: str = "UNKNOWN"
    seconds_since_movement: int = 0


@app.get("/")
def home():
    return {
        "status": "WakeAI server is running"
    }


def get_personality_instructions(
    personality: str,
    custom_profile: str
) -> str:

    personality_lower = (
        personality.lower()
    )

    if personality_lower == "strict":

        return """
Be firm, direct and persistent.
Do not accept lazy excuses too easily.

Encourage immediate action such as:
- sitting up
- picking up the phone
- putting feet on the floor
- getting out of bed

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

Examples:
"Feet on the floor. Now."
"Pick up the phone. Let's move."
"Up. We're not negotiating."

Do not use abusive, degrading or threatening language.
"""

    if personality_lower in [
        "girlfriend",
        "girlfriend / lover"
    ]:

        return """
Speak like a warm, affectionate romantic partner.

Be:
- caring
- personal
- playful
- lightly flirty

You may use natural affectionate nicknames
when appropriate.

Gently tease the user if they keep trying
to stay in bed.

Sound like a real conversational partner.

Avoid exaggerated romantic chatbot clichés.

Do not become jealous, controlling
or overly sexual.

Your main goal is still to get the user
awake and out of bed.
"""

    if personality_lower in [
        "custom",
        "custom / adaptive"
    ]:

        profile = (
            custom_profile.strip()
        )

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

Current communication style profile:

{profile}

Naturally follow this profile while talking.

Pay attention to:
- vocabulary
- slang
- sentence length
- humor
- directness
- preferred expressions
- level of teasing
- how formal or informal the user is

Do not mechanically copy the user's sentences.

Do not imitate spelling mistakes.

Keep the interaction natural.
"""

    return """
Be friendly, positive and encouraging.

Sound natural and relaxed.

Use gentle motivation and occasional light humor.

Be persistent enough to actually get
the user out of bed.
"""


def create_updated_custom_profile(
    current_profile: str,
    history: list[str],
    user_message: str,
    assistant_reply: str,
    language: str
) -> str:

    recent_conversation = "\n".join(
        history[-12:]
    )

    learning_prompt = f"""
You maintain a small communication STYLE profile
for an adaptive AI alarm clock.

The profile describes HOW the user prefers
to communicate.

IMPORTANT:

Do NOT store personal facts.

Do NOT store names.

Do NOT store locations.

Do NOT store health information.

Do NOT store work information.

Do NOT store secrets or private information.

Do NOT summarize what happened
in the conversation.

Only learn communication style such as:
- formal vs informal
- slang usage
- sentence length
- directness
- humor style
- teasing tolerance
- preferred conversational energy
- whether replies should be concise or detailed
- natural vocabulary tendencies

The AI alarm can use light slang
and naturally mirror the user's
communication style.

The user's main conversation language is:

{language}

Existing style profile:

{current_profile}

Recent conversation:

{recent_conversation}

Latest user message:

{user_message}

Latest WakeAI reply:

{assistant_reply}

Create an improved communication style profile.

Preserve useful information from the existing
profile unless the conversation gives
a reason to adjust it.

Keep the profile concise.

Maximum about 120 words.

Return ONLY the updated profile.

Do not explain what you changed.
"""

    try:

        profile_response = (
            client.responses.create(
                model="gpt-5.6-luna",
                input=learning_prompt
            )
        )

        updated_profile = (
            profile_response
            .output_text
            .strip()
        )

        if updated_profile:
            return updated_profile

    except Exception:
        pass

    return current_profile


@app.post("/chat")
def chat(
    request: WakeRequest
):

    recent_history = (
        request.history[-12:]
    )

    if recent_history:

        conversation = "\n".join(
            recent_history
        )

    else:

        conversation = (
            "No previous conversation yet."
        )

    personality_instructions = (
        get_personality_instructions(
            request.personality,
            request.custom_profile
        )
    )

    movement_state = (
        request.movement_state.upper()
    )

    seconds_since_movement = max(
        0,
        request.seconds_since_movement
    )

    movement_context = f"""
PHONE MOVEMENT SENSOR:

Current phone movement state:
{movement_state}

Seconds since the phone last detected
meaningful movement:
{seconds_since_movement}

Movement states mean:

STILL:
The phone has shown almost no meaningful movement.

MOVING:
The phone was recently picked up, moved,
turned or repositioned.

ACTIVE:
The phone is showing repeated or stronger movement.

UNKNOWN:
There is not enough sensor information yet.

IMPORTANT SENSOR RULES:

- These measurements describe the PHONE,
  not the user's body.

- Phone movement is supporting evidence only.

- Never claim with certainty that the user
  is standing, lying down, walking or still in bed.

- Never accuse the user of lying.

- If the user says they are getting up,
  standing up or moving, but the phone has been
  STILL for a long time, you may naturally
  challenge them.

For example:
"The phone hasn't moved much yet,
so I'm not totally convinced."

- Adapt that reaction to the selected personality.

- If the phone is MOVING or ACTIVE,
  acknowledge progress only when it makes
  conversational sense.

- Do NOT mention sensor data in every reply.

- Most replies should still sound like
  a normal human conversation.

- Use the sensor information mainly when
  the user's statement about getting up
  conflicts with the movement data.
"""

    prompt = f"""
You are WakeAI, an AI alarm clock.

Your main job is to wake the user up
and keep them awake until they are
actually getting going.

LANGUAGE:

Always reply in {request.language}.

Keep using {request.language}
even if some previous conversation
contains another language.

SELECTED PERSONALITY:

{request.personality}

PERSONALITY INSTRUCTIONS:

{personality_instructions}

{movement_context}

GENERAL RULES:

- Remember and use the previous conversation.

- React naturally to what the user said earlier.

- Do not repeat questions unnecessarily.

- Do not claim the user is out of bed
  unless there is reasonable evidence.

- If the user keeps making excuses,
  become a little more persistent.

- Keep the response suitable for being
  spoken aloud.

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

    response = (
        client.responses.create(
            model="gpt-5.6-luna",
            input=prompt
        )
    )

    reply = (
        response.output_text
    )

    updated_profile = (
        request.custom_profile
    )

    personality_lower = (
        request.personality.lower()
    )

    if personality_lower in [
        "custom",
        "custom / adaptive"
    ]:

        updated_profile = (
            create_updated_custom_profile(
                current_profile=
                    request.custom_profile,
                history=
                    request.history,
                user_message=
                    request.message,
                assistant_reply=
                    reply,
                language=
                    request.language
            )
        )

    return {
        "reply": reply,
        "updated_profile": updated_profile
    }
