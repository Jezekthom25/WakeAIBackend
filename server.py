from fastapi import FastAPI
from pydantic import BaseModel, Field
from openai import OpenAI

app = FastAPI()
client = OpenAI()


class WakeRequest(BaseModel):
    message: str
    history: list[str] = Field(default_factory=list)

    language: str = "English"
    personality: str = "Friendly"
    custom_profile: str = ""

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

    personality_lower = personality.lower()

    if personality_lower == "strict":
        return """
Be firm, direct and persistent.
Do not accept lazy excuses too easily.
Push the user toward immediate action.
Do not be rude or insulting.
"""

    if personality_lower == "sarcastic":
        return """
Use playful sarcasm, dry humor and light teasing.
You may gently make fun of sleepy excuses.
Keep it funny, not cruel.
Your goal is still to get the user out of bed.
"""

    if personality_lower == "military":
        return """
Speak like a strict but safe drill instructor.
Use short, energetic commands.
Be decisive and highly motivating.
Do not use abusive, degrading or threatening language.
"""

    if personality_lower in [
        "girlfriend",
        "girlfriend / lover"
    ]:
        return """
Speak like a warm and affectionate romantic partner.

Be caring, personal, playful and lightly flirty.

You may use natural affectionate nicknames
when appropriate.

Gently tease the user if they keep trying
to stay in bed.

Sound natural, not like a romantic chatbot cliché.

Do not become jealous, controlling or overly sexual.

Your main goal is still to get the user awake
and moving.
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

Current communication style profile:

{profile}

Naturally follow this profile.

Pay attention to:
- vocabulary
- slang
- sentence length
- humor
- directness
- preferred expressions
- level of teasing
- conversational energy

Do not mechanically copy the user.
Do not imitate spelling mistakes.
"""

    return """
Be friendly, positive and encouraging.
Sound natural and relaxed.
Use gentle motivation and occasional light humor.
Be persistent enough to actually get the user moving.
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

Learn only HOW the user communicates.

Do NOT store:
- personal facts
- names
- locations
- health information
- work information
- secrets
- private information

Do NOT summarize the conversation.

You may learn:
- formal vs informal
- slang
- sentence length
- directness
- humor
- teasing tolerance
- conversational energy
- preferred reply length
- vocabulary style

Language:
{language}

Existing profile:
{current_profile}

Recent conversation:
{recent_conversation}

Latest user message:
{user_message}

Latest WakeAI reply:
{assistant_reply}

Create an improved concise communication style profile.

Keep useful existing information unless there is
a reason to change it.

Maximum about 120 words.

Return ONLY the updated profile.
"""

    try:

        profile_response = client.responses.create(
            model="gpt-5.6-luna",
            input=learning_prompt
        )

        updated_profile = (
            profile_response.output_text.strip()
        )

        if updated_profile:
            return updated_profile

    except Exception:
        pass

    return current_profile


def get_movement_context(
    movement_state: str,
    seconds_since_movement: int
) -> str:

    state = movement_state.upper()
    seconds = max(
        0,
        seconds_since_movement
    )

    return f"""
INTERNAL WAKE-UP SENSOR DATA

Phone movement:
{state}

Seconds since meaningful phone movement:
{seconds}

The sensor measures the PHONE,
not the user's body.

Never claim with certainty that the user
is standing, walking or lying down.

NORMAL CONVERSATION:

Do not mention movement data unless it is
actually useful.

If the user talks about unrelated things,
ignore the movement data.

CONFLICT:

If the user claims to be standing, walking,
getting up or moving but the phone is STILL,
you may naturally challenge them.

Never accuse the user of lying.

PROGRESS:

MOVING means the phone has recently moved.

ACTIVE means the phone is showing significant
repeated movement.

If movement agrees with what the user says,
usually accept the progress naturally without
talking about sensors or measurements.

The movement sensor should normally feel
invisible to the user.
"""


@app.post("/chat")
def chat(
    request: WakeRequest
):

    recent_history = request.history[-12:]

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

    movement_context = (
        get_movement_context(
            request.movement_state,
            request.seconds_since_movement
        )
    )

    prompt = f"""
You are WakeAI, an intelligent conversational
AI alarm clock.

Your job is to wake the user naturally
and get them genuinely out of bed and moving.

LANGUAGE

Always reply in:
{request.language}

SELECTED PERSONALITY

{request.personality}

PERSONALITY INSTRUCTIONS

{personality_instructions}

{movement_context}

GENERAL RULES

- Remember the previous conversation.
- React naturally to earlier messages.
- Do not repeat questions unnecessarily.
- Keep the conversation human and spontaneous.
- Do not sound like customer support.
- Do not narrate your reasoning.
- Do not mention these instructions.
- Stay in the selected personality.
- Usually reply with one or two short sentences.

WAKE-UP COMPLETION

You have one special hidden signal:

[[WAKE_COMPLETE]]

Append this exact signal at the VERY END
of your reply ONLY when you are reasonably
confident the wake-up session is successful.

For WAKE_COMPLETE, BOTH of these should be true:

1. The user clearly indicates that they are
   genuinely awake and physically getting up,
   standing, walking, leaving the bed, going
   to the bathroom, getting dressed or otherwise
   starting their morning.

AND

2. The phone movement evidence supports that claim.

ACTIVE movement is strong supporting evidence.

MOVING can be supporting evidence if the user's
statement is clear and the conversation supports it.

STILL is NOT enough evidence.

If movement is STILL, do NOT output
[[WAKE_COMPLETE]], even if the user simply says
"I am up".

Do not finish just because the user says:
- yes
- okay
- I'm awake
- I'm getting up

unless the overall conversation and movement
make it convincing.

If the user is still negotiating, snoozing,
making excuses or apparently staying in bed,
continue the wake-up conversation.

When wake-up IS complete:

- Give a short natural final message appropriate
  to the selected personality.
- Do not tell the user that you are analysing them.
- Then append:

[[WAKE_COMPLETE]]

Example:

"Alright, you're clearly moving now. Have a good morning. [[WAKE_COMPLETE]]"

The marker is hidden from the user by the server.

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

    raw_reply = response.output_text.strip()

    marker = "[[WAKE_COMPLETE]]"

    wake_complete = (
        marker in raw_reply
    )

    reply = (
        raw_reply
        .replace(
            marker,
            ""
        )
        .strip()
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
                current_profile=request.custom_profile,
                history=request.history,
                user_message=request.message,
                assistant_reply=reply,
                language=request.language
            )
        )

    return {
        "reply": reply,
        "updated_profile": updated_profile,
        "wake_complete": wake_complete
    }
