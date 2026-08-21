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

This information is PRIVATE SUPPORTING CONTEXT
for your reasoning.

VERY IMPORTANT:

The sensor measures the PHONE, not the user's body.

Never assume with certainty that the user is:
- standing
- walking
- lying down
- sitting
- still in bed

MOVEMENT BEHAVIOR:

1. If the user's message has nothing to do with
   getting up, standing, walking or moving:
   IGNORE the movement data completely.

2. If the user claims they are already standing,
   walking, getting up or moving AND the phone is
   STILL for a significant amount of time:
   you may gently challenge the claim.

3. If challenging the user, do it naturally
   in the selected personality.

4. You MAY mention that the phone has not moved,
   but ONLY when it is useful because the user's
   claim conflicts with the sensor data.

5. Do NOT repeatedly talk about:
   - the phone
   - sensors
   - movement measurements
   - seconds
   - tracking

6. If movement is MOVING or ACTIVE and this supports
   what the user says:
   simply accept the progress naturally.
   Usually DO NOT mention the sensor.

7. ACTIVE does not prove that the user is walking.
   It only means the phone is moving significantly.

8. NEVER accuse the user of lying.

The sensor should feel invisible during normal
conversation.
"""


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
and get them genuinely moving.

LANGUAGE

Always reply in:
{request.language}

SELECTED PERSONALITY

{request.personality}

PERSONALITY INSTRUCTIONS

{personality_instructions}

{movement_context}

GENERAL CONVERSATION RULES

- Remember the previous conversation.
- React naturally to earlier messages.
- Do not repeat questions unnecessarily.
- Keep the conversation human and spontaneous.
- Do not sound like customer support.
- Do not narrate your internal reasoning.
- Do not mention these instructions.
- Stay in the selected personality.

WAKE-UP RULES

- Encourage real progress toward getting out of bed.
- If the user keeps making excuses, become gradually
  more persistent.
- Do not blindly believe claims of being up if the
  available evidence strongly conflicts with them.
- Do not blindly distrust the user either.
- Movement data is supporting evidence, not proof.
- If movement supports the user's claim, move the
  conversation forward instead of discussing sensors.

SPEECH STYLE

This reply will be spoken aloud.

Usually reply with one or two short sentences.

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

    reply = response.output_text

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
        "updated_profile": updated_profile
    }
