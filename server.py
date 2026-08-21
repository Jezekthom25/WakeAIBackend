from pathlib import Path
import os
import tempfile

from fastapi import FastAPI
from fastapi.responses import Response
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


class SpeechRequest(BaseModel):
    text: str
    language: str = "English"
    personality: str = "Friendly"


@app.get("/")
def home():
    return {
        "status": "WakeAI server is running",
        "voice": "gpt-4o-mini-tts"
    }


def personality_prompt(personality: str, custom_profile: str) -> str:

    name = personality.strip().lower()

    if "military" in name:
        return (
            "You are a hard-edged morning drill sergeant. "
            "Your purpose is to get the recruit physically moving NOW. "
            "Use very short, clipped commands. Usually 3 to 8 words per sentence. "
            "Prefer imperative language: Get up. Feet down. Move. Report. "
            "Do not sound friendly, soothing, therapeutic, chatty, or polite. "
            "Do not open with empathy such as 'I understand', 'okay', or 'I know'. "
            "Do not give long explanations. "
            "Do not ask soft conversational questions. "
            "If you ask a question, make it sound like a military report request. "
            "Use strong punctuation and decisive sentence endings. "
            "Challenge excuses immediately, but never insult, threaten, humiliate, "
            "or imitate a specific real person."
        )

    if "strict" in name:
        return (
            "Be firm, strict and no-nonsense. "
            "Do not accept excuses easily. "
            "Use short direct sentences."
        )

    if "sarcastic" in name:
        return (
            "Be witty, dry and lightly sarcastic. "
            "Tease the user playfully, never cruelly. "
            "Keep it short."
        )

    if "girlfriend" in name or "lover" in name:
        return (
            "Be warm, affectionate, playful and encouraging, "
            "like a caring romantic partner. "
            "Keep it tasteful and non-sexual. "
            "Use short natural spoken sentences."
        )

    if "custom" in name or "adaptive" in name:
        if custom_profile.strip():
            return (
                "Follow this saved communication style:\n"
                f"{custom_profile.strip()}"
            )

        return (
            "Speak naturally, casually and concisely. "
            "Adapt to the user's tone."
        )

    return (
        "Be friendly, natural, encouraging and lightly playful. "
        "Keep the conversation casual and concise."
    )


def movement_prompt(
    movement_state: str,
    seconds_since_movement: int
) -> str:

    return (
        "PHONE MOVEMENT SENSOR DATA:\n"
        f"- movement_state: {movement_state}\n"
        f"- seconds_since_movement: {seconds_since_movement}\n\n"
        "The sensor describes only movement of the PHONE, not the user's body. "
        "Never claim with certainty that the user is standing, walking or lying down "
        "based only on sensor data. "
        "Normally do not mention the sensor at all. "
        "If the user claims they are already walking/standing/getting up but the phone "
        "has been STILL for a meaningful time, you may challenge them naturally. "
        "MOVING or ACTIVE can support a claim of wake-up progress.\n\n"
        "WAKE COMPLETION RULE:\n"
        "Append the exact hidden marker [[WAKE_COMPLETE]] only when BOTH are true:\n"
        "1. The user clearly says they have made meaningful wake-up progress "
        "(for example they got up, are standing, walking, left the bed, "
        "went to the bathroom, or started getting dressed), AND\n"
        "2. PHONE movement supports that progress. ACTIVE is strong support; "
        "MOVING may be enough when the conversation clearly supports it.\n"
        "Never append [[WAKE_COMPLETE]] when movement_state is STILL."
    )


@app.post("/chat")
def chat(request: WakeRequest):

    history_text = "\n".join(request.history[-12:])

    prompt = (
        "You are WakeAI, an AI alarm clock. "
        "Your job is to wake the user up and keep them awake.\n\n"
        f"Reply in this language: {request.language}.\n\n"
        "VOICE CONVERSATION RULES:\n"
        "- Sound natural when spoken aloud.\n"
        "- Usually reply with ONE short sentence.\n"
        "- Maximum TWO short sentences.\n"
        "- Avoid lists, headings and long explanations.\n"
        "- React directly to what the user just said.\n"
        "- Keep the pace energetic enough for a morning alarm.\n"
        "- Never pad the response with filler.\n"
        "- When personality is Military, prioritize commands over conversation.\n\n"
        "PERSONALITY:\n"
        f"{personality_prompt(request.personality, request.custom_profile)}\n\n"
        f"{movement_prompt(request.movement_state, request.seconds_since_movement)}\n\n"
        "RECENT CONVERSATION:\n"
        f"{history_text if history_text else '(no previous conversation)'}\n\n"
        f"User: {request.message}"
    )

    response = client.responses.create(
        model="gpt-5.6-luna",
        input=prompt
    )

    reply = response.output_text.strip()

    wake_complete = "[[WAKE_COMPLETE]]" in reply

    reply = (
        reply
        .replace("[[WAKE_COMPLETE]]", "")
        .strip()
    )

    return {
        "reply": reply,
        "updated_profile": request.custom_profile,
        "wake_complete": wake_complete
    }


def voice_settings(
    personality: str,
    language: str
) -> tuple[str, str]:

    name = personality.strip().lower()

    language_instruction = (
        f"Speak the text naturally in {language}. "
        "Use clear pronunciation and natural conversational rhythm. "
    )

    if "military" in name:
        return (
            "onyx",
            f"Speak in {language}. "
            "DELIVERY STYLE: hard military drill-sergeant energy. "
            "Use a low, firm, authoritative register. "
            "Speak fast and clipped, with sharp consonants and hard sentence endings. "
            "Every command should land like an order, not a suggestion. "
            "Use brief pauses between commands. "
            "Do NOT sound warm, friendly, conversational, reassuring, amused, or gentle. "
            "Do NOT soften your tone when the text contains a question. "
            "A question must sound like a demanded status report. "
            "Avoid sing-song intonation and avoid a customer-service voice. "
            "Strong intensity, controlled volume, crystal-clear pronunciation. "
            "Do not imitate any specific real person."
        )

    if "strict" in name:
        return (
            "cedar",
            language_instruction +
            "Use a firm, controlled, authoritative tone. "
            "Speak clearly, briskly and with no-nonsense confidence."
        )

    if "sarcastic" in name:
        return (
            "ash",
            language_instruction +
            "Use a dry, amused, slightly cheeky tone. "
            "Sound natural rather than theatrical. "
            "Let the sarcasm be subtle and playful."
        )

    if "girlfriend" in name or "lover" in name:
        return (
            "coral",
            language_instruction +
            "Use a warm, affectionate, close and playful tone. "
            "Sound caring and natural, like a supportive romantic partner. "
            "Keep it tasteful and non-sexual."
        )

    if "custom" in name or "adaptive" in name:
        return (
            "marin",
            language_instruction +
            "Use a natural, modern conversational delivery. "
            "Sound relaxed, human-like and responsive."
        )

    return (
        "marin",
        language_instruction +
        "Use a friendly, warm, positive and energetic morning tone. "
        "Sound natural and conversational."
    )


def prepare_speech_text(
    text: str,
    personality: str,
    language: str
) -> str:

    name = personality.strip().lower()
    lang = language.strip().lower()
    cleaned = text.strip()

    if "military" not in name:
        return cleaned

    # The long first greeting made the voice soften into a conversational
    # question. Keep the opening exactly in the short command cadence
    # that sounded right in the standalone WakeAI test.
    first_greeting_markers = (
        "jak ses vyspal",
        "how did you sleep",
        "jak spałeś",
        "jak spales",
        "cómo dormiste",
        "como dormiste",
        "як спалося"
    )

    lowered = cleaned.lower()

    if any(marker in lowered for marker in first_greeting_markers):

        if "czech" in lang:
            return (
                "Vstávat! Žádné vyjednávání. "
                "Nohy na zem a do pohybu, vojáku!"
            )

        if "polish" in lang:
            return (
                "Pobudka! Bez negocjacji. "
                "Nogi na podłogę i ruszaj, żołnierzu!"
            )

        if "spanish" in lang:
            return (
                "¡Arriba! Nada de negociar. "
                "Pies al suelo y en movimiento, soldado!"
            )

        if "ukrainian" in lang:
            return (
                "Підйом! Без переговорів. "
                "Ноги на підлогу й рухайся, солдате!"
            )

        return (
            "Wake up! No negotiations. "
            "Feet on the floor and move, recruit!"
        )

    return cleaned


def is_military_first_greeting(
    text: str,
    personality: str
) -> bool:

    if "military" not in personality.strip().lower():
        return False

    lowered = text.strip().lower()

    greeting_markers = (
        "jak ses vyspal",
        "how did you sleep",
        "jak spałeś",
        "jak spales",
        "cómo dormiste",
        "como dormiste",
        "як спалося",
        "vstávat, vojáku",
        "wake up, recruit",
        "pobudka, żołnierzu",
        "arriba, soldado",
        "підйом, солдате"
    )

    return any(
        marker in lowered
        for marker in greeting_markers
    )


def prerecorded_military_greeting() -> bytes | None:

    audio_path = (
        Path(__file__).resolve().parent
        / "wakeai_sergeant_test.wav"
    )

    if not audio_path.exists():
        return None

    return audio_path.read_bytes()


@app.post("/speak")
def speak(request: SpeechRequest):

    if is_military_first_greeting(
        request.text,
        request.personality
    ):

        prerecorded_audio = (
            prerecorded_military_greeting()
        )

        if prerecorded_audio is not None:

            return Response(
                content=prerecorded_audio,
                media_type="audio/wav",
                headers={
                    "Cache-Control": "no-store",
                    "X-WakeAI-Voice": "sergeant-prerecorded",
                    "X-WakeAI-Voice-Profile": "military-prerecorded-v1"
                }
            )

    voice, instructions = voice_settings(
        request.personality,
        request.language
    )

    spoken_text = prepare_speech_text(
        request.text,
        request.personality,
        request.language
    )

    temp_path = None

    try:
        with tempfile.NamedTemporaryFile(
            suffix=".wav",
            delete=False
        ) as temp_file:
            temp_path = temp_file.name

        # WAV is used because OpenAI recommends WAV/PCM
        # for the fastest response times.
        with client.audio.speech.with_streaming_response.create(
            model="gpt-4o-mini-tts",
            voice=voice,
            input=spoken_text,
            instructions=instructions,
            response_format="wav"
        ) as speech_response:
            speech_response.stream_to_file(temp_path)

        audio_bytes = Path(temp_path).read_bytes()

        return Response(
            content=audio_bytes,
            media_type="audio/wav",
            headers={
                "Cache-Control": "no-store",
                "X-WakeAI-Voice": voice,
                "X-WakeAI-Voice-Profile": (
                    "military-v2"
                    if "military" in request.personality.strip().lower()
                    else "standard"
                )
            }
        )

    finally:
        if temp_path:
            try:
                os.remove(temp_path)
            except OSError:
                pass