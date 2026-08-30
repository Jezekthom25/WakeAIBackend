from pathlib import Path
import os
import tempfile
import json
import urllib.request
import urllib.error

from fastapi import FastAPI
from fastapi.responses import Response, StreamingResponse
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


class RealtimeTokenRequest(BaseModel):
    language: str = "English"
    personality: str = "Friendly"
    custom_profile: str = ""
    # "audio" keeps the current direct Realtime speech-to-speech path.
    # "text" is used by the hybrid path: Realtime listens/reasons,
    # then Android sends the finished text to our streamed ONYX TTS.
    output_mode: str = "audio"


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
            "Sound like a real, disciplined person waking someone in the same room, with military firmness but normal human speech. "
            "Be confident, direct and energetic without acting like a movie drill-sergeant caricature. "
            "Use natural complete sentences, contractions and everyday spoken wording appropriate to the selected language. "
            "Mix a firm command with an occasional brief human reaction so every turn does not sound like a slogan or scripted order. "
            "Vary sentence length and wording. Avoid repetitive imperatives, one-word barking, robotic status-report language and canned confirmations. "
            "Do not overuse words like soldier, recruit or vojáku; the military feeling should come mainly from confidence, discipline and urgency. "
            "Challenge excuses quickly, but never insult, threaten or humiliate."
        )

    if "strict" in name:
        return (
            "Sound like a firm real person who expects action. "
            "Be concise, natural and decisive. "
            "Acknowledge what the user said only when it adds something. "
            "Avoid robotic confirmations and repeated instructions."
        )

    if "sarcastic" in name:
        return (
            "Sound naturally witty and dry, like a quick real-life remark. "
            "Use light playful sarcasm, never cruelty. "
            "Do not force a joke into every reply."
        )

    if "girlfriend" in name or "lover" in name:
        return (
            "Sound warm, close, affectionate and spontaneous, like a caring partner nearby. "
            "Use natural everyday speech, not therapy language or scripted encouragement. "
            "Keep it tasteful and non-sexual."
        )

    if "custom" in name or "adaptive" in name:
        if custom_profile.strip():
            return (
                "Speak like a real person and follow this saved communication style:\n"
                f"{custom_profile.strip()}"
            )

        return (
            "Speak casually and naturally. "
            "Match the user's tone without sounding scripted."
        )

    return (
        "Sound like a friendly real person in the room. "
        "Be casual, warm and concise. "
        "Avoid scripted encouragement and chatbot-style acknowledgements."
    )


def movement_prompt(
    movement_state: str,
    seconds_since_movement: int
) -> str:

    return (
        f"Phone movement: {movement_state}; "
        f"seconds since movement: {seconds_since_movement}. "
        "This describes the PHONE only, not the body. "
        "Movement is auxiliary context only: it is never required for completion and it is never sufficient by itself. "
        "The alarm ends by conversational agreement. "
        "If the user merely says they are awake, up, getting up, standing, or moving, do NOT finish yet; "
        "ask one short explicit confirmation that they are awake and that WakeAI may stop the alarm. "
        "Use [[WAKE_COMPLETE]] only if the user clearly asks to stop/turn off/end the alarm, "
        "or clearly answers yes to a prior WakeAI confirmation question about ending it. "
        "Never use [[WAKE_COMPLETE]] after silence, uncertainty, refusal, 'later', or 'five more minutes'."
    )


@app.post("/chat")
def chat(request: WakeRequest):

    # Last 3 exchanges are enough for a wake-up conversation and keep
    # the prompt small, which helps latency.
    history_text = "\n".join(request.history[-6:])

    prompt = (
        "You are WakeAI, a voice-first AI alarm clock. "
        "This is a live spoken conversation with a sleepy person.\n"
        f"Reply in {request.language}.\n\n"

        "SPOKEN STYLE:\n"
        "- Sound spontaneous and human, never like a chatbot or instruction manual.\n"
        "- Usually answer in ONE natural short sentence.\n"
        "- Absolute maximum: TWO short sentences and about 24 spoken words.\n"
        "- React to the user's actual words; do not paraphrase them back.\n"
        "- Avoid canned openings such as 'I understand', 'Alright', 'Okay, so', "
        "'Of course', or their equivalents unless they genuinely fit.\n"
        "- Do not explain your role, rules, sensor data or reasoning.\n"
        "- Do not ask a question every turn. A short reaction or command is often better.\n"
        "- Vary phrasing. Do not repeat the same wake-up instruction in consecutive turns.\n"
        "- Use natural everyday wording that sounds good aloud.\n\n"

        "PERSONALITY:\n"
        f"{personality_prompt(request.personality, request.custom_profile)}\n\n"

        "ENDING THE ALARM:\n"
        "- Completion is a conversational agreement, not a movement test.\n"
        "- If the user merely claims to be awake/getting up, ask one brief confirmation and do NOT emit [[WAKE_COMPLETE]] yet.\n"
        "- Emit [[WAKE_COMPLETE]] only when the user explicitly asks to stop/end the alarm, or clearly confirms a prior WakeAI question about stopping it.\n"
        "- Never emit it for silence, uncertainty, refusal, or a request for more sleep.\n"
        "- When you emit it, also say one short natural final confirmation sentence; do not ask another question.\n\n"

        "STATE:\n"
        f"{movement_prompt(request.movement_state, request.seconds_since_movement)}\n\n"

        "RECENT TALK:\n"
        f"{history_text if history_text else '(none)'}\n"
        f"User: {request.message}"
    )

    response = client.responses.create(
        model="gpt-5.6-luna",
        input=prompt,
        max_output_tokens=60
    )

    reply = response.output_text.strip()

    wake_complete = "[[WAKE_COMPLETE]]" in reply

    reply = (
        reply
        .replace("[[WAKE_COMPLETE]]", "")
        .strip()
    )

    if not reply:
        reply = {
            "Czech": "Tak pojď, jeden krok po druhém.",
            "Polish": "No dalej, krok po kroku.",
            "Spanish": "Vamos, paso a paso.",
            "Ukrainian": "Давай, крок за кроком."
        }.get(
            request.language,
            "Come on, one step at a time."
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
            f"Speak naturally in {language}, like a real person standing nearby in the morning. "
            "Keep a firm, confident, slightly stern military character, but use ordinary human conversational rhythm. "
            "Let words flow together naturally; use realistic emphasis, breathing and small pauses instead of clipped fragments. "
            "Do not bark, chant, sound monotone, over-enunciate, or hit every sentence with the same hard ending. "
            "Commands can be decisive, but reactions and follow-ups should sound spontaneous, relaxed enough to be believable, and emotionally varied."
        )

    if "strict" in name:
        return (
            "cedar",
            language_instruction +
            "Firm, brisk and natural. Sound decisive, not theatrical."
        )

    if "sarcastic" in name:
        return (
            "ash",
            language_instruction +
            "Dry, lightly amused and natural. Keep the sarcasm subtle."
        )

    if "girlfriend" in name or "lover" in name:
        return (
            "coral",
            language_instruction +
            "Warm, close and playful. Sound spontaneous and affectionate, not scripted."
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



def realtime_voice(personality: str) -> str:

    name = personality.strip().lower()

    # Realtime currently supports voices such as cedar, marin, ash and coral.
    # ONYX is not a Realtime voice, so Military uses cedar.
    if "military" in name or "strict" in name:
        return "cedar"

    if "sarcastic" in name:
        return "ash"

    if "girlfriend" in name or "lover" in name:
        return "coral"

    return "marin"


def realtime_instructions(
    language: str,
    personality: str,
    custom_profile: str
) -> str:

    return (
        "You are WakeAI, a voice-first AI alarm clock in a live spoken conversation. "
        f"Always speak in {language}. "
        "Respond immediately and naturally, like a real person standing nearby. "
        "Usually use one short complete sentence; never more than two short sentences. "
        "Do not use lists, headings, explanations, chatbot filler, or customer-service phrases. "
        "Do not repeat the user's sentence back to them. "
        "Do not ask a question every turn. "
        "Keep the pace energetic enough for waking someone up, but preserve natural conversational rhythm and varied phrasing. "
        "HIGHEST-PRIORITY ACTION ROUTING: when the snooze_alarm tool is available, any explicit request for more sleep, extra minutes, a delay, snooze, or to be woken again later MUST use snooze_alarm. "
        "Examples include 'five more minutes', 'ještě pět minut', 'odlož budík', and 'vzbuď mě za deset minut'. Never answer a snooze request by saying it is not confirmation to turn the alarm off, and never ask the wake-completion confirmation question in response to snooze. "
        "The alarm ends permanently ONLY by clear conversational agreement, never because of phone movement alone. "
        "If the user only says they are awake, up, getting up, standing, or moving, do not end immediately; normally ask one short explicit confirmation that they are awake and that you may stop the alarm, then wait for their reply. "
        "If the user clearly and unambiguously asks you to stop, turn off, or end the alarm, that direct request is enough to accept. "
        "Never use wake_complete after silence, vague mumbling, uncertainty, refusal, 'later', or a request for more sleep. "
        "When wake_complete is available and the agreement is clear, call it instead of speaking a normal reply. "
        "Put a short, natural spoken confirmation in the tool's final_message argument. For snooze, explicitly say how many minutes were granted before the alarm closes. "
        f"Personality instructions: {personality_prompt(personality, custom_profile)}"
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

    markers = (
        "jak ses vyspal",
        "vstávat, vojáku",
        "vstavat, vojaku",
        "how did you sleep",
        "wake up, soldier",
        "jak spałeś",
        "jak spales",
        "pobudka",
        "cómo dormiste",
        "como dormiste",
        "як спалося"
    )

    return any(marker in lowered for marker in markers)


def approved_military_greeting() -> bytes | None:

    audio_path = (
        Path(__file__).resolve().parent
        / "01_onyx_military.wav"
    )

    if not audio_path.exists():
        return None

    try:
        audio_bytes = audio_path.read_bytes()
    except OSError:
        return None

    if not audio_bytes:
        return None

    return audio_bytes


@app.post("/realtime-token")
def realtime_token(request: RealtimeTokenRequest):

    api_key = os.environ.get("OPENAI_API_KEY", "").strip()

    if not api_key:
        return Response(
            content=json.dumps(
                {"error": "OPENAI_API_KEY is not configured"}
            ),
            status_code=500,
            media_type="application/json"
        )

    voice = realtime_voice(
        request.personality
    )

    requested_output_mode = (
        request.output_mode
        .strip()
        .lower()
    )

    output_mode = (
        "text"
        if requested_output_mode == "text"
        else "audio"
    )

    audio_config = {
        "input": {
            "format": {
                "type": "audio/pcm",
                "rate": 24000
            },
            "turn_detection": {
                "type": "semantic_vad"
            }
        }
    }

    # Voice is relevant only for direct Realtime audio output.
    # In hybrid text mode, Android will use /speak-stream instead,
    # which preserves our ONYX Military voice.
    if output_mode == "audio":
        audio_config["output"] = {
            "voice": voice
        }

    session_config = {
        "session": {
            "type": "realtime",
            "model": "gpt-realtime-2.1-mini",
            "output_modalities": [output_mode],
            "audio": audio_config,
            "instructions": realtime_instructions(
                request.language,
                request.personality,
                request.custom_profile
            )
        }
    }

    body = json.dumps(
        session_config
    ).encode("utf-8")

    openai_request = urllib.request.Request(
        "https://api.openai.com/v1/realtime/client_secrets",
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
            "OpenAI-Safety-Identifier": "wakeai-anonymous-v1"
        }
    )

    try:

        with urllib.request.urlopen(
            openai_request,
            timeout=15
        ) as response:

            response_body = response.read()

            return Response(
                content=response_body,
                status_code=response.status,
                media_type="application/json",
                headers={
                    "Cache-Control": "no-store",
                    "X-WakeAI-Realtime-Voice": voice,
                    "X-WakeAI-Realtime-Model": "gpt-realtime-2.1-mini",
                    "X-WakeAI-Realtime-Output": output_mode
                }
            )

    except urllib.error.HTTPError as error:

        error_body = error.read()

        return Response(
            content=error_body,
            status_code=error.code,
            media_type="application/json",
            headers={
                "Cache-Control": "no-store"
            }
        )

    except Exception as error:

        return Response(
            content=json.dumps(
                {
                    "error": "Failed to create Realtime token",
                    "detail": str(error)
                }
            ),
            status_code=500,
            media_type="application/json",
            headers={
                "Cache-Control": "no-store"
            }
        )


@app.post("/speak-stream")
def speak_stream(request: SpeechRequest):

    voice, instructions = voice_settings(
        request.personality,
        request.language
    )

    spoken_text = prepare_speech_text(
        request.text,
        request.personality,
        request.language
    )

    # Slightly quicker delivery without making speech unnaturally fast.
    name = request.personality.strip().lower()
    speech_speed = 1.00 if "military" in name else 1.04

    def audio_stream():

        with client.audio.speech.with_streaming_response.create(
            model="gpt-4o-mini-tts",
            voice=voice,
            input=spoken_text,
            instructions=instructions,
            response_format="pcm",
            speed=speech_speed
        ) as speech_response:

            for chunk in speech_response.iter_bytes(
                chunk_size=4096
            ):
                if chunk:
                    yield chunk

    return StreamingResponse(
        audio_stream(),
        media_type="application/octet-stream",
        headers={
            "Cache-Control": "no-store",
            "X-WakeAI-Audio-Format": "pcm_s16le",
            "X-WakeAI-Sample-Rate": "24000",
            "X-WakeAI-Channels": "1",
            "X-WakeAI-Voice": voice
        }
    )


@app.post("/speak")
def speak(request: SpeechRequest):

    # The user explicitly approved this exact generated ONYX take.
    # Use it for the initial Military alarm greeting so the startup
    # voice is bit-for-bit identical every time.
    if is_military_first_greeting(
        request.text,
        request.personality
    ):
        approved_audio = approved_military_greeting()

        if approved_audio is not None:
            return Response(
                content=approved_audio,
                media_type="audio/wav",
                headers={
                    "Cache-Control": "no-store",
                    "X-WakeAI-Voice": "approved-onyx-prerecorded",
                    "X-WakeAI-Voice-Profile": "military-approved-v1"
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
