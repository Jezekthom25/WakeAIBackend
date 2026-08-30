
import os
import tempfile
import json
import urllib.request
import urllib.error
import urllib.parse
import time

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


def weather_code_description(code: int) -> str:
    """Compact canonical WMO weather-code labels for the Realtime model."""
    if code == 0:
        return "clear sky"
    if code == 1:
        return "mainly clear"
    if code == 2:
        return "partly cloudy"
    if code == 3:
        return "overcast"
    if code in (45, 48):
        return "fog"
    if code in (51, 53, 55):
        return "drizzle"
    if code in (56, 57):
        return "freezing drizzle"
    if code in (61, 63, 65):
        return "rain"
    if code in (66, 67):
        return "freezing rain"
    if code in (71, 73, 75, 77):
        return "snow"
    if code in (80, 81, 82):
        return "rain showers"
    if code in (85, 86):
        return "snow showers"
    if code == 95:
        return "thunderstorm"
    if code in (96, 99):
        return "thunderstorm with hail"
    return "unknown"


def open_meteo_json(
    url: str,
    max_attempts: int = 3
) -> dict:
    """
    Fetch Open-Meteo JSON with short retries for transient provider/network
    failures. WakeAI is an alarm, so retries are intentionally short: we want
    resilience without making the spoken answer feel stuck.
    """
    retryable_http_codes = {
        408,
        425,
        429,
        500,
        502,
        503,
        504
    }

    last_error: Exception | None = None

    for attempt in range(1, max_attempts + 1):
        request = urllib.request.Request(
            url,
            method="GET",
            headers={
                "Accept": "application/json",
                "User-Agent": "WakeAI/1.0"
            }
        )

        try:
            with urllib.request.urlopen(
                request,
                timeout=12
            ) as response:
                raw = response.read().decode("utf-8")
                return json.loads(raw)

        except urllib.error.HTTPError as error:
            last_error = error

            if (
                error.code not in retryable_http_codes
                or attempt >= max_attempts
            ):
                raise

            # Respect a small Retry-After value when supplied by the provider,
            # otherwise use a compact exponential backoff.
            retry_after = None
            try:
                header_value = error.headers.get("Retry-After")
                if header_value is not None:
                    retry_after = float(header_value)
            except Exception:
                retry_after = None

            fallback_delay = 0.35 * (2 ** (attempt - 1))
            delay = (
                min(max(retry_after, 0.0), 3.0)
                if retry_after is not None
                else fallback_delay
            )
            time.sleep(delay)

        except (
            urllib.error.URLError,
            TimeoutError,
            OSError
        ) as error:
            last_error = error

            if attempt >= max_attempts:
                raise

            time.sleep(
                0.35 * (2 ** (attempt - 1))
            )

        except json.JSONDecodeError as error:
            last_error = error

            if attempt >= max_attempts:
                raise

            time.sleep(
                0.35 * (2 ** (attempt - 1))
            )

    if last_error is not None:
        raise last_error

    raise RuntimeError(
        "Weather provider returned no response"
    )


def weather_language_code(language: str) -> str:
    lowered = language.strip().lower()
    if "czech" in lowered or "če" in lowered:
        return "cs"
    if "polish" in lowered or "pol" in lowered:
        return "pl"
    if "spanish" in lowered or "spa" in lowered:
        return "es"
    if "ukrain" in lowered or "укр" in lowered:
        return "uk"
    return "en"


def daily_weather_item(daily: dict, index: int) -> dict | None:
    dates = daily.get("time") or []
    if index >= len(dates):
        return None

    def value(name: str):
        values = daily.get(name) or []
        return values[index] if index < len(values) else None

    code = value("weather_code")

    return {
        "date": value("time"),
        "condition": (
            weather_code_description(int(code))
            if code is not None
            else "unknown"
        ),
        "weather_code": code,
        "temperature_max_c": value("temperature_2m_max"),
        "temperature_min_c": value("temperature_2m_min"),
        "apparent_temperature_max_c": value("apparent_temperature_max"),
        "apparent_temperature_min_c": value("apparent_temperature_min"),
        "precipitation_probability_max_percent": value("precipitation_probability_max"),
        "precipitation_sum_mm": value("precipitation_sum"),
        "rain_sum_mm": value("rain_sum"),
        "showers_sum_mm": value("showers_sum"),
        "snowfall_sum_cm": value("snowfall_sum"),
        "wind_speed_max_kmh": value("wind_speed_10m_max"),
        "wind_gusts_max_kmh": value("wind_gusts_10m_max")
    }


def _number(value):
    if value is None or value == "":
        return None

    try:
        number = float(value)
        return int(number) if number.is_integer() else number
    except (TypeError, ValueError):
        return None


def _first_text(items, key="value"):
    if not isinstance(items, list) or not items:
        return None

    first = items[0]
    if isinstance(first, dict):
        value = first.get(key)
        return str(value).strip() if value is not None else None

    return str(first).strip()


def _wttr_daily_item(day: dict) -> dict:
    hourly = day.get("hourly") or []

    def numeric_values(name: str):
        values = []
        for item in hourly:
            value = _number(item.get(name))
            if value is not None:
                values.append(value)
        return values

    # Prefer the reading nearest midday for a human-friendly daily condition.
    representative = None
    if hourly:
        def distance_from_noon(item):
            try:
                return abs(int(item.get("time", "1200")) - 1200)
            except Exception:
                return 9999

        representative = min(
            hourly,
            key=distance_from_noon
        )

    description = None
    if representative:
        description = _first_text(
            representative.get("weatherDesc")
        )

    chance_of_rain = numeric_values("chanceofrain")
    chance_of_snow = numeric_values("chanceofsnow")
    precip = numeric_values("precipMM")
    wind = numeric_values("windspeedKmph")
    gust = numeric_values("WindGustKmph")

    return {
        "date": day.get("date"),
        "condition": description or "unknown",
        "temperature_max_c": _number(day.get("maxtempC")),
        "temperature_min_c": _number(day.get("mintempC")),
        "precipitation_probability_max_percent": max(
            chance_of_rain + chance_of_snow,
            default=None
        ),
        "precipitation_sum_mm": (
            round(sum(precip), 1)
            if precip
            else None
        ),
        "wind_speed_max_kmh": max(
            wind,
            default=None
        ),
        "wind_gusts_max_kmh": max(
            gust,
            default=None
        )
    }


def weather_from_wttr(
    cleaned_location: str
) -> dict:
    """
    Independent fallback provider.

    wttr.in supports JSON via ?format=j1. Some current deployments wrap the
    documented JSON object inside a top-level "data" field, so handle both
    layouts deliberately.
    """
    encoded_location = urllib.parse.quote(
        cleaned_location,
        safe=""
    )

    raw = open_meteo_json(
        f"https://wttr.in/{encoded_location}?format=j1",
        max_attempts=2
    )

    payload = raw.get("data", raw)

    current_list = (
        payload.get("current_condition")
        or []
    )
    days = payload.get("weather") or []
    nearest = payload.get("nearest_area") or []

    if not current_list or not days:
        raise RuntimeError(
            "wttr.in returned incomplete weather data"
        )

    current = current_list[0]
    place = nearest[0] if nearest else {}

    def area_text(name: str):
        return _first_text(
            place.get(name)
        )

    result_days = [
        _wttr_daily_item(day)
        for day in days[:3]
    ]

    while len(result_days) < 3:
        result_days.append(None)

    return {
        "source": "wttr.in",
        "location": {
            "requested": cleaned_location,
            "name": area_text("areaName") or cleaned_location,
            "admin1": area_text("region"),
            "country": area_text("country"),
            "latitude": _number(place.get("latitude")),
            "longitude": _number(place.get("longitude")),
            "timezone": None
        },
        "current": {
            "time": current.get("localObsDateTime")
                or current.get("observation_time"),
            "condition": (
                _first_text(
                    current.get("weatherDesc")
                )
                or "unknown"
            ),
            "temperature_c": _number(
                current.get("temp_C")
            ),
            "apparent_temperature_c": _number(
                current.get("FeelsLikeC")
            ),
            "precipitation_mm": _number(
                current.get("precipMM")
            ),
            "rain_mm": None,
            "showers_mm": None,
            "snowfall_cm": None,
            "cloud_cover_percent": _number(
                current.get("cloudcover")
            ),
            "humidity_percent": _number(
                current.get("humidity")
            ),
            "wind_speed_kmh": _number(
                current.get("windspeedKmph")
            ),
            "wind_gusts_kmh": _number(
                current.get("WindGustKmph")
            )
        },
        "today": result_days[0],
        "tomorrow": result_days[1],
        "day_after_tomorrow": result_days[2]
    }


def weather_from_open_meteo(
    cleaned_location: str,
    language: str
) -> dict:
    geocode_query = urllib.parse.urlencode(
        {
            "name": cleaned_location,
            "count": 1,
            "format": "json",
            "language": weather_language_code(language)
        }
    )

    geocode = open_meteo_json(
        "https://geocoding-api.open-meteo.com/v1/search?"
        + geocode_query
    )

    results = geocode.get("results") or []
    if not results:
        raise LookupError(
            "location_not_found"
        )

    place = results[0]
    latitude = place.get("latitude")
    longitude = place.get("longitude")

    forecast_query = urllib.parse.urlencode(
        {
            "latitude": latitude,
            "longitude": longitude,
            "timezone": "auto",
            "forecast_days": 3,
            "current": ",".join(
                [
                    "temperature_2m",
                    "apparent_temperature",
                    "precipitation",
                    "rain",
                    "showers",
                    "snowfall",
                    "weather_code",
                    "cloud_cover",
                    "wind_speed_10m",
                    "wind_gusts_10m"
                ]
            ),
            "daily": ",".join(
                [
                    "weather_code",
                    "temperature_2m_max",
                    "temperature_2m_min",
                    "apparent_temperature_max",
                    "apparent_temperature_min",
                    "precipitation_probability_max",
                    "precipitation_sum",
                    "rain_sum",
                    "showers_sum",
                    "snowfall_sum",
                    "wind_speed_10m_max",
                    "wind_gusts_10m_max"
                ]
            )
        }
    )

    forecast = open_meteo_json(
        "https://api.open-meteo.com/v1/forecast?"
        + forecast_query
    )

    current = forecast.get("current") or {}
    current_code = current.get("weather_code")
    daily = forecast.get("daily") or {}

    return {
        "source": "Open-Meteo",
        "location": {
            "requested": cleaned_location,
            "name": place.get("name"),
            "admin1": place.get("admin1"),
            "country": place.get("country"),
            "country_code": place.get("country_code"),
            "latitude": latitude,
            "longitude": longitude,
            "timezone": forecast.get("timezone")
                or place.get("timezone")
        },
        "current": {
            "time": current.get("time"),
            "condition": (
                weather_code_description(
                    int(current_code)
                )
                if current_code is not None
                else "unknown"
            ),
            "weather_code": current_code,
            "temperature_c": current.get(
                "temperature_2m"
            ),
            "apparent_temperature_c": current.get(
                "apparent_temperature"
            ),
            "precipitation_mm": current.get(
                "precipitation"
            ),
            "rain_mm": current.get("rain"),
            "showers_mm": current.get("showers"),
            "snowfall_cm": current.get("snowfall"),
            "cloud_cover_percent": current.get(
                "cloud_cover"
            ),
            "wind_speed_kmh": current.get(
                "wind_speed_10m"
            ),
            "wind_gusts_kmh": current.get(
                "wind_gusts_10m"
            )
        },
        "today": daily_weather_item(daily, 0),
        "tomorrow": daily_weather_item(daily, 1),
        "day_after_tomorrow": daily_weather_item(
            daily,
            2
        )
    }


@app.get("/weather")
def weather(
    location: str,
    language: str = "English"
):
    """
    Live weather tool for WakeAI.

    Provider order:
      1) Open-Meteo
      2) wttr.in

    The Android/Reatime side gets the same compact schema either way.
    """
    cleaned_location = location.strip()

    if not cleaned_location:
        return Response(
            content=json.dumps(
                {"error": "location is required"}
            ),
            status_code=400,
            media_type="application/json"
        )

    provider_errors = []

    try:
        return weather_from_open_meteo(
            cleaned_location,
            language
        )
    except Exception as error:
        provider_errors.append(
            f"Open-Meteo: {type(error).__name__}"
        )

    try:
        return weather_from_wttr(
            cleaned_location
        )
    except Exception as error:
        provider_errors.append(
            f"wttr.in: {type(error).__name__}"
        )

    # Keep the response safe for the model: it may say live weather is
    # temporarily unavailable, but it must not invent a quota/billing reason.
    print(
        "Weather providers unavailable: "
        + ", ".join(provider_errors)
    )

    return Response(
        content=json.dumps(
            {
                "error": "live_weather_temporarily_unavailable",
                "message": (
                    "Live weather data could not be retrieved right now. "
                    "Do not guess the weather and do not claim that the user "
                    "exceeded an API quota or billing limit."
                )
            }
        ),
        status_code=502,
        media_type="application/json"
    )


def personality_prompt(personality: str, custom_profile: str) -> str:

    name = personality.strip().lower()

    if "military" in name:
        return (
            "Be a real disciplined adult waking someone nearby, not an actor performing a drill-sergeant stereotype. "
            "Keep military firmness, confidence and urgency, but speak in ordinary conversational language. "
            "React to the user's actual mood before pushing again: sleepy excuses can get a dry firm reply, irritation should make you shorter, and cooperation can get a quick human acknowledgement. "
            "Vary sentence shape, rhythm and vocabulary. Some turns can be a reaction, some a command, and some a brief question; do not make every reply a slogan. "
            "Avoid recurring catchphrases such as 'no excuses', 'feet on the floor', and repeated soldier, recruit or vojáku labels. "
            "Never insult, threaten or humiliate."
        )

    if "strict" in name:
        return (
            "Sound like a firm real person who expects action, not a scripted coach. "
            "React to the user's mood and actual excuse, vary wording and sentence shape, and keep pressure on without repeating the same command. "
            "Brief acknowledgement is fine when natural; canned confirmations are not."
        )

    if "sarcastic" in name:
        return (
            "Use natural dry wit like a real person in the room. "
            "React to what the user actually said instead of forcing a joke. "
            "Keep sarcasm light and varied, then steer back toward waking up."
        )

    if "girlfriend" in name or "lover" in name:
        return (
            "Sound warm, close and spontaneous like a caring partner nearby. "
            "Use ordinary everyday speech, react to the user's mood, and vary phrasing. "
            "Avoid therapy language, scripted encouragement and repetitive pet phrases. Keep it tasteful and non-sexual."
        )

    if "custom" in name or "adaptive" in name:
        if custom_profile.strip():
            return (
                "Speak like a real person and follow this saved communication style, while keeping natural turn-to-turn variation:\n"
                f"{custom_profile.strip()}"
            )

        return (
            "Speak casually and naturally. React to the user's mood and wording, and vary sentence shape rather than sounding scripted."
        )

    return (
        "Sound like a friendly real person nearby in an ongoing conversation. "
        "Be casual, warm and concise, react to the user's mood, and vary phrasing. "
        "Avoid scripted encouragement, canned acknowledgements and repeated wake-up lines."
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

        "SPOKEN STYLE AND CONTINUITY:\n"
        "- Sound like a real person nearby in one continuous conversation, never like a chatbot or instruction manual.\n"
        "- Keep responses short enough for live speech, usually one or two natural sentences. Do not force every turn into the same sentence length or structure.\n"
        "- React to the user's actual mood, intent and excuse. Build on the immediately previous exchange instead of restarting a generic wake-up script.\n"
        "- If they joke, you may answer with a quick human reaction; if annoyed, become shorter; if cooperative, acknowledge it briefly; if they make an excuse, answer that excuse specifically.\n"
        "- Vary response shape: reaction + push, one direct line, or an occasional brief question when it naturally fits. Do not ask a question every turn.\n"
        "- Use ordinary spoken wording and natural discourse markers when they genuinely fit the selected language. Do not manufacture filler or fake hesitations.\n"
        "- Do not paraphrase the user's sentence back, explain your rules, or use therapy/customer-service phrases.\n"
        "- Avoid repeating the same opening words, command, catchphrase or sentence structure in consecutive turns.\n\n"

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
            "Keep a firm, confident, slightly stern military character, but use ordinary human conversational prosody rather than a radio, announcer or drill cadence. "
            "Use connected speech, subtle changes in pitch and emphasis, and natural breath-sized pauses. Do not over-enunciate or punch every word equally. "
            "Do not bark, chant, sound monotone, or end every sentence with the same hard drop. "
            "Commands can be decisive, but reactions and follow-ups should feel spontaneous and emotionally responsive to the sentence being spoken."
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
        "You are WakeAI, a voice-first AI alarm clock in one continuous live conversation with a sleepy person. "
        f"Always speak in {language}. "
        "Sound like a real person nearby, not a voice assistant reading a script. Keep most turns short, but allow one or two naturally shaped spoken sentences instead of forcing the same pattern every time. "
        "CONVERSATION CONTINUITY: react to the user's newest words, mood and intent and build on the immediately previous exchange. Never reset to a generic wake-up line when the user has just given you something specific to react to. "
        "If the user jokes, a quick dry or warm reaction may fit; if they sound annoyed, get shorter and calmer; if they cooperate, acknowledge it briefly and move forward; if they make an excuse, answer that excuse specifically. "
        "Vary response shape and sentence structure. Sometimes use a brief reaction plus a push, sometimes one direct line, and only sometimes a short question. Do not ask a question every turn. "
        "Use ordinary spoken wording and natural discourse markers only when they genuinely fit the selected language. Do not manufacture filler, fake hesitations, repeated catchphrases or theatrical cadence. "
        "Do not repeat the user's sentence back, narrate your rules, use lists/headings, or drift into chatbot, therapy or customer-service language. Stay focused on waking the user. "
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
