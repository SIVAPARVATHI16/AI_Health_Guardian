"""
Optional Google Gemini integration.

This module is entirely optional and defensive: if no API key is
configured, the `google-generativeai` package is not installed, or any API
call fails for any reason (network, quota, malformed response), every
public function here fails soft (returns None / empty / a fallback
string) rather than raising. The rest of the application must keep
working exactly as before when Gemini is unavailable -- Gemini only adds
capability, it is never a hard dependency.

To enable it, set an API key from Google AI Studio
(https://aistudio.google.com/apikey -- free tier available) as either:
  - the environment variable GEMINI_API_KEY, or
  - st.secrets["GEMINI_API_KEY"] in a Streamlit secrets.toml file.

Two things use this module:
  1. symptom_analyzer.predict_from_text -- as a fallback symptom extractor
     ONLY when the rule-based normalizer finds nothing. It is constrained
     to return codes from the model's own known symptom vocabulary, so it
     can never inject a symptom/disease the model wasn't trained on.
  2. chat_assistant.chat_response -- as a fallback for open-ended health
     questions that don't match a known intent, with a strict system
     prompt that forbids diagnosis/prescriptions and enforces the same
     safety boundaries as the rest of the app.
"""
import concurrent.futures
import json
import os
import re
import sys
import time
import traceback

_MODEL_NAME = "gemini-3.8-flash"
_TIMEOUT = 12  # seconds; keep chat/voice interactions snappy
_MAX_RETRIES = 2         # extra attempts after the first, for transient errors only
_RETRY_BACKOFF_SECONDS = 1.5   # multiplied by attempt number: 1.5s, then 3s

# Every public function in this module fails soft (returns None/[]/False)
# so the rest of the app never crashes when Gemini is unavailable. That's
# the right behavior for end users, but it means real errors (missing
# package, bad key, network block, etc.) were previously invisible.
# _log_error prints the real exception to the terminal running
# `streamlit run` (stderr), without changing the fail-soft return values,
# so a misconfigured key/package is easy to diagnose. Set
# GEMINI_DEBUG=0 in the environment to silence this again.
_DEBUG = os.environ.get("GEMINI_DEBUG", "1") != "0"

# Set when the most recent request failed specifically due to quota
# exhaustion on EVERY configured key (see _is_quota_exhausted and
# _generate_with_failover below), so the UI can show a clear, specific
# "today's free quota is used up" message instead of the vague generic
# "basic mode" text it would otherwise show (gemini_available() only
# checks that a key/package are present -- it can't know quota status
# without making a call).
_last_quota_exhausted_at = None

# Set when the most recent request failed because Gemini simply took too
# long to respond (see _TIMEOUT / _is_timeout below). Deliberately kept
# separate from quota/unavailable: on a timeout we do NOT want to silently
# substitute the rule-based assistant's answer (that could look like a
# real, considered reply when actually nothing was checked) -- instead the
# caller should tell the person to try again.
_last_timeout_at = None


def quota_recently_exhausted(window_seconds=600):
    """True if the most recent Gemini call failed because every configured
    key's quota was exhausted, within the last `window_seconds` (default
    10 minutes). Lets the UI show an accurate status instead of implying
    something is broken."""
    if _last_quota_exhausted_at is None:
        return False
    return (time.time() - _last_quota_exhausted_at) < window_seconds


def call_recently_timed_out(window_seconds=15):
    """True if the request that JUST completed (moments ago, in the same
    request/response cycle) failed because Gemini didn't respond in time.
    A short default window on purpose -- this is meant to be checked
    immediately after a single chat_reply/agentic_chat_reply call returns
    None, not as a lingering status."""
    if _last_timeout_at is None:
        return False
    return (time.time() - _last_timeout_at) < window_seconds

# Free-tier Gemini calls occasionally hit a transient, load-related error
# (503 UNAVAILABLE / "high demand", or a bare 500) that typically clears up
# within a couple of seconds -- these are worth a couple of short retries.
# A 404 (wrong/retired model name), 401/403 (bad key), or 400 (malformed
# request) will never succeed on retry, so those fail immediately instead
# of needlessly delaying every real failure.
_TRANSIENT_MARKERS = ("503", "UNAVAILABLE", "500 ", "INTERNAL")

# 429 RESOURCE_EXHAUSTED is NOT treated as transient: on the free tier it
# means the *daily* request quota for that specific key (as of writing, 20
# requests/day/project for this model) is used up, and Google's own error
# tells you to retry after 10-60+ seconds -- a short local backoff can
# never succeed against that. Instead of retrying, _generate_with_failover
# (below) reacts to this by switching to the next configured backup key,
# if any, and only gives up once every key has hit this same error.
_QUOTA_MARKERS = ("RESOURCE_EXHAUSTED", "429", "quota")

# A slow/hanging request (bounded by the timeout configured on the client,
# see _client() below) surfaces as an httpx timeout exception. This is
# deliberately NOT retried multiple times or papered over with a fallback
# answer -- see _last_timeout_at above for why.
_TIMEOUT_MARKERS = ("timeout", "timed out", "deadline")


def _is_quota_exhausted(exc):
    text = str(exc)
    return any(marker in text for marker in _QUOTA_MARKERS)


def _is_transient(exc):
    text = str(exc)
    return any(marker in text for marker in _TRANSIENT_MARKERS)


def _is_timeout(exc):
    text = (str(exc) + " " + type(exc).__name__).lower()
    return any(marker in text for marker in _TIMEOUT_MARKERS)


# A plain, SDK-version-independent way to bound how long we wait for a
# response: run the (blocking) call in a worker thread and give up after
# _TIMEOUT seconds. The background request can't be forcibly killed (Python
# threads can't be), but the important part -- not leaving the person
# staring at a spinner indefinitely -- works the same regardless of which
# google-genai version is installed, unlike http_options timeout (see
# _client's docstring for why that's avoided).
_executor = concurrent.futures.ThreadPoolExecutor(max_workers=4, thread_name_prefix="gemini-call")


def _call_with_timeout(fn, *args, timeout=_TIMEOUT, **kwargs):
    future = _executor.submit(fn, *args, **kwargs)
    try:
        return future.result(timeout=timeout)
    except concurrent.futures.TimeoutError:
        raise TimeoutError(f"Gemini request timed out after {timeout}s")


def _generate_with_retry(client, **kwargs):
    """Call client.models.generate_content, with:
      - up to _MAX_RETRIES short-backoff retries for genuinely transient
        errors (_is_transient) on THIS client/key,
      - exactly one immediate retry for a timeout (_is_timeout), since a
        single slow response is common and often succeeds on a second
        try, but no more than that,
      - NO retry at all for quota exhaustion (_is_quota_exhausted) -- that
        is handled one level up, by switching keys, not by retrying.
    Any other exception (auth, bad request, retired model, etc.) is raised
    immediately. This function has no knowledge of multiple keys; that is
    _generate_with_failover's job."""
    last_exc = None
    for attempt in range(_MAX_RETRIES + 1):
        try:
            return _call_with_timeout(client.models.generate_content, **kwargs)
        except Exception as exc:
            last_exc = exc
            if _is_quota_exhausted(exc):
                raise
            if _is_timeout(exc):
                if attempt >= 1:
                    raise
                if _DEBUG:
                    print("[gemini_assistant] request timed out, retrying once...",
                          file=sys.stderr)
                continue
            if attempt == _MAX_RETRIES or not _is_transient(exc):
                raise
            if _DEBUG:
                print(
                    f"[gemini_assistant] transient error ({exc}), "
                    f"retrying (attempt {attempt + 2}/{_MAX_RETRIES + 1})...",
                    file=sys.stderr,
                )
            time.sleep(_RETRY_BACKOFF_SECONDS * (attempt + 1))
    # Unreachable in practice: every branch above either returns a response
    # or re-raises before the loop can run out of attempts on its own (the
    # attempt == _MAX_RETRIES check always raises on the final iteration).
    # Kept only as a defensive backstop with a real exception type, never
    # `None`, so it can't itself raise a TypeError if some future edit
    # breaks that invariant.
    raise RuntimeError(f"generate_content failed with no captured exception (last: {last_exc!r})")


def _generate_with_failover(**kwargs):
    """The entry point every call site below uses instead of building a
    client and calling _generate_with_retry directly. Tries the primary
    key (GEMINI_API_KEY); if -- and only if -- that key's request fails
    specifically on quota exhaustion, automatically rotates through any
    configured backup keys (GEMINI_API_KEY_2/3/4) and retries the exact
    same request on each, in order, until one succeeds or every configured
    key has hit the same quota error for the day. A timeout is never
    retried across keys (a different key doesn't fix a slow response) and
    is surfaced immediately via _last_timeout_at. With only one key
    configured (the common case), this behaves exactly like calling
    _generate_with_retry(_client(), **kwargs) directly."""
    primary_key = _get_api_key()
    primary_client = _client(primary_key)
    if primary_client is None:
        raise RuntimeError("Gemini is not configured (no GEMINI_API_KEY set).")

    global _last_timeout_at, _last_quota_exhausted_at
    tried_keys = {primary_key}
    try:
        return _generate_with_retry(primary_client, **kwargs)
    except Exception as exc:
        if _is_timeout(exc):
            _last_timeout_at = time.time()
            raise
        if not _is_quota_exhausted(exc):
            raise
        last_exc = exc

    backup_keys = [k for k in _get_all_api_keys() if k not in tried_keys]
    for i, key in enumerate(backup_keys, start=1):
        tried_keys.add(key)
        client = _client(key)
        if client is None:
            continue
        if _DEBUG:
            print(
                f"[gemini_assistant] key #{i}'s daily quota is exhausted, "
                f"switching to backup key #{i + 1} of {len(backup_keys) + 1}...",
                file=sys.stderr,
            )
        try:
            return _generate_with_retry(client, **kwargs)
        except Exception as exc:
            if _is_timeout(exc):
                _last_timeout_at = time.time()
                raise
            last_exc = exc
            if not _is_quota_exhausted(exc):
                raise

    # Every configured key hit quota exhaustion -- nothing left to try.
    _last_quota_exhausted_at = time.time()
    if _DEBUG:
        print(
            f"[gemini_assistant] all {len(tried_keys)} configured key(s) are "
            "quota-exhausted for today -- falling back.",
            file=sys.stderr,
        )
    raise last_exc



def _log_error(context, exc):
    if not _DEBUG:
        return
    print(f"\n[gemini_assistant] {context} failed: {exc!r}", file=sys.stderr)
    traceback.print_exc(file=sys.stderr)
    print("[gemini_assistant] (set GEMINI_DEBUG=0 to hide these)\n", file=sys.stderr)


def _get_api_key():
    """The primary (first-priority) key."""
    key = os.environ.get("GEMINI_API_KEY")
    if key:
        return key
    try:
        import streamlit as st
        return st.secrets.get("GEMINI_API_KEY")
    except Exception:
        return None


def _get_all_api_keys():
    """All configured keys, in priority order, for automatic failover when
    one key's daily quota is exhausted. Configure up to 4 as either
    environment variables or entries in .streamlit/secrets.toml:
        GEMINI_API_KEY, GEMINI_API_KEY_2, GEMINI_API_KEY_3, GEMINI_API_KEY_4
    Only GEMINI_API_KEY is required; the others are optional extras used
    purely as backups. Blank/duplicate values are dropped; order is kept.
    """
    names = ["GEMINI_API_KEY", "GEMINI_API_KEY_2", "GEMINI_API_KEY_3", "GEMINI_API_KEY_4"]
    keys = []
    try:
        import streamlit as st
        secrets = st.secrets
    except Exception:
        secrets = {}
    for name in names:
        val = os.environ.get(name)
        if not val:
            try:
                val = secrets.get(name)
            except Exception:
                val = None
        if val and val not in keys:
            keys.append(val)
    return keys


def gemini_available():
    """Cheap, side-effect-free check the UI can use to show a status badge."""
    if not _get_api_key():
        return False
    try:
        import google.genai  # noqa: F401
        return True
    except Exception as exc:
        _log_error("gemini_available (import google.genai)", exc)
        return False


def _client(key=None):
    """Build a genai.Client for `key`, or for the primary configured key
    when key is None.

    Deliberately does NOT set a timeout via http_options=types.HttpOptions
    (timeout=...) here -- that SDK feature is unreliable across versions:
    it can silently fail to apply, or (worse) itself trigger a confusing
    generic "500 INTERNAL" error instead of the intended timeout
    (https://github.com/googleapis/python-genai/issues/1330 /
    https://github.com/googleapis/python-genai/issues/911). Instead, every
    call goes through _call_with_timeout (below), a plain Python
    thread-based deadline that behaves the same on every SDK version."""
    key = key or _get_api_key()
    if not key:
        return None
    try:
        from google import genai
        return genai.Client(api_key=key)
    except Exception as exc:
        _log_error("_client (genai.Client construction)", exc)
        return None


def _extract_json(raw_text):
    """Gemini sometimes wraps JSON in ```json fences; strip them safely."""
    if not raw_text:
        return None
    cleaned = re.sub(r"^```(?:json)?|```$", "", raw_text.strip(), flags=re.MULTILINE).strip()
    try:
        return json.loads(cleaned)
    except Exception:
        return None


def transcribe_audio_gemini(audio_bytes, mime_type="audio/wav", language_hint=None):
    """
    Transcribe recorded speech using Gemini's native audio understanding,
    as a more reliable alternative to the free/unofficial Google Web
    Speech endpoint (SpeechRecognition's recognize_google), which is not
    an officially supported API and is prone to returning garbled or
    unrelated "best guess" text for anything less than very clean, close-
    mic'd audio -- and has no real multilingual reliability guarantees.

    Returns (text, None) on success, or (None, error_message) on failure
    (including "not configured", so the caller can fall back cleanly).
    """
    if _client() is None:
        return None, "Gemini is not configured (no GEMINI_API_KEY set)."
    if not audio_bytes:
        return None, "No audio data."

    lang_instruction = (
        f"The speaker is expected to be speaking {language_hint}. " if language_hint else ""
    )
    prompt = (
        "Transcribe the speech in this audio clip exactly as spoken, in its "
        "original language and script (do not translate). " + lang_instruction +
        "Output ONLY the transcribed text, with no quotes, labels, or commentary. "
        "If the audio is silent, unintelligible, or contains no speech, output "
        "exactly: [NO_SPEECH]"
    )
    try:
        from google.genai import types
        resp = _generate_with_failover(
            model=_MODEL_NAME,
            contents=[
                types.Part.from_bytes(data=audio_bytes, mime_type=mime_type),
                prompt,
            ],
            config=types.GenerateContentConfig(temperature=0, max_output_tokens=300),
        )
        text = (getattr(resp, "text", None) or "").strip()
        if not text or text == "[NO_SPEECH]" or "[NO_SPEECH]" in text:
            return None, "No clear speech detected in the recording."
        return text, None
    except Exception as exc:
        _log_error("transcribe_audio_gemini", exc)
        return None, f"Gemini transcription failed: {exc}"


def suggest_symptoms_from_text(text, known_symptom_columns):
    """
    Ask Gemini to map free-text into ONLY the given canonical symptom
    vocabulary. Used strictly as a fallback when the rule-based extractor
    (symptom_normalizer) matches nothing, so it never overrides or
    second-guesses the deterministic path -- it only helps when that path
    would otherwise come back empty.

    Returns a list of symptom codes (subset of known_symptom_columns), or
    an empty list on any failure/unavailability.
    """
    if _client() is None or not text or not text.strip():
        return []

    vocab = ", ".join(known_symptom_columns)
    prompt = (
        "You are a symptom-extraction function, not a medical assistant. "
        "Given a user's free-text description of how they feel, return ONLY "
        "a JSON array of symptom codes chosen from this exact vocabulary "
        f"(use these exact strings, nothing else): [{vocab}]. "
        "Only include a code if the text clearly indicates that symptom is "
        "PRESENT (not denied/negated). If nothing in the vocabulary applies, "
        "return []. Do not explain, do not add codes outside the vocabulary.\n\n"
        f"Text: {text}"
    )
    try:
        from google.genai import types
        resp = _generate_with_failover(
            model=_MODEL_NAME,
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0, max_output_tokens=200),
        )
        data = _extract_json(getattr(resp, "text", None))
        if not isinstance(data, list):
            return []
        known = set(known_symptom_columns)
        return [c for c in data if isinstance(c, str) and c in known]
    except Exception as exc:
        _log_error("suggest_symptoms_from_text", exc)
        return []


_SAFETY_SYSTEM_PROMPT = (
    "You are a health-awareness assistant inside an educational app. "
    "You are NOT a doctor and must never diagnose, never name a specific "
    "disease as confirmed, never prescribe or recommend a specific "
    "medication or dosage, and never contradict the app's own risk "
    "screening. Give general, cautious health-awareness information in 2-4 "
    "short sentences. If the question describes anything potentially "
    "urgent (severe chest pain, trouble breathing, fainting, stroke signs, "
    "severe bleeding/allergic reaction, suicidal thoughts), tell the person "
    "to seek emergency care immediately instead of answering normally. "
    "Always gently suggest consulting a qualified doctor for anything "
    "beyond general awareness."
)


def chat_reply(user_message, context_note=""):
    """
    Free-form health-awareness Q&A fallback for the chat assistant, used
    only when the deterministic intent matcher in chat_assistant.py does
    not recognize the message. Returns None on any failure so the caller
    can fall back to its own generic response.
    """
    if _client() is None or not user_message or not user_message.strip():
        return None
    prompt = _SAFETY_SYSTEM_PROMPT
    if context_note:
        prompt += f"\n\nRelevant app context: {context_note}"
    prompt += f"\n\nUser message: {user_message}\n\nReply:"
    try:
        from google.genai import types
        resp = _generate_with_failover(
            model=_MODEL_NAME,
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.3, max_output_tokens=220),
        )
        text = getattr(resp, "text", None)
        return text.strip() if text and text.strip() else None
    except Exception as exc:
        _log_error("chat_reply", exc)
        return None


# ===========================================================================
# Agentic chat: a general-purpose, conversational assistant (like ChatGPT /
# Claude / Gemini apps) with full multi-turn memory and the ability to
# actively call the app's own tools (symptom analysis, recovery/adherence
# lookup) mid-conversation, instead of only answering from rule-based
# intents. Emergency detection is deliberately NOT exposed as an optional
# tool here -- it stays a hard, always-run pre-check in chat_assistant.py,
# because safety-critical logic must never depend on the model remembering
# to call something.
# ===========================================================================

_AGENTIC_SYSTEM_PROMPT = (
    "You are the AI Health Companion's chat assistant. You are a helpful, "
    "general-purpose conversational assistant -- like ChatGPT, Claude, or "
    "Gemini -- and you can discuss absolutely anything the person brings "
    "up, not only health topics. Be warm, direct, and concise; use plain "
    "language and short paragraphs unless the person's question calls for "
    "more depth.\n\n"
    "For anything about how the person is feeling physically, their "
    "symptoms, their recovery progress, or medicine adherence: actively "
    "use the tools available to you (analyze_symptoms, get_recovery_status) "
    "to ground your answer in the app's own model/data rather than "
    "guessing -- the way a research or coding assistant would run a tool "
    "before answering instead of assuming. Call analyze_symptoms whenever "
    "the person describes a new symptom, even in passing during a broader "
    "conversation.\n\n"
    "Hard safety rules, which always apply regardless of how the "
    "conversation is going: you are NOT a doctor. Never state a disease as "
    "confirmed or diagnosed -- describe model output as 'a possible "
    "pattern match', not a diagnosis. Never prescribe or recommend a "
    "specific medication, dosage, or dosage change. Never contradict the "
    "app's own risk screening. For anything beyond general awareness, "
    "gently suggest seeing a qualified doctor. If a message describes "
    "anything that sounds like a medical emergency, prioritize telling the "
    "person to seek urgent/emergency care over continuing the "
    "conversation normally."
)


def _history_to_contents(history, max_turns=24):
    """Convert the app's chat history format
    ([{"role": "user"/"assistant", "content": "..."}]) into Gemini's
    `contents` format ([{"role": "user"/"model", "parts": [...]}]),
    keeping only the most recent `max_turns` so token usage/cost stays
    bounded on long-running conversations."""
    contents = []
    for turn in (history or [])[-max_turns:]:
        role = "model" if turn.get("role") == "assistant" else "user"
        text = turn.get("content")
        if not text:
            continue
        contents.append({"role": role, "parts": [{"text": str(text)}]})
    return contents


def _build_tools(uid):
    """Build the tool functions for this conversation, closing over `uid`
    so the recovery-status tool looks up the right person's data. Google's
    genai SDK supports "automatic function calling": passing plain,
    type-hinted, well-documented Python callables directly as `tools` lets
    the SDK infer their schema, call them itself when the model requests,
    and feed the results back -- no manual request/response loop needed
    here."""
    effective_uid = uid or "guest"

    def analyze_symptoms(description: str) -> dict:
        """Run the app's own symptom-checking model on a description of how
        someone feels, to get matched symptoms, the closest disease
        patterns with probabilities, and a risk level. Call this whenever
        the person describes symptoms, pain, or how they are feeling
        physically, so the answer is grounded in the app's real screening
        model instead of a guess.

        Args:
            description: The symptoms as described by the person, in their
                own words (can be the current message or a summary of
                several messages).

        Returns:
            A dict with matched_symptoms, denied_symptoms,
            top_disease_patterns (list of {disease, probability}),
            risk_level, low_confidence (bool), and followup_questions
            (a list of questions to ask if evidence is thin).
        """
        from symptom_analyzer import predict_from_text
        result, matched = predict_from_text(description)
        return {
            "matched_symptoms": sorted(matched),
            "denied_symptoms": result.get("denied_symptoms", []),
            "top_disease_patterns": result.get("top_diseases", []),
            "risk_level": result.get("risk_level"),
            "low_confidence": result.get("low_confidence"),
            "followup_questions": result.get("followup_questions", []),
        }

    def get_recovery_status() -> dict:
        """Get the current person's recovery-tracking summary: how many
        days they have logged, their symptom trend (Improving/Stable/
        Worsening), medicine adherence percentage, and latest health
        score. Call this when the person asks about their recovery,
        progress, or whether they've been taking their medicine.

        Returns:
            A dict with days_tracked, trend, adherence_pct, and
            latest_health_score.
        """
        import recovery_tracker as rt
        s = rt.summary(effective_uid)
        return {
            "days_tracked": s.get("days_tracked"),
            "trend": s.get("trend"),
            "adherence_pct": s.get("adherence_pct"),
            "latest_health_score": s.get("latest_health_score"),
        }

    return [analyze_symptoms, get_recovery_status]


def agentic_chat_reply(user_message, history=None, uid=None, app_context=None):
    """
    Full ChatGPT/Claude-style conversational reply: complete multi-turn
    history is sent every time (Gemini itself is stateless between calls),
    and the model can actively call app tools (symptom analysis, recovery
    status) mid-conversation via automatic function calling.

    Returns the reply text, or None on any failure/unavailability so the
    caller can fall back to the deterministic rule-based chat assistant.
    """
    if _client() is None or not user_message or not user_message.strip():
        return None

    system_instruction = _AGENTIC_SYSTEM_PROMPT
    if app_context:
        system_instruction += (
            f"\n\nCurrent app context (from the person's most recent Symptom "
            f"Analyzer result, if any): last pattern match = "
            f"{app_context.get('disease')}, risk level = "
            f"{app_context.get('risk_level')}."
        )

    contents = _history_to_contents(history) + [
        {"role": "user", "parts": [{"text": user_message}]}
    ]

    try:
        from google.genai import types
        resp = _generate_with_failover(
            model=_MODEL_NAME,
            contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                tools=_build_tools(uid),
                temperature=0.4,
                max_output_tokens=500,
            ),
        )
        text = getattr(resp, "text", None)
        return text.strip() if text and text.strip() else None
    except Exception as exc:
        _log_error("agentic_chat_reply", exc)
        return None