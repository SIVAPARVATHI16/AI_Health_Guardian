"""
app.py — AI Health Guardian (Version 2.0)
Full-featured: Symptom AI, multilingual voice assistant, risk assessment,
image analysis, tracking, recovery dashboard, medicine tracker, chat and online emergency tools.
"""

import sys, os, json
from datetime import datetime
from PIL import Image
import streamlit as st
import io
import speech_recognition as sr_audio
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

# Load GEMINI_API_KEY (and any other secrets) from a local .env file if
# python-dotenv is installed and a .env file is present. This is entirely
# optional and silent when absent -- .streamlit/secrets.toml or a real
# shell environment variable work just as well and don't need this. Either
# way, the actual key value is never read from or written to source code.
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

sys.path.append(os.path.join(os.path.dirname(__file__), "..", "modules"))

from symptom_analyzer import predict_from_text
from image_analyzer import classify_image, compare_conditions, compare_analysis_results, LABEL_DISPLAY
import recovery_tracker as rt
from genai_explainer import generate_explanation, followup_recommendation
from chat_assistant import chat_response
from safety import detect_emergency, emergency_message
from health_risk import assess as assess_health_risk
from session_manager import (
    list_sessions, create_session, load_session, save_session,
    add_log_entry, add_chat_message, clear_chat_history, set_medicines, set_symptom_result,
    set_image_result, close_session,
    add_image_analysis, get_image_history, get_previous_image_analysis,
    get_image_path, add_recovery_entry, get_recovery_history,
    load_general_chat_history, add_general_chat_message, clear_general_chat_history
)

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Health Guardian",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded",
)

RISK_COLOR = {"Low": "#2E8B57", "Medium": "#D9A300", "High": "#C0392B"}
RISK_BG    = {"Low": "#e6f4ea", "Medium": "#f5f4f0", "High": "#f4f2f2"}
RISK_EMOJI = {"Low": "🟢", "Medium": "🟡", "High": "🔴"}


def render_severity_guidance(result):
    """Render a plain-language severity guide similar to the requested cough table."""
    sg = result.get("severity_guidance") or {}
    if not sg:
        return

    risk = result.get("risk_level", "Medium")
    color = RISK_COLOR.get(risk, RISK_COLOR["Medium"])
    bg = RISK_BG.get(risk, RISK_BG["Medium"])
    title = sg.get("display_level", f"{risk} Risk")
    emoji = sg.get("emoji", RISK_EMOJI.get(risk, "🟡"))
    source = result.get("severity_source", "ML model")
    prob = result.get("risk_model_probability")

    prob_text = (f" · ML probability for selected level: {prob*100:.1f}%"
                  if isinstance(prob, (int, float)) else "")
    current_card = f"""<div style="background:{bg};border:1px solid {color};border-left:7px solid {color};
    padding:16px 18px;border-radius:10px;margin:14px 0 10px 0">
    <div style="font-size:25px;font-weight:700;color:{color}">{emoji} {title}</div>
    <div style="margin-top:7px;font-size:16px">{sg.get('description','')}</div>
    <div style="margin-top:8px;font-size:14px"><b>Why this level:</b> {sg.get('reason','')}</div>
    <div style="margin-top:5px;font-size:13px;opacity:.85">Assessment source: {source}{prob_text}</div>
    </div>"""
    st.markdown(current_card, unsafe_allow_html=True)

    if sg.get("is_cough_guide"):
        st.markdown("### 🫁 Cough Severity Guide")
        st.caption("These are screening/awareness descriptions, not a diagnosis. A cough can have many causes.")
    else:
        st.markdown("### Severity Guide")

    guide = sg.get("guide", {})
    rows = []
    for level in ("Low", "Medium", "High"):
        item = guide.get(level)
        if not item:
            continue
        rows.append(
            f"""<tr>
            <td style=\"padding:14px 12px;font-weight:700;white-space:nowrap\">{RISK_EMOJI[level]} {item['display']}</td>
            <td style=\"padding:14px 12px;line-height:1.5\">{item['examples']}</td>
            <td style=\"padding:14px 12px;line-height:1.5\">{item['action']}</td>
            </tr>"""
        )
    table = """<table style="width:100%;border-collapse:collapse;border:1px solid #ddd;border-radius:8px;overflow:hidden">
    <thead><tr style="background:#f5f5f5">
    <th style="text-align:left;padding:12px">Risk level</th>
    <th style="text-align:left;padding:12px">Typical symptom pattern</th>
    <th style="text-align:left;padding:12px">What to do</th>
    </tr></thead><tbody>""" + "".join(rows) + "</tbody></table>"
    st.markdown(table, unsafe_allow_html=True)

    if risk == "High":
        st.error("🚨 High-risk warning signs were detected. Do not rely on this app alone; seek urgent medical evaluation, and use local emergency services for an emergency.")

# ── Session state ─────────────────────────────────────────────────────────────
for k, v in [("user_id","patient_001"), ("active_session",None),
             ("last_symptom_result",None), ("last_image_result",None),
             ("last_prev_image_result",None)]:
    if k not in st.session_state:
        st.session_state[k] = v

# ── Voice input helper ────────────────────────────────────────────────────────
def _voice_language_candidates(language):
    """Return Google Speech recognition locale fallbacks for the selected language."""
    aliases = {
        "en-IN": ["en-IN", "en-US", "en"],
        "hi-IN": ["hi-IN", "hi"],
        "te-IN": ["te-IN", "te"],
        "ta-IN": ["ta-IN", "ta"],
        "kn-IN": ["kn-IN", "kn"],
        "bn-IN": ["bn-IN", "bn"],
        "mr-IN": ["mr-IN", "mr"],
        "gu-IN": ["gu-IN", "gu"],
        "ml-IN": ["ml-IN", "ml"],
        "pa-IN": ["pa-IN", "pa"],
        "ur-IN": ["ur-IN", "ur-PK", "ur"],
        "ar-SA": ["ar-SA", "ar-EG", "ar"],
        "fr-FR": ["fr-FR", "fr"],
        "es-ES": ["es-ES", "es-MX", "es"],
    }
    return aliases.get(language, [language])


def _expected_script(language):
    """Unicode ranges used to verify that Google returned the requested script."""
    if language.startswith("hi") or language.startswith("mr"):
        return "devanagari"
    if language.startswith("te"):
        return "telugu"
    if language.startswith("ta"):
        return "tamil"
    if language.startswith("kn"):
        return "kannada"
    if language.startswith("bn"):
        return "bengali"
    if language.startswith("gu"):
        return "gujarati"
    if language.startswith("ml"):
        return "malayalam"
    if language.startswith("pa"):
        return "gurmukhi"
    if language.startswith("ur") or language.startswith("ar"):
        return "arabic"
    return None


def _has_script(text, script):
    if not text or not script:
        return True
    ranges = {
        "devanagari": (0x0900, 0x097F),
        "bengali": (0x0980, 0x09FF),
        "gurmukhi": (0x0A00, 0x0A7F),
        "gujarati": (0x0A80, 0x0AFF),
        "tamil": (0x0B80, 0x0BFF),
        "telugu": (0x0C00, 0x0C7F),
        "kannada": (0x0C80, 0x0CFF),
        "malayalam": (0x0D00, 0x0D7F),
        "arabic": (0x0600, 0x06FF),
    }
    lo, hi = ranges[script]
    return any(lo <= ord(ch) <= hi for ch in text)


def transcribe_audio(uploaded_audio, language="en-IN"):
    """Transcribe Streamlit microphone audio.

    Tries Gemini's native audio transcription first when configured (it is
    both more accurate and more reliable across languages than the free,
    unofficial Google Web Speech endpoint used as the fallback below).
    Falls back to the locale-retry Google Speech approach when Gemini is
    not configured or fails for any reason, so voice input keeps working
    either way.
    """
    if uploaded_audio is None:
        return None, "No recording received. Click the microphone, speak, then stop recording."

    try:
        audio_bytes = uploaded_audio.getvalue()
        if not audio_bytes:
            return None, "The recording is empty. Please record again."

        # --- Preferred path: Gemini (if configured) ---
        try:
            from gemini_assistant import transcribe_audio_gemini, gemini_available
            if gemini_available():
                lang_name = LANG_NAME_BY_CODE.get(language, language)
                text, error = transcribe_audio_gemini(
                    audio_bytes, mime_type=getattr(uploaded_audio, "type", "audio/wav"),
                    language_hint=lang_name,
                )
                if text:
                    return text, None
                # fall through to the Google Speech path below on failure
        except Exception:
            pass  # Gemini is optional; never break voice input on failure.

        # --- Fallback path: free Google Web Speech endpoint with locale retries ---
        recognizer = sr_audio.Recognizer()
        recognizer.dynamic_energy_threshold = True
        recognizer.pause_threshold = 0.8
        recognizer.non_speaking_duration = 0.3

        errors = []
        script = _expected_script(language)

        for candidate in _voice_language_candidates(language):
            try:
                # Create a fresh AudioFile stream for every retry.
                with sr_audio.AudioFile(io.BytesIO(audio_bytes)) as source:
                    audio = recognizer.record(source)

                text = recognizer.recognize_google(audio, language=candidate)
                text = (text or "").strip()
                if not text:
                    continue

                # For non-Latin selected languages, reject an obvious wrong-script
                # result and let the next locale attempt try again.
                if script and not _has_script(text, script):
                    errors.append(f"{candidate}: wrong script returned")
                    continue

                return text, None
            except sr_audio.UnknownValueError:
                errors.append(f"{candidate}: speech not understood")
                continue
            except sr_audio.RequestError as exc:
                return None, (
                    "Google speech recognition is unavailable. Check your internet "
                    f"connection and try again. Details: {exc}"
                )
            except Exception as exc:
                errors.append(f"{candidate}: {exc}")
                continue

        return None, (
            f"I could not recognize speech in {language}. "
            "Please select the language you are actually speaking, speak clearly for "
            "2-8 seconds close to the microphone, and record again."
        )
    except Exception as exc:
        return None, f"Could not process the recording: {exc}"


def voice_input(label, key, language):
    """Record from the browser microphone and persist the transcription.

    IMPORTANT: st.audio_input's returned value persists across Streamlit
    reruns until a new recording is made (it does not reset to None just
    because the app rerun). Without deduping on the recording's file_id,
    the same audio clip would be re-sent to the speech API and
    re-transcribed on every single rerun of the whole app (any button
    click anywhere), not just once per recording -- silently hammering the
    (rate-limited, unofficial) free Google speech endpoint and making
    "Clear voice text" appear broken, since the very next rerun would
    immediately re-populate the text from the still-attached recording.
    """
    audio = st.audio_input(label, key=key)
    state_key = f"{key}_text"
    last_id_key = f"{key}_last_file_id"

    if state_key not in st.session_state:
        st.session_state[state_key] = ""

    if audio is not None and st.session_state.get(last_id_key) != audio.file_id:
        st.session_state[last_id_key] = audio.file_id
        with st.spinner("Transcribing..."):
            text, error = transcribe_audio(audio, language)
        if text:
            st.session_state[state_key] = text
            st.success(f"🎤 Heard ({language}): {text}")
        elif error:
            st.session_state[state_key] = ""
            st.warning(error)

    return st.session_state[state_key]


def speak_text(text, language, key="tts"):
    """Speak text in the selected language using the browser SpeechSynthesis API.

    IMPORTANT: this renders inside a sandboxed iframe (st.components.v1.html).
    Calling speechSynthesis.speak() automatically when the script loads is
    NOT a genuine user gesture from that iframe's point of view, and modern
    browsers silently block autoplaying speech/audio without one -- which is
    why "Read this aloud" could appear to do nothing even though no error
    was shown. Rendering an actual clickable button INSIDE the iframe (so
    the click event originates in that frame) satisfies the browser's
    user-gesture requirement.
    """
    import json as _json
    safe_text = _json.dumps(str(text)[:1200], ensure_ascii=False)
    safe_lang = _json.dumps(language)
    st.components.v1.html(
        f"""
        <div style="font-family:sans-serif">
          <button id="{key}_play" style="padding:8px 14px;border-radius:6px;border:1px solid #ccc;
              background:#f0f2f6;cursor:pointer;font-size:14px;">🔊 Play</button>
          <button id="{key}_stop" style="padding:8px 14px;border-radius:6px;border:1px solid #ccc;
              background:#f0f2f6;cursor:pointer;font-size:14px;margin-left:6px;">⏹ Stop</button>
          <span id="{key}_status" style="margin-left:8px;color:#888;font-size:13px;"></span>
        </div>
        <script>
          (() => {{
            const text = {safe_text};
            const lang = {safe_lang};
            const statusEl = document.getElementById('{key}_status');
            const playBtn = document.getElementById('{key}_play');
            const stopBtn = document.getElementById('{key}_stop');
            if (!('speechSynthesis' in window)) {{
              statusEl.textContent = 'Speech is not supported in this browser.';
              playBtn.disabled = true;
              return;
            }}
            playBtn.addEventListener('click', () => {{
              window.speechSynthesis.cancel();
              const u = new SpeechSynthesisUtterance(text);
              u.lang = lang;
              u.rate = 0.95;
              u.onstart = () => {{ statusEl.textContent = 'Speaking...'; }};
              u.onend = () => {{ statusEl.textContent = 'Done.'; }};
              u.onerror = (e) => {{ statusEl.textContent = 'Could not play audio (' + e.error + ').'; }};
              window.speechSynthesis.speak(u);
            }});
            stopBtn.addEventListener('click', () => {{
              window.speechSynthesis.cancel();
              statusEl.textContent = 'Stopped.';
            }});
          }})();
        </script>
        """,
        height=50,
    )

LANG_OPTIONS = {
    "English (India)": "en-IN",
    "Hindi": "hi-IN",
    "Telugu": "te-IN",
    "Tamil": "ta-IN",
    "Kannada": "kn-IN",
    "Bengali": "bn-IN",
    "Marathi": "mr-IN",
    "Gujarati": "gu-IN",
    "Malayalam": "ml-IN",
    "Punjabi": "pa-IN",
    "Urdu": "ur-IN",
    "Arabic": "ar-SA",
    "French": "fr-FR",
    "Spanish": "es-ES",
}
LANG_NAME_BY_CODE = {code: name for name, code in LANG_OPTIONS.items()}

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏥 AI Health Guardian")
    st.caption("Intelligent Symptom Tracking & Recovery Assistant")
    st.markdown("---")

    uid = st.text_input("Patient / User ID", value=st.session_state.user_id)
    st.session_state.user_id = uid

    lang_name = st.selectbox("Language / Voice", list(LANG_OPTIONS.keys()))
    lang_code  = LANG_OPTIONS[lang_name]

    st.markdown("---")
    st.markdown("**Sessions**")
    sessions = list_sessions(uid)
    session_names = ["➕ Start new session"] + [
        f"{'🟢' if s['status']=='active' else '📁'} {s['name']} ({s['started'][:10]}) · {s.get('entry_count',0)} logs · {s.get('image_count',0)} photos · {s.get('chat_count',0)} chats"
        for s in sessions
    ]
    sel = st.selectbox("My problems", session_names)

    if sel == "➕ Start new session":
        new_name = st.text_input("Session name", placeholder="e.g. Dengue Oct 2025")
        if st.button("Create", type="primary") and new_name:
            sid = create_session(uid, new_name)
            st.session_state.active_session = sid
            # A brand-new session has no prior results yet -- clear any
            # leftover state from whatever session was active before, so a
            # stale result from a *different* problem doesn't leak in.
            st.session_state.last_symptom_result = None
            st.session_state.last_image_result = None
            st.session_state.last_prev_image_result = None
            st.rerun()
    else:
        idx = session_names.index(sel) - 1
        if idx >= 0:
            sid = sessions[idx]["session_id"]
            if st.session_state.active_session != sid:
                st.session_state.active_session = sid
            sdata = load_session(uid, sid)
            # Always sync to the selected session's stored value (including
            # clearing it back to None when that session has none) -- not
            # just when a value happens to be present -- otherwise switching
            # from a session that HAD a result to one that doesn't would
            # keep showing the previous session's stale result.
            st.session_state.last_symptom_result = (sdata or {}).get("symptom_result")
            st.session_state.last_image_result = (sdata or {}).get("image_result")
            _selected_history = get_image_history(uid, sid) if sdata else []
            st.session_state.last_prev_image_result = (
                _selected_history[-2].get("analysis") if len(_selected_history) >= 2 else None
            )

    st.markdown("---")
    page = st.radio("Navigate", [
        "🧾 Symptom Analyzer",
        "❤️ Health Risk Assessment",
        "🚨 Emergency & Hospitals",
        "📸 Image Analysis",
        "📅 Daily Tracking",
        "📊 Recovery Dashboard",
        "💊 Medicine Tracker",
        "💬 Chat Assistant",
        "ℹ️ About",
    ])

    if st.button("🗑️ Reset tracking data"):
        if st.session_state.active_session:
            sd = load_session(uid, st.session_state.active_session) or {}
            sd["recovery_history"] = []
            sd["daily_log"] = []
            save_session(uid, st.session_state.active_session, sd)
            st.success("Tracking data cleared for the active session only.")
        else:
            rt.reset_user(uid); st.success("Legacy tracking data cleared.")

    st.markdown("---")
    st.caption("⚠️ For health awareness only. Not a medical diagnosis tool.")

# ── Active session banner ─────────────────────────────────────────────────────
if st.session_state.active_session:
    sdata = load_session(uid, st.session_state.active_session)
    if sdata:
        st.info(f"📂 Active session: **{sdata['name']}** — started {sdata['started'][:10]}")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 1: Symptom Analyzer
# ═══════════════════════════════════════════════════════════════════════════════
if page == "🧾 Symptom Analyzer":
    st.title("🧾 Symptom Analyzer")
    st.caption("Module 1 — NLP keyword extraction + Machine Learning (Random Forest)")


    col1, col2 = st.columns([2, 1])
    with col1:
        # IMPORTANT: Do not write to st.session_state["sym_text"] after the
        # text_area widget with key="sym_text" has been created. Streamlit
        # treats widget keys as owned by the widget for the current run and
        # raises StreamlitAPIException if they are modified afterwards.
        text_input = st.text_area(
            "Describe your symptoms in any language",
            placeholder="e.g. fever, headache, body pain for 2 days",
            height=100, key="sym_text"
        )

        st.markdown("#### 🎤 Voice Input")
        voice_symptoms = voice_input("Click here to record your symptoms", "symptom_voice", lang_code)

        # Keep voice text in a SEPARATE state key. It is intentionally not
        # copied into the text_area widget's key. This survives the rerun
        # caused by clicking the Analyze button.
        if voice_symptoms:
            st.session_state["voice_symptom_text"] = voice_symptoms

        stored_voice = st.session_state.get("voice_symptom_text", "")
        if stored_voice:
            st.success(f"🎤 Heard: {stored_voice}")
            if st.button("Clear voice text", key="clear_symptom_voice"):
                st.session_state.pop("voice_symptom_text", None)
                st.rerun()
    with col2:
        quick = st.multiselect("Or quick-select", [
            "fever", "headache", "body pain", "cough", "sore throat",
            "vomiting", "diarrhea", "rash", "fatigue", "dizziness",
            "breathlessness", "chest pain", "joint pain", "nausea"
        ])

    if st.button("🔍 Analyze Symptoms", type="primary"):
        # Use manual text plus the last successfully transcribed voice text.
        # The voice text is stored separately so Streamlit widget state is
        # never modified after the text_area has been instantiated.
        stored_voice = st.session_state.get("voice_symptom_text", "")
        combined = " ".join(
            part for part in [text_input, stored_voice, " ".join(quick)] if part
        )
        if not combined.strip():
            st.warning("Please enter or select at least one symptom.")
        else:
            # Safety first: urgent-symptom check overrides normal analysis.
            is_emergency, emergency_category = detect_emergency(combined)
            if is_emergency:
                st.error(f"🚨 {emergency_message(emergency_category)}")
            result, matched = predict_from_text(combined)
            if not matched:
                st.warning("No recognized symptoms matched. Try common terms like 'fever', 'headache', 'knee pain', or describe where it hurts and for how long.")
                if result.get("followup_questions"):
                    st.markdown("**A few questions that might help me understand better:**")
                    for q in result["followup_questions"]:
                        st.write("•", q)
            else:
                st.session_state.last_symptom_result = result
                if st.session_state.active_session:
                    set_symptom_result(uid, st.session_state.active_session, result)

                if result.get("denied_symptoms"):
                    st.caption("Noted as **not present**: " + ", ".join(
                        s.replace("_", " ") for s in result["denied_symptoms"]))

                if result.get("ai_assisted"):
                    st.caption("🤖 Gemini helped interpret this description (no exact keyword match was found).")

                risk = result["risk_level"]
                bg = RISK_BG[risk]; col = RISK_COLOR[risk]

                if result.get("low_confidence"):
                    st.warning(
                        "⚠️ There isn't enough matching evidence yet for a confident pattern match "
                        f"({result['matched_symptom_count']} relevant symptom(s) detected). "
                        "The result below is shown for reference only — please answer a few "
                        "follow-up questions or add more detail for a better result."
                    )
                    if result.get("followup_questions"):
                        st.markdown("**Follow-up questions:**")
                        for q in result["followup_questions"]:
                            st.write("•", q)

                # Risk banner
                st.markdown(
                    f"""<div style="background:{bg};border-left:6px solid {col};
                    padding:16px;border-radius:8px;margin:12px 0">
                    <h2 style="color:{col};margin:0">{RISK_EMOJI[risk]} Risk Level: {risk}</h2>
                    <p style="margin:4px 0 0;color:{col}">
                    {result['matched_symptom_count']} symptoms matched</p></div>""",
                    unsafe_allow_html=True
                )

                render_severity_guidance(result)

                if risk == "High":
                    st.error("🚨 HIGH RISK DETECTED — Please go to the Emergency page immediately!")

                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("#### Top Probable Conditions")
                    for d in result["top_diseases"]:
                        st.progress(d["probability"],
                            text=f"{d['disease']} — {d['probability']*100:.1f}%")
                with c2:
                    st.markdown("#### Risk Probability Breakdown")
                    rdf = pd.DataFrame({
                        "Risk": list(result["risk_probabilities"].keys()),
                        "Probability": list(result["risk_probabilities"].values())
                    })
                    fig = px.bar(rdf, x="Risk", y="Probability", color="Risk",
                        color_discrete_map=RISK_COLOR, range_y=[0,1])
                    fig.update_layout(showlegend=False, height=250, margin=dict(t=10,b=10))
                    st.plotly_chart(fig, use_container_width=True)

                top_disease = result["top_diseases"][0]["disease"]
                expl = generate_explanation(top_disease, risk_level=risk, trend="Stable",
                                            adherence_pct=100, days_tracked=1)
                st.markdown("---")
                st.markdown("#### 🤖 AI Guidance")
                st.info(expl["summary"])
                d1, d2, d3 = st.columns(3)
                for col_el, (meal, food) in zip([d1,d2,d3], expl["diet_plan"].items()):
                    with col_el:
                        st.markdown(f"**{meal.capitalize()}**")
                        st.write(food)

                st.warning(f"⚠️ {expl['safety_note']}")

                # TTS readback -- a self-contained Play/Stop control (see
                # speak_text docstring for why this can't be behind a
                # separate Streamlit button that triggers a rerun first).
                summary_text = f"Risk level is {risk}. {expl['summary'][:200]}"
                speak_text(summary_text, lang_code, key="read_result_aloud")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2: Health Risk Assessment
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "❤️ Health Risk Assessment":
    st.title("❤️ Health Risk Assessment")
    st.caption("Transparent wellness screening based on the information you enter — not a clinical risk calculator.")

    c1, c2, c3 = st.columns(3)
    with c1:
        age = st.number_input("Age", min_value=1, max_value=120, value=20)
        height = st.number_input("Height (cm)", min_value=50.0, max_value=250.0, value=160.0, step=0.5)
    with c2:
        weight = st.number_input("Weight (kg)", min_value=10.0, max_value=300.0, value=60.0, step=0.5)
        systolic = st.number_input("Systolic BP (mmHg)", min_value=70, max_value=250, value=120)
    with c3:
        activity = st.selectbox("Physical activity", ["Low", "Moderate", "High"])
        smoking = st.checkbox("Current smoking")
        family_history = st.checkbox("Relevant family history")

    if st.button("📊 Calculate Wellness Risk", type="primary"):
        rr = assess_health_risk(age, height, weight, systolic, activity, smoking, family_history)
        st.session_state["last_wellness_risk"] = rr
        level = rr["level"]
        banner = {"Low": ("🟢", RISK_COLOR["Low"], RISK_BG["Low"]), "Medium": ("🟡", RISK_COLOR["Medium"], RISK_BG["Medium"]), "High": ("🔴", RISK_COLOR["High"], RISK_BG["High"])}[level]
        e, color, bg = banner
        st.markdown(f'''<div style="background:{bg};border-left:6px solid {color};padding:18px;border-radius:10px"><h2 style="color:{color};margin:0">{e} {level} screening level</h2><p style="margin:6px 0 0">Score: <b>{rr["score"]}</b>. This score is an educational screening aid, not a diagnosis.</p></div>''', unsafe_allow_html=True)
        m1, m2 = st.columns(2)
        m1.metric("BMI", rr["bmi"] if rr["bmi"] is not None else "—", rr["bmi_label"])
        m2.metric("Systolic BP", f"{systolic} mmHg")
        st.markdown("### Factors contributing to the score")
        if rr["factors"]:
            for factor in rr["factors"]:
                st.write("•", factor)
        else:
            st.success("No extra screening factors were detected from the entered information.")
        if level == "High":
            st.warning("Please discuss these results with a qualified healthcare professional, especially if you have symptoms or abnormal readings.")
        else:
            st.info("Use this as a wellness-monitoring aid. Recheck measurements under consistent conditions and seek professional advice for persistent concerns.")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 2: Emergency & Hospitals
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "🚨 Emergency & Hospitals":
    st.title("🚨 Emergency Response Centre")
    result = st.session_state.last_symptom_result
    risk = result["risk_level"] if result else "Medium"
    disease = result["top_diseases"][0]["disease"] if result else "Unknown condition"

    if risk == "High":
        st.error("🚨 HIGH EMERGENCY RISK — seek professional emergency care immediately.")
    elif risk == "Medium":
        st.warning(f"⚠️ MEDIUM RISK — monitor closely. Condition: **{disease}**")
    else:
        st.success(f"✅ LOW RISK — continue monitoring. Condition: **{disease}**")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("### 🚑 Call Emergency Services")
        ambulance_no = st.text_input("Emergency number", value="112", key="online_ambulance")
        st.link_button("📞 Open phone dialler", f"tel:{ambulance_no}", use_container_width=True)
        st.caption("India: 112 is the national emergency number; 108 is an ambulance service in many states.")
    with c2:
        st.markdown("### 🌐 Online Hospital Search")
        area = st.text_input("City / area", placeholder="e.g. Visakhapatnam", key="online_area")
        if area:
            q = area.replace(" ", "+")
            st.link_button("🗺️ Open OpenStreetMap", f"https://www.openstreetmap.org/search?query=hospital+{q}", use_container_width=True)
            st.link_button("📍 Open Google Maps", f"https://www.google.com/maps/search/hospitals+near+{q}", use_container_width=True)

    st.markdown("---")
    st.markdown("### 🩺 First Aid While Waiting")
    if result:
        d = disease.lower()
        if "dengue" in d:
            st.info("Rest, hydrate, and seek urgent care for bleeding, severe abdominal pain, persistent vomiting, breathing difficulty, confusion or marked weakness. Avoid aspirin/ibuprofen unless a clinician has told you otherwise.")
        elif "asthma" in d:
            st.info("Sit upright and use your prescribed rescue inhaler according to your action plan. If severe breathing difficulty persists, call emergency services immediately.")
        else:
            st.info("Keep the person safe and comfortable, monitor breathing and consciousness, and contact emergency services for severe or rapidly worsening symptoms.")
    else:
        st.info("Complete the Symptom Analyzer first for condition-specific guidance.")

elif page == "📸 Image Analysis":
    st.title("📸 Skin Progress & Image Analysis")
    st.caption("CNN screening + repeat-photo comparison + persistent session history")

    st.markdown(
        "Upload **today's photo only**. Once you analyze it, the app stores the "
        "result and the image inside the active health session. On the next day, "
        "your new photo is automatically compared with the most recent stored photo."
    )

    if not st.session_state.active_session:
        st.warning("Create/select a health session in the sidebar first. The session is what lets the app remember yesterday's result.")
    else:
        sdata = load_session(uid, st.session_state.active_session) or {}
        history = get_image_history(uid, st.session_state.active_session)
        previous_record = get_previous_image_analysis(uid, st.session_state.active_session)
        latest_record = history[-1] if history else None

        # Show the last stored baseline before a new upload.
        if latest_record:
            st.info(
                f"📌 Previous stored analysis: {latest_record.get('timestamp','')[:16].replace('T',' ')} — "
                f"**{latest_record.get('analysis',{}).get('display_label', 'Skin analysis')}**. "
                "You do not need to upload that image again."
            )

        tf = st.file_uploader(
            "Upload today's skin image",
            type=["png", "jpg", "jpeg"],
            key="today_skin_persistent",
            help="For meaningful comparison, use the same body area, similar distance, angle and lighting each day.",
        )

        st.markdown("### 📷 Photo quality checklist")
        q1, q2, q3, q4 = st.columns(4)
        q1.write("✓ Same body area")
        q2.write("✓ Similar lighting")
        q3.write("✓ Similar distance")
        q4.write("✓ No beauty filter")

        if st.button("🔬 Analyze & Save Today's Image", type="primary", disabled=(tf is None)):
            try:
                today_img = Image.open(tf).convert("RGB")

                with st.spinner("Analyzing the image..."):
                    current_result = classify_image(today_img)

                # Compare against the immediately previous stored analysis AND
                # the actual previous photo. The direct photo comparison is what
                # lets the app detect a visibly larger/smaller region even when
                # the CNN gives the same broad class on both days.
                comparison = None
                previous_image = None
                if latest_record:
                    previous_analysis = latest_record.get("analysis")
                    previous_path = get_image_path(uid, st.session_state.active_session, latest_record)
                    if previous_path:
                        try:
                            previous_image = Image.open(previous_path).convert("RGB")
                        except Exception:
                            previous_image = None
                    comparison = compare_analysis_results(
                        previous_analysis,
                        current_result,
                        previous_image=previous_image,
                        current_image=today_img,
                    )

                record = add_image_analysis(
                    uid,
                    st.session_state.active_session,
                    today_img,
                    current_result,
                    comparison=comparison,
                )

                st.session_state.last_image_result = current_result
                st.session_state.last_prev_image_result = (
                    latest_record.get("analysis") if latest_record else None
                )

                st.success("✅ Today's analysis was saved to this health session.")

                a, b = st.columns([1, 1])
                with a:
                    st.image(today_img, caption="Today", use_container_width=True)

                with b:
                    label = current_result["display_label"]
                    conf = current_result["confidence"]
                    certainty = current_result.get("certainty", 0)
                    color = {
                        "healthy_skin": "#2E8B57",
                        "healing_rash": "#D9A300",
                        "inflamed_rash": "#C0392B",
                    }.get(current_result["predicted_class"], "#666")

                    st.markdown(
                        f'<div style="background:{color}22;border-left:5px solid {color};'
                        f'padding:12px;border-radius:8px">'
                        f'<h3 style="color:{color};margin:0">{label}</h3>'
                        f'<p>Model confidence: {conf*100:.1f}%</p>'
                        f'<p>Model certainty: {certainty*100:.1f}%</p></div>',
                        unsafe_allow_html=True,
                    )

                    prob_df = pd.DataFrame({
                        "Condition": [
                            LABEL_DISPLAY.get(k, k)
                            for k in current_result["all_probs"]
                        ],
                        "Probability": list(current_result["all_probs"].values()),
                    })
                    fig = px.bar(
                        prob_df, x="Probability", y="Condition",
                        orientation="h", range_x=[0, 1],
                    )
                    fig.update_layout(height=220, margin=dict(t=5, b=5))
                    st.plotly_chart(fig, use_container_width=True)

                vm = current_result.get("visual_metrics", {})
                st.markdown("### 🔎 Current image quality / visual measurements")
                m1, m2, m3, m4 = st.columns(4)
                m1.metric("Brightness", f"{vm.get('brightness', 0):.2f}")
                m2.metric("Contrast", f"{vm.get('contrast', 0):.2f}")
                m3.metric("Redness signal", f"{vm.get('redness_index', 0):.3f}")
                m4.metric(
                    "Visible affected-region estimate",
                    f"{(vm.get('affected_region') or {}).get('affected_area_pct', 0):.2f}%",
                    help="Adaptive visual estimate only; not a clinical lesion measurement.",
                )
                st.caption(
                    f"Region-estimate confidence: {(vm.get('affected_region') or {}).get('region_confidence', 0)*100:.0f}% · "
                    "used only when sufficiently reliable. It is not a diagnosis."
                )

                if comparison and previous_image is not None:
                    st.markdown("---")
                    st.markdown("## 📈 Automatic Day-to-Day Comparison")

                    x1, x2, x3 = st.columns([1, 1, 1])
                    with x1:
                        st.image(
                            previous_image,
                            caption=f"Previous: {LABEL_DISPLAY.get(comparison.get('previous_class'), comparison.get('previous_class'))}",
                            use_container_width=True,
                        )
                    with x2:
                        st.image(today_img, caption="Today", use_container_width=True)
                    with x3:
                        trend = comparison["trend"]
                        icons = {"Improving": "✅", "Stable": "➖", "Worsening": "⚠️", "Uncertain": "❓"}
                        colors = {"Improving": "#2E8B57", "Stable": "#777", "Worsening": "#C0392B", "Uncertain": "#B07A00"}
                        tc = colors.get(trend, "#777")
                        st.markdown(
                            f'<div style="background:{tc}22;border-left:6px solid {tc};'
                            f'padding:20px;border-radius:8px;text-align:center">'
                            f'<h2 style="color:{tc}">{icons.get(trend,"")} {trend}</h2>'
                            f'<p>Comparison confidence: {comparison["confidence"]*100:.0f}%</p>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric(
                        "Visual model score",
                        f"{comparison['current_score']:.1f}",
                        delta=f"{comparison['score_delta']:+.1f}",
                    )
                    c2.metric(
                        "Visible region estimate",
                        ((f"{comparison.get('current_affected_area_pct'):.2f}%" if comparison.get('current_affected_area_pct') is not None else "Legacy / unavailable")),
                        delta=(f"{comparison.get('affected_area_delta_pp', 0):+.2f} pp" if comparison.get('region_comparison_available') else None),
                        delta_color="inverse",
                    )
                    c3.metric(
                        "Changed image region",
                        f"{comparison.get('direct_change', {}).get('changed_region_fraction', 0)*100:.2f}%",
                        help="Fraction of the comparable photo containing the strongest localized visual change.",
                    )
                    c4.metric(
                        "Photo comparability",
                        f"{comparison['comparability']*100:.0f}%",
                    )

                    st.info(comparison["message"])

                    direct = comparison.get("direct_change") or {}
                    if direct:
                        d1, d2, d3 = st.columns(3)
                        d1.metric("Changed-region confidence", f"{direct.get('changed_region_confidence', 0)*100:.0f}%")
                        d2.metric("Redness change in changed region", f"{direct.get('changed_region_red_delta', 0):+.3f}")
                        d3.metric("Darkness change in changed region", f"{direct.get('changed_region_dark_delta', 0):+.3f}")
                        st.caption(
                            "The trend engine compares the stored photos directly as well as the CNN probabilities. "
                            "This prevents identical CNN labels from automatically producing 'Stable'."
                        )
                    st.markdown("**Evidence used:**")
                    for reason in comparison.get("reasons", []):
                        st.write(f"• {reason}")

                    if trend == "Worsening":
                        st.warning(
                            "The images show a potentially unfavorable visual trend. "
                            "This is not a diagnosis. If the skin problem is rapidly "
                            "spreading, very painful, blistering/open, associated with "
                            "fever, or involves the eyes/mouth, seek medical care promptly."
                        )
                    elif trend == "Improving":
                        st.success(
                            "The stored visual measurements are more consistent with improvement. "
                            "This does not confirm that an underlying condition is cured."
                        )
                    elif trend == "Uncertain":
                        st.warning(
                            "Retake the next photo with similar lighting, distance and framing "
                            "before relying on the trend."
                        )
                    else:
                        st.info("No consistent enough visual change was detected.")

                elif latest_record:
                    st.warning(
                        "The previous analysis exists, but its stored image could not be opened. "
                        "Today's result was still saved; the next comparison will use today's image."
                    )
                else:
                    st.info(
                        "This is the first stored image for this session. "
                        "It will become the baseline for your next photo."
                    )

            except Exception as exc:
                st.error(f"Image analysis failed: {exc}")

        # Persistent history is visible even after a Streamlit rerun.
        history = get_image_history(uid, st.session_state.active_session)
        if history:
            st.markdown("---")
            st.markdown("## 🗓️ Stored Skin Analysis History")
            rows = []
            for item in reversed(history):
                a = item.get("analysis", {})
                cmp = item.get("comparison") or {}
                rows.append({
                    "Date / time": item.get("timestamp", "")[:16].replace("T", " "),
                    "Visual result": a.get("display_label", a.get("predicted_class", "")),
                    "Confidence": f"{a.get('confidence', 0)*100:.1f}%",
                    "Trend vs previous": cmp.get("trend", "Baseline"),
                    "Comparison confidence": (
                        f"{cmp.get('confidence', 0)*100:.0f}%"
                        if cmp else "—"
                    ),
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        st.caption(
            "⚠️ This module is a visual monitoring/screening feature. The CNN was trained "
            "on synthetic demo images and cannot diagnose a skin disease, measure true "
            "disease severity, or determine by itself whether a doctor is required."
        )

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 4: Daily Tracking
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📅 Daily Tracking":
    st.title("📅 Daily Health Tracking")
    st.caption("Module 3 — Recovery log. Each entry builds your health score trend.")

    with st.form("daily_form"):
        severity = st.slider("Overall symptom severity today", 1, 10, 5,
            help="1 = very mild, 10 = extremely severe")
        med_taken = st.radio("Did you take your medicine today?", ["Yes","No"], horizontal=True)
        notes = st.text_input("Notes (optional)", placeholder="e.g. Felt better after lunch")

        use_img = False
        if st.session_state.last_image_result:
            use_img = st.checkbox(
                f"Include image result ({LABEL_DISPLAY[st.session_state.last_image_result['predicted_class']]})"
                f" from Image Analysis", value=True)

        submitted = st.form_submit_button("✅ Save Today's Entry", type="primary")

    if submitted:
        img_cls = img_conf = None
        if use_img and st.session_state.last_image_result:
            img_cls  = st.session_state.last_image_result["predicted_class"]
            img_conf = st.session_state.last_image_result["confidence"]

        entry = rt.add_entry(uid, severity=severity,
            medicine_taken=(med_taken=="Yes"),
            image_class=img_cls, image_confidence=img_conf, notes=notes)

        if st.session_state.active_session:
            add_recovery_entry(uid, st.session_state.active_session, entry)

        st.success(f"Entry saved! Today's Health Score: **{entry['health_score']}/100**")
        st.balloons()

    st.markdown("---")
    st.markdown("#### Medicine Reminder")
    st.info("💊 Did you take your medicine? Log it above. Your adherence % updates automatically on the dashboard.")

    entries = (get_recovery_history(uid, st.session_state.active_session) if st.session_state.active_session else rt.load_log(uid))
    if entries:
        st.markdown("#### Recent Entries")
        df = pd.DataFrame(entries[::-1])
        st.dataframe(df[["timestamp","severity","medicine_taken","image_class","health_score","notes"]],
            use_container_width=True, hide_index=True)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 5: Recovery Dashboard
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "📊 Recovery Dashboard":
    st.title("📊 Recovery Dashboard")
    st.caption("Modules 3, 4 & 6 — Trend analytics + GenAI explainer + Follow-up")

    if st.session_state.active_session:
        entries = get_recovery_history(uid, st.session_state.active_session)
        if entries:
            # Build the same dashboard summary, but scoped to the selected session.
            import statistics
            scores = [e.get("health_score", 0) for e in entries]
            trend = "Not enough data"
            if len(scores) >= 2:
                m = len(scores)//2
                delta = statistics.mean(scores[m:]) - statistics.mean(scores[:m] if m else scores[:1])
                trend = "Improving" if delta >= 5 else "Worsening" if delta <= -5 else "Stable"
            adherence = round(100*sum(1 for e in entries[-7:] if e.get("medicine_taken"))/min(7,len(entries)),1) if entries else 100.0
            summary = {"days_tracked":len(entries),"trend":trend,"adherence_pct":adherence,"latest_health_score":scores[-1],"history":entries}
        else:
            summary = {"days_tracked":0,"trend":"Not enough data","adherence_pct":100.0,"latest_health_score":None,"history":[]}
    else:
        summary = rt.summary(uid)

    if summary["days_tracked"] == 0:
        st.info("No data yet. Go to **Daily Tracking** to log your first entry.")
    else:
        m1,m2,m3,m4 = st.columns(4)
        m1.metric("Days Tracked", summary["days_tracked"])
        m2.metric("Latest Score", f"{summary['latest_health_score']}/100")
        trend_icon = {"Improving":"📈","Stable":"➡️","Worsening":"📉"}.get(summary["trend"],"")
        m3.metric("Trend", f"{trend_icon} {summary['trend']}")
        m4.metric("Medicine Adherence", f"{summary['adherence_pct']}%")

        st.markdown("---")
        hdf = pd.DataFrame(summary["history"])
        hdf["entry"] = range(1, len(hdf)+1)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown("#### Health Score Over Time")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=hdf["entry"], y=hdf["health_score"],
                mode="lines+markers", line=dict(color="#2E8B57", width=3), marker=dict(size=10)))
            fig.update_layout(xaxis_title="Entry #", yaxis_title="Health Score (0-100)",
                yaxis_range=[0,100], height=300, margin=dict(t=10,b=10))
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            st.markdown("#### Symptom Severity Trend")
            fig2 = go.Figure()
            fig2.add_trace(go.Bar(x=hdf["entry"], y=hdf["severity"], marker_color="#C0392B"))
            fig2.update_layout(xaxis_title="Entry #", yaxis_title="Severity (1-10)",
                yaxis_range=[0,10], height=300, margin=dict(t=10,b=10))
            st.plotly_chart(fig2, use_container_width=True)

        st.markdown("#### Medicine Compliance")
        adh = summary["adherence_pct"]
        color = "#2E8B57" if adh >= 80 else "#D9A300" if adh >= 50 else "#C0392B"
        st.markdown(f'<div style="background:#f5f5f5;border-radius:8px;padding:8px">'
            f'<div style="background:{color};height:24px;width:{adh}%;border-radius:6px"></div>'
            f'<p style="text-align:center;margin:4px 0"><b>{adh}%</b> adherence</p></div>',
            unsafe_allow_html=True)

        st.markdown("---")
        sr = st.session_state.last_symptom_result
        disease = sr["top_diseases"][0]["disease"] if sr else "Common Cold"
        risk    = sr["risk_level"] if sr else "Low"

        expl = generate_explanation(disease, risk_level=risk, trend=summary["trend"],
            adherence_pct=summary["adherence_pct"], days_tracked=summary["days_tracked"])

        st.markdown("### 🤖 AI Recovery Explanation")
        st.info(expl["summary"])

        dc1,dc2,dc3 = st.columns(3)
        for col_el,(meal,food) in zip([dc1,dc2,dc3], expl["diet_plan"].items()):
            with col_el:
                st.markdown(f"**{meal.capitalize()}**")
                st.write(food)

        st.markdown("---")
        decision, reasoning = followup_recommendation(
            summary["trend"], risk, summary["days_tracked"], summary["adherence_pct"])
        dec_color = {"No checkup needed":"success","Monitor for 2 more days":"warning","Visit doctor recommended":"error"}[decision]
        getattr(st, dec_color)(f"**{decision}** — {reasoning}")

        text = f"Recovery trend is {summary['trend']}. {expl['summary'][:200]}. Recommendation: {decision}."
        speak_text(text, lang_code, key="read_summary_aloud")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 6: Medicine Tracker
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "💊 Medicine Tracker":
    st.title("💊 Medicine Tracker & Alarm")
    st.caption("Module 5 — Scan prescription, set timings, get diet guidance")

    st.markdown("#### Add Medicines from Your Prescription")
    st.info("📸 Tip: In a real app, upload a photo of your prescription here to auto-extract medicine names (OCR). For this demo, enter them manually below.")

    if st.session_state.active_session:
        sdata = load_session(uid, st.session_state.active_session)
        medicines = sdata.get("medicines", []) if sdata else []
    else:
        medicines = []

    with st.form("med_form"):
        med_name = st.text_input("Medicine name", placeholder="e.g. Paracetamol 500mg")
        times = st.multiselect("When to take", ["Morning 8 AM","Afternoon 2 PM","Evening 6 PM","Night 10 PM"])
        with_food = st.radio("Take with food?", ["Yes — after meals","No — before meals","Doesn't matter"], horizontal=True)
        duration  = st.number_input("For how many days?", min_value=1, max_value=60, value=5)
        add_med   = st.form_submit_button("➕ Add Medicine")

    if add_med and med_name:
        entry = {"name": med_name, "times": times, "with_food": with_food, "duration": int(duration), "added": datetime.now().strftime("%Y-%m-%d")}
        medicines.append(entry)
        if st.session_state.active_session:
            set_medicines(uid, st.session_state.active_session, medicines)
        st.success(f"Added **{med_name}**! Set browser notifications for the selected times.")

    if medicines:
        st.markdown("#### Your Medicines")
        for m in medicines:
            col1, col2 = st.columns([3,1])
            with col1:
                times_str = ", ".join(m["times"]) if m["times"] else "No times set"
                st.markdown(f"""
                <div style="border:1px solid #ddd;border-radius:8px;padding:12px;margin:6px 0">
                <b>💊 {m['name']}</b><br>
                ⏰ {times_str}<br>
                🍽️ {m['with_food']}<br>
                📅 For {m['duration']} days from {m['added']}
                </div>""", unsafe_allow_html=True)
            with col2:
                if st.button(f"🔔 Set alarm", key=f"alarm_{m['name']}"):
                    time_str = m["times"][0] if m["times"] else "8 AM"
                    st.components.v1.html(f"""
                    <script>
                    if ('Notification' in window) {{
                        Notification.requestPermission().then(p => {{
                            if (p === 'granted') {{
                                new Notification('💊 Medicine Reminder', {{
                                    body: 'Time to take {m["name"]} — {time_str}',
                                    icon: '🏥'
                                }});
                            }}
                        }});
                    }} else {{
                        alert('Medicine reminder: Take {m["name"]} at {time_str}');
                    }}
                    </script>""", height=0)
                    st.success(f"Alarm set for {m['name']}")
    else:
        st.info("No medicines added yet. Add them using the form above.")

    st.markdown("---")
    st.markdown("#### 🍎 Diet Guidance While on Medication")
    st.info("Eat light, nutritious meals. Avoid alcohol while on antibiotics. Drink plenty of water. Follow your doctor's specific diet instructions.")

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 7: Chat Assistant
# ═══════════════════════════════════════════════════════════════════════════════
elif page == "💬 Chat Assistant":
    # ── ChatGPT/Claude-style reskin of Streamlit's native chat widgets ──
    # (best-effort cosmetic CSS; every selector below is optional polish,
    # so if a future Streamlit version renames an internal test-id this
    # simply falls back to Streamlit's normal chat look, nothing breaks).
    st.markdown(
        """
        <style>
        div.block-container { padding-top: 1.6rem; max-width: 900px; }
        div[data-testid="stChatInput"] textarea { border-radius: 20px !important; }
        div[data-testid="stChatMessage"] { padding: 10px 0; }
        div[data-testid="stChatMessageContent"] p { font-size: 15.5px; line-height: 1.55; }
        div[data-testid="stButton"] button, div[data-testid="stDownloadButton"] button {
            min-height: 2.4rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # Each check is separate so we can tell the person the REAL reason for
    # basic mode (previously one failed import silently disabled Gemini
    # even when the key was configured correctly).
    _llm_on = False
    _quota_hit = False
    _gemini_problem = None
    try:
        import gemini_assistant as _ga
    except Exception as _exc:
        _ga = None
        _gemini_problem = f"Could not import gemini_assistant.py: {_exc!r}"

    if _ga is not None:
        try:
            _llm_on = bool(_ga.gemini_available())
        except Exception as _exc:
            _gemini_problem = f"gemini_available() failed: {_exc!r}"
        if not _llm_on and _gemini_problem is None:
            try:
                _has_key = bool(_ga._get_api_key())
            except Exception as _exc:
                _has_key = False
                _gemini_problem = f"Could not read the API key: {_exc!r}"
            if _gemini_problem is None:
                if not _has_key:
                    _gemini_problem = (
                        "No GEMINI_API_KEY was found. Streamlit reads `.streamlit/secrets.toml` from the "
                        "folder you run `streamlit run` in (or ~/.streamlit/). Put "
                        "`GEMINI_API_KEY = \"...\"` at the top level of that file and restart."
                    )
                else:
                    try:
                        import google.genai  # noqa: F401
                    except Exception as _exc:
                        _gemini_problem = (
                            f"Key found, but the `google-genai` package can't be imported: {_exc!r}. "
                            "Run `pip install google-genai` in the same environment as Streamlit, then restart."
                        )
        try:
            _quota_hit = bool(_ga.quota_recently_exhausted())
        except Exception:
            _quota_hit = False

    sr = st.session_state.last_symptom_result
    disease = sr["top_diseases"][0]["disease"] if sr else "Common Cold"
    risk    = sr["risk_level"] if sr else "Low"

    def _get_history():
        """Single source of truth for this chat's history -- reloaded
        fresh each run so 'New chat' and every new turn are reflected
        immediately, and passed in full to chat_response() so every reply
        (typed, voice, or quick-question) is grounded in everything said
        earlier in *this same chat*. Persisted to disk either way: to the
        active session's chat_history if one is selected, or otherwise to
        a general per-user chat file that's saved automatically -- no
        "Start new session" click required -- so it survives app restarts
        and browser refreshes too, not just page navigation."""
        if st.session_state.active_session:
            sdata = load_session(uid, st.session_state.active_session)
            return list((sdata or {}).get("chat_history", []))
        return load_general_chat_history(uid)

    def _persist(user_msg, reply):
        if st.session_state.active_session:
            add_chat_message(uid, st.session_state.active_session, "user", user_msg)
            add_chat_message(uid, st.session_state.active_session, "assistant", reply)
        else:
            add_general_chat_message(uid, "user", user_msg)
            add_general_chat_message(uid, "assistant", reply)

    def _send(msg):
        """One shared send path for the text box, voice, and quick-question
        buttons, so all three ways of talking to the assistant land in the
        same message list and the same memory -- previously, quick
        questions bypassed history entirely and were invisible to later
        turns."""
        if not msg or not str(msg).strip():
            return
        hist = _get_history()
        with st.spinner("Thinking..."):
            reply = chat_response(msg, disease=disease, risk_level=risk, history=hist,
                                   language=lang_code, uid=uid)
        _persist(msg, reply)
        st.session_state["_chat_tts_pending"] = reply
        st.rerun()

    history = _get_history()

    def _export_chat_text():
        chat_title = (
            (load_session(uid, st.session_state.active_session) or {}).get("name", "Chat")
            if st.session_state.active_session else "General chat"
        )
        lines = [
            "AI Health Companion -- Chat export",
            f"Chat: {chat_title}",
            f"User: {uid}",
            f"Exported: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "-" * 40, "",
        ]
        for msg in history:
            who = "You" if msg.get("role") == "user" else "Assistant"
            t = msg.get("time")
            lines.append(f"[{t}] {who}: {msg.get('content', '')}" if t else f"{who}: {msg.get('content', '')}")
            lines.append("")
        return "\n".join(lines)

    st.markdown("### 💬 AI Health Chat")

    status_col, dl_col, new_col = st.columns([6, 2, 2], vertical_alignment="center")
    with status_col:
        if _quota_hit:
            st.caption("⏳ **Daily free quota used up**")
        elif _llm_on:
            st.caption("🧠 **Full conversational mode**")
        else:
            st.caption("💡 **Basic mode** — no Gemini key")
    with dl_col:
        st.download_button(
            "⬇️ Download", data=_export_chat_text(), file_name="chat_export.txt",
            mime="text/plain", use_container_width=True, disabled=not history,
        )
    with new_col:
        if st.button("🗑️ New chat", use_container_width=True):
            if st.session_state.active_session:
                clear_chat_history(uid, st.session_state.active_session)
            else:
                clear_general_chat_history(uid)
            st.rerun()

    if _quota_hit:
        st.caption(
            "Google's Gemini free tier caps requests per day (as of writing, 20/day for this "
            "model). Today's quota is used up, so I'm answering with the basic assistant until "
            "it resets — usually within 24 hours, or sooner if Google resets it more often. "
            "Everything still works, just without open-ended AI conversation for now."
        )
    elif not _llm_on:
        if _gemini_problem:
            st.error(f"Why Gemini is off: {_gemini_problem}")
        st.caption(
            "Add a free `GEMINI_API_KEY` (from [Google AI Studio](https://aistudio.google.com/apikey)) "
            "to unlock open-ended, ChatGPT/Claude-style conversation that remembers this whole chat "
            "and can pull in your real symptom/recovery data as we talk. Until then, I can handle "
            "known health questions and commands."
        )
    st.caption(f"Context for this chat: **{disease}** · Risk: **{risk}** · "
               f"{'Saved to this session' if st.session_state.active_session else 'Auto-saved as your general chat'}")
    st.divider()

    # ── message list ──
    if not history:
        st.markdown(
            "<div style='text-align:center;color:#888;padding:56px 0 28px;'>"
            "👋 Ask me anything about how you're feeling — or chat about anything else."
            "<br>I'll remember this whole conversation as we go.</div>",
            unsafe_allow_html=True,
        )
    else:
        for msg in history:
            avatar = "🧑" if msg.get("role") == "user" else "🏥"
            with st.chat_message(msg.get("role", "assistant"), avatar=avatar):
                st.markdown(msg.get("content", ""))
                if msg.get("time"):
                    st.caption(msg["time"])

    # Speak only the reply that was *just* generated, not the whole
    # history on every rerun (see speak_text's own docstring on why this
    # needs a real button click, not an autoplay attempt).
    if st.session_state.get("_chat_tts_pending"):
        speak_text(st.session_state.pop("_chat_tts_pending"), lang_code, key="tts_chat")

    # ── voice input, tucked away so the main view stays clean ──
    with st.expander("🎤 Ask by voice instead"):
        st.caption(f"Voice recognition and spoken replies use **{lang_name}**. "
                    "Change the language from the sidebar before recording.")
        voice_msg = voice_input("Record your question", "chat_voice", lang_code)
        if voice_msg:
            st.info(f"Heard: {voice_msg}")
            vc1, vc2 = st.columns(2, vertical_alignment="center")
            with vc1:
                if st.button("Send", type="primary", key="send_voice_question", use_container_width=True):
                    # Clear first so the same transcript can't be re-sent on
                    # a later rerun of the page.
                    st.session_state["chat_voice_text"] = ""
                    _send(voice_msg)
            with vc2:
                if st.button("Clear", key="clear_chat_voice", use_container_width=True):
                    st.session_state["chat_voice_text"] = ""
                    st.rerun()

    # ── quick questions -- now go through the same _send() path, so they
    #    show up in the message list and are remembered like any other turn ──
    st.markdown("###### Quick questions")
    quick_qs = [
        "Can I exercise?", "What should I eat?", "Can I go to work?",
        "What are the do's and don'ts?", "When to see a doctor?", "Can I travel?"
    ]
    qcols = st.columns(2, vertical_alignment="center")
    for i, q in enumerate(quick_qs):
        with qcols[i % 2]:
            if st.button(q, key=f"qq{i}", use_container_width=True):
                _send(q)

    # ── main input, pinned to the bottom of the page like ChatGPT/Claude ──
    user_msg = st.chat_input("Message AI Health Chat...")
    if user_msg:
        _send(user_msg)

# ═══════════════════════════════════════════════════════════════════════════════
# PAGE 8: About
# ═══════════════════════════════════════════════════════════════════════════════
else:
    st.title("ℹ️ About AI Health Guardian")
    st.markdown("""
This is a full-stack multi-AI healthcare companion system.

| Module | AI Technique | Key Feature |
|---|---|---|
| Symptom Analyzer | NLP + ML (Random Forest) | Free-text to disease + risk |
| Image Analysis | Deep Learning (CNN) | Day-over-day skin comparison |
| Recovery Tracking | Data Science | Health score trend graphs |
| GenAI Explainer | Template + LLM-ready | Diet plan + recovery advice |
| Medicine Tracker | Rule-based + Browser API | Alarms + diet guidance |
| Emergency Response | Rule engine + Online Maps | Red zone alert + ambulance call |
| Chat Assistant | Knowledge-base NLU | Multi-turn Q&A on do's & don'ts |
| Session History | JSON store | Resume any previous problem |

**Unique Features:**
- ✅ Core ML inference runs locally (internet is required for Google speech transcription)
- ✅ Voice input + TTS readback in 11 languages
- ✅ For uneducated users: voice assistant in regional languages
- ✅ Emergency ambulance one-tap call
- ✅ Hospital map via OpenStreetMap (works without Google Maps)
- ✅ Session archive — revisit and resume any past problem
    """)

    try:
        with open(os.path.join(os.path.dirname(__file__),"..","models","training_report.json")) as f:
            ml = json.load(f)
        with open(os.path.join(os.path.dirname(__file__),"..","models","dl_training_report.json")) as f:
            dl = json.load(f)
        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Disease Model Accuracy", f"{ml['disease_model']['metrics']['accuracy']*100:.1f}%")
        c2.metric("Risk Model Accuracy",    f"{ml['risk_model']['metrics']['accuracy']*100:.1f}%")
        c3.metric("Image CNN Accuracy",     f"{dl['val_accuracy']*100:.1f}%")
        c4.metric("Diseases Covered",        ml["dataset"]["diseases"])

        st.markdown("### 🧪 Model Comparison")
        st.caption("Same stratified 80/20 test split for all candidate models. These are project-dataset metrics, not clinical diagnostic accuracy.")
        comp_path = os.path.join(os.path.dirname(__file__), "..", "models", "model_comparison.csv")
        if os.path.exists(comp_path):
            comp = pd.read_csv(comp_path)
            for task, title in [("disease", "Disease classification"), ("risk", "Risk classification")]:
                st.markdown(f"**{title}**")
                view = comp[comp["task"] == task].copy()
                view["accuracy"] = (view["accuracy"] * 100).round(2).astype(str) + "%"
                view["weighted_f1"] = (view["weighted_f1"] * 100).round(2).astype(str) + "%"
                st.dataframe(view[["model", "accuracy", "weighted_f1"]], use_container_width=True, hide_index=True)
        else:
            st.info("Run models/compare_models.py to generate the model comparison table.")
    except Exception as exc:
        st.warning(f"Run the training scripts first to see model metrics. ({exc})")

    st.markdown("---")
    st.caption("⚠️ This system is for health awareness and monitoring only, not a medical diagnosis tool. Always consult a qualified doctor for medical concerns.")
