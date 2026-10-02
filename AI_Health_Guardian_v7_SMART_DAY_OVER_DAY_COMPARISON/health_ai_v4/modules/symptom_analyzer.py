"""
symptom_analyzer.py — v4 (severity-aware)
Same symptom (e.g. "cough") → different risk/disease based on severity qualifiers:
  "mild cough"          → Common Cold (Low)
  "persistent wet cough"→ Bronchitis (Medium)
  "cough with blood"    → TB/Pneumonia (High)
"""
import json, os, re, joblib
import numpy as np
import pandas as pd

try:
    from symptom_normalizer import (
        extract_body_location_symptoms,
        apply_negation_to_matches,
        extract_context_clues,
        generate_followup_questions,
    )
except Exception:
    extract_body_location_symptoms = None
    apply_negation_to_matches = None
    extract_context_clues = None
    generate_followup_questions = None

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
DATA_DIR  = os.path.join(os.path.dirname(__file__), "..", "data")

_dm  = joblib.load(os.path.join(MODEL_DIR, "disease_model.joblib"))
_rm  = joblib.load(os.path.join(MODEL_DIR, "risk_model.joblib"))
_re  = joblib.load(os.path.join(MODEL_DIR, "risk_label_encoder.joblib"))

with open(os.path.join(MODEL_DIR, "symptom_columns.json"))  as f: SC   = json.load(f)
with open(os.path.join(MODEL_DIR, "symptom_synonyms.json")) as f: SYNS = json.load(f)
with open(os.path.join(DATA_DIR,  "disease_meta.json"))     as f: META = json.load(f)

_SP = sorted(SYNS.keys(), key=len, reverse=True)
_LAST_DENIED_SYMPTOMS = set()
_LAST_CONTEXT_CLUES = None

# ── Severity keyword sets ──────────────────────────────────────────────────────
SEVERITY_ESCALATORS = {
    # Words that push risk UP to High regardless of ML prediction
    "high": [
        "can't breathe","cannot breathe","difficulty breathing","no breath",
        "coughing blood","blood in cough","cough blood","blood cough",
        "chest pain","crushing pain","severe pain","worst pain",
        "blue lips","lips blue","unconscious","fainted","not responding",
        "stiff neck","neck stiff","high fever cough","swollen face",
        "left arm pain","jaw pain","sweating chest",
        "lower right stomach severe","severe stomach pain",
        "weeks of cough","months of cough","cough for weeks","night sweats",
        "severe headache sudden","worst headache ever",
        "rash difficulty breathing","hives breathless",
        "very high fever","104","105","106",
        "cyclic fever","fever with chills","chills and sweating",
        "shivering fever","malaria","high fever",
        "lower right pain","right side stomach pain","appendix",
    ],
    # Words that keep/push to Medium
    "medium": [
        "persistent cough","cough with mucus","wet cough","productive cough",
        "frequent cough","cough with phlegm","cough for days",
        "moderate fever","fever 101","fever 102","fever 103",
        "repeated vomiting","vomiting diarrhea",
        "chest tightness cough","wheezing",
        "burning urination","frequent urination",
        "breathless walking","breathless stairs",
    ],
    # Words that keep/push to Low
    "low": [
        "mild cough","dry cough","little cough","occasional cough",
        "slight cough","minor cough","tiny cough",
        "mild fever","low fever","slight fever","99","100",
        "minor headache","small rash","tiny rash","mild pain",
        "light cold","slight cold",
    ],
}


# ── Human-readable severity guidance ──────────────────────────────────────────
# This is a safety-oriented explanation layer, not a diagnostic rule engine.
# The ML model still produces the underlying disease/risk probabilities; these
# guides explain what the selected screening level means in plain language.
GENERAL_SEVERITY_GUIDE = {
    "Low": {
        "display": "Low Risk",
        "dot": "🟢",
        "description": "Symptoms appear mild based on the information provided, with no detected high-risk warning signs.",
        "examples": "Mild or occasional symptoms; minor irritation; no breathing difficulty or other emergency warning signs.",
        "action": "Monitor symptoms, rest and hydrate as appropriate. If symptoms persist, worsen, or new warning signs appear, seek medical advice.",
    },
    "Medium": {
        "display": "Moderate Risk",
        "dot": "🟡",
        "description": "Symptoms may need closer monitoring or medical advice, especially if they persist or become more severe.",
        "examples": "Persistent/frequent symptoms, fever with symptoms, wheezing, chest discomfort, or symptoms lasting several days.",
        "action": "Monitor closely and consider contacting a qualified healthcare professional, particularly if symptoms are worsening or persistent.",
    },
    "High": {
        "display": "High Risk",
        "dot": "🔴",
        "description": "A potentially urgent warning pattern was detected from the information provided.",
        "examples": "Difficulty breathing, significant bleeding, blue lips/face, severe chest pain, confusion, fainting, or rapidly worsening symptoms.",
        "action": "Seek urgent medical evaluation. If there is severe breathing difficulty, blue lips/face, major bleeding, loss of consciousness, or another emergency sign, contact local emergency services now.",
    },
}

COUGH_SEVERITY_GUIDE = {
    "Low": {
        "display": "Low Risk",
        "dot": "🟢",
        "examples": "Mild dry cough, occasional cough, or cough from minor throat irritation; no fever or breathing difficulty.",
        "action": "Monitor the cough and watch for fever, breathing difficulty, blood, chest pain, or worsening symptoms.",
    },
    "Medium": {
        "display": "Moderate Risk",
        "dot": "🟡",
        "examples": "Persistent wet/productive cough, frequent coughing, cough with fever, wheezing, chest discomfort, or symptoms lasting several days.",
        "action": "Monitor closely and consider medical advice, especially when the cough is persistent, worsening, or accompanied by fever/wheezing.",
    },
    "High": {
        "display": "High Risk",
        "dot": "🔴",
        "examples": "Cough with difficulty breathing, significant blood, blue lips/face, severe chest pain, confusion, or rapidly worsening symptoms.",
        "action": "Seek urgent medical evaluation. Severe breathing difficulty, blue lips/face, significant bleeding, confusion, or rapidly worsening symptoms require urgent/emergency attention.",
    },
}

def _build_severity_guidance(text, risk, matched, source):
    """Return a UI-ready explanation of the screening level.

    The guide deliberately describes warning patterns instead of claiming that
    a risk level is a diagnosis. Cough gets a more specific guide because the
    app supports cough-severity phrases such as dry, persistent, productive,
    blood, and breathing difficulty.
    """
    t = (text or "").lower()
    is_cough = "cough" in t or any(c in matched for c in {
        "mild_cough", "persistent_cough", "cough_with_phlegm", "coughing_blood", "cough_at_night"
    })
    guide = COUGH_SEVERITY_GUIDE if is_cough else GENERAL_SEVERITY_GUIDE
    current = guide.get(risk, GENERAL_SEVERITY_GUIDE["Medium"])

    high_hits = [p for p in SEVERITY_ESCALATORS["high"] if p in t]
    medium_hits = [p for p in SEVERITY_ESCALATORS["medium"] if p in t]
    low_hits = [p for p in SEVERITY_ESCALATORS["low"] if p in t]
    detected = high_hits or medium_hits or low_hits

    if source == "keyword override":
        reason = "Safety/severity phrase detected: " + ", ".join(detected[:4]) if detected else "Safety/severity phrase detected in the description."
    else:
        reason = "Level estimated from the machine-learning symptom pattern; no explicit severity override was triggered."

    model_prob = None
    if risk in ("Low", "Medium", "High"):
        model_prob = None  # filled by predict_from_text from risk_probabilities

    return {
        "display_level": current["display"],
        "emoji": current["dot"],
        "description": current["description"] if "description" in current else current.get("examples", ""),
        "examples": current["examples"],
        "action": current["action"],
        "reason": reason,
        "is_cough_guide": bool(is_cough),
        "guide": {
            level: {
                "display": item["display"],
                "examples": item["examples"],
                "action": item["action"],
            }
            for level, item in guide.items()
        },
        "detected_high_signals": high_hits,
        "detected_medium_signals": medium_hits,
        "detected_low_signals": low_hits,
    }

def _detect_severity_override(text):
    """
    Scans the user's raw text for severity escalators.
    Returns 'High', 'Medium', 'Low', or None (let ML decide).
    """
    t = text.lower()
    for phrase in SEVERITY_ESCALATORS["high"]:
        if phrase in t: return "High"
    for phrase in SEVERITY_ESCALATORS["medium"]:
        if phrase in t: return "Medium"
    for phrase in SEVERITY_ESCALATORS["low"]:
        if phrase in t: return "Low"
    return None

# ── Multilingual translation ───────────────────────────────────────────────────
MULTILANG = {
    # Telugu
    "జ్వరం":"fever","జలుబు":"cold runny nose","దగ్గు":"cough",
    "తలనొప్పి":"headache","వాంతులు":"vomiting","విరేచనాలు":"diarrhea",
    "దద్దుర్లు":"rash","అలసట":"fatigue","మైకం":"dizziness",
    "గొంతు నొప్పి":"sore throat","ఉమ్మడి నొప్పి":"joint pain",
    "శ్వాస తీసుకోవడం కష్టం":"difficulty breathing",
    "వికారం":"nausea","ఛాతీ నొప్పి":"chest pain",
    "శరీర నొప్పి":"body pain","కడుపు నొప్పి":"stomach pain",
    "వణుకు":"chills","తుమ్ములు":"sneezing","బలహీనత":"weakness",
    "రక్తం దగ్గు":"coughing blood","రాత్రి చెమట":"night sweats",
    # Hindi
    "बुखार":"fever","खांसी":"cough","सिरदर्द":"headache",
    "उल्टी":"vomiting","दस्त":"diarrhea","थकान":"fatigue",
    "चक्कर":"dizziness","गले में दर्द":"sore throat",
    "जोड़ों में दर्द":"joint pain","सांस लेने में तकलीफ":"difficulty breathing",
    "मतली":"nausea","सीने में दर्द":"chest pain","शरीर में दर्द":"body pain",
    "टांडा":"chills","ठंड":"chills","कंपकंपी":"chills",
    "जुकाम":"runny nose","ठंड":"cold runny nose","बंद नाक":"congestion","खांसी में खून":"coughing blood",
    "रात को पसीना":"night sweats","तेज बुखार":"high fever",
    "पेट दर्द":"stomach pain","कमजोरी":"weakness",
    # French / Spanish
    "fièvre":"fever","toux":"cough","mal de tête":"headache",
    "fiebre":"fever","tos":"cough","dolor de cabeza":"headache",
    # Malayalam / Punjabi / Urdu
    "പനി":"fever","ചുമ":"cough","തലവേദന":"headache",
    "ਬੁਖਾਰ":"fever","ਖੰਘ":"cough","ਸਿਰ ਦਰਦ":"headache",
    "بخار":"fever","کھانسی":"cough","سر درد":"headache",
    # Tamil
    "காய்ச்சல்":"fever","இருமல்":"cough","தலைவலி":"headache",
    "வாந்தி":"vomiting","சோர்வு":"fatigue","மூச்சு திணறல்":"difficulty breathing",
    "தடிப்பு":"rash","குமட்டல்":"nausea","மார்பு வலி":"chest pain",
    # Kannada
    "ಜ್ವರ":"fever","ಕೆಮ್ಮು":"cough","ತಲೆನೋವು":"headache",
    "ವಾಂತಿ":"vomiting","ಭೇದಿ":"diarrhea","ಆಯಾಸ":"fatigue",
}

def _translate_regional(text):
    for regional, english in MULTILANG.items():
        if regional in text:
            text = text.replace(regional, " " + english + " ")
    return text

def extract_symptoms_from_text(text):
    """Extract canonical symptoms from free text using two complementary layers.

    1) literal/synonym phrases from ``symptom_synonyms.json``;
    2) the scalable body-location normalizer for phrases such as
       ``my knee hurts`` or ``मेरे घुटने में दर्द है``.

    Negated symptoms are returned through ``last_denied_symptoms`` so the
    prediction layer can explain what the user explicitly said was absent.
    """
    global _LAST_DENIED_SYMPTOMS, _LAST_CONTEXT_CLUES
    text = text or ""
    translated = _translate_regional(text)
    t = " " + re.sub(r"[^a-z0-9\s]", " ", translated.lower()) + " "
    matched = set()
    phrase_spans = []

    for p in _SP:
        phrase = p.lower()
        for m in re.finditer(rf"(?<!\w){re.escape(phrase)}(?!\w)", translated.lower()):
            for code in SYNS[p]:
                phrase_spans.append((code, m.start(), m.end()))

    if apply_negation_to_matches is not None:
        literal_matched, literal_denied = apply_negation_to_matches(
            translated, phrase_spans, "en"
        )
        matched.update(literal_matched)
        denied = set(literal_denied)
    else:
        for p in _SP:
            if f" {p} " in t:
                matched.update(SYNS[p])
        denied = set()

    # Body-part composition handles natural variants not present in the
    # literal dictionary. Run English after translation; also run Telugu and
    # Hindi against the original text so agglutinated forms are preserved.
    body_parts = set()
    if extract_body_location_symptoms is not None:
        for source_text, lang in ((translated, "en"), (text, "te"), (text, "hi")):
            nr = extract_body_location_symptoms(source_text, lang)
            matched.update(nr.matched)
            denied.update(nr.denied)
            body_parts.update(nr.body_parts_detected)

    _LAST_DENIED_SYMPTOMS = denied
    _LAST_CONTEXT_CLUES = extract_context_clues(translated) if extract_context_clues else None
    return matched

def symptoms_to_vector(codes):
    # Models are trained on NumPy arrays; keep inference on the same representation.
    return np.asarray([[1 if s in codes else 0 for s in SC]], dtype=np.int8)

def predict_from_text(text, top_k=3, lang=None):
    matched = extract_symptoms_from_text(text)
    result  = predict_from_symptoms(matched, top_k=top_k)
    denied = set(_LAST_DENIED_SYMPTOMS)
    clues = _LAST_CONTEXT_CLUES
    result["denied_symptoms"] = sorted(denied)
    result["severity"] = getattr(clues, "severity", None) if clues else None
    body_parts = set()
    if extract_body_location_symptoms is not None:
        for source_text, source_lang in ((text or "", "en"), (text or "", "te"), (text or "", "hi")):
            try:
                body_parts.update(extract_body_location_symptoms(source_text, source_lang).body_parts_detected)
            except Exception:
                pass
    result["followup_questions"] = (
        generate_followup_questions(matched, body_parts, clues)
        if generate_followup_questions is not None else []
    )
    top_prob = float(result["top_diseases"][0]["probability"]) if result.get("top_diseases") else 0.0
    result["low_confidence"] = bool(len(matched) < 2 or top_prob < 0.35)

    # Apply severity override AFTER ML prediction
    severity_override = _detect_severity_override(text)
    if severity_override:
        result["risk_level"]         = severity_override
        result["severity_detected"]  = True
        result["severity_source"]    = "keyword override"

        # If High override, promote high-risk diseases to top
        if severity_override == "High":
            high_risk_diseases = [d for d,m in META.items() if m["risk"]=="High"]
            # Re-rank: put high risk ones first
            high_top = [d for d in result["top_diseases"] if d["disease"] in high_risk_diseases]
            low_top  = [d for d in result["top_diseases"] if d["disease"] not in high_risk_diseases]
            if high_top:
                result["top_diseases"] = high_top + low_top
    else:
        result["severity_detected"] = False
        result["severity_source"]   = "ML model"

    result["severity_guidance"] = _build_severity_guidance(
        text, result["risk_level"], matched, result.get("severity_source", "ML model")
    )
    # Keep the model probability visible separately from any safety override.
    result["risk_model_probability"] = round(
        float(result.get("risk_probabilities", {}).get(result["risk_level"], 0.0)), 3
    )
    return result, matched

def predict_from_symptoms(codes, top_k=3):
    vec = symptoms_to_vector(codes)
    dp  = _dm.predict_proba(vec)[0]
    dc  = _dm.classes_
    top = [{"disease":dc[i],"probability":round(float(dp[i]),3)}
           for i in np.argsort(dp)[::-1][:top_k]]
    rp  = _rm.predict_proba(vec)[0]
    rc  = _re.inverse_transform(_rm.classes_)
    ri  = int(np.argmax(rp))
    rl  = str(rc[ri])
    rpd = {str(rc[i]):round(float(p),3) for i,p in enumerate(rp)}
    return {
        "top_diseases": top,
        "risk_level": rl,
        "risk_probabilities": rpd,
        "matched_symptom_count": int(sum(1 for s in codes if s in SC)),
        "severity_detected": False,
        "severity_source": "ML model",
        "risk_model_probability": round(float(rpd.get(rl, 0.0)), 3),
        "severity_guidance": _build_severity_guidance("", rl, codes, "ML model"),
        "denied_symptoms": [],
        "followup_questions": [],
        "low_confidence": False,
        "severity": None,
    }

def all_known_symptom_phrases():
    return sorted(_SP)

SYNS_update = {
    "mild cough":          ["mild_cough"],
    "dry cough":           ["mild_cough"],
    "occasional cough":    ["mild_cough"],
    "persistent cough":    ["persistent_cough"],
    "wet cough":           ["cough_with_phlegm","persistent_cough"],
    "productive cough":    ["cough_with_phlegm","persistent_cough"],
    "cough with phlegm":   ["cough_with_phlegm"],
    "cough with mucus":    ["cough_with_phlegm"],
    "coughing blood":      ["coughing_blood"],
    "blood in cough":      ["coughing_blood"],
    "cough for weeks":     ["persistent_cough","coughing_blood"],
    "night sweats":        ["night_sweats"],
    "wheezing":            ["wheezing","breathlessness"],
    "rapid breathing":     ["breathlessness","rapid_breathing"],
    "difficulty breathing":["breathlessness","chest_tightness"],
    "can't breathe":       ["breathlessness","chest_tightness"],
    "left arm pain":       ["left_arm_pain"],
    "jaw pain":            ["jaw_pain"],
    "neck stiff":          ["neck_stiffness"],
    "stiff neck":          ["neck_stiffness"],
    "severe headache":     ["severe_headache"],
    "sudden headache":     ["severe_headache"],
    "confusion":           ["confusion"],
    "rapid heartbeat":     ["rapid_heartbeat"],
    "swollen legs":        ["swelling"],
    "cyclic fever":        ["high_fever","chills","sweating"],
    "fever with chills":   ["high_fever","chills"],
    "prolonged fever":     ["prolonged_fever"],
    "lower right pain":    ["severe_abdominal_pain"],
    "cough at night":      ["cough_at_night"],
    "breathless walking":  ["breathlessness"],
    "breathless stairs":   ["breathlessness"],
}
SYNS.update(SYNS_update)
# Re-sort after update
_SP = sorted(SYNS.keys(), key=len, reverse=True)

if __name__ == "__main__":
    tests = [
        ("mild cough no fever", "→ expect Low (Common Cold)"),
        ("persistent wet cough with mucus", "→ expect Medium (Bronchitis)"),
        ("coughing blood night sweats weight loss", "→ expect High (TB)"),
        ("high fever body pain joint pain rash", "→ expect High (Dengue)"),
        ("mild fever runny nose sneezing", "→ expect Low (Cold)"),
        ("severe chest pain left arm pain sweating", "→ expect High (Heart Attack)"),
        ("chest pain after food heartburn", "→ expect Low (Acid Reflux)"),
        ("severe headache stiff neck high fever", "→ expect High (Meningitis)"),
        ("lower right stomach severe pain fever", "→ expect High (Appendicitis)"),
        ("slight headache one side light sensitivity", "→ expect Low (Migraine)"),
    ]
    for text, expected in tests:
        result, matched = predict_from_text(text)
        top = result["top_diseases"][0]["disease"]
        risk = result["risk_level"]
        src = result["severity_source"]
        print(f"Input:    {text}")
        print(f"Expected: {expected}")
        print(f"Got:      {top} | Risk: {risk} | Source: {src}")
        print()

# Fix: "chest pain after food" / "heartburn" should NOT escalate to High
# These were getting pulled up by "chest pain" in the HIGH escalators.
# Solution: add context-aware de-escalation for food-related chest pain.
_CONTEXT_LOW = [
    "after food", "after eating", "after meal", "heartburn",
    "acid", "acidity", "burning after", "reflux",
]
_orig_detect = _detect_severity_override
def _detect_severity_override_v2(text):
    t = text.lower()
    # If context clearly points to acid reflux / indigestion, keep Low
    if any(k in t for k in _CONTEXT_LOW) and "chest" in t:
        return "Low"
    # "high fever" alone should trigger High
    if "high fever" in t or "very high fever" in t:
        return "High"
    return _orig_detect(text)

# Monkey-patch
import sys
this = sys.modules[__name__]
setattr(this, "_detect_severity_override", _detect_severity_override_v2)
