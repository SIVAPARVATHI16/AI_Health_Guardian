"""Multilingual health chat assistant.
Primarily rule-based (deterministic, auditable) for known intents and
safety-critical paths. An optional Gemini fallback (gemini_assistant.py)
is used only for open-ended questions that match no known intent, and
only when a free API key is configured -- the app works identically
without it. Voice transcription is handled by app.py; this module detects
intents in multiple languages and returns a response in that language.
"""
import json
import os
import re

from safety import detect_emergency, emergency_message
from symptom_analyzer import predict_from_text

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
with open(os.path.join(DATA_DIR, "disease_meta.json"), encoding="utf-8") as f:
    DISEASE_META = json.load(f)

# Intent keywords. Keep these phrases short and forgiving because speech
# recognition can insert spaces/punctuation differently across languages.
INTENT_PHRASES = {
    "exercise": {
        "en": ["exercise", "workout", "gym", "walk"],
        "hi": ["व्यायाम", "कसरत", "जिम", "चलना"],
        "te": ["వ్యాయామం", "వర్కౌట్", "జిమ్", "నడవడం"],
        "ta": ["உடற்பயிற்சி", "வொர்க்அவுட்", "ஜிம்", "நடக்க"],
        "kn": ["ವ್ಯಾಯಾಮ", "ವರ್ಕೌಟ್", "ಜಿಮ್", "ನಡೆಯಲು"],
        "bn": ["ব্যায়াম", "শরীরচর্চা", "জিম", "হাঁটা"],
        "mr": ["व्यायाम", "कसरत", "जिम", "चालणे"],
        "gu": ["કસરત", "વ્યાયામ", "જિમ", "ચાલવું"],
        "ar": ["تمرين", "رياضة", "الجيم", "المشي"],
        "fr": ["exercice", "sport", "salle de sport", "marcher"],
        "es": ["ejercicio", "entrenamiento", "gimnasio", "caminar"],
    },
    "food": {
        "en": ["eat", "food", "diet", "meal"],
        "hi": ["खाना", "भोजन", "डाइट", "खाऊ"],
        "te": ["తిన", "ఆహారం", "డైట్", "భోజనం"],
        "ta": ["சாப்பிட", "உணவு", "டயட்", "உணவு"],
        "kn": ["ತಿನ್ನ", "ಆಹಾರ", "ಡಯಟ್", "ಊಟ"],
        "bn": ["খাবার", "খেতে", "ডায়েট", "খাওয়া"],
        "mr": ["खाणे", "अन्न", "डाएट", "जेवण"],
        "gu": ["ખાવું", "ખોરાક", "ડાયેટ", "ભોજન"],
        "ar": ["آكل", "طعام", "حمية", "وجبة"],
        "fr": ["manger", "nourriture", "régime", "repas"],
        "es": ["comer", "comida", "dieta", "comida"],
    },
    "drink": {
        "en": ["drink", "water", "alcohol"],
        "hi": ["पीना", "पानी", "शराब"],
        "te": ["తాగ", "నీరు", "మద్యం"],
        "ta": ["குடிக்க", "தண்ணீர்", "மது"],
        "kn": ["ಕುಡಿಯ", "ನೀರು", "ಮದ್ಯ"],
        "bn": ["পান", "জল", "পানি", "মদ"],
        "mr": ["पिणे", "पाणी", "दारू"],
        "gu": ["પીવું", "પાણી", "દારૂ"],
        "ar": ["أشرب", "ماء", "كحول"],
        "fr": ["boire", "eau", "alcool"],
        "es": ["beber", "agua", "alcohol"],
    },
    "work": {
        "en": ["work", "office", "job"],
        "hi": ["काम", "ऑफिस", "नौकरी"],
        "te": ["పని", "ఆఫీస్", "ఉద్యోగం"],
        "ta": ["வேலை", "அலுவலகம்"],
        "kn": ["ಕೆಲಸ", "ಆಫೀಸ್", "ಉದ್ಯೋಗ"],
        "bn": ["কাজ", "অফিস", "চাকরি"],
        "mr": ["काम", "ऑफिस", "नोकरी"],
        "gu": ["કામ", "ઓફિસ", "નોકરી"],
        "ar": ["عمل", "مكتب", "وظيفة"],
        "fr": ["travail", "bureau", "emploi"],
        "es": ["trabajo", "oficina", "empleo"],
    },
    "travel": {
        "en": ["travel", "trip", "journey"],
        "hi": ["यात्रा", "सफर", "ट्रैवल"],
        "te": ["ప్రయాణం", "ట్రావెల్", "యాత్ర"],
        "ta": ["பயணம்", "சுற்றுலா"],
        "kn": ["ಪ್ರಯಾಣ", "ಟ್ರಾವೆಲ್"],
        "bn": ["ভ্রমণ", "যাত্রা", "ট্রাভেল"],
        "mr": ["प्रवास", "यात्रा", "ट्रॅव्हल"],
        "gu": ["પ્રવાસ", "મુસાફરી", "ટ્રાવેલ"],
        "ar": ["سفر", "رحلة", "السفر"],
        "fr": ["voyager", "voyage", "trajet"],
        "es": ["viajar", "viaje", "trayecto"],
    },
    "doctor": {
        "en": ["doctor", "hospital", "when to see", "medical"],
        "hi": ["डॉक्टर", "अस्पताल", "कब डॉक्टर", "चिकित्सा"],
        "te": ["డాక్టర్", "ఆసుపత్రి", "హాస్పిటల్", "ఎప్పుడు డాక్టర్"],
        "ta": ["மருத்துவர்", "மருத்துவமனை", "டாக்டர்"],
        "kn": ["ವೈದ್ಯ", "ಆಸ್ಪತ್ರೆ", "ಡಾಕ್ಟರ್"],
        "bn": ["ডাক্তার", "হাসপাতাল", "চিকিৎসা"],
        "mr": ["डॉक्टर", "रुग्णालय", "दवाखाना"],
        "gu": ["ડોક્ટર", "હોસ્પિટલ", "દવાખાનું"],
        "ar": ["طبيب", "مستشفى", "دكتور", "علاج"],
        "fr": ["médecin", "hôpital", "docteur", "médical"],
        "es": ["médico", "hospital", "doctor", "tratamiento"],
    },
    "symptoms": {
        "en": ["symptom", "symptoms", "sign", "feel"],
        "hi": ["लक्षण", "महसूस", "संकेत"],
        "te": ["లక్షణాలు", "లక్షణం", "అనిపిస్తుంది"],
        "ta": ["அறிகுறி", "அறிகுறிகள்", "உணர்கிறேன்"],
        "kn": ["ಲಕ್ಷಣ", "ಲಕ್ಷಣಗಳು", "ಅನಿಸುತ್ತದೆ"],
        "bn": ["উপসর্গ", "লক্ষণ", "অনুভব"],
        "mr": ["लक्षण", "लक्षणे", "वाटते"],
        "gu": ["લક્ષણ", "લક્ષણો", "લાગે"],
        "ar": ["أعراض", "عرض", "أشعر"],
        "fr": ["symptôme", "symptômes", "signe", "ressens"],
        "es": ["síntoma", "síntomas", "señal", "siento"],
    },
    "medicine": {
        "en": ["medicine", "drug", "tablet", "pill", "medication"],
        "hi": ["दवा", "गोली", "औषधि"],
        "te": ["మందు", "టాబ్లెట్", "ఔషధం"],
        "ta": ["மருந்து", "மாத்திரை"],
        "kn": ["ಔಷಧಿ", "ಮಾತ್ರೆ", "ಮದ್ದು"],
        "bn": ["ওষুধ", "ট্যাবলেট", "ঔষধ"],
        "mr": ["औषध", "गोळी", "औषधि"],
        "gu": ["દવા", "ગોળી", "ઔષધ"],
        "ar": ["دواء", "دواء", "حبوب", "علاج"],
        "fr": ["médicament", "comprimé", "pilule"],
        "es": ["medicina", "medicamento", "pastilla"],
    },
    "emergency": {
        "en": ["emergency", "ambulance", "critical", "severe", "serious"],
        "hi": ["आपातकाल", "एम्बुलेंस", "गंभीर"],
        "te": ["అత్యవసరం", "అంబులెన్స్", "తీవ్రం", "గంభీర"],
        "ta": ["அவசரம்", "ஆம்புலன்ஸ்", "கடுமையான"],
        "kn": ["ತುರ್ತು", "ಆಂಬ್ಯುಲೆನ್ಸ್", "ಗಂಭೀರ"],
        "bn": ["জরুরি", "অ্যাম্বুলেন্স", "গুরুতর"],
        "mr": ["आपत्कालीन", "रुग्णवाहिका", "गंभीर"],
        "gu": ["કટોકટી", "એમ્બ્યુલન્સ", "ગંભીર"],
        "ar": ["طوارئ", "إسعاف", "خطير", "شديد"],
        "fr": ["urgence", "ambulance", "grave", "sévère"],
        "es": ["emergencia", "ambulancia", "grave", "severo"],
    },
    "tracking": {
        "en": ["how am i recovering", "recovery", "adherence", "did i take my medicine", "am i improving", "my progress"],
        "hi": ["मेरी रिकवरी", "मेरी प्रगति", "क्या मैंने दवा ली"],
        "te": ["నా కోలుకోవడం", "నా పురోగతి", "మందు వేసుకున్నానా"],
    },
}

COMMON = {
    "en-IN": {
        "exercise_high": "Avoid strenuous exercise and seek medical advice.",
        "exercise_med": "Light activity may be reasonable if you feel well; avoid heavy workouts.",
        "exercise_low": "Gentle activity such as walking is generally reasonable if you feel well.",
        "food": "Choose light, balanced meals and stay hydrated. Follow any diet instructions from your doctor.",
        "drink": "Stay hydrated with water or suitable fluids. Avoid alcohol while unwell.",
        "work_high": "Rest and seek medical care rather than going to work.",
        "work_med": "If possible, rest or work from home. Avoid work if symptoms are worsening.",
        "work_low": "You may be able to work if you feel well enough; take breaks and monitor your symptoms.",
        "travel_high": "Avoid travel and seek medical care nearby.",
        "travel_med": "Avoid unnecessary long travel until you improve.",
        "travel_low": "Short necessary travel may be reasonable if you feel well.",
        "doctor_high": "Please seek medical care promptly because the screening risk is high.",
        "doctor_med": "Seek medical advice if symptoms worsen or do not improve within a few days.",
        "doctor_low": "Consider medical advice if symptoms persist, recur, or become worse.",
        "medicine": "I cannot prescribe medicines. Use the Medicine Tracker for your prescribed medicines and follow your doctor's instructions.",
        "symptoms": "The Symptom Analyzer can compare your reported symptoms with the model's symptom patterns. It is not a diagnosis.",
        "emergency": "If this feels like an emergency, use the Emergency page and contact local emergency services immediately.",
        "fallback": "Ask me about exercise, food, fluids, work, travel, medicines, symptoms, or when to see a doctor.",
    },
    "hi-IN": {
        "exercise_high": "भारी व्यायाम से बचें और चिकित्सकीय सलाह लें।", "exercise_med": "यदि आप ठीक महसूस कर रहे हैं तो हल्की गतिविधि कर सकते हैं, लेकिन भारी व्यायाम से बचें।", "exercise_low": "यदि आप ठीक महसूस कर रहे हैं तो हल्की सैर जैसी गतिविधि सामान्यतः ठीक हो सकती है।",
        "food": "हल्का और संतुलित भोजन करें तथा पर्याप्त पानी पिएँ। डॉक्टर की सलाह का पालन करें।", "drink": "पर्याप्त पानी या उचित तरल लें। बीमारी के दौरान शराब से बचें।",
        "work_high": "आराम करें और काम पर जाने के बजाय चिकित्सकीय सहायता लें।", "work_med": "संभव हो तो आराम करें या घर से काम करें। लक्षण बढ़ें तो काम न करें।", "work_low": "यदि आप ठीक महसूस कर रहे हैं तो काम कर सकते हैं, लेकिन आराम करें और लक्षण देखें।",
        "travel_high": "यात्रा से बचें और पास में चिकित्सा सहायता लें।", "travel_med": "ठीक होने तक अनावश्यक लंबी यात्रा से बचें।", "travel_low": "यदि आप ठीक महसूस कर रहे हैं तो छोटी आवश्यक यात्रा संभव हो सकती है।",
        "doctor_high": "जोखिम अधिक है, इसलिए जल्द चिकित्सकीय सहायता लें।", "doctor_med": "लक्षण बिगड़ें या कुछ दिनों में सुधार न हो तो डॉक्टर से मिलें।", "doctor_low": "लक्षण बने रहें, लौटें या बिगड़ें तो डॉक्टर से सलाह लें।",
        "medicine": "मैं दवा लिख नहीं सकता। अपनी निर्धारित दवाओं के लिए Medicine Tracker का उपयोग करें और डॉक्टर की सलाह मानें।", "symptoms": "Symptom Analyzer आपके लक्षणों की तुलना मॉडल के पैटर्न से करता है; यह निदान नहीं है।", "emergency": "यदि यह आपातकाल लगता है तो Emergency पेज खोलें और तुरंत स्थानीय आपातकालीन सेवा से संपर्क करें।", "fallback": "व्यायाम, भोजन, पानी, काम, यात्रा, दवा, लक्षण या डॉक्टर से मिलने के बारे में पूछें।",
    },
    "te-IN": {
        "exercise_high": "భారీ వ్యాయామం చేయకండి మరియు వైద్య సలహా తీసుకోండి.", "exercise_med": "మీకు బాగుంటే తేలికపాటి వ్యాయామం చేయవచ్చు; భారీ వ్యాయామం వద్దు.", "exercise_low": "మీకు బాగుంటే నడక వంటి తేలికపాటి వ్యాయామం సాధారణంగా చేయవచ్చు.",
        "food": "తేలికపాటి, సమతుల్య ఆహారం తీసుకుని తగినంత నీరు తాగండి. డాక్టర్ సూచనలను పాటించండి.", "drink": "తగినంత నీరు లేదా ద్రవాలు తీసుకోండి. అనారోగ్యంగా ఉన్నప్పుడు మద్యం వద్దు.",
        "work_high": "విశ్రాంతి తీసుకుని వైద్య సహాయం పొందండి; పనికి వెళ్లకండి.", "work_med": "సాధ్యమైతే విశ్రాంతి తీసుకోండి లేదా ఇంటి నుంచి పని చేయండి. లక్షణాలు పెరిగితే పని చేయకండి.", "work_low": "మీకు బాగుంటే పని చేయవచ్చు; మధ్యలో విశ్రాంతి తీసుకుని లక్షణాలను గమనించండి.",
        "travel_high": "ప్రయాణం చేయకండి; సమీపంలో వైద్య సహాయం పొందండి.", "travel_med": "కోలుకునే వరకు అవసరం లేని దీర్ఘ ప్రయాణాలను నివారించండి.", "travel_low": "మీకు బాగుంటే చిన్న అవసరమైన ప్రయాణం చేయవచ్చు.",
        "doctor_high": "ప్రమాద స్థాయి ఎక్కువగా ఉంది; వెంటనే వైద్య సహాయం పొందండి.", "doctor_med": "లక్షణాలు పెరిగితే లేదా కొన్ని రోజుల్లో తగ్గకపోతే డాక్టర్‌ను కలవండి.", "doctor_low": "లక్షణాలు కొనసాగితే, మళ్లీ వస్తే లేదా పెరిగితే డాక్టర్‌ను సంప్రదించండి.",
        "medicine": "నేను మందులు సూచించలేను. డాక్టర్ సూచించిన మందుల కోసం Medicine Tracker ఉపయోగించండి.", "symptoms": "Symptom Analyzer మీ లక్షణాలను మోడల్ నమూనాలతో పోలుస్తుంది; ఇది వైద్య నిర్ధారణ కాదు.", "emergency": "ఇది అత్యవసర పరిస్థితి అనిపిస్తే Emergency పేజీని తెరిచి వెంటనే స్థానిక అత్యవసర సేవలను సంప్రదించండి.", "fallback": "వ్యాయామం, ఆహారం, నీరు, పని, ప్రయాణం, మందులు, లక్షణాలు లేదా డాక్టర్ గురించి అడగండి.",
    },
}

# Fallback to English for less common response languages while recognition still works.

def _lang(language):
    return language if language in COMMON else "en-IN"

def _has(text, phrase):
    if not phrase:
        return False
    t = str(text).casefold()
    p = str(phrase).casefold()
    # Unicode-safe substring is appropriate for speech-recognition phrases.
    return p in t

def _intent(text, language):
    lang = language.split("-")[0]
    # Prefer the selected language, then English.
    for intent, groups in INTENT_PHRASES.items():
        phrases = groups.get(lang, []) + groups.get("en", [])
        for phrase in phrases:
            if _has(text, phrase):
                return intent
    return None

def chat_response(user_message, disease="Common Cold", risk_level="Low", history=None,
                   language="en-IN", uid=None):
    """
    Response priority (highest first):
      1. Emergency/urgent-symptom detection -- always overrides everything
         else, and is a hard, deterministic pre-check that runs regardless
         of which path (Gemini or rule-based) handles the rest of the
         message. This is intentionally NEVER delegated to the LLM, so it
         can't be skipped by the model forgetting to check.
      2. When Gemini is configured: a full, general-purpose, ChatGPT/
         Claude/Gemini-style conversational reply with complete multi-turn
         memory, able to answer anything (not just health topics), and
         able to actively call the app's own tools mid-conversation
         (symptom analysis, recovery/adherence lookup) -- see
         gemini_assistant.agentic_chat_reply.
      3. Fallback (no Gemini key configured, or the call fails for any
         reason): the original deterministic, rule-based assistant below,
         so the chat still works without any external API.

    `uid` is optional; when provided, tracking questions/tools can be
    answered with the person's real adherence/recovery numbers instead of
    a generic pointer to the dashboard.
    """
    # 1. Safety first -- this check runs regardless of language/intent match,
    #    and regardless of which path handles the rest of the message.
    is_emergency, category = detect_emergency(user_message)
    if is_emergency:
        return emergency_message(category)

    # 2. Primary path: general-purpose, tool-using Gemini conversation.
    try:
        from gemini_assistant import agentic_chat_reply, gemini_available, call_recently_timed_out
        if gemini_available():
            reply = agentic_chat_reply(
                user_message, history=history, uid=uid,
                app_context={"disease": disease, "risk_level": risk_level},
            )
            if reply:
                return reply
            # A timeout is deliberately NOT allowed to fall through to the
            # rule-based assistant below: that could look like a real,
            # considered answer when nothing was actually checked. Tell
            # the person to retry instead of silently substituting a
            # different (and possibly less relevant) kind of answer.
            if call_recently_timed_out():
                return (
                    "⏳ I'm taking too long to respond right now. "
                    "Please try sending that again in a moment."
                )
    except Exception:
        pass  # Gemini is optional; never break the chat on failure.

    # 3. Fallback: deterministic rule-based assistant (no external API).
    return _rule_based_chat_response(user_message, disease, risk_level, history, language, uid)


def _rule_based_chat_response(user_message, disease, risk_level, history, language, uid):
    """
    The original, fully deterministic chat assistant, used when Gemini is
    not configured (or a call to it fails). Priority order:
      1. Known deterministic intents (exercise/food/work/travel/doctor/
         medicine/emergency keyword/tracking).
      2. If the message itself describes symptoms, run the real symptom
         pipeline (normalization -> ML prediction) and answer with it,
         including follow-up questions when evidence is thin -- this is
         what makes "I have knee pain" or "my hand hurts" typed into chat
         actually useful instead of a generic canned reply.
      3. Single-shot Gemini fallback for open-ended questions (kept as a
         secondary fallback for when the agentic path in chat_response
         wasn't reached at all, e.g. it's being called directly/in tests).
      4. Generic fallback text.
    """
    language = _lang(language)
    t = COMMON[language]

    intent = _intent(user_message, language)

    if intent == "exercise":
        return t[f"exercise_{risk_level.lower()}"]
    if intent == "food":
        return t["food"]
    if intent == "drink":
        return t["drink"]
    if intent == "work":
        return t[f"work_{risk_level.lower()}"]
    if intent == "travel":
        return t[f"travel_{risk_level.lower()}"]
    if intent == "doctor":
        return t[f"doctor_{risk_level.lower()}"]
    if intent == "medicine":
        return t["medicine"]
    if intent == "emergency":
        return t["emergency"]
    if intent == "tracking":
        return _tracking_response(uid, language)

    # Try the real symptom pipeline before falling back to a canned
    # "symptoms" message or generic text. This covers both the explicit
    # "symptoms" intent and messages that mention symptoms without using
    # the word "symptom" at all (e.g. "I have knee pain").
    symptom_reply = _symptom_aware_response(user_message, language, explicit_intent=(intent == "symptoms"))

    # If the current message alone doesn't describe a symptom, but the
    # assistant's last turn was a follow-up question (e.g. "How long have
    # you had this?"), the current message is very likely the answer to
    # that question ("since yesterday", "no swelling", "both knees"). Retry
    # by combining it with the original symptom message so context isn't
    # lost turn-to-turn. This only triggers on that specific signal (the
    # bot just asked something) rather than merging arbitrary history, to
    # avoid stitching together unrelated topics.
    if not symptom_reply and _looks_like_followup_context(history):
        prior_user_message = _last_user_message(history)
        if prior_user_message:
            combined_text = f"{prior_user_message}. {user_message}"
            symptom_reply = _symptom_aware_response(combined_text, language, explicit_intent=False)

    if symptom_reply:
        return symptom_reply
    if intent == "symptoms":
        return t["symptoms"]

    # Single-shot Gemini fallback for open-ended health-awareness questions.
    try:
        from gemini_assistant import chat_reply, gemini_available
        if gemini_available():
            context_note = f"Current symptom-analyzer context: disease pattern={disease}, risk={risk_level}."
            ai_reply = chat_reply(user_message, context_note=context_note)
            if ai_reply:
                return ai_reply
    except Exception:
        pass  # Gemini is optional; never break the chat on failure.

    # Disease-specific English context is retained when a localized
    # response is not available; this avoids inventing medical translations.
    meta = DISEASE_META.get(disease, {})
    precaution = meta.get("precaution", "")
    if precaution and language == "en-IN":
        return f"For {disease}: {precaution}\n\n{t['fallback']}"
    return t["fallback"]


def _looks_like_followup_context(history):
    """True if the assistant's last turn asked a question, i.e. the current
    user message is very likely answering it rather than starting a new
    topic. Kept as a narrow, explicit signal rather than always merging
    recent history, so unrelated turns are never stitched together."""
    if not history:
        return False
    last = history[-1]
    return last.get("role") == "assistant" and "?" in (last.get("content") or "")


def _last_user_message(history):
    for turn in reversed(history or []):
        if turn.get("role") == "user":
            return turn.get("content", "")
    return None


def _symptom_aware_response(user_message, language, explicit_intent=False):
    """
    Run the normalized symptom-extraction + ML pipeline on the chat message
    itself. Returns None if the message doesn't actually describe any
    recognizable symptom (so the caller can fall through to other logic),
    otherwise returns a safety-bounded, appropriately uncertain reply.
    """
    lang_code = language.split("-")[0]
    result, matched = predict_from_text(user_message, top_k=3, lang=lang_code if lang_code in ("te", "hi") else None)
    if not matched and not explicit_intent:
        return None

    if not matched:
        return ("I couldn't clearly identify a specific symptom from that message. "
                "Could you describe what you're feeling in a bit more detail "
                "(for example, where it hurts, since when, and how severe)?")

    lines = []
    if result.get("denied_symptoms"):
        lines.append(f"Noted as NOT present: {', '.join(s.replace('_', ' ') for s in result['denied_symptoms'])}.")
    if result.get("severity") == "severe":
        lines.append("Noted that this sounds severe -- if it's intense, sudden, or unlike anything you've felt before, please don't wait on a screening result to seek care.")

    if result["low_confidence"]:
        lines.append(
            "Based on what you've shared, I don't have enough information yet for even a "
            "tentative pattern match. Here are a few things that would help:"
        )
        for q in result["followup_questions"] or [
            "How long have you had this?", "Is it getting better, worse, or the same?",
            "Do you have any other symptoms alongside this?"
        ]:
            lines.append(f"- {q}")
    else:
        top = result["top_diseases"][0]
        lines.append(
            f"Based on the symptoms mentioned, the model's closest pattern match is "
            f"**{top['disease']}** (model confidence: {round(top['probability']*100)}%), "
            f"with an overall risk screening of **{result['risk_level']}**."
        )
        lines.append("This is a pattern match from a screening model, not a medical diagnosis.")
        if result["risk_level"] == "High":
            lines.append("Please consider seeking medical care given the risk level.")

    lines.append("Open the Symptom Analyzer page for a full breakdown, or add more detail here.")
    return "\n".join(lines)


def _tracking_response(uid, language):
    """Answer recovery/adherence questions using real tracker data when a
    uid is available; otherwise point to the dashboard."""
    if uid:
        try:
            import recovery_tracker as rt
            s = rt.summary(uid)
            if s.get("days_tracked"):
                parts = []
                if s.get("trend"):
                    parts.append(f"Your recent recovery trend looks **{s['trend'].lower()}**.")
                if s.get("adherence_pct") is not None:
                    parts.append(f"Your medicine adherence is **{s['adherence_pct']}%**.")
                if parts:
                    return " ".join(parts) + " See the Recovery Dashboard for full details."
        except Exception:
            pass
    return "Open the Recovery Dashboard to see your symptom trend, medicine adherence, and tracking history."