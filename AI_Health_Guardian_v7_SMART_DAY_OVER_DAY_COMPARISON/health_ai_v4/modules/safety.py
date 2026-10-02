"""
Emergency/urgent-symptom detection.

Kept as a small, explicit, easy-to-audit module rather than folded into
chat_assistant or symptom_analyzer, because this logic must take priority
over ordinary disease-prediction/chat flow and should be easy for a future
maintainer to review in isolation.

This is intentionally keyword/pattern based rather than ML-based: for
safety-critical overrides, predictable and auditable behavior is preferred
over a model that could silently miss a pattern.
"""
import re

# Each entry: (canonical_label, English trigger phrases, Telugu, Hindi)
# A hit on ANY phrase is enough to raise the urgent flag -- these are
# deliberately high-sensitivity (favor false alarms over missed emergencies).
_EMERGENCY_PATTERNS = {
    "severe_chest_pain_or_breathing": {
        "en": ["severe chest pain", "crushing chest pain", "can't breathe",
               "cant breathe", "difficulty breathing", "struggling to breathe",
               "not able to breathe", "gasping for air", "chest pain and sweating"],
        "te": ["ఛాతిలో తీవ్ర నొప్పి", "శ్వాస తీసుకోలేకపోతున్నాను", "శ్వాస ఆడటం లేదు"],
        "hi": ["सीने में तेज दर्द", "सांस नहीं आ रही", "सांस लेने में बहुत तकलीफ"],
    },
    "stroke_signs": {
        "en": ["face drooping", "slurred speech", "sudden numbness one side",
               "sudden confusion", "sudden severe headache", "can't speak properly",
               "one side weakness", "vision loss suddenly"],
        "te": ["ముఖం వంగిపోవడం", "మాట తడబడటం", "అకస్మాత్తుగా మాట్లాడలేకపోవడం"],
        "hi": ["चेहरा टेढ़ा होना", "बोलने में लड़खड़ाहट", "अचानक बेहोशी"],
    },
    "loss_of_consciousness": {
        "en": ["lost consciousness", "passed out", "fainted", "unresponsive",
               "not waking up", "unconscious"],
        "te": ["స్పృహ కోల్పోయాను", "మూర్ఛపోయాను"],
        "hi": ["बेहोश हो गया", "होश खो दिया"],
    },
    "severe_allergic_reaction": {
        "en": ["severe allergic reaction", "throat closing", "swelling of throat",
               "anaphylaxis", "face swelling and difficulty breathing",
               "hives and trouble breathing"],
        "te": ["తీవ్ర అలర్జీ ప్రతిచర్య", "గొంతు మూసుకుపోతోంది"],
        "hi": ["गंभीर एलर्जी प्रतिक्रिया", "गला बंद हो रहा है"],
    },
    "severe_bleeding_or_trauma": {
        "en": ["severe bleeding", "won't stop bleeding", "wont stop bleeding",
               "major injury", "deep wound", "head injury with vomiting"],
        "te": ["తీవ్ర రక్తస్రావం", "రక్తం ఆగడం లేదు"],
        "hi": ["तेज खून बह रहा है", "खून रुक नहीं रहा"],
    },
    "suicidal_thoughts": {
        "en": ["suicidal", "want to end my life", "want to kill myself",
               "self harm", "no reason to live"],
        "te": ["ఆత్మహత్య", "నా జీవితాన్ని ముగించాలని"],
        "hi": ["आत्महत्या", "अपनी जान लेना"],
    },
}

EMERGENCY_MESSAGE_EN = (
    "This may be a medical emergency. Please call your local emergency "
    "number or go to the nearest emergency room right now. Do not wait for "
    "an app-based prediction. If you are with someone else, ask them to "
    "help you get care immediately."
)
CRISIS_MESSAGE_EN = (
    "It sounds like you may be going through something very difficult "
    "right now. Please reach out to a crisis helpline or emergency "
    "services in your area immediately, or go to the nearest emergency "
    "room. You do not have to go through this alone."
)


def detect_emergency(text):
    """
    Returns (is_emergency, category) where category is one of the keys in
    _EMERGENCY_PATTERNS, or (False, None) if nothing matched.
    """
    if not text:
        return False, None
    t = text.casefold()
    for category, langs in _EMERGENCY_PATTERNS.items():
        for phrases in langs.values():
            for phrase in phrases:
                if phrase.casefold() in t:
                    return True, category
    return False, None


def emergency_message(category):
    if category == "suicidal_thoughts":
        return CRISIS_MESSAGE_EN
    return EMERGENCY_MESSAGE_EN
