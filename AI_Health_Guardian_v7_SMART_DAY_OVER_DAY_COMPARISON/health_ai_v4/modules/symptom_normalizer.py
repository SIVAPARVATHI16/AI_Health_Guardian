"""
Symptom normalization engine.

This module is the scalable core of the "understand natural language ->
canonical symptom" pipeline. It intentionally avoids hard-coding every
possible phrase. Instead it composes two small, easily-extended
vocabularies:

    BODY_PARTS  : body-part word -> canonical symptom column it maps to
    PAIN_WORDS  : words that indicate pain/soreness/stiffness at a location

and generates matches dynamically for ANY body-part/pain-word pair present
in the text (e.g. "wrist" + "aching" -> wrist_pain), instead of requiring a
literal phrase like "wrist pain" to be pre-registered.

It also implements clause-scoped negation detection so that
"I have fever but no cough" extracts {fever} and marks {cough} as denied
rather than matched.

New body parts, languages or pain words can be added by extending the
dictionaries below -- no changes to the matching logic are required.
"""
from __future__ import annotations
import re
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple

# ---------------------------------------------------------------------------
# 1. Vocabulary configuration (data, not logic -- extend freely)
# ---------------------------------------------------------------------------

# canonical_symptom_column -> {lang: [surface forms / stems]}
BODY_PARTS: Dict[str, Dict[str, List[str]]] = {
    "headache":       {"en": ["head"],
                       "te": ["తల", "తలకి"],
                       "hi": ["सिर"]},
    "sore_throat":    {"en": ["throat"],
                       "te": ["గొంతు"],
                       "hi": ["गले", "गला"]},
    "facial_pain":    {"en": ["face", "jaw", "tooth", "teeth", "ear", "sinus"],
                       "te": ["ముఖం", "దవడ"],
                       "hi": ["चेहरे", "जबड़े", "दांत"]},
    "knee_pain":     {"en": ["knee", "knees"],
                       "te": ["మోకాలి", "మోకాళ్ళ", "మోకాలు", "మోకాళ్ళు"],
                       "hi": ["घुटने", "घुटना", "घुटनों"]},
    "hand_pain":      {"en": ["hand", "hands"],
                       "te": ["చేతి", "చేతికి", "చేతులు", "చేయి"],
                       "hi": ["हाथ", "हाथों"]},
    "wrist_pain":     {"en": ["wrist", "wrists"],
                       "te": ["మణికట్టు"],
                       "hi": ["कलाई"]},
    "shoulder_pain":  {"en": ["shoulder", "shoulders"],
                       "te": ["భుజం", "భుజానికి", "భుజాలు"],
                       "hi": ["कंधे", "कंधा", "कंधों"]},
    "elbow_pain":     {"en": ["elbow", "elbows"],
                       "te": ["మోచేయి", "మోచేతి"],
                       "hi": ["कोहनी"]},
    "neck_pain":      {"en": ["neck"],
                       "te": ["మెడ", "మెడకి"],
                       "hi": ["गर्दन"]},
    "back_pain":      {"en": ["back", "lower back", "upper back", "spine", "tailbone"],
                       "te": ["వెన్ను", "వెన్నుకి", "నడుము"],
                       "hi": ["कमर", "पीठ"]},
    "hip_pain":       {"en": ["hip", "hips"],
                       "te": ["తుంటి"],
                       "hi": ["कूल्हे", "कूल्हा"]},
    "ankle_pain":     {"en": ["ankle", "ankles"],
                       "te": ["చీలమండ"],
                       "hi": ["टखने", "टखना"]},
    "foot_pain":      {"en": ["foot", "feet", "toe", "toes"],
                       "te": ["పాదం", "పాదాలు"],
                       "hi": ["पैर के तलवे"]},
    "heel_pain":      {"en": ["heel", "heels"],
                       "te": ["మడమ"],
                       "hi": ["एड़ी"]},
    "leg_pain":       {"en": ["leg", "legs", "thigh", "calf"],
                       "te": ["కాలు", "కాళ్ళు"],
                       "hi": ["टांग", "पैर", "टांगों"]},
    "arm_pain":       {"en": ["arm", "arms"],
                       "te": ["చేయి", "చేతులు"],
                       "hi": ["बांह", "बाजू"]},
    "finger_pain":    {"en": ["finger", "fingers"],
                       "te": ["వేలు", "వేళ్ళు"],
                       "hi": ["उंगली", "उंगलियों"]},
    "pelvic_pain":    {"en": ["pelvis", "pelvic", "groin"],
                       "te": ["కటి"],
                       "hi": ["श्रोणि"]},
    "chest_pain":     {"en": ["chest", "rib", "ribs"],
                       "te": ["ఛాతి", "గుండె"],
                       "hi": ["सीने", "छाती"]},
    "abdominal_pain": {"en": ["stomach", "abdomen", "belly", "tummy"],
                       "te": ["కడుపు"],
                       "hi": ["पेट"]},
    "joint_pain":     {"en": ["joint", "joints"],
                       "te": ["కీళ్ళ", "కీళ్లు"],
                       "hi": ["जोड़ों", "जोड़"]},
    "muscle_pain":    {"en": ["muscle", "muscles"],
                       "te": ["కండరాల", "కండరం"],
                       "hi": ["मांसपेशियों", "मांसपेशी"]},
}

PAIN_WORDS: Dict[str, List[str]] = {
    "en": ["pain", "ache", "aches", "aching", "hurt", "hurts", "hurting",
           "sore", "soreness", "painful", "stiff", "stiffness", "throbbing"],
    "te": ["నొప్పి", "నొప్పిగా", "నొప్పులు"],
    "hi": ["दर्द", "दर्द हो", "दर्द है"],
}

SWELLING_WORDS: Dict[str, List[str]] = {
    "en": ["swelling", "swollen", "swell", "puffy"],
    "te": ["వాపు"],
    "hi": ["सूजन"],
}

# Numbness/tingling are, on their own (without the word "pain"), still a
# clinically meaningful location-specific symptom -- e.g. "my wrist is
# tingling and numb" should still compose to wrist_pain-adjacent evidence,
# and matter on their own even without a body part in the same clause.
NUMBNESS_TINGLING_WORDS: Dict[str, List[str]] = {
    "en": ["numb", "numbness", "tingling", "tingly", "pins and needles"],
    "te": ["తిమ్మిరి"],
    "hi": ["सुन्न", "झुनझुनी"],
}

# Negation trigger words per language. Kept intentionally short: negation
# scope is resolved structurally (same clause), not by an exhaustive phrase
# list, so this stays small even as the symptom vocabulary grows.
NEGATION_WORDS: Dict[str, Set[str]] = {
    "en": {"no", "not", "dont", "don't", "doesnt", "doesn't", "never",
           "without", "didnt", "didn't", "isnt", "isn't", "havent",
           "haven't", "hasnt", "hasn't", "arent", "aren't", "n't"},
    "te": {"లేదు", "కాదు", "లేవు"},
    "hi": {"नहीं", "ना", "मत"},
}

# Clause boundaries: sentence punctuation + coordinating/contrasting
# conjunctions that typically separate an affirmed symptom from a denied
# one ("I have fever but no cough"). Plain commas are intentionally NOT a
# boundary: "my toe is swollen, red and painful" describes one symptom
# location across a comma-separated adjective list, and splitting on the
# comma would separate the body part from its pain word.
_CLAUSE_SPLIT_RE = re.compile(
    r"[.!?;]+|\bbut\b|\bhowever\b|\balthough\b|\bexcept\b", re.IGNORECASE
)


# ---------------------------------------------------------------------------
# 2. Result container
# ---------------------------------------------------------------------------

@dataclass
class NormalizationResult:
    matched: Set[str] = field(default_factory=set)
    denied: Set[str] = field(default_factory=set)
    body_parts_detected: Set[str] = field(default_factory=set)
    swelling_detected: bool = False


# ---------------------------------------------------------------------------
# 3. Matching logic
# ---------------------------------------------------------------------------

def _normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def _split_clauses(text: str) -> List[str]:
    parts = _CLAUSE_SPLIT_RE.split(text)
    return [p.strip() for p in parts if p and p.strip()]


def _clause_has_negation(clause_lower: str, lang: str) -> bool:
    """Unicode-aware token check for negation words, plus a substring
    fallback for non-Latin scripts (Telugu/Hindi negation particles are
    sometimes attached directly to the previous word with no space in
    casual transcripts). The substring fallback is NOT used for English to
    avoid false positives like "no" inside "know"/"snow"."""
    words = NEGATION_WORDS.get(lang, set()) | NEGATION_WORDS.get("en", set())
    tokens = re.findall(r"[\w']+", clause_lower, flags=re.UNICODE)
    if any(tok in words for tok in tokens):
        return True
    if lang in ("te", "hi"):
        return any(w in clause_lower for w in NEGATION_WORDS.get(lang, set()))
    return False


def _body_part_hits(clause_lower: str, lang: str) -> Set[str]:
    """Return canonical symptom codes for body parts mentioned in the clause.
    English uses word-boundary matching so short body-part words like "arm"
    don't spuriously match inside unrelated words like "warm"/"farm".
    Telugu/Hindi are agglutinative (case/postposition suffixes attach
    directly to the stem, e.g. మోకాలి + కి -> మోకాలికి), so a strict
    trailing boundary would miss real matches there; substring containment
    is used for those instead, matching the literal-phrase matcher's
    approach elsewhere in the project."""
    hits = set()
    for canonical, forms_by_lang in BODY_PARTS.items():
        forms = forms_by_lang.get(lang, [])
        for form in forms:
            form_l = form.lower()
            if lang == "en":
                if re.search(rf"(?<!\w){re.escape(form_l)}(?!\w)", clause_lower):
                    hits.add(canonical)
                    break
            else:
                if form_l in clause_lower:
                    hits.add(canonical)
                    break
    return hits


def _has_any(clause_lower: str, words: List[str]) -> bool:
    """Word-boundary match for ASCII (English) words; substring containment
    for non-Latin scripts, for the same agglutination reason as
    _body_part_hits above."""
    for w in words:
        w_l = w.lower()
        if w_l.isascii():
            if re.search(rf"(?<!\w){re.escape(w_l)}(?!\w)", clause_lower):
                return True
        else:
            if w_l in clause_lower:
                return True
    return False


def extract_body_location_symptoms(text: str, lang: str = "en") -> NormalizationResult:
    """
    Scan `text` for body-part + pain-word co-occurrence and negation, per
    clause, in the given language ("en", "te", "hi"). Any body part paired
    with a pain indicator in the same clause is emitted as the matching
    canonical symptom (e.g. knee_pain), unless the clause is negated.

    This generalizes: it is not limited to the literal example phrases in
    the spec (e.g. "knee pain", "my knee hurts", "pain in knee" all work,
    as would any future body part added to BODY_PARTS without further
    code changes).
    """
    result = NormalizationResult()
    if not text:
        return result

    text = _normalize_text(text)
    clauses = _split_clauses(text) or [text]

    pain_words = PAIN_WORDS.get(lang, []) + (PAIN_WORDS["en"] if lang != "en" else [])
    swelling_words = SWELLING_WORDS.get(lang, []) + (SWELLING_WORDS["en"] if lang != "en" else [])
    numbness_words = NUMBNESS_TINGLING_WORDS.get(lang, []) + (NUMBNESS_TINGLING_WORDS["en"] if lang != "en" else [])

    for clause in clauses:
        clause_lower = clause.lower()
        body_hits = _body_part_hits(clause_lower, lang)
        if not body_hits:
            # also try English body-part words even for non-English text
            # (mixed-language input is common in speech transcripts)
            if lang != "en":
                body_hits = _body_part_hits(clause_lower, "en")

        negated = _clause_has_negation(clause_lower, lang)
        has_numbness = _has_any(clause_lower, numbness_words)

        # Numbness/tingling matter on their own even with no body part
        # mentioned in the same clause (report as generic symptoms).
        if has_numbness:
            for generic_code, trigger_words in (("numbness", ["numb", "numbness", "सुन्न"]),
                                                  ("tingling", ["tingling", "tingly", "pins and needles", "झुनझुनी", "తిమ్మిరి"])):
                hit = False
                for w in trigger_words:
                    w_l = w.lower()
                    if w_l.isascii():
                        if re.search(rf"(?<!\w){re.escape(w_l)}(?!\w)", clause_lower):
                            hit = True
                            break
                    elif w_l in clause_lower:
                        hit = True
                        break
                if hit:
                    if negated:
                        result.denied.add(generic_code)
                    else:
                        result.matched.add(generic_code)

        if not body_hits:
            continue

        has_pain = _has_any(clause_lower, pain_words)
        has_swelling = _has_any(clause_lower, swelling_words)

        if not (has_pain or has_swelling or has_numbness):
            # a body part mentioned with no pain/swelling/numbness cue is
            # not a symptom (e.g. "I went for a knee replacement consult")
            continue

        for canonical in body_hits:
            result.body_parts_detected.add(canonical)
            if has_pain:
                if negated:
                    result.denied.add(canonical)
                else:
                    result.matched.add(canonical)

        if has_swelling and not negated:
            result.swelling_detected = True

    return result


def apply_negation_to_matches(text: str, phrase_spans: List[Tuple[str, int, int]], lang: str = "en") -> Tuple[Set[str], Set[str]]:
    """
    Given a list of (canonical_symptom, start_idx, end_idx) phrase match
    spans found by a literal-phrase matcher (see symptom_analyzer.py),
    resolve which are negated using the same clause-scoped negation logic,
    so the two matching strategies (literal-phrase and body-part
    composition) share one consistent negation rule.
    Returns (matched, denied).
    """
    matched, denied = set(), set()
    if not phrase_spans:
        return matched, denied
    text = _normalize_text(text)
    # map each clause to its character range in the original text so we can
    # test which clause a given span falls into.
    clauses = []
    pos = 0
    for m in _CLAUSE_SPLIT_RE.finditer(text):
        clauses.append((pos, m.start()))
        pos = m.end()
    clauses.append((pos, len(text)))

    for canonical, start, end in phrase_spans:
        clause_text = ""
        for c_start, c_end in clauses:
            if c_start <= start < c_end or (c_start <= start and c_end >= end):
                clause_text = text[c_start:c_end].lower()
                break
        if not clause_text:
            clause_text = text.lower()
        if _clause_has_negation(clause_text, lang):
            denied.add(canonical)
        else:
            matched.add(canonical)
    return matched, denied


def generate_followup_questions(matched_symptoms: Set[str], body_parts_detected: Set[str],
                                 context_clues: "ContextClues" = None) -> List[str]:
    """
    Produce a short list of clinically-relevant follow-up questions when the
    available evidence (e.g. an isolated body-location pain report) is too
    sparse for a confident prediction. Only asks about things that would
    actually help narrow down risk/urgency -- not an exhaustive checklist.

    `context_clues` (see extract_context_clues) lets this skip questions the
    person has already answered in their message -- e.g. don't ask "how
    long have you had this?" if they already said "for three days".
    """
    clues = context_clues or ContextClues()
    questions: List[str] = []
    if body_parts_detected:
        if not clues.duration_mentioned:
            questions.append("How long have you had this pain (hours, days, or longer)?")
        if not clues.swelling_mentioned:
            questions.append("Is there any swelling, redness, or warmth at the site?")
        if not clues.injury_mentioned:
            questions.append("Did the pain start after an injury, fall, or overuse?")
        if not clues.stiffness_mentioned:
            questions.append("Is the area stiff, or is it hard to move normally?")
        if ("back_pain" in body_parts_detected or "leg_pain" in body_parts_detected) \
                and "numbness" not in matched_symptoms and "tingling" not in matched_symptoms:
            questions.append("Is there any numbness, tingling, or weakness?")
        if len(body_parts_detected) == 1 and not clues.sidedness_mentioned:
            questions.append("Is the pain on one side only, or both?")
        if not clues.fever_context_mentioned:
            questions.append("Do you also have fever or feel generally unwell?")
    return questions[:6]


# ---------------------------------------------------------------------------
# 4. Context clues: duration / severity / sidedness / injury mentions.
# These don't change WHICH symptoms are matched, but let the follow-up
# question generator avoid re-asking something the person already told us,
# and let the caller show a lightweight severity/duration summary.
# ---------------------------------------------------------------------------

_DURATION_RE = re.compile(
    r"\b(for|since|from)\b.{0,25}?\b(\d+|a|an|one|two|three|four|five|six|seven|"
    r"couple\s+of|few)\b.{0,15}?\b(hour|hours|day|days|week|weeks|month|months)\b"
    r"|\bsince\s+(yesterday|last\s+night|this\s+morning|today)\b"
    r"|\b(hours|days|weeks|months)\s+ago\b",
    re.IGNORECASE,
)

SEVERITY_WORDS = {
    "severe": ["severe", "excruciating", "unbearable", "intense", "extreme", "worst"],
    "moderate": ["moderate", "noticeable", "persistent"],
    "mild": ["mild", "slight", "minor", "a little", "occasional"],
}

_INJURY_RE = re.compile(
    r"\b(fell|fall|injury|injured|accident|twisted|sprained|hit|hurt myself|overuse|"
    r"lifting|strain(ed)?)\b", re.IGNORECASE,
)
_SIDEDNESS_RE = re.compile(r"\b(left|right|both sides|one side|either side)\b", re.IGNORECASE)
_FEVER_CONTEXT_RE = re.compile(r"\b(fever|temperature|chills|feverish|unwell|well)\b", re.IGNORECASE)


@dataclass
class ContextClues:
    duration_mentioned: bool = False
    severity: str = None  # "severe" | "moderate" | "mild" | None
    injury_mentioned: bool = False
    sidedness_mentioned: bool = False
    fever_context_mentioned: bool = False
    swelling_mentioned: bool = False
    stiffness_mentioned: bool = False


def extract_context_clues(text: str) -> ContextClues:
    """
    Cheap regex-based extraction of duration/severity/sidedness/injury
    mentions from free text. Used to (a) avoid re-asking a follow-up
    question the person already answered, and (b) optionally surface a
    one-line "noted: severe, right-sided, for 3 days" summary in the UI.
    This intentionally does not attempt to parse an exact number of days
    into a structured value -- for a self-care awareness tool, "was a
    duration mentioned at all" is enough signal to skip the question.
    """
    text = text or ""
    tl = text.lower()
    clues = ContextClues()
    clues.duration_mentioned = bool(_DURATION_RE.search(tl))
    clues.injury_mentioned = bool(_INJURY_RE.search(tl))
    clues.sidedness_mentioned = bool(_SIDEDNESS_RE.search(tl))
    clues.fever_context_mentioned = bool(_FEVER_CONTEXT_RE.search(tl))
    clues.swelling_mentioned = any(w in tl for w in SWELLING_WORDS["en"]) or "no swelling" in tl
    clues.stiffness_mentioned = "stiff" in tl or "stiffness" in tl or "range of motion" in tl
    for level, words in SEVERITY_WORDS.items():
        if any(re.search(rf"(?<!\w){re.escape(w)}(?!\w)", tl) for w in words):
            clues.severity = level
            break
    return clues
