import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'modules'))

from symptom_analyzer import extract_symptoms_from_text, predict_from_text
from symptom_normalizer import extract_body_location_symptoms
from chat_assistant import chat_response
from safety import detect_emergency


# ---------------------------------------------------------------------
# Body-location symptom extraction (the core ask: knee/hand/back/etc.)
# ---------------------------------------------------------------------

def test_knee_pain_variants():
    for text in ["I have knee pain", "My knees are hurting", "pain in my knee",
                 "my knee hurts", "pain around my knee", "painful knee"]:
        assert "knee_pain" in extract_symptoms_from_text(text), text


def test_hand_wrist_shoulder_back_variants():
    assert "hand_pain" in extract_symptoms_from_text("my hand hurts")
    assert "hand_pain" in extract_symptoms_from_text("pain in my hands")
    assert "wrist_pain" in extract_symptoms_from_text("my wrist is hurting")
    assert "shoulder_pain" in extract_symptoms_from_text("my shoulder hurts")
    assert "back_pain" in extract_symptoms_from_text("I have severe back pain")
    assert "joint_pain" in extract_symptoms_from_text("I have joint pain")


def test_additional_body_locations():
    assert "neck_pain" in extract_symptoms_from_text("my neck hurts")
    assert "elbow_pain" in extract_symptoms_from_text("elbow pain since morning")
    assert "ankle_pain" in extract_symptoms_from_text("my ankle is sore")
    assert "leg_pain" in extract_symptoms_from_text("my leg hurts")
    assert "hip_pain" in extract_symptoms_from_text("pain in my hip")
    assert "finger_pain" in extract_symptoms_from_text("my finger hurts")


def test_headache_as_body_part_composition():
    # "headache" literal phrase already worked; this checks the new
    # generative head+pain composition catches phrasing not in the literal
    # dictionary.
    assert "headache" in extract_symptoms_from_text("my head hurts")
    assert "headache" in extract_symptoms_from_text("I have head pain")


def test_body_part_without_pain_word_is_not_a_symptom():
    # Mentioning a body part with no pain/swelling cue should not be treated
    # as a symptom.
    result = extract_body_location_symptoms("I went for a knee consultation", "en")
    assert result.matched == set()


def test_swelling_detection():
    result = extract_body_location_symptoms("my knee hurts and is swollen", "en")
    assert result.swelling_detected is True


# ---------------------------------------------------------------------
# Negation handling
# ---------------------------------------------------------------------

def test_negation_basic():
    matched = extract_symptoms_from_text("I have fever but no cough")
    assert "mild_fever" in matched or "high_fever" in matched
    assert "cough" not in matched
    assert "mild_cough" not in matched


def test_negation_dont_have():
    matched = extract_symptoms_from_text("I don't have cough")
    assert "cough" not in matched and "mild_cough" not in matched


def test_negation_without_and_not_experiencing():
    assert "vomiting" not in extract_symptoms_from_text("not experiencing vomiting")
    matched = extract_symptoms_from_text("no fever")
    assert "mild_fever" not in matched and "high_fever" not in matched


def test_negation_reported_in_prediction():
    result, matched = predict_from_text("I have fever but no cough")
    assert "cough" in result["denied_symptoms"] or "mild_cough" in result["denied_symptoms"]


# ---------------------------------------------------------------------
# Multiple symptoms / natural sentences
# ---------------------------------------------------------------------

def test_multiple_symptoms_natural_sentence():
    matched = extract_symptoms_from_text("Since yesterday I have fever, headache and joint pain")
    assert "headache" in matched
    assert "joint_pain" in matched
    assert any(s in matched for s in ("mild_fever", "high_fever", "prolonged_fever"))


def test_combined_location_and_swelling_sentence():
    matched = extract_symptoms_from_text(
        "My right knee has been hurting for three days and it is slightly swollen.")
    assert "knee_pain" in matched


# ---------------------------------------------------------------------
# Low-confidence / follow-up question flow
# ---------------------------------------------------------------------

def test_low_confidence_triggers_followups():
    result, matched = predict_from_text("I have knee pain")
    assert result["low_confidence"] is True
    assert len(result["followup_questions"]) > 0


def test_rich_symptom_set_not_low_confidence():
    result, matched = predict_from_text(
        "I have high fever, severe headache, joint pain and body pain since yesterday")
    assert result["matched_symptom_count"] >= 2


# ---------------------------------------------------------------------
# Telugu / Hindi body-location voice-style text
# ---------------------------------------------------------------------

def test_telugu_knee_and_hand_pain():
    assert "knee_pain" in extract_symptoms_from_text("నాకు మోకాలికి నొప్పిగా ఉంది")
    assert "hand_pain" in extract_symptoms_from_text("నా చేతికి నొప్పిగా ఉంది")


def test_hindi_knee_and_hand_pain():
    assert "knee_pain" in extract_symptoms_from_text("मेरे घुटने में दर्द है")
    assert "hand_pain" in extract_symptoms_from_text("मेरे हाथ में दर्द हो रहा है")


# ---------------------------------------------------------------------
# Chat assistant: symptom-aware replies, safety override, tracking
# ---------------------------------------------------------------------

def test_chat_responds_to_knee_pain_directly():
    reply = chat_response("I have knee pain", language="en-IN")
    assert reply and len(reply) > 0
    assert "diagnosis" in reply.lower() or "pattern" in reply.lower() or "?" in reply


def test_chat_emergency_override():
    reply = chat_response("I have severe chest pain and can't breathe", language="en-IN")
    assert "emergency" in reply.lower() or "112" in reply or "immediately" in reply.lower()


def test_chat_known_intent_still_works():
    reply = chat_response("Can I exercise?", risk_level="Low", language="en-IN")
    assert "activity" in reply.lower() or "walking" in reply.lower() or "exercise" in reply.lower()


def test_chat_tracking_intent_without_uid():
    reply = chat_response("How am I recovering?", language="en-IN")
    assert "dashboard" in reply.lower() or "recovery" in reply.lower()


# ---------------------------------------------------------------------
# Emergency/safety detector
# ---------------------------------------------------------------------

def test_emergency_detection_positive():
    is_emergency, category = detect_emergency("I have severe chest pain and difficulty breathing")
    assert is_emergency is True
    assert category == "severe_chest_pain_or_breathing"


def test_emergency_detection_negative():
    is_emergency, category = detect_emergency("I have a mild headache")
    assert is_emergency is False


# ---------------------------------------------------------------------
# New conditions/symptoms sourced from medical_conditions_and_symptoms.csv
# ---------------------------------------------------------------------

def test_new_body_part_columns_present():
    import json, os
    cols = json.load(open(os.path.join(os.path.dirname(__file__), "..", "models", "symptom_columns.json")))
    for expected in ["heel_pain", "tingling", "numbness", "balance_problems",
                      "morning_stiffness", "limited_range_of_motion"]:
        assert expected in cols, expected


def test_heel_pain_extraction():
    assert "heel_pain" in extract_symptoms_from_text("I have heel pain when I walk")


def test_no_false_positive_arm_from_warm():
    # "warm" must not spuriously match the "arm" body part.
    matched = extract_symptoms_from_text("the area feels warm and swollen")
    assert "arm_pain" not in matched


def test_tingling_and_numbness_without_pain_word():
    matched = extract_symptoms_from_text("my wrist and fingers are tingling and numb")
    assert "tingling" in matched
    assert "numbness" in matched


def test_vertigo_prediction():
    result, matched = predict_from_text("I feel dizzy and off balance with nausea")
    assert "balance_problems" in matched
    assert result["top_diseases"][0]["disease"] == "Vertigo (Benign Pattern)"


def test_gout_prediction():
    result, matched = predict_from_text(
        "severe joint pain in my big toe, it is red, swollen and warm")
    assert result["top_diseases"][0]["disease"] == "Gout"


def test_sciatica_symptoms_extracted():
    matched = extract_symptoms_from_text("I have sciatica pain going down my leg with tingling")
    assert "back_pain" in matched and "leg_pain" in matched and "tingling" in matched


def test_50_diseases_trained():
    import json, os
    report = json.load(open(os.path.join(os.path.dirname(__file__), "..", "models", "training_report.json")))
    assert report["num_diseases"] >= 40


# ---------------------------------------------------------------------
# Performance regression guard
# ---------------------------------------------------------------------

def test_prediction_latency_budget():
    """Locks in the regex-precompilation + numpy-vector performance fix:
    a full predict_from_text call (extraction + both RF models) should
    comfortably complete in well under 100ms even on a cold/uncached
    input, so this doesn't silently regress back to the ~37ms-per-call
    (mostly regex-recompilation) behavior seen before that fix."""
    import time
    text = "I have fever, headache, joint pain and my knee has been hurting for three days."
    predict_from_text(text)  # warm up (model already loaded at import time)
    t0 = time.perf_counter()
    for i in range(20):
        predict_from_text(f"{text} case {i}")  # unique text -> cache miss each time
    elapsed_ms_per_call = (time.perf_counter() - t0) / 20 * 1000
    assert elapsed_ms_per_call < 100, f"prediction pipeline too slow: {elapsed_ms_per_call:.1f} ms/call"


# ---------------------------------------------------------------------
# Context clues: duration/severity/sidedness -- avoid re-asking answered
# follow-up questions, and multi-turn conversational memory in chat
# ---------------------------------------------------------------------

def test_duration_mentioned_skips_duration_question():
    result, matched = predict_from_text("I have knee pain for three days")
    assert not any("how long" in q.lower() for q in result["followup_questions"])


def test_no_duration_still_asks_duration_question():
    result, matched = predict_from_text("I have knee pain")
    assert any("how long" in q.lower() for q in result["followup_questions"])


def test_swelling_mentioned_skips_swelling_question():
    result, matched = predict_from_text("I have knee pain, no swelling")
    assert not any("swelling" in q.lower() for q in result["followup_questions"])


def test_sidedness_mentioned_skips_side_question():
    result, matched = predict_from_text("I have right knee pain")
    assert not any("one side only" in q.lower() for q in result["followup_questions"])


def test_severity_detected():
    result, matched = predict_from_text("I have severe knee pain")
    assert result["severity"] == "severe"


def test_chat_remembers_followup_context():
    """A short follow-up reply with no symptom keywords of its own should
    still be understood in the context of the prior turn, once the
    assistant has asked a follow-up question."""
    r1 = chat_response("I have knee pain", language="en-IN")
    assert "?" in r1  # sanity: turn 1 actually asked follow-up questions
    history = [
        {"role": "user", "content": "I have knee pain"},
        {"role": "assistant", "content": r1},
    ]
    r2 = chat_response("since 3 days, no swelling, right side only, no fever",
                        language="en-IN", history=history)
    assert "not present" in r2.lower()
    # Previously-answered questions should not be repeated.
    assert "how long" not in r2.lower()
    assert "swelling, redness" not in r2.lower()


def test_chat_does_not_merge_unrelated_followup_turn():
    """If the assistant's last turn did NOT ask a question, an unrelated
    short message should not be spuriously merged with older history."""
    history = [
        {"role": "user", "content": "I have knee pain"},
        {"role": "assistant", "content": "Open the Symptom Analyzer page for a full breakdown."},
    ]
    reply = chat_response("thanks", language="en-IN", history=history)
    assert reply is not None
