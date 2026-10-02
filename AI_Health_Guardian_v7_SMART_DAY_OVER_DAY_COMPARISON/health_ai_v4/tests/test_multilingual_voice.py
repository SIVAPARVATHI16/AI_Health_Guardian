import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'modules'))
from symptom_analyzer import extract_symptoms_from_text
from chat_assistant import chat_response


def test_telugu_symptoms():
    found = extract_symptoms_from_text('జలుబు దగ్గు')
    assert 'cough' in found
    assert 'runny_nose' in found


def test_hindi_symptoms():
    found = extract_symptoms_from_text('बुखार खांसी')
    assert 'mild_fever' in found
    assert 'cough' in found


def test_tamil_symptoms():
    found = extract_symptoms_from_text('காய்ச்சல் இருமல்')
    assert 'mild_fever' in found
    assert 'cough' in found


def test_french_symptoms():
    found = extract_symptoms_from_text('fièvre toux')
    assert 'mild_fever' in found
    assert 'cough' in found


def test_spanish_symptoms():
    found = extract_symptoms_from_text('fiebre tos')
    assert 'mild_fever' in found
    assert 'cough' in found


def test_telugu_chat():
    text = chat_response('నేను వ్యాయామం చేయవచ్చా?', risk_level='Low', language='te-IN')
    assert text
    assert any(ch in text for ch in 'వ్యాయామం')


def test_hindi_chat():
    text = chat_response('क्या मैं व्यायाम कर सकता हूं?', risk_level='Low', language='hi-IN')
    assert text
    assert 'हल्की' in text or 'व्यायाम' in text


def test_malayalam_symptoms():
    found = extract_symptoms_from_text("പനി ചുമ")
    assert "mild_fever" in found and "cough" in found

def test_punjabi_symptoms():
    found = extract_symptoms_from_text("ਬੁਖਾਰ ਖੰਘ")
    assert "mild_fever" in found and "cough" in found

def test_urdu_symptoms():
    found = extract_symptoms_from_text("بخار کھانسی")
    assert "mild_fever" in found and "cough" in found


def test_hindi_cold_colloquial():
    matched = extract_symptoms_from_text("ठंड")
    assert "runny_nose" in matched or "congestion" in matched


def test_hindi_cold_sentence():
    matched = extract_symptoms_from_text("मुझे ठंड लग रही है और खाँसी है")
    assert "runny_nose" in matched or "congestion" in matched
    assert "cough" in matched or "mild_cough" in matched
