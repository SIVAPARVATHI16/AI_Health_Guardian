import os, sys
from unittest.mock import MagicMock, patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'modules'))

import gemini_assistant as g


def _fake_client_with_response(text):
    """Build a mock google.genai Client whose models.generate_content
    returns an object with a `.text` attribute, matching the real SDK's
    response shape."""
    client = MagicMock()
    response = MagicMock()
    response.text = text
    client.models.generate_content.return_value = response
    return client


# ---------------------------------------------------------------------
# Regression tests for the old-SDK/new-SDK mismatch bug: these functions
# used to call `model.generate_content(...)` directly on a `genai.Client`
# instance (a method that doesn't exist on Client -- only on the old,
# deprecated `google-generativeai` GenerativeModel object), which raised
# AttributeError internally and was silently swallowed by the blanket
# except clause, making every Gemini-assisted path a silent no-op.
# ---------------------------------------------------------------------

def test_suggest_symptoms_uses_correct_sdk_call_shape():
    fake_client = _fake_client_with_response('["mild_fever", "headache"]')
    with patch.object(g, "_client", return_value=fake_client):
        result = g.suggest_symptoms_from_text("I feel awful", ["mild_fever", "headache", "cough"])
    assert result == ["mild_fever", "headache"]
    # Confirms the call went through client.models.generate_content(...),
    # not the old (nonexistent-on-Client) client.generate_content(...).
    fake_client.models.generate_content.assert_called_once()


def test_chat_reply_uses_correct_sdk_call_shape():
    fake_client = _fake_client_with_response("Please stay hydrated and rest.")
    with patch.object(g, "_client", return_value=fake_client):
        result = g.chat_reply("what should I do for a mild cold?")
    assert result == "Please stay hydrated and rest."
    fake_client.models.generate_content.assert_called_once()


def test_transcribe_audio_gemini_uses_correct_sdk_call_shape():
    fake_client = _fake_client_with_response("I have knee pain")
    with patch.object(g, "_client", return_value=fake_client):
        text, error = g.transcribe_audio_gemini(b"fake-wav-bytes", mime_type="audio/wav")
    assert text == "I have knee pain"
    assert error is None
    fake_client.models.generate_content.assert_called_once()


def test_transcribe_audio_gemini_handles_no_speech_marker():
    fake_client = _fake_client_with_response("[NO_SPEECH]")
    with patch.object(g, "_client", return_value=fake_client):
        text, error = g.transcribe_audio_gemini(b"silence", mime_type="audio/wav")
    assert text is None
    assert error is not None


def test_transcribe_audio_gemini_not_configured():
    with patch.object(g, "_client", return_value=None):
        text, error = g.transcribe_audio_gemini(b"bytes")
    assert text is None
    assert "not configured" in error.lower()


def test_suggest_symptoms_never_returns_out_of_vocabulary_codes():
    # Even if the model hallucinates a code outside the known vocabulary,
    # it must never leak through.
    fake_client = _fake_client_with_response('["mild_fever", "made_up_disease_code"]')
    with patch.object(g, "_client", return_value=fake_client):
        result = g.suggest_symptoms_from_text("text", ["mild_fever"])
    assert result == ["mild_fever"]


def test_functions_fail_soft_on_exception():
    fake_client = MagicMock()
    fake_client.models.generate_content.side_effect = RuntimeError("network down")
    with patch.object(g, "_client", return_value=fake_client):
        assert g.suggest_symptoms_from_text("text", ["mild_fever"]) == []
        assert g.chat_reply("hello") is None
        text, error = g.transcribe_audio_gemini(b"bytes")
        assert text is None and error is not None


# ---------------------------------------------------------------------
# Agentic chat (ChatGPT/Claude-style): full history + tool-calling
# ---------------------------------------------------------------------

def test_history_to_contents_conversion():
    history = [
        {"role": "user", "content": "I have knee pain"},
        {"role": "assistant", "content": "How long have you had it?"},
        {"role": "user", "content": "3 days"},
    ]
    contents = g._history_to_contents(history)
    assert contents == [
        {"role": "user", "parts": [{"text": "I have knee pain"}]},
        {"role": "model", "parts": [{"text": "How long have you had it?"}]},
        {"role": "user", "parts": [{"text": "3 days"}]},
    ]


def test_history_to_contents_truncates_long_history():
    history = [{"role": "user", "content": f"msg {i}"} for i in range(100)]
    contents = g._history_to_contents(history, max_turns=10)
    assert len(contents) == 10
    assert contents[-1]["parts"][0]["text"] == "msg 99"


def test_build_tools_analyze_symptoms_returns_real_prediction():
    tools = g._build_tools(uid="tool_test_user")
    by_name = {t.__name__: t for t in tools}
    result = by_name["analyze_symptoms"]("I have knee pain and fever")
    assert "knee_pain" in result["matched_symptoms"]
    assert result["risk_level"] in {"Low", "Medium", "High"}
    assert isinstance(result["top_disease_patterns"], list)


def test_build_tools_recovery_status_handles_no_data():
    tools = g._build_tools(uid="brand_new_user_no_data")
    by_name = {t.__name__: t for t in tools}
    result = by_name["get_recovery_status"]()
    assert result["days_tracked"] == 0


def test_build_tools_recovery_status_handles_none_uid():
    """uid=None (anonymous session) must not crash the tool."""
    tools = g._build_tools(uid=None)
    by_name = {t.__name__: t for t in tools}
    result = by_name["get_recovery_status"]()
    assert result["days_tracked"] == 0


def test_agentic_chat_reply_passes_history_and_tools():
    fake_client = _fake_client_with_response("Sure, I can help with that.")
    with patch.object(g, "_client", return_value=fake_client):
        reply = g.agentic_chat_reply(
            "hello",
            history=[{"role": "user", "content": "earlier"}, {"role": "assistant", "content": "earlier reply"}],
            uid="u1",
            app_context={"disease": "Flu", "risk_level": "Low"},
        )
    assert reply == "Sure, I can help with that."
    kwargs = fake_client.models.generate_content.call_args.kwargs
    assert kwargs["contents"][-1] == {"role": "user", "parts": [{"text": "hello"}]}
    assert len(kwargs["config"].tools) == 2
    assert "Flu" in kwargs["config"].system_instruction


def test_agentic_chat_reply_none_when_not_configured():
    with patch.object(g, "_client", return_value=None):
        assert g.agentic_chat_reply("hello") is None


def test_agentic_chat_reply_fails_soft():
    fake_client = MagicMock()
    fake_client.models.generate_content.side_effect = RuntimeError("boom")
    with patch.object(g, "_client", return_value=fake_client):
        assert g.agentic_chat_reply("hello") is None


# ---------------------------------------------------------------------
# chat_assistant.chat_response routing: emergency check must run before
# (and regardless of) the Gemini path; Gemini path used when available,
# rule-based fallback used otherwise.
# ---------------------------------------------------------------------

def test_emergency_overrides_even_when_gemini_available():
    import chat_assistant
    fake_client = _fake_client_with_response("I would just chat about this normally.")
    with patch.object(g, "_client", return_value=fake_client), \
         patch("gemini_assistant.gemini_available", return_value=True):
        reply = chat_assistant.chat_response(
            "I have severe chest pain and can't breathe", language="en-IN")
    # The emergency message must win, not a normal Gemini reply -- and
    # Gemini should never even be called for this message.
    assert "emergency" in reply.lower() or "immediately" in reply.lower()
    fake_client.models.generate_content.assert_not_called()


def test_chat_response_uses_agentic_path_when_gemini_available():
    import chat_assistant
    fake_client = _fake_client_with_response("Here's a general, conversational answer.")
    with patch.object(g, "_client", return_value=fake_client), \
         patch("gemini_assistant.gemini_available", return_value=True):
        reply = chat_assistant.chat_response("What's a good stretch for a stiff neck?", language="en-IN")
    assert reply == "Here's a general, conversational answer."
    fake_client.models.generate_content.assert_called_once()


def test_chat_response_falls_back_to_rule_based_when_gemini_unavailable():
    import chat_assistant
    with patch("gemini_assistant.gemini_available", return_value=False):
        reply = chat_assistant.chat_response("Can I exercise?", risk_level="Low", language="en-IN")
    assert "exercise" in reply.lower() or "activity" in reply.lower() or "walking" in reply.lower()


def test_chat_response_falls_back_when_agentic_reply_is_none():
    import chat_assistant
    with patch("gemini_assistant.gemini_available", return_value=True), \
         patch("gemini_assistant.agentic_chat_reply", return_value=None):
        reply = chat_assistant.chat_response("Can I exercise?", risk_level="Low", language="en-IN")
    assert "exercise" in reply.lower() or "activity" in reply.lower() or "walking" in reply.lower()
