import os, sys, shutil
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'modules'))

from session_manager import (
    create_session, load_session, set_image_result, set_symptom_result,
    list_sessions, add_log_entry,
)


def _cleanup(uid):
    d = os.path.join(os.path.dirname(__file__), "..", "data", "sessions", uid)
    shutil.rmtree(d, ignore_errors=True)


def test_rapid_session_creation_does_not_collide():
    """Regression: create_session used to derive the session_id (and
    therefore the filename) from a second-resolution timestamp only, so
    two sessions created within the same second silently overwrote each
    other -- real data loss."""
    uid = "test_rapid_collision"
    _cleanup(uid)
    try:
        sid_a = create_session(uid, "Session A")
        sid_b = create_session(uid, "Session B")
        assert sid_a != sid_b
        assert load_session(uid, sid_a) is not None
        assert load_session(uid, sid_b) is not None
        names = {s["name"] for s in list_sessions(uid)}
        assert names == {"Session A", "Session B"}
    finally:
        _cleanup(uid)


def test_image_result_persists_to_session():
    """Regression: there was no session-level storage for image analysis
    results at all -- only an ephemeral, cross-session st.session_state
    value that was never saved to the session file."""
    uid = "test_image_persist"
    _cleanup(uid)
    try:
        sid = create_session(uid, "Test session")
        assert load_session(uid, sid)["image_result"] is None
        set_image_result(uid, sid, {"predicted_class": "healing_rash", "confidence": 0.8})
        reloaded = load_session(uid, sid)
        assert reloaded["image_result"] == {"predicted_class": "healing_rash", "confidence": 0.8}
    finally:
        _cleanup(uid)


def test_image_and_symptom_results_isolated_per_session():
    """Regression: without per-session storage, an image/symptom result
    from one session could leak into a different session's Daily Tracking
    entry (the reported "image class not working" bug)."""
    uid = "test_isolation"
    _cleanup(uid)
    try:
        sid_a = create_session(uid, "Problem A")
        set_image_result(uid, sid_a, {"predicted_class": "inflamed_rash", "confidence": 0.9})
        set_symptom_result(uid, sid_a, {"top_diseases": [{"disease": "Gout", "probability": 0.6}]})

        sid_b = create_session(uid, "Problem B")
        data_b = load_session(uid, sid_b)
        assert data_b["image_result"] is None
        assert data_b["symptom_result"] is None

        data_a = load_session(uid, sid_a)
        assert data_a["image_result"]["predicted_class"] == "inflamed_rash"
    finally:
        _cleanup(uid)


def test_add_log_entry_works_on_legacy_session_missing_daily_log_key():
    """Regression: add_log_entry used data["daily_log"].append(...) with
    no default, which would KeyError on any session dict missing that key
    (e.g. a session file from an older schema version)."""
    uid = "test_legacy_schema"
    _cleanup(uid)
    try:
        sid = create_session(uid, "Legacy")
        # Simulate an older/partial session file missing "daily_log".
        from session_manager import load_session as _load, save_session as _save
        data = _load(uid, sid)
        del data["daily_log"]
        _save(uid, sid, data)

        add_log_entry(uid, sid, {"severity": 3})
        reloaded = _load(uid, sid)
        assert reloaded["daily_log"] == [{"severity": 3}]
    finally:
        _cleanup(uid)
