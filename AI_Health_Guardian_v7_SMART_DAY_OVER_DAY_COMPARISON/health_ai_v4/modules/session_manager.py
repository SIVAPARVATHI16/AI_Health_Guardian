"""
Persistent session storage for AI Health Guardian.

Each health session stores symptom results, daily logs, chat history and a
chronological skin-image analysis history. Skin images are copied into the
session directory so tomorrow's image can automatically be compared with the
most recent previous image without asking the user to upload yesterday's
photo again.
"""
import json
import os
import uuid
import shutil
from datetime import datetime

SESSIONS_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "sessions")
os.makedirs(SESSIONS_DIR, exist_ok=True)

# Reserved filename (leading underscore keeps it out of list_sessions/the
# sidebar's session picker) used to auto-persist casual chat that isn't
# tied to any named problem/session, so it survives app restarts and
# browser refreshes without the person needing to click "Start new
# session" first.
_GENERAL_CHAT_FILE = "_general_chat.json"


def _user_dir(user_id):
    safe = "".join(c for c in str(user_id) if c.isalnum() or c in "-_") or "guest"
    d = os.path.join(SESSIONS_DIR, safe)
    os.makedirs(d, exist_ok=True)
    return d


def _session_path(user_id, session_id):
    return os.path.join(_user_dir(user_id), f"{session_id}.json")


def _images_dir(user_id, session_id):
    d = os.path.join(_user_dir(user_id), "skin_images", str(session_id))
    os.makedirs(d, exist_ok=True)
    return d


def list_sessions(user_id):
    d = _user_dir(user_id)
    sessions = []
    for f in sorted(os.listdir(d)):
        if f.endswith(".json") and f != _GENERAL_CHAT_FILE:
            try:
                with open(os.path.join(d, f), encoding="utf-8") as fp:
                    data = json.load(fp)
                sessions.append({
                    "session_id": f[:-5],
                    "name": data.get("name", f[:-5]),
                    "started": data.get("started", ""),
                    "disease": data.get("disease_context", "Unknown"),
                    "status": data.get("status", "active"),
                    "entry_count": len(data.get("daily_log", [])),
                    "symptom_count": len(data.get("symptom_history", [])),
                    "chat_count": len(data.get("chat_history", [])),
                    "image_count": len(data.get("image_history", [])),
                })
            except (OSError, json.JSONDecodeError):
                continue
    return sessions[::-1]


def create_session(user_id, name, disease_context="Unknown"):
    # A short random suffix avoids collisions when two sessions are created
    # within the same second (session_id doubles as the filename).
    session_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    data = {
        "session_id": session_id,
        "name": name,
        "started": datetime.now().strftime("%Y-%m-%d %H:%M"),
        "disease_context": disease_context,
        "status": "active",
        "daily_log": [],
        "symptom_history": [],
        "image_history": [],
        "recovery_history": [],
        "medicines": [],
        "chat_history": [],
        "symptom_result": None,
        "image_result": None,
    }
    with open(_session_path(user_id, session_id), "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    _images_dir(user_id, session_id)
    return session_id


def load_session(user_id, session_id):
    p = _session_path(user_id, session_id)
    if not os.path.exists(p):
        return None
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def save_session(user_id, session_id, data):
    with open(_session_path(user_id, session_id), "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def add_log_entry(user_id, session_id, entry):
    data = load_session(user_id, session_id)
    if not data:
        return
    data.setdefault("daily_log", []).append(entry)
    save_session(user_id, session_id, data)


def add_chat_message(user_id, session_id, role, content):
    data = load_session(user_id, session_id)
    if not data:
        return
    data.setdefault("chat_history", []).append({
        "role": role,
        "content": content,
        "time": datetime.now().strftime("%H:%M"),
    })
    save_session(user_id, session_id, data)


def clear_chat_history(user_id, session_id):
    """Wipe just the chat_history for a session (used by the chat UI's
    'New chat' button), leaving daily_log/medicines/symptom_result/
    image_result/image_history untouched -- starting a new conversation
    shouldn't lose tracking data for the same problem."""
    data = load_session(user_id, session_id)
    if not data:
        return
    data["chat_history"] = []
    save_session(user_id, session_id, data)


def set_medicines(user_id, session_id, medicines):
    data = load_session(user_id, session_id)
    if not data:
        return
    data["medicines"] = medicines
    save_session(user_id, session_id, data)


def close_session(user_id, session_id):
    data = load_session(user_id, session_id)
    if not data:
        return
    data["status"] = "closed"
    save_session(user_id, session_id, data)


def set_symptom_result(user_id, session_id, result):
    """Save the latest symptom result without losing older analyses."""
    data = load_session(user_id, session_id)
    if not data:
        return
    data["symptom_result"] = result
    data.setdefault("symptom_history", []).append({
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "result": result,
    })
    save_session(user_id, session_id, data)

def get_symptom_history(user_id, session_id):
    data = load_session(user_id, session_id) or {}
    return data.get("symptom_history", [])

def add_recovery_entry(user_id, session_id, entry):
    data = load_session(user_id, session_id)
    if not data:
        return
    data.setdefault("recovery_history", []).append(entry)
    data.setdefault("daily_log", []).append(entry)
    save_session(user_id, session_id, data)

def get_recovery_history(user_id, session_id):
    data = load_session(user_id, session_id) or {}
    return data.get("recovery_history", data.get("daily_log", []))


def set_image_result(user_id, session_id, result):
    data = load_session(user_id, session_id)
    if not data:
        return
    data["image_result"] = result
    save_session(user_id, session_id, data)


def add_image_analysis(user_id, session_id, pil_image, result, comparison=None):
    """Persist an analyzed skin image and its complete result.

    Returns the history record. The image is saved as JPEG in the session's
    private local data folder; the JSON session stores only its relative path.
    """
    data = load_session(user_id, session_id)
    if not data:
        return None

    image_id = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
    filename = f"{image_id}.jpg"
    image_dir = _images_dir(user_id, session_id)
    image_path = os.path.join(image_dir, filename)

    # JPEG keeps the stored history reasonably small. RGB conversion avoids
    # errors from PNG alpha/paletted modes.
    pil_image.convert("RGB").save(image_path, "JPEG", quality=92, optimize=True)

    history = data.setdefault("image_history", [])
    previous_record = history[-1] if history else None

    record = {
        "image_id": image_id,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "image_path": os.path.relpath(image_path, _user_dir(user_id)),
        "analysis_version": "v7_visual_region_compare",
        "previous_image_id": previous_record.get("image_id") if previous_record else None,
        "analysis": result,
        "comparison": comparison,
    }

    history.append(record)
    # Keep the legacy field synchronized with the newest analysis.
    data["image_result"] = result
    save_session(user_id, session_id, data)
    return record


def get_image_history(user_id, session_id):
    data = load_session(user_id, session_id) or {}
    return data.get("image_history", [])


def get_latest_image_analysis(user_id, session_id):
    history = get_image_history(user_id, session_id)
    return history[-1] if history else None


def get_previous_image_analysis(user_id, session_id):
    history = get_image_history(user_id, session_id)
    return history[-2] if len(history) >= 2 else None


def get_image_path(user_id, session_id, record):
    if not record:
        return None
    rel = record.get("image_path")
    if not rel:
        return None
    # Records are generated by this module; still normalize and ensure they
    # remain inside the user's session directory.
    base = os.path.abspath(_user_dir(user_id))
    path = os.path.abspath(os.path.join(base, rel))
    if not path.startswith(base + os.sep):
        return None
    return path if os.path.exists(path) else None


def reset_session_images(user_id, session_id):
    data = load_session(user_id, session_id)
    if not data:
        return
    data["image_history"] = []
    data["image_result"] = None
    img_dir = os.path.join(_user_dir(user_id), "skin_images", str(session_id))
    if os.path.isdir(img_dir):
        shutil.rmtree(img_dir, ignore_errors=True)
    save_session(user_id, session_id, data)


# ---------------------------------------------------------------------------
# General chat -- persisted automatically for casual conversations that
# aren't tied to any named problem/session (i.e. when the person hasn't
# clicked "Start new session"). Stored to disk exactly like a real session's
# chat_history, just under one reserved, always-the-same filename per user
# instead of a new file per problem, so it survives restarts/refreshes.
# ---------------------------------------------------------------------------

def _general_chat_path(user_id):
    return os.path.join(_user_dir(user_id), _GENERAL_CHAT_FILE)


def load_general_chat_history(user_id):
    p = _general_chat_path(user_id)
    if not os.path.exists(p):
        return []
    try:
        with open(p, encoding="utf-8") as f:
            return json.load(f).get("chat_history", [])
    except (OSError, json.JSONDecodeError):
        return []


def add_general_chat_message(user_id, role, content):
    history = load_general_chat_history(user_id)
    history.append({
        "role": role,
        "content": content,
        "time": datetime.now().strftime("%H:%M"),
    })
    with open(_general_chat_path(user_id), "w", encoding="utf-8") as f:
        json.dump({"chat_history": history}, f, indent=2)
    return history


def clear_general_chat_history(user_id):
    p = _general_chat_path(user_id)
    if os.path.exists(p):
        os.remove(p)
