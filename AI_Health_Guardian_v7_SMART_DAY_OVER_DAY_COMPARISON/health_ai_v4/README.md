# AI Health Guardian

A Streamlit healthcare-awareness project combining NLP, machine learning, optional CNN image analysis, multilingual voice interaction, health-risk screening, recovery tracking, medicine reminders, session history, and online emergency/hospital tools.

> This project is for educational/health-awareness use only. It is not a medical diagnosis or treatment system.

## Setup: enabling the full AI chat assistant (Gemini)

The Chat Assistant works in a basic, rule-based mode out of the box with no
setup. To unlock the full ChatGPT/Claude/Gemini-style experience -- open-
ended conversation on anything, full multi-turn memory, and the assistant
actively calling this app's own symptom-analysis model and your real
recovery data mid-conversation -- add a free Gemini API key:

1. Get a free key from [Google AI Studio](https://aistudio.google.com/apikey).
2. Set it as the `GEMINI_API_KEY` environment variable, e.g.:
   ```bash
   export GEMINI_API_KEY="your-key-here"
   ```
   or add it to `.streamlit/secrets.toml`:
   ```toml
   GEMINI_API_KEY = "your-key-here"
   ```
3. Restart the app. The Chat Assistant page shows a banner confirming
   whether full conversational mode is active.

The same key also improves voice transcription accuracy (see "Voice input"
below) and assists symptom-phrase understanding as a fallback. Nothing else
in the app requires this key -- every feature works without it, just with
the more limited rule-based chat instead of a full conversational one.

## Features

- Symptom Analyzer: free text + quick selection + Random Forest disease/risk prediction
- Multilingual symptom extraction: English, Hindi, Telugu, Tamil, Kannada, Bengali, Marathi, Gujarati, Arabic, French and Spanish

## Dataset & Model

The symptom→disease model is trained on a synthetically generated dataset
(`data/generate_dataset.py`) built from a hand-curated knowledge base of
**50 diseases/conditions** and **91 symptom columns**, including body-location
pain (knee, hand, wrist, shoulder, back, neck, hip, ankle, foot, heel, leg,
arm, finger, pelvis, etc.), each with a standard, well-established textbook
symptom pattern.

An expanded set of conditions and symptom terms was incorporated from a
user-supplied `medical_conditions_and_symptoms.csv` taxonomy. That file is a
flat vocabulary list (name/category/type) with **no disease↔symptom
relationship data** — it does not say which symptoms belong to which
condition. Two things were done with it:

1. **All of its ~67 symptom terms** were added to the symptom-recognition
   vocabulary (`modules/symptom_normalizer.py` and the synonym dictionary
   in `models/train_all.py`), so phrases like "sciatica pain", "carpal
   tunnel", "plantar fasciitis", "TMJ pain", "menstrual cramps", "dysuria",
   etc. are now recognized and mapped to canonical symptom codes.
2. A **curated subset of ~27 common, well-established, low/medium-risk
   conditions** from that file (mostly musculoskeletal, plus a few
   digestive/EENT/dermatological/balance conditions) were added as
   trainable disease classes, using standard, widely documented clinical
   symptom patterns for each — e.g. Sciatica, Osteoarthritis, Rheumatoid
   Arthritis, Gout, Carpal Tunnel Syndrome, Plantar Fasciitis, Tension
   Headache, Vertigo (Benign Pattern), Gastritis/Indigestion, Eczema or
   Contact Dermatitis, Tonsillitis/Strep Throat (Pattern).

**Deliberately excluded** from disease-prediction training: cancers/tumors,
rare genetic and congenital disorders, outbreak/exotic infectious diseases
(Ebola, anthrax, plague, rabies, smallpox, cholera...), acute emergencies
(stroke, sepsis, meningitis, pulmonary embolism, appendicitis...), and
psychiatric diagnoses. The source file had no real symptom-association data
for these, and fabricating one so a self-report checkbox tool could output
"possible melanoma" or "possible ALS" would be medically irresponsible for
an educational, non-diagnostic app. Where relevant, their common symptom
*terms* are still recognized in text (so the app doesn't ignore what a user
typed) — they just don't drive a disease prediction. Urgent-sounding
descriptions are instead caught by the separate `modules/safety.py`
emergency detector, which prioritizes "seek care now" guidance over any
pattern match.

Retrain at any time with:
```bash
cd data && python3 generate_dataset.py
cd ../models && python3 train_all.py --skip-cnn   # omit --skip-cnn to also retrain the image model if TensorFlow is installed
```

## Voice input & text-to-speech: bugs found and fixed

Several real, reproducible bugs were found in the voice/TTS pipeline:

1. **`st.audio_input`'s value persists across every Streamlit rerun**, not
   just the run where you record. The code was re-sending the same
   recording to the speech API and re-transcribing it on *every single
   rerun of the whole app* (any button click anywhere), not once per
   recording. This is why "Clear voice text" appeared broken -- the very
   next rerun re-populated it from the still-attached recording -- and
   likely contributed to poor-quality/garbled results from repeatedly
   hammering the same audio at an unofficial, rate-limited endpoint.
   Fixed by deduping on the recording's stable `file_id` in
   `app/app.py::voice_input`, so transcription now runs exactly once per
   new recording.
2. **The free `recognize_google` endpoint** (SpeechRecognition's
   `recognize_google`, no API key) is an unofficial, reverse-engineered
   Google endpoint with no accuracy or availability guarantees, and it can
   return unrelated "best guess" text for anything less than very clean
   audio -- which is consistent with the odd, unrelated transcriptions
   reported. When a `GEMINI_API_KEY` is configured, voice input now tries
   **Gemini's native audio transcription first** (`modules/gemini_assistant.
   py::transcribe_audio_gemini`), which is both more accurate and far more
   reliable across languages, falling back to the old locale-retry Google
   Speech path only if Gemini isn't configured or fails.
3. **The Gemini integration itself was silently broken**: `chat_reply` and
   `suggest_symptoms_from_text` called `model.generate_content(...)`, a
   method that only exists on the old, deprecated `google-generativeai`
   SDK's model object -- not on the current `google-genai` SDK's `Client`
   object this app actually uses. Every call raised `AttributeError`
   internally, which was silently swallowed by the fallback `except`
   clause, so any Gemini-assisted symptom extraction or chat fallback was
   a silent no-op even with a valid key configured. Fixed to use
   `client.models.generate_content(model=..., contents=..., config=...)`,
   the current SDK's actual call shape.
4. **"Read this aloud" / "Read summary aloud" did nothing.** Two separate
   causes: (a) the summary-dashboard button injected `<script>speak(...)
   </script>` calling a JavaScript function that was never defined anywhere
   in that isolated iframe -- a guaranteed `ReferenceError`, invisible since
   iframe console errors don't surface in the Streamlit UI; (b) more
   generally, calling `speechSynthesis.speak()` automatically when a
   `components.v1.html` iframe's script loads is not a genuine user
   gesture from that frame's point of view, and modern browsers silently
   block autoplaying speech without one -- so even the "working" button
   could fail silently depending on the browser. Fixed by rendering an
   actual clickable Play/Stop button *inside* the iframe itself
   (`app/app.py::speak_text`), so the click is a real, in-frame user
   gesture, with on-screen status text ("Speaking...", "Done.", or an
   explicit error) instead of failing invisibly.
5. A transcribed voice **chat question kept resurfacing** after being sent
   once, since its text stayed in session state after use; now cleared
   immediately after sending.

## Symptom Analyzer & Chat Assistant: recent improvements

- **Fixed a dead parameter**: `chat_response(..., history=...)` was accepted
  but never used, so every chat message was analyzed in total isolation.
  A short follow-up reply like "since 3 days, no swelling" after the
  assistant asked clarifying questions would lose all context of the
  original "I have knee pain" message. Now, when the assistant's last turn
  ended in a question, the next user message is combined with the prior
  one before re-running the symptom pipeline -- this only triggers on that
  specific signal, so unrelated turns are never stitched together.
- **Duration/severity/sidedness awareness**
  (`symptom_normalizer.extract_context_clues`): follow-up questions no
  longer re-ask something the person already said. "I have knee pain for
  three days" skips the "how long?" question; mentioning "no swelling" or
  "right side only" skips those questions too. Severity words ("severe",
  "mild"...) are detected and a severe report adds an explicit
  "don't wait on a screening result" nudge.

## Performance

Profiling (`cProfile`) surfaced two real bottlenecks in the symptom
pipeline, both fixed:
- **Regex recompilation on every phrase check.** The literal-phrase
  matcher used to build and compile a fresh regex per phrase per clause on
  every call (~300 phrases × multiple clauses). All patterns are now
  precompiled once at module import (`modules/symptom_analyzer.py`), and
  clause text is normalized once instead of once per phrase.
- **Pandas DataFrame construction on every prediction.** The feature
  vector fed to the RandomForest models is now built as a plain numpy
  array via a precomputed symptom→index map, instead of constructing a
  new DataFrame per call. Models are also now fit on numpy arrays (no
  stored feature-name metadata), removing a `sklearn` warning that fired
  on every prediction.
- Added an in-process LRU cache (`functools.lru_cache`, 512 entries) on
  the symptom-extraction step, since the same message is often
  re-analyzed more than once per interaction (Streamlit reruns, chat +
  Symptom Analyzer both processing the same text).
- Reduced `n_estimators` (200→120 disease model, 150→100 risk model)
  after confirming accuracy held essentially steady (94.0%/94.8% vs.
  94.7%/94.4%), since a single-row RandomForest prediction is dominated by
  per-tree Python/joblib dispatch overhead, not tree depth.

Net effect: `predict_from_text` went from **~37ms to ~9-10ms per call**
(cache miss) / **~8ms** (cache hit) on the same hardware — roughly a
4x improvement. A regression test (`test_prediction_latency_budget` in
`tests/test_symptom_normalization.py`) guards against this silently
regressing back.

Model loading itself (`joblib.load` for the RF models) already only
happens once per server process, since Python caches imported modules —
Streamlit reruns do not reload them.

- Multilingual voice input using the selected speech-recognition locale
- Multilingual Chat Assistant: text or voice questions with localized responses
- Browser text-to-speech in the selected language
- Health Risk Assessment: BMI, blood pressure, activity and lifestyle screening
- Emergency & Hospitals: emergency dialler and online OpenStreetMap/Google Maps hospital search
- Optional CNN image analysis and image comparison
- Daily symptom/medicine tracking
- Recovery dashboard with Plotly charts
- Medicine tracker and browser reminders
- Session history for symptoms, logs, medicines and chat

## Important: Emergency Offline Mode is excluded

This version intentionally does **not** include the new Emergency Offline Mode. There is no offline hospital database, offline routing, offline GPS emergency workflow, SMS queue or satellite integration in this version. Those can be added later to a native mobile/final-year version.

## Structure

```text
health_ai_guardian/
├── app/app.py
├── modules/
│   ├── symptom_analyzer.py
│   ├── image_analyzer.py
│   ├── recovery_tracker.py
│   ├── genai_explainer.py
│   ├── health_risk.py
│   ├── chat_assistant.py
│   └── session_manager.py
├── data/
├── models/
├── tests/
├── requirements.txt
├── requirements-dl.txt
├── run_windows.bat
└── setup_and_run.sh
```

## Windows

Use Python 3.11. Then run:

```powershell
.venv\Scripts\activate
python -m pip install -r requirements.txt
streamlit run app\app.py
```

Or double-click `run_windows.bat`.

TensorFlow is optional and is only needed for the CNN/image-analysis feature.

## Multilingual Voice

Select **Language / Voice** in the sidebar before recording. Supported recognition locales are:

- English (India)
- Hindi
- Telugu
- Tamil
- Kannada
- Bengali
- Marathi
- Gujarati
- Arabic
- French
- Spanish

For symptom analysis: record → stop → click **Analyze Symptoms**.

For chat: record the question → click **Send voice question**. The assistant responds in the selected language where localized responses are available.

Speech-to-text uses Google's SpeechRecognition service, so internet access is normally required for transcription. Browser text-to-speech uses the device's available speech voices.

### Examples

```text
Telugu: జలుబు దగ్గు
→ cough + runny nose/congestion

Hindi: बुखार खांसी
→ fever + cough

Tamil: காய்ச்சல் இருமல்
→ fever + cough
```

## Model note

The included model artifacts and datasets are intended for project demonstration. Accuracy on generated/synthetic data must not be presented as clinical accuracy. For a final-year project, document the dataset source, preprocessing, train/test split, evaluation metrics and limitations.

## Symptom Analyzer severity update (v4.1)

The Symptom Analyzer now has a human-readable **Severity Guide** in addition to the ML risk level. When cough is detected, the UI shows a three-row guide for **Low Risk / Moderate Risk / High Risk** with example cough patterns and what to do next. High-risk warning patterns are also surfaced separately from the model probability.

The analyzer uses two layers:

1. **ML layer:** Random Forest disease and risk models trained on the binary symptom vector.
2. **Safety/severity layer:** explicit warning phrases such as difficulty breathing, significant bleeding, blue lips/face, severe chest pain, confusion, or rapidly worsening symptoms can trigger a safety-oriented severity override.

The UI shows the source of the level and keeps the ML probability visible separately when a rule-based safety override is used.

## Model comparison for project review

Run from `models/`:

```powershell
python compare_models.py
```

This evaluates Logistic Regression, Decision Tree, KNN, SVM (RBF), the production Random Forest, Extra Trees, HistGradientBoosting, and XGBoost on the same stratified 80/20 split. Results are written to:

- `models/model_comparison.csv`
- `models/model_comparison_report.json`
- `models/MODEL_TRAINING_AND_SELECTION.md`
- `models/confusion_matrix_risk_random_forest.csv`
- `models/confusion_matrix_disease_random_forest.csv`

The metrics are **project-dataset/test-split metrics, not clinical diagnostic accuracy**.

## Improved persistent skin progress tracking

The skin image module now:
- stores each analyzed image and its full CNN result inside the active session;
- automatically uses the most recent stored image as tomorrow's comparison baseline;
- compares CNN probability distributions instead of relying only on a class-label change;
- records non-diagnostic visual measurements such as brightness, contrast, redness signal, warm-pixel fraction, texture and sharpness;
- calculates photo comparability and reports **Uncertain** when lighting/framing differences make the comparison unreliable;
- keeps a chronological skin-analysis history that survives Streamlit reruns and app restarts;
- uses cautious language: the module reports visual trends and screening signals, not a diagnosis or a claim that a disease has been cured.

For meaningful longitudinal comparisons, photograph the same body area at a similar distance, angle and lighting.

## Improved day-over-day skin comparison (v7 visual-region engine)

The persistent multi-session image workflow now stores the complete analysis for every uploaded photo and links each record to the immediately previous image in the same session.

The image analysis layer now combines:

1. **CNN class probabilities** — retained as supporting evidence only.
2. **Adaptive visible-region estimate** — estimates the fraction of skin-like pixels belonging to a compact colour/texture anomaly. This is a visual estimate, not a clinical lesion measurement.
3. **Direct photo comparison** — the previous and current photos are resized, high-pass filtered, aligned with a small translation search, and compared pixel-by-pixel to find the strongest localized change.
4. **Directional change analysis** — the strongest changed region is checked for whether it became visually redder/darker or less red/dark.
5. **Photo comparability checks** — brightness, contrast, aspect ratio and skin-region coverage reduce confidence when photos are not comparable.
6. **Session persistence** — each image record stores its analysis, comparison result, previous image ID, visual metrics and timestamp, so tomorrow's upload automatically compares against today's stored image.

This prevents a rule such as `Healthy == Healthy -> Stable`. A matching CNN label can coexist with a larger visible changed region, and the comparison engine can therefore report a visual worsening/improvement signal instead of automatically returning Stable.

> **Safety:** These measurements are for visual progress monitoring only. They are not a medical diagnosis, lesion measurement, or clinical severity score. Rapidly worsening skin symptoms, severe pain, blistering/open sores, fever/illness, eye/mouth/genital involvement, or breathing/swallowing difficulty should be evaluated by a qualified clinician/emergency service as appropriate.

## v7 image-recovery comparison update

The skin image module now uses a layered day-over-day comparison instead of treating an identical CNN class as "Stable".

Each stored image analysis contains:

- CNN class probabilities and certainty.
- Brightness, contrast, redness, warmth, texture and sharpness measurements.
- A heuristic visible affected-region estimate based on local redness/darkness and compactness.
- Region confidence, normalized location and area as a percentage of detected skin pixels.
- A version tag (`v7-visible-region-v1`) so stored analyses remain traceable.

When a new photo is uploaded, the active session automatically loads the newest previous image and analysis. The comparison engine combines:

1. Visible affected-region area change.
2. Change in the affected-region redness signal.
3. Direct localized visual difference between the two stored photos.
4. CNN probability-weighted visual score.
5. Photo comparability from framing/lighting-related measurements.

A strong visible-area increase can therefore produce **Worsening even when both CNN predictions are `Healthy / Normal`**. Conversely, a decrease in the visible affected region can support **Improving**. If the photos are not sufficiently comparable, the result is **Uncertain** rather than an invented trend.

The previous day's analysis and image are persisted inside the selected health session and are automatically reused as the next day's baseline. Older v6 records that do not contain the new region metric are supported through direct photo comparison and model/redness evidence during the transition.

This remains a visual monitoring/screening feature using synthetic demo data. The affected-region percentage is not a clinical lesion measurement and the CNN is not a diagnostic model.
