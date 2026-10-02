"""
Image analysis for AI Health Guardian.

This module provides visual monitoring/screening only. It is NOT a diagnostic
system. The CNN predicts one of the project's synthetic demo classes, while a
second, model-independent layer measures visible image changes so that two
photos can be compared across days.

The important recovery-tracking rule is:
    same CNN class != stable condition

The day-over-day decision therefore uses:
    1. CNN probability distribution
    2. Visible affected-region estimate
    3. Redness/warm-pixel measurements
    4. Photo comparability / quality

The affected-region estimate is an image-analysis heuristic. It is deliberately
reported as a *visible affected-region estimate*, not a medical lesion area.
"""
import json
import os
from datetime import datetime

import numpy as np
from PIL import Image, ImageFilter

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
IMG_SIZE = 128
MODEL_PATH = os.path.join(MODEL_DIR, "skin_cnn.keras")
CLASS_PATH = os.path.join(MODEL_DIR, "skin_class_names.json")

LABEL_DISPLAY = {
    "healthy_skin": "Healthy / Normal",
    "healing_rash": "Healing / Fading inflammation",
    "inflamed_rash": "Active inflammation / irritation",
}

CLASS_HEALTH_SCORE = {
    "inflamed_rash": 20.0,
    "healing_rash": 60.0,
    "healthy_skin": 100.0,
}

ANALYSIS_VERSION = "v7-visible-region-v1"
_MEASURE_SIZE = 256
_MODEL = None
_TF_ERROR = None


def _load_model():
    global _MODEL, _TF_ERROR
    if _MODEL is not None:
        return _MODEL
    if _TF_ERROR is not None:
        return None
    try:
        import tensorflow as tf
        _MODEL = tf.keras.models.load_model(MODEL_PATH, compile=False)
        return _MODEL
    except Exception as exc:
        _TF_ERROR = f"TensorFlow/CNN unavailable: {exc}"
        return None


def is_available():
    return _load_model() is not None


def unavailable_reason():
    _load_model()
    return _TF_ERROR or "CNN model is unavailable."


def _class_names():
    try:
        with open(CLASS_PATH, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return ["healing_rash", "healthy_skin", "inflamed_rash"]


def _safe_float(x, default=0.0):
    try:
        return float(x)
    except Exception:
        return default


def _binary_dilate(mask):
    out = np.zeros_like(mask, dtype=bool)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            y0, y1 = max(0, dy), min(mask.shape[0], mask.shape[0] + dy)
            x0, x1 = max(0, dx), min(mask.shape[1], mask.shape[1] + dx)
            sy0, sy1 = max(0, -dy), min(mask.shape[0], mask.shape[0] - dy)
            sx0, sx1 = max(0, -dx), min(mask.shape[1], mask.shape[1] - dx)
            out[y0:y1, x0:x1] |= mask[sy0:sy1, sx0:sx1]
    return out


def _binary_erode(mask):
    out = np.ones_like(mask, dtype=bool)
    for dy in (-1, 0, 1):
        for dx in (-1, 0, 1):
            y0, y1 = max(0, dy), min(mask.shape[0], mask.shape[0] + dy)
            x0, x1 = max(0, dx), min(mask.shape[1], mask.shape[1] + dx)
            sy0, sy1 = max(0, -dy), min(mask.shape[0], mask.shape[0] - dy)
            sx0, sx1 = max(0, -dx), min(mask.shape[1], mask.shape[1] - dx)
            tmp = np.zeros_like(mask, dtype=bool)
            tmp[y0:y1, x0:x1] = mask[sy0:sy1, sx0:sx1]
            out &= tmp
    return out


def _connected_components(mask, score, red_excess, gray):
    """Return compact candidate regions without requiring OpenCV/scikit-image."""
    h, w = mask.shape
    seen = np.zeros_like(mask, dtype=bool)
    components = []

    ys, xs = np.where(mask)
    for start_y, start_x in zip(ys.tolist(), xs.tolist()):
        if seen[start_y, start_x]:
            continue

        stack = [(start_y, start_x)]
        seen[start_y, start_x] = True
        pts = []

        while stack:
            y, x = stack.pop()
            pts.append((y, x))
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ny, nx = y + dy, x + dx
                if 0 <= ny < h and 0 <= nx < w and mask[ny, nx] and not seen[ny, nx]:
                    seen[ny, nx] = True
                    stack.append((ny, nx))

        if len(pts) < 20:
            continue

        py = np.fromiter((p[0] for p in pts), dtype=np.int32)
        px = np.fromiter((p[1] for p in pts), dtype=np.int32)
        n = len(pts)
        mean_score = float(score[py, px].mean())
        mean_red = float(red_excess[py, px].mean())
        mean_gray = float(gray[py, px].mean())
        bbox = [int(px.min()), int(py.min()), int(px.max()), int(py.max())]
        cx = float(px.mean() / max(1, w - 1))
        cy = float(py.mean() / max(1, h - 1))

        # Compactness helps reject long edges/shadows.  The heuristic is only
        # used to rank visible candidate regions, not to diagnose disease.
        bw = max(1, bbox[2] - bbox[0] + 1)
        bh = max(1, bbox[3] - bbox[1] + 1)
        bbox_area = bw * bh
        compactness = n / bbox_area
        # Compactness is strongly weighted so long lighting/shadow edges do not
        # beat a small but genuinely localized red/dark region.
        redness_bonus = 0.70 + 0.30 * min(1.0, max(0.0, mean_red / 0.25))
        ranking = mean_score * np.sqrt(n) * compactness * redness_bonus

        components.append({
            "pixels": n,
            "mean_score": mean_score,
            "mean_red_excess": mean_red,
            "mean_gray": mean_gray,
            "bbox": bbox,
            "center_x": cx,
            "center_y": cy,
            "compactness": float(compactness),
            "ranking": float(ranking),
        })

    return components


def _estimate_affected_region(pil_image):
    """Estimate the visually abnormal/affected region.

    This is intentionally conservative and explainable. It looks for a compact
    region that differs from surrounding skin in local redness/darkness. It is
    not segmentation of a disease lesion and must not be presented as one.
    """
    img = pil_image.convert("RGB").resize((_MEASURE_SIZE, _MEASURE_SIZE))
    arr = np.asarray(img).astype(np.float32) / 255.0
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    mx = arr.max(axis=2)
    mn = arr.min(axis=2)
    saturation = (mx - mn) / (mx + 1e-6)

    # Broad skin mask. It deliberately allows different skin tones and rejects
    # most white/gray background and very dark clothing.
    skin = (
        (gray > 0.18)
        & (r > g * 0.82)
        & (g > b * 0.82)
        & (saturation > 0.08)
        & (r > 0.12)
    )

    # Ignore a small border because camera/background edges create large false
    # regions in many photos.
    border = max(6, int(_MEASURE_SIZE * 0.06))
    interior = np.zeros_like(skin, dtype=bool)
    interior[border:-border, border:-border] = True
    skin &= interior

    skin_count = int(skin.sum())
    if skin_count < 500:
        return {
            "affected_area_pct": 0.0,
            "affected_area_image_pct": 0.0,
            "affected_pixel_count": 0,
            "skin_pixel_count": skin_count,
            "region_detected": False,
            "region_confidence": 0.0,
            "region_bbox_norm": None,
            "region_center_norm": None,
            "region_mean_red_excess": 0.0,
            "region_mean_anomaly": 0.0,
            "region_compactness": 0.0,
            "method": "insufficient_skin_pixels",
        }

    # Local contrast is more useful than a global redness threshold because
    # lighting may change between days. A red/dark spot remains locally unusual.
    blurred = np.asarray(img.filter(ImageFilter.GaussianBlur(radius=10))).astype(np.float32) / 255.0
    br, bg, bb = blurred[..., 0], blurred[..., 1], blurred[..., 2]
    blurred_gray = 0.299 * br + 0.587 * bg + 0.114 * bb

    local_red = np.clip((r - g) - (br - bg), 0.0, 1.0)
    local_dark = np.clip(blurred_gray - gray, 0.0, 1.0)
    red_excess = np.clip(r - (g + b) / 2.0, 0.0, 1.0)

    # Combine local red/dark contrast with a small absolute redness signal.
    anomaly_score = 0.65 * local_red + 0.35 * local_dark + 0.20 * red_excess

    values = anomaly_score[skin]
    med = float(np.median(values))
    mad = float(np.median(np.abs(values - med)) + 1e-4)
    threshold = max(0.055, med + 3.0 * mad)
    candidate = skin & (anomaly_score > threshold)

    # One light close operation fills tiny holes while retaining compact spots.
    candidate = _binary_erode(_binary_dilate(candidate))

    # Only consider the central 80% when ranking regions. This prevents fingers,
    # clothing and image borders near the top/edges from winning over a skin spot.
    yy, xx = np.indices(candidate.shape)
    roi = (xx >= 0.10 * _MEASURE_SIZE) & (xx <= 0.90 * _MEASURE_SIZE) & (
        (yy >= 0.10 * _MEASURE_SIZE) & (yy <= 0.90 * _MEASURE_SIZE)
    )
    candidate &= roi

    components = _connected_components(candidate, anomaly_score, red_excess, gray)
    if not components:
        return {
            "affected_area_pct": 0.0,
            "affected_area_image_pct": 0.0,
            "affected_pixel_count": 0,
            "skin_pixel_count": skin_count,
            "region_detected": False,
            "region_confidence": 0.0,
            "region_bbox_norm": None,
            "region_center_norm": None,
            "region_mean_red_excess": 0.0,
            "region_mean_anomaly": 0.0,
            "region_compactness": 0.0,
            "method": "no_compact_region",
        }

    # Rank by intensity and size, while penalizing very elongated/non-compact
    # regions. The top component is the visible region tracked across days.
    best = max(components, key=lambda c: c["ranking"])
    pixels = int(best["pixels"])
    area_pct_of_skin = 100.0 * pixels / max(1, skin_count)
    area_pct_of_image = 100.0 * pixels / float(_MEASURE_SIZE * _MEASURE_SIZE)

    # Confidence is deliberately capped: this is heuristic image analysis.
    region_conf = min(
        0.95,
        max(
            0.0,
            0.35
            + min(0.35, best["mean_score"] * 1.6)
            + min(0.20, np.sqrt(pixels / 65536.0) * 2.0)
            + min(0.10, best["compactness"] * 0.10),
        ),
    )

    return {
        "affected_area_pct": round(float(area_pct_of_skin), 4),
        "affected_area_image_pct": round(float(area_pct_of_image), 4),
        "affected_pixel_count": pixels,
        "skin_pixel_count": skin_count,
        "region_detected": True,
        "region_confidence": round(float(region_conf), 3),
        "region_bbox_norm": [round(v / (_MEASURE_SIZE - 1), 4) for v in best["bbox"]],
        "region_center_norm": [round(best["center_x"], 4), round(best["center_y"], 4)],
        "region_mean_red_excess": round(best["mean_red_excess"], 5),
        "region_mean_anomaly": round(best["mean_score"], 5),
        "region_compactness": round(best["compactness"], 4),
        "method": "local_redness_darkness_compact_region",
    }


def _visual_metrics(pil_image):
    """Extract reproducible, non-diagnostic image measurements."""
    img = pil_image.convert("RGB").resize((_MEASURE_SIZE, _MEASURE_SIZE))
    arr = np.asarray(img).astype(np.float32) / 255.0
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]

    gray = 0.299 * r + 0.587 * g + 0.114 * b
    brightness = float(gray.mean())
    contrast = float(gray.std())

    red_excess = np.clip(r - (g + b) / 2.0, 0.0, 1.0)
    warm_mask = (r > g * 1.08) & (r > b * 1.05) & (red_excess > 0.035)
    redness_index = float(red_excess.mean())
    warm_pixel_fraction = float(warm_mask.mean())

    gx = np.abs(gray[:, 1:] - gray[:, :-1]).mean()
    gy = np.abs(gray[1:, :] - gray[:-1, :]).mean()
    texture_index = float((gx + gy) / 2.0)

    lap = (
        -4.0 * gray[1:-1, 1:-1]
        + gray[:-2, 1:-1] + gray[2:, 1:-1]
        + gray[1:-1, :-2] + gray[1:-1, 2:]
    )
    sharpness = float(np.var(lap))

    affected = _estimate_affected_region(pil_image)

    return {
        "brightness": round(brightness, 5),
        "contrast": round(contrast, 5),
        "redness_index": round(redness_index, 5),
        "warm_pixel_fraction": round(warm_pixel_fraction, 5),
        "texture_index": round(texture_index, 5),
        "sharpness": round(sharpness, 5),
        "width": int(pil_image.width),
        "height": int(pil_image.height),
        "aspect_ratio": round(pil_image.width / max(1, pil_image.height), 4),
        "affected_region": affected,
        "analysis_version": ANALYSIS_VERSION,
    }


def _preprocess(pil_image):
    img = pil_image.convert("RGB").resize((IMG_SIZE, IMG_SIZE))
    return np.expand_dims(np.asarray(img).astype("float32"), axis=0)


def classify_image(pil_image):
    """Run CNN inference and attach visual measurements."""
    if pil_image is None:
        raise ValueError("No image supplied.")

    model = _load_model()
    if model is None:
        raise RuntimeError(unavailable_reason())

    classes = _class_names()
    preds = np.asarray(model.predict(_preprocess(pil_image), verbose=0)[0], dtype=float)

    if len(preds) != len(classes):
        raise RuntimeError(
            f"CNN output has {len(preds)} values but {len(classes)} class names were found."
        )

    preds = np.clip(preds, 0, None)
    total = float(preds.sum())
    if total <= 0:
        preds[:] = 1.0 / len(preds)
    else:
        preds /= total

    idx = int(np.argmax(preds))
    confidence = float(preds[idx])
    p = np.clip(preds, 1e-8, 1.0)
    entropy = float(-(p * np.log(p)).sum() / np.log(len(p))) if len(p) > 1 else 0.0
    certainty = max(0.0, min(1.0, 1.0 - entropy))

    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "predicted_class": classes[idx],
        "display_label": LABEL_DISPLAY.get(classes[idx], classes[idx]),
        "confidence": round(confidence, 4),
        "certainty": round(certainty, 4),
        "all_probs": {classes[i]: round(float(preds[i]), 4) for i in range(len(classes))},
        "visual_metrics": _visual_metrics(pil_image),
        "model_note": "Demo CNN trained on synthetic skin images; not a clinical diagnostic model.",
        "analysis_version": ANALYSIS_VERSION,
    }


def _health_score(result):
    """Probability-weighted demo visual score, 0-100, for trend support only."""
    probs = result.get("all_probs", {})
    if probs:
        score = sum(
            _safe_float(p) * CLASS_HEALTH_SCORE.get(cls, 50.0)
            for cls, p in probs.items()
        )
        return max(0.0, min(100.0, score))
    return CLASS_HEALTH_SCORE.get(result.get("predicted_class"), 50.0)


def _comparability(prev_metrics, curr_metrics):
    """Estimate whether photos are similar enough for a visual comparison."""
    if not prev_metrics or not curr_metrics:
        return 0.0

    checks = []
    ar1 = _safe_float(prev_metrics.get("aspect_ratio"), 1)
    ar2 = _safe_float(curr_metrics.get("aspect_ratio"), 1)
    checks.append(max(0.0, 1.0 - abs(ar1 - ar2) / max(ar1, ar2, 1e-6)))

    for key, scale in (("brightness", 0.35), ("contrast", 0.25)):
        a = _safe_float(prev_metrics.get(key))
        b = _safe_float(curr_metrics.get(key))
        checks.append(max(0.0, 1.0 - abs(a - b) / scale))

    # The affected-region center is useful as a soft framing check. Do not make
    # it mandatory because a healing spot can legitimately move slightly due to
    # camera angle/cropping.
    p_region = prev_metrics.get("affected_region") or {}
    c_region = curr_metrics.get("affected_region") or {}
    p_center = p_region.get("region_center_norm")
    c_center = c_region.get("region_center_norm")
    if p_center and c_center and len(p_center) == 2 and len(c_center) == 2:
        dist = float(np.hypot(float(p_center[0]) - float(c_center[0]), float(p_center[1]) - float(c_center[1])))
        checks.append(max(0.0, 1.0 - dist / 0.45))

    return round(float(np.mean(checks)), 3)



def _skin_mask_for_direct(arr):
    """Broad skin mask for direct two-photo change estimation."""
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    gray = 0.299 * r + 0.587 * g + 0.114 * b
    mx = arr.max(axis=2)
    mn = arr.min(axis=2)
    sat = (mx - mn) / (mx + 1e-6)
    return (
        (gray > 0.18)
        & (r > g * 0.82)
        & (g > b * 0.82)
        & (sat > 0.08)
        & (r > 0.12)
    )


def _direct_photo_change(previous_image, current_image):
    """Estimate localized visual change between the two actual stored photos.

    The images are resized to a common frame and compared using normalized
    redness and brightness rather than raw RGB. This reduces sensitivity to
    ordinary exposure differences. It is still only a visual-change heuristic.
    """
    if previous_image is None or current_image is None:
        return None

    size = 256
    p_img = previous_image.convert("RGB").resize((size, size))
    c_img = current_image.convert("RGB").resize((size, size))
    p = np.asarray(p_img).astype(np.float32) / 255.0
    c = np.asarray(c_img).astype(np.float32) / 255.0

    p_gray = 0.299 * p[..., 0] + 0.587 * p[..., 1] + 0.114 * p[..., 2]
    c_gray = 0.299 * c[..., 0] + 0.587 * c[..., 1] + 0.114 * c[..., 2]

    # Normalize broad illumination differences using each image's median and
    # contrast. The values remain bounded and are only used for comparison.
    def normalize_gray(x):
        med = float(np.median(x))
        std = float(np.std(x)) + 1e-5
        return np.clip((x - med) / std, -3.0, 3.0) / 3.0

    pg = normalize_gray(p_gray)
    cg = normalize_gray(c_gray)

    # Red-vs-green chroma is less affected by absolute exposure than raw red.
    p_red = (p[..., 0] - p[..., 1]) / (p[..., 0] + p[..., 1] + 0.08)
    c_red = (c[..., 0] - c[..., 1]) / (c[..., 0] + c[..., 1] + 0.08)

    # Blur the comparison so single-pixel noise does not become a lesion.
    def blur_channel(x):
        # Convert normalized channel to an 8-bit image for Pillow's fast blur.
        q = np.clip((x + 1.0) * 127.5, 0, 255).astype(np.uint8)
        bq = np.asarray(Image.fromarray(q).filter(ImageFilter.GaussianBlur(radius=5)), dtype=np.float32)
        return bq / 127.5 - 1.0

    pg = blur_channel(pg)
    cg = blur_channel(cg)
    p_red = blur_channel(p_red)
    c_red = blur_channel(c_red)

    direct_diff = 0.55 * np.abs(c_red - p_red) + 0.45 * np.abs(cg - pg)
    skin = _skin_mask_for_direct(p) & _skin_mask_for_direct(c)

    # Ignore edges where tiny framing differences dominate.
    border = int(size * 0.08)
    interior = np.zeros_like(skin, dtype=bool)
    interior[border:-border, border:-border] = True
    skin &= interior

    if int(skin.sum()) < 500:
        return {
            "changed_region_fraction": 0.0,
            "changed_region_confidence": 0.0,
            "changed_region_red_delta": 0.0,
            "changed_region_dark_delta": 0.0,
            "method": "insufficient_common_skin",
        }

    values = direct_diff[skin]
    med = float(np.median(values))
    mad = float(np.median(np.abs(values - med)) + 1e-4)
    threshold = max(0.075, med + 3.0 * mad)
    changed = skin & (direct_diff > threshold)

    # Remove tiny specks and connect close pixels.
    changed = _binary_erode(_binary_dilate(changed))
    changed_values = direct_diff[changed]
    changed_fraction = float(changed.sum() / max(1, skin.sum()))

    if changed_values.size:
        red_delta_map = c_red - p_red
        dark_delta_map = pg - cg  # positive = current appears locally darker
        changed_red_delta = float(np.mean(red_delta_map[changed]))
        changed_dark_delta = float(np.mean(dark_delta_map[changed]))
        mean_change = float(np.mean(changed_values))
    else:
        changed_red_delta = 0.0
        changed_dark_delta = 0.0
        mean_change = 0.0

    confidence = min(
        0.90,
        max(0.0, 0.35 + min(0.35, mean_change * 2.0) + min(0.20, changed_fraction * 3.0)),
    )

    return {
        "changed_region_fraction": round(changed_fraction, 5),
        "changed_region_confidence": round(confidence, 3),
        "changed_region_red_delta": round(changed_red_delta, 5),
        "changed_region_dark_delta": round(changed_dark_delta, 5),
        "threshold": round(threshold, 5),
        "method": "normalized_redness_brightness_local_difference",
    }


def compare_analysis_results(previous, current, previous_image=None, current_image=None):
    """Compare consecutive stored image analyses using visible-region evidence.

    The affected-region estimate is the primary day-over-day signal. CNN class
    and probability changes support it but cannot overrule a strong visible
    area change merely because both images have the same broad class.
    """
    if not previous or not current:
        return {
            "trend": "Not enough data",
            "confidence": 0.0,
            "message": "A previous analysis is required for comparison.",
            "reasons": [],
            "comparison_version": ANALYSIS_VERSION,
        }

    old_score = _health_score(previous)
    new_score = _health_score(current)
    score_delta = new_score - old_score

    pm = previous.get("visual_metrics", {})
    cm = current.get("visual_metrics", {})
    comparability = _comparability(pm, cm)

    direct_change = _direct_photo_change(previous_image, current_image)

    prev_region = pm.get("affected_region") or {}
    curr_region = cm.get("affected_region") or {}
    region_comparison_available = bool(prev_region.get("method")) and bool(curr_region.get("method"))
    prev_area = _safe_float(prev_region.get("affected_area_pct")) if region_comparison_available else None
    curr_area = _safe_float(curr_region.get("affected_area_pct")) if region_comparison_available else None
    if region_comparison_available:
        area_delta_pp = curr_area - prev_area
        if prev_area > 0.01:
            area_relative_pct = 100.0 * area_delta_pp / prev_area
        elif curr_area > 0.01:
            area_relative_pct = 100.0
        else:
            area_relative_pct = 0.0
    else:
        area_delta_pp = 0.0
        area_relative_pct = 0.0

    prev_red = _safe_float(prev_region.get("region_mean_red_excess"))
    curr_red = _safe_float(curr_region.get("region_mean_red_excess"))
    region_red_delta = curr_red - prev_red

    redness_delta = _safe_float(cm.get("redness_index")) - _safe_float(pm.get("redness_index"))
    warm_delta = _safe_float(cm.get("warm_pixel_fraction")) - _safe_float(pm.get("warm_pixel_fraction"))
    texture_delta = _safe_float(cm.get("texture_index")) - _safe_float(pm.get("texture_index"))

    class_old = previous.get("predicted_class")
    class_new = current.get("predicted_class")

    reasons = []
    evidence = []

    if direct_change:
        changed_fraction = _safe_float(direct_change.get("changed_region_fraction"))
        if changed_fraction >= 0.01:
            evidence.append(f"direct photo comparison found {changed_fraction*100:.2f}% of common skin area changed visibly")
        else:
            evidence.append("direct photo comparison found little localized visual change")

    if region_comparison_available:
        if curr_region.get("region_detected"):
            evidence.append(f"visible affected-region estimate today: {curr_area:.2f}% of detected skin area")
        else:
            evidence.append("no compact visible affected region was detected today")
        evidence.append(
            f"affected-region change: {area_delta_pp:+.2f} percentage points"
            + (f" ({area_relative_pct:+.0f}% vs previous)" if prev_area > 0.01 else "")
        )
    else:
        evidence.append("affected-region comparison unavailable for the previous legacy record; model/redness evidence used for this transition")

    if abs(region_red_delta) >= 0.015:
        evidence.append(
            f"affected-region redness signal {'increased' if region_red_delta > 0 else 'decreased'}"
        )
    if score_delta >= 8:
        evidence.append("CNN probability-weighted visual score increased")
    elif score_delta <= -8:
        evidence.append("CNN probability-weighted visual score decreased")
    else:
        evidence.append("CNN probability-weighted visual score changed only slightly")

    if class_old != class_new:
        evidence.append(
            f"CNN class changed from {LABEL_DISPLAY.get(class_old, class_old)} "
            f"to {LABEL_DISPLAY.get(class_new, class_new)}"
        )

    # Strong affected-area evidence is intentionally able to override an equal
    # CNN label. This fixes the original false "Stable" behavior.
    area_worsening = region_comparison_available and ((area_delta_pp >= 0.20 and area_relative_pct >= 25) or area_delta_pp >= 0.50)
    area_improving = region_comparison_available and ((area_delta_pp <= -0.15 and area_relative_pct <= -20) or area_delta_pp <= -0.50)
    red_worsening = region_red_delta >= 0.015 or redness_delta >= 0.015
    red_improving = region_red_delta <= -0.015 or redness_delta <= -0.015
    model_worsening = score_delta <= -8
    model_improving = score_delta >= 8

    direct_worsening = bool(direct_change and (
        _safe_float(direct_change.get("changed_region_fraction")) >= 0.012
        and _safe_float(direct_change.get("changed_region_red_delta")) >= 0.015
    ))
    direct_improving = bool(direct_change and (
        _safe_float(direct_change.get("changed_region_fraction")) >= 0.012
        and _safe_float(direct_change.get("changed_region_red_delta")) <= -0.015
    ))

    if comparability < 0.55:
        trend = "Uncertain"
        message = (
            "The two photos differ substantially in framing/lighting, so the visible-area "
            "change is not reliable enough to classify as improvement or worsening."
        )
    elif area_worsening and (red_worsening or model_worsening or area_relative_pct >= 60):
        trend = "Worsening"
        message = (
            f"The visible affected-region estimate increased from {prev_area:.2f}% to "
            f"{curr_area:.2f}% of detected skin area. The change is large enough that "
            "the broad CNN class alone should not be treated as evidence of stability."
        )
    elif area_improving and (red_improving or model_improving or abs(area_relative_pct) >= 50):
        trend = "Improving"
        message = (
            f"The visible affected-region estimate decreased from {prev_area:.2f}% to "
            f"{curr_area:.2f}% of detected skin area, with supporting visual/model evidence."
        )
    elif area_worsening:
        trend = "Worsening"
        message = (
            f"The visible affected-region estimate increased from {prev_area:.2f}% to "
            f"{curr_area:.2f}%. Because the increase is visible but supporting signals are "
            "mixed, this is flagged as a cautious worsening trend."
        )
    elif area_improving:
        trend = "Improving"
        message = (
            f"The visible affected-region estimate decreased from {prev_area:.2f}% to "
            f"{curr_area:.2f}%. This is a visual monitoring result, not proof of healing."
        )
    elif direct_worsening and (area_worsening or not region_comparison_available):
        trend = "Worsening"
        message = "The stored photos show a localized visible change toward a redder/darker affected region, so the trend is flagged for review even if the CNN class stayed the same."
    elif direct_improving and (area_improving or not region_comparison_available):
        trend = "Improving"
        message = "The stored photos show a localized visible change toward a less red/dark region. This is a visual monitoring result, not proof of healing."
    elif model_worsening and red_worsening:
        trend = "Worsening"
        message = "The CNN probability distribution and redness measurements both moved in an unfavorable direction."
    elif model_improving and red_improving:
        trend = "Improving"
        message = "The CNN probability distribution and redness measurements both moved in a favorable direction."
    else:
        trend = "Stable"
        message = "No consistent enough visible-region or supporting visual change was detected."

    if comparability >= 0.75:
        reasons.append("Photo comparability: good")
    elif comparability >= 0.55:
        reasons.append("Photo comparability: moderate")
    else:
        reasons.append("Photo comparability: low")

    evidence.append(
        f"CNN class: {LABEL_DISPLAY.get(class_old, class_old)} -> {LABEL_DISPLAY.get(class_new, class_new)}"
    )

    confidence = min(
        0.95,
        max(
            0.0,
            comparability
            * (0.55 + 0.25 * current.get("certainty", 0.0))
            * (0.70 + 0.30 * max(_safe_float(curr_region.get("region_confidence")), _safe_float(prev_region.get("region_confidence")))),
        ),
    )

    return {
        "comparison_version": ANALYSIS_VERSION,
        "trend": trend,
        "confidence": round(float(confidence), 3),
        "message": message,
        "reasons": reasons + evidence,
        "previous_class": class_old,
        "current_class": class_new,
        "previous_score": round(old_score, 2),
        "current_score": round(new_score, 2),
        "score_delta": round(score_delta, 2),
        "region_comparison_available": region_comparison_available,
        "previous_affected_area_pct": round(prev_area, 4) if prev_area is not None else None,
        "current_affected_area_pct": round(curr_area, 4) if curr_area is not None else None,
        "affected_area_delta_pp": round(area_delta_pp, 4),
        "affected_area_relative_change_pct": round(area_relative_pct, 2),
        "previous_region_confidence": round(_safe_float(prev_region.get("region_confidence")), 3),
        "current_region_confidence": round(_safe_float(curr_region.get("region_confidence")), 3),
        "region_red_delta": round(region_red_delta, 5),
        "redness_delta": round(redness_delta, 5),
        "warm_pixel_fraction_delta": round(warm_delta, 5),
        "texture_delta": round(texture_delta, 5),
        "direct_change": direct_change,
        "comparability": comparability,
    }


def compare_conditions(prev, curr):
    """Backward-compatible class-only comparison."""
    order = {"inflamed_rash": 0, "healing_rash": 1, "healthy_skin": 2}
    if prev not in order or curr not in order:
        return "Stable"
    delta = order[curr] - order[prev]
    return "Improving" if delta > 0 else "Worsening" if delta < 0 else "Stable"
