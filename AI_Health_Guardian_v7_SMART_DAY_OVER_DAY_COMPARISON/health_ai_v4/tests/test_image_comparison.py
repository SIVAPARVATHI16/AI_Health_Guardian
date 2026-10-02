import os
import sys

from PIL import Image, ImageDraw

ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(ROOT, "modules"))

from image_analyzer import _estimate_affected_region, _visual_metrics, compare_analysis_results


def _analysis_from_metrics(metrics):
    return {
        "predicted_class": "healthy_skin",
        "display_label": "Healthy / Normal",
        "confidence": 0.80,
        "certainty": 0.60,
        "all_probs": {
            "healthy_skin": 0.80,
            "healing_rash": 0.15,
            "inflamed_rash": 0.05,
        },
        "visual_metrics": metrics,
    }


def test_visible_region_grows_when_red_area_grows():
    base = Image.new("RGB", (400, 400), (220, 180, 160))

    small = base.copy()
    ImageDraw.Draw(small).ellipse((185, 185, 215, 215), fill=(150, 45, 35))

    large = base.copy()
    ImageDraw.Draw(large).ellipse((155, 155, 245, 245), fill=(150, 45, 35))

    small_region = _estimate_affected_region(small)
    large_region = _estimate_affected_region(large)

    assert small_region["region_detected"]
    assert large_region["region_detected"]
    assert large_region["affected_area_pct"] > small_region["affected_area_pct"]


def test_same_cnn_label_does_not_force_stable_when_visible_area_increases():
    base = Image.new("RGB", (400, 400), (220, 180, 160))

    small = base.copy()
    ImageDraw.Draw(small).ellipse((185, 185, 215, 215), fill=(150, 45, 35))

    large = base.copy()
    ImageDraw.Draw(large).ellipse((155, 155, 245, 245), fill=(150, 45, 35))

    previous = _analysis_from_metrics(_visual_metrics(small))
    current = _analysis_from_metrics(_visual_metrics(large))

    comparison = compare_analysis_results(
        previous,
        current,
        previous_image=small,
        current_image=large,
    )

    assert comparison["trend"] == "Worsening"
    assert comparison["region_comparison_available"] is True
    assert comparison["current_affected_area_pct"] > comparison["previous_affected_area_pct"]
    assert comparison["affected_area_delta_pp"] > 0


def test_direct_photo_comparison_can_flag_change_even_for_legacy_previous_record():
    base = Image.new("RGB", (400, 400), (220, 180, 160))

    previous_image = base.copy()
    ImageDraw.Draw(previous_image).ellipse((185, 185, 215, 215), fill=(150, 45, 35))

    current_image = base.copy()
    ImageDraw.Draw(current_image).ellipse((155, 155, 245, 245), fill=(150, 45, 35))

    previous = _analysis_from_metrics(_visual_metrics(previous_image))
    # Simulate a v6 stored result that predates the visible-region metric.
    previous["visual_metrics"].pop("affected_region", None)
    current = _analysis_from_metrics(_visual_metrics(current_image))

    comparison = compare_analysis_results(
        previous,
        current,
        previous_image=previous_image,
        current_image=current_image,
    )

    assert comparison["region_comparison_available"] is False
    assert comparison["direct_change"] is not None
    assert comparison["direct_change"]["changed_region_fraction"] > 0
