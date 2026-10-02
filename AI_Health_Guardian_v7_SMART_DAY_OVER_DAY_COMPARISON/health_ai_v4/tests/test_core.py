import os, sys
ROOT = os.path.dirname(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(ROOT, 'modules'))

from symptom_analyzer import predict_from_text
from health_risk import assess
from recovery_tracker import compute_trend


def test_symptom_prediction_shape():
    result, matched = predict_from_text('fever headache body pain')
    assert matched
    assert result['top_diseases']
    assert result['risk_level'] in {'Low', 'Medium', 'High'}


def test_health_risk():
    result = assess(20, 160, 60, 120, 'Moderate', False, False)
    assert result['bmi'] is not None
    assert result['level'] in {'Low', 'Medium', 'High'}


def test_trend():
    entries = [
        {'health_score': 40}, {'health_score': 45},
        {'health_score': 70}, {'health_score': 80}
    ]
    assert compute_trend(entries) == 'Improving'
