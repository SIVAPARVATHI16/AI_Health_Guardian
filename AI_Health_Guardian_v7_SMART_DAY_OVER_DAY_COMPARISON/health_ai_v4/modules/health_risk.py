"""Simple transparent wellness-risk scoring layer.
This is a screening/education feature, not a clinical risk calculator.
"""

def bmi(height_cm, weight_kg):
    if not height_cm or not weight_kg or height_cm <= 0 or weight_kg <= 0:
        return None
    return round(weight_kg / ((height_cm / 100) ** 2), 1)


def bmi_label(value):
    if value is None:
        return "Unknown"
    if value < 18.5:
        return "Underweight"
    if value < 25:
        return "Healthy range"
    if value < 30:
        return "Overweight"
    return "Obesity range"


def assess(age, height_cm, weight_kg, systolic, activity, smoking, family_history):
    score = 0
    factors = []
    b = bmi(height_cm, weight_kg)
    if b is not None and (b < 18.5 or b >= 25):
        score += 1; factors.append(f"BMI {b} ({bmi_label(b)})")
    if systolic >= 140:
        score += 3; factors.append("Systolic blood pressure ≥ 140 mmHg")
    elif systolic >= 130:
        score += 2; factors.append("Systolic blood pressure 130–139 mmHg")
    if age >= 60:
        score += 2; factors.append("Age 60+")
    elif age >= 45:
        score += 1; factors.append("Age 45+")
    if activity == "Low":
        score += 2; factors.append("Low physical activity")
    elif activity == "Moderate":
        score += 1; factors.append("Moderate physical activity")
    if smoking:
        score += 3; factors.append("Smoking")
    if family_history:
        score += 2; factors.append("Relevant family history")
    level = "Low" if score <= 2 else "Medium" if score <= 5 else "High"
    return {"score": score, "level": level, "bmi": b, "bmi_label": bmi_label(b), "factors": factors}
