import json,os,random
DATA_DIR=os.path.join(os.path.dirname(__file__),"..","data")
with open(os.path.join(DATA_DIR,"disease_meta.json")) as f: DISEASE_META=json.load(f)
TREND_OPENERS={"Improving":["Good news — your tracked data shows a positive recovery trend.","Your recovery indicators are trending upward.","Things are moving in the right direction."],"Stable":["Your condition appears stable — no major change either way.","Recovery indicators are holding steady right now."],"Worsening":["Your recent entries show signs of worsening — please take this seriously.","The trend suggests your symptoms may be intensifying."]}
ADHERENCE_NOTES={"high":["Your medicine adherence has been excellent — keep it up.","Great consistency with your medication schedule."],"medium":["Your medicine adherence has been inconsistent — try to take doses on schedule.","A few missed doses were recorded; consistent timing will help."],"low":["Several doses were missed — this can slow recovery.","Low medicine adherence detected. Please follow the prescribed schedule."]}
RISK_SAFETY={"Low":"This appears low-risk based on your inputs, but continue monitoring for any changes.","Medium":"This is a moderate-risk pattern. Keep monitoring closely and rest as needed.","High":"This pattern is flagged higher-risk. Please consult a doctor promptly."}
def _ab(p): return "high" if p>=85 else "medium" if p>=50 else "low"
def generate_explanation(disease,risk_level=None,trend="Stable",adherence_pct=100,days_tracked=1,seed=None):
    if seed is not None: random.seed(seed)
    meta=DISEASE_META.get(disease,{})
    risk_level=risk_level or meta.get("risk","Medium")
    opener=random.choice(TREND_OPENERS.get(trend,TREND_OPENERS["Stable"]))
    anote=random.choice(ADHERENCE_NOTES[_ab(adherence_pct)])
    diet=meta.get("diet",["Stay hydrated","Eat light balanced meals","Avoid processed food"])
    prec=meta.get("precaution","Monitor symptoms and rest as needed.")
    safety=RISK_SAFETY.get(risk_level,RISK_SAFETY["Medium"])
    return {"summary":f"{opener} Based on {days_tracked} day(s) of tracking for '{disease}'. {anote} {prec}","diet_plan":{"morning":diet[0] if diet else "Light food","afternoon":diet[1] if len(diet)>1 else "Balanced meal","night":diet[2] if len(diet)>2 else "Light dinner"},"precaution":prec,"safety_note":safety}
def followup_recommendation(trend,risk_level,days_tracked,adherence_pct):
    if risk_level=="High" and trend in ("Worsening","Stable"): return "Visit doctor recommended","High-risk pattern with non-improving trend. We recommend an in-person medical evaluation."
    if trend=="Worsening": return "Monitor for 2 more days","Symptoms appear to be worsening. If no improvement in 2 days, please see a doctor."
    if trend=="Improving" and days_tracked>=3: return "No checkup needed","Your recovery trend looks positive. Continue current care and complete medication."
    if adherence_pct<50: return "Monitor for 2 more days","Low medicine adherence may be affecting recovery. Improve consistency and reassess."
    return "Monitor for 2 more days","Continue tracking your symptoms daily for a clearer recovery picture."
