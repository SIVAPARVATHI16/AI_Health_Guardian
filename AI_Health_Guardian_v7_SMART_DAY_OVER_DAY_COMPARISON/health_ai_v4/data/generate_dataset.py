"""
generate_dataset.py — v4 (severity-aware symptom risk)
Key change: same symptom type (e.g. cough) maps to different risk levels
depending on severity qualifiers entered by the user.
Low cough  → Common Cold (Low)
Medium cough → Bronchitis (Medium)
High cough → Pneumonia/TB (High)
Same logic for fever, breathlessness, headache, skin rash, etc.
"""
import json, random, csv
random.seed(42)

DISEASE_KB = {
    # ════════════════ COUGH SPECTRUM ═══════════════════════════════════════
    "Common Cold": {
        "risk": "Low",
        "symptoms": ["mild_cough","runny_nose","sneezing","sore_throat","congestion","mild_fever"],
        "diet": ["Warm fluids (soup, herbal tea)","Vitamin C rich fruits","Avoid cold drinks"],
        "precaution": "Rest, stay hydrated, use steam inhalation.",
        "severity_keywords": ["mild cough","dry cough","little cough","occasional cough","cold"],
    },
    "Acute Bronchitis": {
        "risk": "Medium",
        "symptoms": ["persistent_cough","cough_with_phlegm","mild_fever","chest_tightness","fatigue","sore_throat"],
        "diet": ["Warm fluids","Honey ginger tea","Avoid cold food"],
        "precaution": "Rest, stay hydrated, steam inhalation. See doctor if cough lasts more than 3 weeks.",
        "severity_keywords": ["persistent cough","wet cough","productive cough","cough with mucus","frequent cough"],
    },
    "Pneumonia": {
        "risk": "High",
        "symptoms": ["high_fever","cough_with_phlegm","breathlessness","chest_pain","fatigue","chills","rapid_breathing"],
        "diet": ["High-protein soft diet","Warm fluids","Avoid cold food"],
        "precaution": "Requires prompt medical evaluation and antibiotics. Do not delay hospital visit.",
        "severity_keywords": ["cough with blood","difficulty breathing","chest pain with cough","severe cough","high fever cough"],
    },
    "Tuberculosis (TB)": {
        "risk": "High",
        "symptoms": ["persistent_cough","coughing_blood","night_sweats","weight_loss","fatigue","mild_fever","chest_pain"],
        "diet": ["High protein diet","Nutritious balanced meals","Avoid alcohol"],
        "precaution": "EMERGENCY: Coughing blood + weight loss + night sweats = TB signs. Go to hospital IMMEDIATELY for testing.",
        "severity_keywords": ["coughing blood","blood in cough","cough for weeks","cough for months","night sweats cough"],
    },
    "Asthma": {
        "risk": "High",
        "symptoms": ["breathlessness","wheezing","chest_tightness","persistent_cough","fatigue","cough_at_night"],
        "diet": ["Anti-inflammatory foods","Avoid known triggers","Stay hydrated"],
        "precaution": "Use prescribed inhaler. Seek urgent care if breathing worsens. HIGH EMERGENCY if lips turn blue.",
        "severity_keywords": ["wheezing","breathless cough","can't breathe","cough at night","tight chest"],
    },

    # ════════════════ FEVER SPECTRUM ════════════════════════════════════════
    "Seasonal Flu": {
        "risk": "Medium",
        "symptoms": ["high_fever","body_pain","fatigue","headache","chills","mild_cough","sore_throat"],
        "diet": ["Light khichdi/porridge","Warm fluids","Avoid oily/fried food"],
        "precaution": "Rest, monitor temperature, isolate from others.",
        "severity_keywords": ["high fever","body pain fever","fever with chills","flu"],
    },
    "Dengue": {
        "risk": "High",
        "symptoms": ["high_fever","severe_body_pain","joint_pain","rash","nausea","fatigue","headache_behind_eyes"],
        "diet": ["Papaya leaf extract (as advised)","ORS / fluids","Avoid NSAIDs like ibuprofen"],
        "precaution": "EMERGENCY: Dengue can drop platelets rapidly. Go to hospital immediately for blood test.",
        "severity_keywords": ["fever with rash","high fever joint pain","dengue","fever behind eyes"],
    },
    "Typhoid": {
        "risk": "High",
        "symptoms": ["prolonged_fever","abdominal_pain","weakness","loss_of_appetite","headache","constipation_or_diarrhea"],
        "diet": ["Soft bland diet","Plenty of fluids","Avoid spicy/oily food"],
        "precaution": "Complete prescribed antibiotic course. Medical supervision required.",
        "severity_keywords": ["fever for many days","prolonged fever","fever with stomach pain","typhoid"],
    },
    "Malaria": {
        "risk": "High",
        "symptoms": ["high_fever","chills","sweating","headache","nausea","muscle_pain","fatigue"],
        "diet": ["ORS and fluids","Light nutritious food","Avoid oily heavy food"],
        "precaution": "EMERGENCY: Fever with chills and sweating in cycles = possible malaria. Get blood test immediately.",
        "severity_keywords": ["fever with chills","cyclic fever","malaria","shivering fever"],
    },

    # ════════════════ BREATHLESSNESS SPECTRUM ═══════════════════════════════
    "Anxiety": {
        "risk": "Low",
        "symptoms": ["restlessness","rapid_heartbeat","sweating","difficulty_concentrating","fatigue","sleep_trouble","breathlessness"],
        "diet": ["Reduce caffeine","Magnesium-rich foods","Regular hydration"],
        "precaution": "Practice relaxation techniques. Consult a professional if persistent.",
        "severity_keywords": ["anxiety breathlessness","nervous breathing","stress breathless","panic"],
    },
    "Anemia": {
        "risk": "Medium",
        "symptoms": ["fatigue","weakness","pale_skin","dizziness","breathlessness","rapid_heartbeat"],
        "diet": ["Iron-rich foods (leafy greens, legumes)","Vitamin C to aid absorption","Avoid tea/coffee with meals"],
        "precaution": "Get hemoglobin levels tested. Consult physician.",
        "severity_keywords": ["breathless on walking","breathless without activity","tired breathless","pale"],
    },
    "Heart Failure Risk": {
        "risk": "High",
        "symptoms": ["breathlessness","chest_pain","rapid_heartbeat","fatigue","swelling","dizziness"],
        "diet": ["Low sodium diet","Fluid restriction if advised","Light nutritious meals"],
        "precaution": "EMERGENCY: Breathlessness with chest pain + swollen legs = possible heart issue. Go to hospital NOW.",
        "severity_keywords": ["breathless lying down","breathless at rest","severe breathlessness","breathless with chest pain"],
    },

    # ════════════════ HEADACHE SPECTRUM ════════════════════════════════════
    "Migraine": {
        "risk": "Low",
        "symptoms": ["headache","nausea","sensitivity_to_light","blurred_vision","dizziness"],
        "diet": ["Stay hydrated","Avoid caffeine excess","Regular small meals"],
        "precaution": "Rest in a dark quiet room. Avoid screen time.",
        "severity_keywords": ["migraine","one side headache","throbbing headache","headache with light sensitivity"],
    },
    "Hypertension": {
        "risk": "Medium",
        "symptoms": ["headache","dizziness","blurred_vision","chest_discomfort","fatigue"],
        "diet": ["Low sodium diet","Reduce processed food","Fresh fruits and vegetables"],
        "precaution": "Monitor blood pressure regularly. Consult physician if persistent.",
        "severity_keywords": ["severe headache","back of head pain","headache with dizziness","BP headache"],
    },
    "Meningitis Risk": {
        "risk": "High",
        "symptoms": ["severe_headache","high_fever","neck_stiffness","sensitivity_to_light","vomiting","confusion"],
        "diet": ["Medical treatment required","IV fluids as prescribed"],
        "precaution": "EMERGENCY: Severe headache + stiff neck + high fever = possible meningitis. Go to hospital IMMEDIATELY.",
        "severity_keywords": ["severe headache fever neck stiff","worst headache","sudden severe headache","stiff neck headache"],
    },

    # ════════════════ SKIN SPECTRUM ════════════════════════════════════════
    "Allergic Rhinitis": {
        "risk": "Low",
        "symptoms": ["sneezing","itchy_eyes","runny_nose","congestion","skin_rash"],
        "diet": ["Anti-inflammatory foods","Avoid known allergens","Stay hydrated"],
        "precaution": "Identify and avoid triggers. Consider antihistamines.",
        "severity_keywords": ["mild rash","itchy skin","allergy rash","small rash"],
    },
    "Fungal Skin Infection": {
        "risk": "Low",
        "symptoms": ["skin_rash","itching","skin_peeling","redness","discoloration"],
        "diet": ["Reduce sugar intake","Stay hydrated","Probiotic foods"],
        "precaution": "Keep area dry. Avoid sharing towels/clothing.",
        "severity_keywords": ["ring shaped rash","itchy rash","peeling skin","fungal"],
    },
    "Chickenpox": {
        "risk": "Medium",
        "symptoms": ["rash","itching","mild_fever","fatigue","loss_of_appetite","blisters"],
        "diet": ["Soft cool foods","Plenty of fluids","Avoid spicy food"],
        "precaution": "Isolate to prevent spread. Avoid scratching blisters.",
        "severity_keywords": ["blisters rash","chickenpox","rash with fever","fluid filled rash"],
    },
    "Skin Infection (Bacterial)": {
        "risk": "Medium",
        "symptoms": ["skin_rash","swelling","redness","warmth_at_site","mild_fever","pain"],
        "diet": ["Protein-rich food for healing","Stay hydrated","Vitamin C foods"],
        "precaution": "Keep area clean and dry. Consult doctor if spreading rapidly.",
        "severity_keywords": ["infected wound","spreading rash","pus rash","hot swollen skin"],
    },
    "Severe Allergic Reaction": {
        "risk": "High",
        "symptoms": ["rash","breathlessness","rapid_heartbeat","swelling","vomiting","dizziness"],
        "diet": ["Medical emergency — no food until treated"],
        "precaution": "EMERGENCY: Rash + difficulty breathing + swelling = anaphylaxis. Call 108 immediately.",
        "severity_keywords": ["rash difficulty breathing","swollen face rash","allergic shock","hives breathless"],
    },

    # ════════════════ STOMACH SPECTRUM ════════════════════════════════════
    "Gastritis or Indigestion": {
        "risk": "Low",
        "symptoms": ["stomach_pain","bloating","nausea","heartburn","sour_taste"],
        "diet": ["Smaller frequent meals","Avoid spicy/oily food","Avoid lying down after eating"],
        "precaution": "Avoid trigger foods. Elevate head while sleeping.",
        "severity_keywords": ["mild stomach pain","gas","acidity","bloating","heartburn"],
    },
    "Food Poisoning": {
        "risk": "Medium",
        "symptoms": ["vomiting","diarrhea","abdominal_cramps","nausea","mild_fever","weakness"],
        "diet": ["ORS solution","BRAT diet (banana, rice, applesauce, toast)","Avoid dairy temporarily"],
        "precaution": "Stay hydrated. Seek care if symptoms persist beyond 2 days.",
        "severity_keywords": ["vomiting after eating","food poisoning","repeated vomiting","diarrhea vomiting"],
    },
    "Appendicitis Risk": {
        "risk": "High",
        "symptoms": ["severe_abdominal_pain","nausea","vomiting","mild_fever","loss_of_appetite"],
        "diet": ["Medical emergency — no food"],
        "precaution": "EMERGENCY: Severe pain in lower right abdomen + fever = possible appendicitis. Go to hospital NOW.",
        "severity_keywords": ["lower right stomach pain","severe stomach pain fever","appendix pain"],
    },

    # ════════════════ CHEST PAIN SPECTRUM ══════════════════════════════════
    "Acid Reflux": {
        "risk": "Low",
        "symptoms": ["heartburn","regurgitation","chest_discomfort","bloating","sour_taste"],
        "diet": ["Avoid spicy/oily food","Smaller frequent meals","Avoid lying down after eating"],
        "precaution": "Avoid trigger foods. Elevate head while sleeping.",
        "severity_keywords": ["chest burn","acid reflux","heartburn","burning chest after food"],
    },
    "Anxiety Chest Pain": {
        "risk": "Low",
        "symptoms": ["chest_discomfort","rapid_heartbeat","restlessness","sweating","breathlessness"],
        "diet": ["Reduce caffeine","Magnesium-rich foods","Stay hydrated"],
        "precaution": "Practice deep breathing. Consult doctor to rule out cardiac causes.",
        "severity_keywords": ["chest pain anxiety","stress chest pain","tight feeling chest","nervous chest"],
    },
    "Heart Attack Risk": {
        "risk": "High",
        "symptoms": ["chest_pain","breathlessness","rapid_heartbeat","sweating","left_arm_pain","jaw_pain","dizziness"],
        "diet": ["Medical emergency — no food"],
        "precaution": "EMERGENCY: Chest pain + left arm pain + sweating = possible heart attack. Call 108 IMMEDIATELY. Do NOT wait.",
        "severity_keywords": ["severe chest pain","crushing chest pain","heart attack","chest pain arm pain","chest pain sweating"],
    },

    # ════════════════ OTHER CONDITIONS ════════════════════════════════════
    "Urinary Tract Infection": {
        "risk": "Medium",
        "symptoms": ["burning_urination","frequent_urination","lower_abdominal_pain","mild_fever","cloudy_urine"],
        "diet": ["Plenty of water","Cranberry juice (unsweetened)","Avoid caffeine"],
        "precaution": "Complete antibiotic course if prescribed. Stay hydrated.",
        "severity_keywords": ["burning urination","frequent urination","UTI","pain while urinating"],
    },
    "Diabetes Risk": {
        "risk": "Medium",
        "symptoms": ["excessive_thirst","frequent_urination","fatigue","blurred_vision","weight_loss","slow_healing"],
        "diet": ["Low glycemic-index foods","Avoid sugary drinks","Balanced fiber-rich meals"],
        "precaution": "Get blood sugar tested. Consult physician for confirmation.",
        "severity_keywords": ["excessive thirst","frequent urination fatigue","slow healing","diabetes symptoms"],
    },
    "Conjunctivitis": {
        "risk": "Low",
        "symptoms": ["itchy_eyes","redness_eyes","watery_eyes","swelling_eyelids","sensitivity_to_light"],
        "diet": ["No specific diet; maintain hygiene","Stay hydrated","Vitamin A rich foods"],
        "precaution": "Avoid touching eyes. Wash hands frequently. Avoid sharing towels.",
        "severity_keywords": ["red eyes","eye discharge","conjunctivitis","pink eye","watery eyes"],
    },
    "Sinusitis": {
        "risk": "Low",
        "symptoms": ["facial_pain","congestion","headache","thick_nasal_discharge","mild_fever"],
        "diet": ["Warm fluids","Steam inhalation","Avoid dairy if it worsens congestion"],
        "precaution": "Steam inhalation. Consult doctor if symptoms exceed 10 days.",
        "severity_keywords": ["sinus pain","face pain","blocked nose","thick nasal discharge"],
    },
    "Gastroenteritis": {
        "risk": "Medium",
        "symptoms": ["diarrhea","vomiting","stomach_pain","mild_fever","dehydration_signs"],
        "diet": ["Fluids + electrolytes","Light food after vomiting subsides","Avoid dairy temporarily"],
        "precaution": "Hydration is key. Watch for signs of dehydration.",
        "severity_keywords": ["stomach virus","gastro","loose stools fever","diarrhea vomiting fever"],
    },
}

ALL_SYMPTOMS = sorted({s for d in DISEASE_KB.values() for s in d["symptoms"]})

def sample_case(disease, kb):
    ts = kb["symptoms"]
    row = {s:0 for s in ALL_SYMPTOMS}
    n = max(2, int(len(ts)*random.uniform(0.6,1.0)))
    for s in random.sample(ts, min(n, len(ts))): row[s]=1
    if random.random()<0.2:
        pool=[s for s in ALL_SYMPTOMS if s not in ts]
        for s in random.sample(pool, random.randint(1,2)): row[s]=1
    row["prognosis"]=disease; row["risk_level"]=kb["risk"]
    return row

rows=[]
for d,kb in DISEASE_KB.items():
    for _ in range(100): rows.append(sample_case(d,kb))
random.shuffle(rows)

with open("symptom_disease_dataset.csv","w",newline="") as f:
    writer=csv.DictWriter(f,fieldnames=ALL_SYMPTOMS+["prognosis","risk_level"])
    writer.writeheader(); writer.writerows(rows)

with open("symptom_columns.json","w") as f: json.dump(ALL_SYMPTOMS,f,indent=2)

meta={d:{"risk":kb["risk"],"diet":kb["diet"],"precaution":kb["precaution"],
          "symptoms":kb["symptoms"],"severity_keywords":kb.get("severity_keywords",[])}
      for d,kb in DISEASE_KB.items()}
with open("disease_meta.json","w") as f: json.dump(meta,f,indent=2)

print(f"Generated {len(rows)} records, {len(DISEASE_KB)} diseases, {len(ALL_SYMPTOMS)} symptoms.")
print("Diseases by risk:")
from collections import Counter
counts = Counter(kb["risk"] for kb in DISEASE_KB.values())
for k,v in sorted(counts.items()): print(f"  {k}: {v}")
