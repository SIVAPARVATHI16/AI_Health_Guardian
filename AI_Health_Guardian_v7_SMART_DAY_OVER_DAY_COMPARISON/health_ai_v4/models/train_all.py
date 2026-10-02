"""Training entry point for the disease/risk Random Forest models and the
optional skin-image CNN.

TensorFlow is imported lazily and guarded: this script can retrain the
symptom/risk models (the part that changes whenever the symptom vocabulary
or dataset changes) even on machines where TensorFlow is not installed.
Pass --skip-cnn to skip the image model explicitly.
"""
import argparse, json, os, sys, joblib, numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.preprocessing import LabelEncoder

parser = argparse.ArgumentParser()
parser.add_argument("--skip-cnn", action="store_true", help="Skip training the skin-image CNN.")
args = parser.parse_args()

os.makedirs("../models",exist_ok=True)

# ML: symptom->disease + risk
df=pd.read_csv("../data/symptom_disease_dataset.csv")
with open("../data/symptom_columns.json") as f: SC=json.load(f)
# Fit on plain numpy arrays (not a DataFrame) so the fitted model carries no
# feature-name metadata. Inference (symptom_analyzer.symptoms_to_vector)
# also builds a plain numpy array for speed, and passing a DataFrame at fit
# time but a numpy array at predict time otherwise triggers a
# "X does not have valid feature names" warning on every single prediction.
X=df[SC].to_numpy(dtype=np.int8); yd=df["prognosis"]; yr=df["risk_level"]
Xtr,Xte,ytr,yte=train_test_split(X,yd,test_size=0.2,random_state=42,stratify=yd)
clf=RandomForestClassifier(n_estimators=120,max_depth=14,random_state=42,class_weight="balanced")
clf.fit(Xtr,ytr)
print(f"Disease acc: {accuracy_score(yte,clf.predict(Xte)):.3f}")
le=LabelEncoder(); yr_enc=le.fit_transform(yr)
Xrtr,Xrte,yrtr,yrte=train_test_split(X,yr_enc,test_size=0.2,random_state=42,stratify=yr_enc)
rcl=RandomForestClassifier(n_estimators=100,max_depth=10,random_state=42,class_weight="balanced")
rcl.fit(Xrtr,yrtr)
racc=accuracy_score(yrte,rcl.predict(Xrte))
print(f"Risk acc: {racc:.3f}")
joblib.dump(clf,"../models/disease_model.joblib")
joblib.dump(rcl,"../models/risk_model.joblib")
joblib.dump(le,"../models/risk_label_encoder.joblib")
with open("../models/symptom_columns.json","w") as f: json.dump(SC,f,indent=2)

# Base literal English synonym phrases. This is intentionally kept small:
# the runtime symptom_normalizer module (modules/symptom_normalizer.py)
# handles the scalable, pattern-based matching (body-part + pain-word
# composition, synonym normalization, negation) on top of this list, so
# this dictionary does not need to enumerate every possible phrasing.
SYNS={"fever":["mild_fever","high_fever","prolonged_fever"],"high fever":["high_fever"],"temperature":["mild_fever","high_fever"],"headache":["headache","headache_behind_eyes"],"body pain":["body_pain","severe_body_pain"],"joint pain":["joint_pain"],"tired":["fatigue","weakness"],"fatigue":["fatigue"],"weak":["weakness"],"weakness":["weakness"],"cough":["mild_cough","cough","cough_with_phlegm"],"sore throat":["sore_throat"],"runny nose":["runny_nose"],"blocked nose":["congestion"],"congestion":["congestion"],"sneezing":["sneezing"],"rash":["rash","skin_rash"],"skin rash":["skin_rash","rash"],"itching":["itching","itchy_eyes"],"vomiting":["vomiting"],"nausea":["nausea"],"diarrhea":["diarrhea"],"loose motion":["diarrhea"],"stomach pain":["stomach_pain","abdominal_pain","abdominal_cramps"],"abdominal pain":["abdominal_pain"],"chest pain":["chest_pain","chest_discomfort","chest_tightness"],"breathlessness":["breathlessness"],"shortness of breath":["breathlessness"],"wheezing":["wheezing"],"dizzy":["dizziness"],"dizziness":["dizziness"],"blurred vision":["blurred_vision"],"burning urination":["burning_urination"],"frequent urination":["frequent_urination"],"thirst":["excessive_thirst"],"weight loss":["weight_loss"],"pale skin":["pale_skin"],"rapid heartbeat":["rapid_heartbeat"],"sweating":["sweating"],"sleep trouble":["sleep_trouble"],"anxious":["restlessness","rapid_heartbeat"],"restless":["restlessness"],"eye redness":["redness_eyes"],"watery eyes":["watery_eyes"],"facial pain":["facial_pain"],"heartburn":["heartburn"],"bloating":["bloating"],"chills":["chills"],"loss of appetite":["loss_of_appetite"],"constipation":["constipation_or_diarrhea"],
"knee pain":["knee_pain"],"hand pain":["hand_pain"],"wrist pain":["wrist_pain"],"shoulder pain":["shoulder_pain"],"elbow pain":["elbow_pain"],"neck pain":["neck_pain"],"back pain":["back_pain"],"hip pain":["hip_pain"],"ankle pain":["ankle_pain"],"foot pain":["foot_pain"],"leg pain":["leg_pain"],"arm pain":["arm_pain"],"finger pain":["finger_pain"],"pelvic pain":["pelvic_pain"],"muscle pain":["muscle_pain"],"stiffness":["stiffness"],"numbness":["numbness"],"swelling":["swelling"],
# Additional literal phrases from medical_conditions_and_symptoms.csv that
# are not simple body-part + pain-word compositions.
"toe pain":["foot_pain"],"heel pain":["heel_pain"],"thigh pain":["leg_pain"],"calf pain":["leg_pain"],"groin pain":["pelvic_pain"],"jaw pain":["facial_pain"],"rib pain":["chest_pain"],"chest wall pain":["chest_pain"],"tailbone pain":["back_pain"],"plantar fasciitis":["foot_pain","heel_pain"],"achilles tendon pain":["heel_pain","leg_pain"],"rotator cuff pain":["shoulder_pain"],"carpal tunnel":["wrist_pain","hand_pain","numbness","tingling"],"arthritis pain":["joint_pain"],"bone pain":["body_pain"],"growing pains":["leg_pain"],"spinal stenosis":["back_pain"],"herniated disc":["back_pain"],"patellofemoral pain":["knee_pain"],"morton's neuroma":["foot_pain"],"bunion pain":["foot_pain"],"hammer toe":["foot_pain"],"tmj pain":["facial_pain"],"jaw joint pain":["facial_pain"],"tension headache":["headache"],"cluster headache":["headache"],"sinus headache":["headache","facial_pain"],"phantom limb pain":["body_pain"],"muscle soreness":["muscle_pain"],"muscle cramps":["muscle_pain","muscle_spasms"],"joint stiffness":["stiffness"],"morning stiffness":["morning_stiffness"],"limited range of motion":["limited_range_of_motion"],"tingling":["tingling"],"pins and needles":["tingling"],"painful swallowing":["sore_throat"],"odynophagia":["sore_throat"],"painful urination":["burning_urination"],"dysuria":["burning_urination"],"menstrual cramps":["abdominal_pain","pelvic_pain"],"dysmenorrhea":["abdominal_pain","pelvic_pain"],"eye pain":["facial_pain"],"ear pain":["facial_pain"],"otalgia":["facial_pain"],"toothache":["facial_pain"],"stomach cramps":["stomach_pain","abdominal_cramps"],
"vertigo":["dizziness","balance_problems"],"balance problems":["balance_problems"],"off balance":["balance_problems"],"loss of balance":["balance_problems"],"unsteady":["balance_problems"],
"gastritis":["stomach_pain","bloating","nausea"],"indigestion":["bloating","heartburn","sour_taste"],"dyspepsia":["bloating","heartburn","stomach_pain"],
"eczema":["rash","skin_rash","itching"],"contact dermatitis":["rash","skin_rash","itching","redness"],"dermatitis":["rash","skin_rash","itching"],
"tonsillitis":["sore_throat","mild_fever","high_fever"],"strep throat":["sore_throat","high_fever"],
"irritable bowel":["abdominal_pain","bloating","diarrhea"],"ibs":["abdominal_pain","bloating"],
"sciatica":["back_pain","leg_pain","numbness","tingling"]}
with open("../models/symptom_synonyms.json","w") as f: json.dump(SYNS,f,indent=2)

d_pred = clf.predict(Xte)
r_pred = rcl.predict(Xrte)
dacc=accuracy_score(yte,d_pred)

def _metric_bundle(y_true, y_pred):
    p,r,f,_ = precision_recall_fscore_support(y_true,y_pred,average="weighted",zero_division=0)
    mp,mr,mf,_ = precision_recall_fscore_support(y_true,y_pred,average="macro",zero_division=0)
    return {
        "accuracy": round(float(accuracy_score(y_true,y_pred)),4),
        "weighted_precision": round(float(p),4),
        "weighted_recall": round(float(r),4),
        "weighted_f1": round(float(f),4),
        "macro_precision": round(float(mp),4),
        "macro_recall": round(float(mr),4),
        "macro_f1": round(float(mf),4),
    }

disease_labels = sorted(yd.unique().tolist())
report = {
    "dataset": {"records": int(len(df)), "symptoms": len(SC), "diseases": int(yd.nunique()), "risk_classes": list(le.classes_)},
    "split": {"test_size": 0.2, "random_state": 42, "stratified": True},
    "disease_model": {
        "type": "RandomForestClassifier",
        "parameters": {"n_estimators": 120, "max_depth": 14, "random_state": 42, "class_weight": "balanced"},
        "metrics": _metric_bundle(yte, d_pred),
        "confusion_matrix": {"labels": disease_labels, "matrix": confusion_matrix(yte,d_pred,labels=disease_labels).tolist()}
    },
    "risk_model": {
        "type": "RandomForestClassifier",
        "parameters": {"n_estimators": 100, "max_depth": 10, "random_state": 42, "class_weight": "balanced"},
        "metrics": _metric_bundle(yrte, r_pred),
        "confusion_matrix": {"labels": list(le.classes_), "matrix": confusion_matrix(yrte,r_pred,labels=list(range(len(le.classes_)))).tolist()}
    },
    # Backward-compatible flat fields for older code/tools.
    "disease_model_accuracy": round(float(dacc),4),
    "risk_model_accuracy": round(float(racc),4),
    "num_diseases": int(yd.nunique()),
    "num_symptoms": len(SC),
    "num_training_records": len(df),
    "warning": "Metrics are performance on this project dataset/test split, not clinical diagnostic accuracy."
}
with open("../models/training_report.json","w") as f:
    json.dump(report,f,indent=2)
print(f"Disease model: {yd.nunique()} diseases, {len(SC)} symptoms, {len(df)} records.")

# DL: skin CNN (optional — only retrained if TensorFlow is available and
# --skip-cnn was not passed). The symptom/risk dataset changes above do not
# affect the image model, so it is safe to leave the existing artifacts in
# place when TensorFlow is unavailable.
if args.skip_cnn:
    print("Skipping CNN training (--skip-cnn).")
else:
    try:
        import tensorflow as tf
        from tensorflow.keras import layers, models
    except Exception as exc:
        print(f"TensorFlow unavailable ({exc}); skipping CNN training. "
              f"Existing models/skin_cnn.keras is left unchanged.")
        tf = None

    if tf is not None:
        IMG=128; SEED=42
        tds=tf.keras.utils.image_dataset_from_directory("../data/skin_images",validation_split=0.2,subset="training",seed=SEED,image_size=(IMG,IMG),batch_size=16)
        vds=tf.keras.utils.image_dataset_from_directory("../data/skin_images",validation_split=0.2,subset="validation",seed=SEED,image_size=(IMG,IMG),batch_size=16)
        cn=tds.class_names
        AU=tf.data.AUTOTUNE
        tds=tds.cache().shuffle(200).prefetch(AU); vds=vds.cache().prefetch(AU)
        aug=models.Sequential([layers.RandomFlip("horizontal"),layers.RandomRotation(0.04)])
        model=models.Sequential([layers.Input(shape=(IMG,IMG,3)),layers.Rescaling(1./255),aug,layers.Conv2D(8,3,padding="same",activation="relu"),layers.MaxPooling2D(),layers.Conv2D(16,3,padding="same",activation="relu"),layers.MaxPooling2D(),layers.Conv2D(32,3,padding="same",activation="relu"),layers.MaxPooling2D(),layers.GlobalAveragePooling2D(),layers.Dropout(0.2),layers.Dense(32,activation="relu"),layers.Dense(len(cn),activation="softmax")])
        model.compile(optimizer=tf.keras.optimizers.Adam(0.001),loss="sparse_categorical_crossentropy",metrics=["accuracy"])
        es=tf.keras.callbacks.EarlyStopping(monitor="val_accuracy",patience=6,restore_best_weights=True)
        model.fit(tds,validation_data=vds,epochs=20,callbacks=[es],verbose=1)
        vl,va=model.evaluate(vds,verbose=0)
        print(f"CNN val acc: {va:.3f}")
        model.save("../models/skin_cnn.keras")
        with open("../models/skin_class_names.json","w") as f: json.dump(cn,f,indent=2)
        with open("../models/dl_training_report.json","w") as f:
            json.dump({"val_accuracy":round(float(va),4),"val_loss":round(float(vl),4),"classes":cn,"image_size":IMG},f,indent=2)

print("All models saved.")
