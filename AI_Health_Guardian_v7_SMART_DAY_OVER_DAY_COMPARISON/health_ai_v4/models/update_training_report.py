import json, os, joblib
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from sklearn.preprocessing import LabelEncoder

BASE=os.path.dirname(__file__); DATA=os.path.join(BASE,'..','data')
df=pd.read_csv(os.path.join(DATA,'symptom_disease_dataset.csv'))
SC=json.load(open(os.path.join(DATA,'symptom_columns.json'),encoding='utf-8'))
X=df[SC].to_numpy(dtype=np.int8)

def metrics(y_true,y_pred):
    p,r,f,_=precision_recall_fscore_support(y_true,y_pred,average='weighted',zero_division=0)
    mp,mr,mf,_=precision_recall_fscore_support(y_true,y_pred,average='macro',zero_division=0)
    return {'accuracy':round(float(accuracy_score(y_true,y_pred)),4),'weighted_precision':round(float(p),4),'weighted_recall':round(float(r),4),'weighted_f1':round(float(f),4),'macro_precision':round(float(mp),4),'macro_recall':round(float(mr),4),'macro_f1':round(float(mf),4)}

# Disease: exact production split/config
Xtr,Xte,ytr,yte=train_test_split(X,df.prognosis,test_size=.2,random_state=42,stratify=df.prognosis)
dm=joblib.load(os.path.join(BASE,'disease_model.joblib')); dp=dm.predict(Xte)
# Risk: exact production split/config
le=joblib.load(os.path.join(BASE,'risk_label_encoder.joblib'))
yr=le.transform(df.risk_level)
Xrtr,Xrte,yrtr,yrte=train_test_split(X,yr,test_size=.2,random_state=42,stratify=yr)
rm=joblib.load(os.path.join(BASE,'risk_model.joblib')); rp=rm.predict(Xrte)

report={
 'dataset': {'records':int(len(df)),'symptoms':len(SC),'diseases':int(df.prognosis.nunique()),'risk_classes':list(le.classes_)},
 'split': {'test_size':0.2,'random_state':42,'stratified':True},
 'disease_model': {'type':'RandomForestClassifier','parameters':{'n_estimators':120,'max_depth':14,'random_state':42,'class_weight':'balanced'},'metrics':metrics(yte,dp),'confusion_matrix':{'labels':sorted(df.prognosis.unique().tolist()),'matrix':confusion_matrix(yte,dp,labels=sorted(df.prognosis.unique())).tolist()}},
 'risk_model': {'type':'RandomForestClassifier','parameters':{'n_estimators':100,'max_depth':10,'random_state':42,'class_weight':'balanced'},'metrics':metrics(yrte,rp),'confusion_matrix':{'labels':list(le.classes_),'matrix':confusion_matrix(yrte,rp,labels=list(range(len(le.classes_)))).tolist()}},
 'disease_model_accuracy':0.0, 'risk_model_accuracy':0.0,
 'num_diseases':int(df.prognosis.nunique()), 'num_symptoms':len(SC), 'num_training_records':len(df),
 'warning':'Metrics are performance on this project dataset/test split, not clinical diagnostic accuracy.'
}
report['disease_model_accuracy'] = report['disease_model']['metrics']['accuracy']
report['risk_model_accuracy'] = report['risk_model']['metrics']['accuracy']
with open(os.path.join(BASE,'training_report.json'),'w',encoding='utf-8') as f: json.dump(report,f,indent=2)

# Save concise confusion matrices for review use.
pd.DataFrame(report['risk_model']['confusion_matrix']['matrix'],index=report['risk_model']['confusion_matrix']['labels'],columns=report['risk_model']['confusion_matrix']['labels']).to_csv(os.path.join(BASE,'confusion_matrix_risk_random_forest.csv'))
labels=report['disease_model']['confusion_matrix']['labels']
pd.DataFrame(report['disease_model']['confusion_matrix']['matrix'],index=labels,columns=labels).to_csv(os.path.join(BASE,'confusion_matrix_disease_random_forest.csv'))
print(json.dumps({'disease':report['disease_model']['metrics'],'risk':report['risk_model']['metrics']},indent=2))
