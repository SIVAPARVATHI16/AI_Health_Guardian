import json,os,statistics
from datetime import datetime
LOG_DIR=os.path.join(os.path.dirname(__file__),"..","data","user_logs")
os.makedirs(LOG_DIR,exist_ok=True)
IMAGE_CLASS_SCORE={"healthy_skin":100,"healing_rash":60,"inflamed_rash":20}
def _lp(uid): return os.path.join(LOG_DIR,f"{''.join(c for c in uid if c.isalnum() or c in'-_') or 'guest'}.json")
def load_log(uid):
    p=_lp(uid)
    if not os.path.exists(p): return []
    with open(p) as f: return json.load(f)
def save_log(uid,e):
    with open(_lp(uid),"w") as f: json.dump(e,f,indent=2)
def add_entry(uid,severity,medicine_taken,image_class=None,image_confidence=None,notes=""):
    entries=load_log(uid)
    sc=max(0,100-(severity*10))
    ic=IMAGE_CLASS_SCORE.get(image_class,None)
    hs=round(0.6*sc+0.4*ic) if ic is not None else sc
    e={"timestamp":datetime.now().strftime("%Y-%m-%d %H:%M"),"severity":severity,"medicine_taken":bool(medicine_taken),"image_class":image_class,"image_confidence":image_confidence,"health_score":hs,"notes":notes}
    entries.append(e); save_log(uid,entries); return e
def compute_trend(entries,lookback=5):
    r=entries[-lookback:]
    if len(r)<2: return "Not enough data"
    scores=[e["health_score"] for e in r]; m=len(scores)//2
    delta=statistics.mean(scores[m:])-statistics.mean(scores[:m] if m>0 else scores[:1])
    return "Improving" if delta>=5 else "Worsening" if delta<=-5 else "Stable"
def medicine_adherence_pct(entries,lookback=7):
    r=entries[-lookback:]
    if not r: return 100.0
    return round(100*sum(1 for e in r if e["medicine_taken"])/len(r),1)
def summary(uid):
    entries=load_log(uid)
    if not entries: return {"days_tracked":0,"trend":"Not enough data","adherence_pct":100.0,"latest_health_score":None,"history":[]}
    return {"days_tracked":len(entries),"trend":compute_trend(entries),"adherence_pct":medicine_adherence_pct(entries),"latest_health_score":entries[-1]["health_score"],"history":entries}
def reset_user(uid):
    p=_lp(uid)
    if os.path.exists(p): os.remove(p)
