"""Invariance matrix from the holistic sweep (60_robustness.pkl, job
180346): per finding, the fraction of configurations where it holds,
broken down by every assumption axis with >25% spread. See
nummeric_eval.md Sec. 24.6 for the verdicts."""
import pickle, sys
from collections import defaultdict
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
res = [r for r in pickle.load(open(ROOT/"tasks_findings/_scripts/60_robustness.pkl","rb")) if r.get("ok")]

def axes_of(cfg):
    lam = cfg["lam"]; het = isinstance(lam,(tuple,list)); lv = lam[0] if het else lam
    if cfg["cost_form"]=="linear":
        lb = "lam<0.3" if lv<0.3 else ("lam0.3-3" if lv<3 else "lam>3")
    else:
        lb = "lamq<500" if lv<500 else "lamq>500"
    return dict(V=cfg["V_form"], cost=cfg["cost_form"], lam=lb,
                G="G"+str(cfg["G"][0]) if cfg["G"][0]==cfg["G"][1] else "Gasym",
                pbar=f"pb{cfg['pbar']:.0e}", box=f"box{cfg['box'][1]:g}",
                het="het" if het else "hom", T=f"T{cfg['T']}")

FINDINGS = {
  "corner-dominated": lambda r: (r["kinds"]["int"]==0) if r.get("kinds") else None,
  "trough exists": lambda r: (r.get("trough_frac",0)>0) if r.get("kinds") else None,
  "spec premium>1e-4": lambda r: ((r.get("premium") or 0)>1e-4) if r.get("kinds") else None,
  "fair wins (premium<-1e-4)": lambda r: ((r.get("premium") or 0)<-1e-4) if r.get("kinds") else None,
  "diag opt interior": lambda r: r.get("diag_interior"),
  "sym gap<1e-3 & interior": lambda r: (min(x for _,x in r["gaps"])<1e-3 and r.get("diag_interior")) if r.get("gaps") else None,
}
two = [r for r in res if len(r["cfg"]["G"])==2]
print(f"invariance matrix over {len(two)} two-player configs")
for name,f in FINDINGS.items():
    vals = [(axes_of(r["cfg"]), f(r)) for r in two]
    vals = [(a,v) for a,v in vals if v is not None]
    print(f"\n{name}: {sum(v for _,v in vals)}/{len(vals)} overall")
    for ax in ("V","cost","lam","G","pbar","box","het","T"):
        cnt = defaultdict(lambda:[0,0])
        for a,v in vals:
            cnt[a[ax]][0]+=v; cnt[a[ax]][1]+=1
        fr = [c[0]/c[1] for c in cnt.values()]
        if max(fr)-min(fr) > 0.25:
            print(f"   by {ax:4s}: " + "  ".join(f"{k}:{c[0]}/{c[1]}" for k,c in sorted(cnt.items())))
for r in [x for x in res if len(x["cfg"]["G"])==3]:
    print(f"\nn=3 {r['cfg']['G']}: diag argmax {r['diag_argmax']:.3f} interior={r['diag_interior']}")
