"""Figures from the holistic robustness sweep (60_robustness.pkl):

Fig A (invariance heatmap): findings x assumption-dial levels, cell =
share of tested worlds where the finding held. Green = held nearly
everywhere, red = rarely; a row that stays green across a dial is
robust to that dial, a row that changes color shows who owns it.

Fig B (the two dials that matter): bar panels for the two sharpest
flips — cost shape owns corner play; group size owns small-T
symmetric moderation.
"""
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

res = [r for r in pickle.load(open(
    ROOT / "tasks_findings/_scripts/60_robustness.pkl", "rb")) if r.get("ok")]
two = [r for r in res if len(r["cfg"]["G"]) == 2]


def axes_of(cfg):
    lam = cfg["lam"]
    het = isinstance(lam, (tuple, list))
    lv = lam[0] if het else lam
    if cfg["cost_form"] == "linear":
        lb = "w small" if lv < 0.3 else ("w mid" if lv < 3 else "w large")
    else:
        lb = "wq small" if lv < 500 else "wq large"
    return {
        "value shape": {"inv_sq": "1/(1+s²)", "inv": "1/(1+s)",
                        "exp": "e^-s"}[cfg["V_form"]],
        "cost shape": cfg["cost_form"],
        "weight": lb,
        "group size": ("500" if cfg["G"][0] == 500 else
                       "5000" if cfg["G"][0] == 5000 else
                       "50000" if cfg["G"][0] == 50000 else "unequal"),
        "floor": f"{cfg['pbar']:.0e}",
        "box top": f"{cfg['box'][1]:g}",
        "players": "unequal w" if het else "equal w",
        "steps": f"T={cfg['T']}",
    }


FINDINGS = {
    "all-or-nothing play\n(no interior best reply)":
        lambda r: (r["kinds"]["int"] == 0) if r.get("kinds") else None,
    "'worst compromise' dip\nexists":
        lambda r: (r.get("trough_frac", 0) > 0) if r.get("kinds") else None,
    "specializing beats\nequal shares":
        lambda r: ((r.get("premium") or 0) > 1e-4) if r.get("kinds") else None,
    "best symmetric point\nis inside (not at a wall)":
        lambda r: r.get("diag_interior"),
    "symmetric moderation\n(near-)stable":
        lambda r: (min(x for _, x in r["gaps"]) < 1e-3
                   and r.get("diag_interior")) if r.get("gaps") else None,
}

# ---- Fig A: heatmap ------------------------------------------------------
col_groups, col_labels = [], []
for ax_name in ("cost shape", "weight", "group size", "value shape",
                "floor", "box top", "players", "steps"):
    levels = sorted({axes_of(r["cfg"])[ax_name] for r in two})
    col_groups.append((ax_name, levels))
    col_labels += [f"{lv}" for lv in levels]

M = np.full((len(FINDINGS), len(col_labels)), np.nan)
counts = np.zeros_like(M, dtype=object)
for fi, (fname, f) in enumerate(FINDINGS.items()):
    ci = 0
    for ax_name, levels in col_groups:
        agg = defaultdict(lambda: [0, 0])
        for r in two:
            v = f(r)
            if v is None:
                continue
            a = axes_of(r["cfg"])[ax_name]
            agg[a][0] += bool(v)
            agg[a][1] += 1
        for lv in levels:
            h, n = agg[lv]
            M[fi, ci] = h / n if n else np.nan
            counts[fi, ci] = f"{h}/{n}"
            ci += 1

fig, ax = plt.subplots(figsize=(15.5, 5.4))
im = ax.imshow(M, cmap="RdYlGn", vmin=0, vmax=1, aspect="auto")
ax.set_xticks(range(len(col_labels)))
ax.set_xticklabels(col_labels, rotation=45, ha="right", fontsize=8)
ax.set_yticks(range(len(FINDINGS)))
ax.set_yticklabels(list(FINDINGS), fontsize=9)
for fi in range(M.shape[0]):
    for ci in range(M.shape[1]):
        if np.isfinite(M[fi, ci]):
            ax.text(ci, fi, f"{100*M[fi,ci]:.0f}%", ha="center",
                    va="center", fontsize=7,
                    color="black" if 0.25 < M[fi, ci] < 0.85 else "white")
x0 = -0.5
for ax_name, levels in col_groups:
    x1 = x0 + len(levels)
    ax.axvline(x1, color="white", lw=2)
    ax.text(0.5 * (x0 + x1), -0.9, ax_name, ha="center", fontsize=9,
            fontweight="bold")
    x0 = x1
ax.set_title("How often each finding held, split by every assumption dial "
             "(191 exact game configurations)", fontsize=11, pad=28)
plt.colorbar(im, ax=ax, label="share of tested worlds where it held")
fig.tight_layout()
fig.savefig(ROOT / "paper/figures/data/foc_landscape/invariance_heatmap.png",
            dpi=160, bbox_inches="tight")
print("wrote invariance_heatmap.png")

# ---- Fig B: the two dials that matter ------------------------------------
fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4.3))
f_corner = FINDINGS["all-or-nothing play\n(no interior best reply)"]
agg = defaultdict(lambda: [0, 0])
for r in two:
    v = f_corner(r)
    if v is None:
        continue
    a = axes_of(r["cfg"])["cost shape"]
    agg[a][0] += bool(v); agg[a][1] += 1
labels = ["linear", "quad"]
vals = [100 * agg[l][0] / agg[l][1] for l in labels]
a1.bar(["linear pricing\n(pain per unit exposure\nis constant)",
        "progressive pricing\n(heavy exposure hurts\ndisproportionately)"],
       vals, color=["#c0504d", "#4f81bd"])
for x, v, l in zip((0, 1), vals, labels):
    a1.text(x, v + 2, f"{v:.0f}%  ({agg[l][0]}/{agg[l][1]})", ha="center")
a1.set_ylim(0, 115)
a1.set_ylabel("share of worlds with all-or-nothing play")
a1.set_title("who owns all-or-nothing play:\nthe SHAPE of the exposure price",
             fontsize=10)

f_mod = FINDINGS["symmetric moderation\n(near-)stable"]
agg = defaultdict(lambda: [0, 0])
for r in two:
    v = f_mod(r)
    if v is None:
        continue
    a = axes_of(r["cfg"])["group size"]
    agg[a][0] += bool(v); agg[a][1] += 1
order = ["500", "5000", "50000", "unequal"]
vals = [100 * agg[o][0] / agg[o][1] if agg[o][1] else 0 for o in order]
a2.bar([f"{o}\nrecords/player" if o != "unequal" else "unequal\nsizes"
        for o in order], vals, color="#9bbb59")
for x, (o, v) in enumerate(zip(order, vals)):
    a2.text(x, v + 0.8, f"{v:.0f}%  ({agg[o][0]}/{agg[o][1]})", ha="center",
            fontsize=9)
a2.set_ylabel("share of worlds with (near-)stable moderation")
a2.set_title("who owns small-T symmetric moderation:\nGROUP SIZE "
             "(per-record exposure intensity)", fontsize=10)
fig.suptitle("The two dials with the sharpest effects (one- and two-step "
             "exact games)", fontsize=11)
fig.tight_layout()
fig.savefig(ROOT / "paper/figures/data/foc_landscape/invariance_dials.png",
            dpi=160, bbox_inches="tight")
print("wrote invariance_dials.png")
