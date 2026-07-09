"""
Task T10 -- Numerical Price-of-Externality-on-Values (PoEV) table.

Reads the K=200 mechanism-grid cache and the C7/C8 caches, computes PoEV at
- the 8 pure NEs of C1 (symmetric, w1=w2=1) and the C1 social optimum,
- every NE returned across C7's symmetric w sweep,
- every NE returned across C8's asymmetric w1/w2 sweep,

against a geometric-midpoint uniform baseline (eps_uni ~ 5.73 at K=200).

PoEV definition (from COLLUSION_GAME_PLAN.md:60-68, analogous to the
Koutsoupias-Papadimitriou / Roughgarden-Tardos Price of Anarchy over an
observable *value* rather than a cost function):

    PoEV(eps*) := ( sum_i A_i(eps*) - sum_i A_i(eps_uni) ) / sum_i A_i(eps_uni)

Positive PoEV means the equilibrium profile inflates aggregate realised
MIA advantage relative to the symmetric uniform baseline; negative PoEV
means it deflates it (a protective-but-redistributive NE).

Outputs:
- tasks_findings/_scripts/_out/10/table_PoEV.tex   (LaTeX booktabs table for T10)
- tasks_findings/_scripts/_out/10/poev_C1.csv       (raw C1 rows)
- tasks_findings/_scripts/_out/10/poev_C7.csv       (C7 sweep)
- tasks_findings/_scripts/_out/10/poev_C8.csv       (C8 sweep)
- tasks_findings/_scripts/_out/10/summary.json      (headline numbers for T22)

The script is idempotent and reads only from _cache_v2/.
"""
from __future__ import annotations

import csv
import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from game.continuous import sweep_from_grid, find_pure_NE_2player  # noqa: E402

CACHE = ROOT / "game_experiments" / "_cache_v2"
OUT = Path(__file__).resolve().parent / "_out" / "10"
OUT.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# PoEV core
# ---------------------------------------------------------------------------
def uniform_baseline_from_grid(grid: dict) -> tuple[int, float, tuple[float, float]]:
    """Return (k_uni, eps_uni, (A1_uni, A2_uni)) at the geometric-midpoint
    feasible diagonal cell. Walk inwards if K//2 is infeasible."""
    K = len(grid["actions"])
    k0 = K // 2
    feas = grid["feasible"]
    A = grid["A"]
    for step in range(K // 2):
        for k in (k0 + step, k0 - step):
            if 0 <= k < K and bool(feas[k, k]):
                return k, float(grid["actions"][k]), (float(A[k, k, 0]),
                                                       float(A[k, k, 1]))
    raise RuntimeError("No feasible diagonal cell in the mechanism grid.")


def compute_poev(A_star_sum: float, A_uniform_sum: float) -> float:
    if A_uniform_sum <= 0:
        return float("nan")
    return (A_star_sum - A_uniform_sum) / A_uniform_sum


def poev_at_ne_list(A_arr: np.ndarray, actions: np.ndarray,
                    nes: list[tuple[int, int]],
                    A_uni_sum: float) -> list[dict]:
    rows = []
    for (i, j) in nes:
        a1 = float(A_arr[i, j, 0]); a2 = float(A_arr[i, j, 1])
        rows.append(dict(
            i=i, j=j,
            eps1=float(actions[i]), eps2=float(actions[j]),
            A1=a1, A2=a2, A_sum=a1 + a2,
            poev=compute_poev(a1 + a2, A_uni_sum),
        ))
    return rows


def social_optimum(sweep: dict) -> tuple[int, int]:
    U = sweep["U"]; feas = sweep["feasible"]
    W = np.where(feas, U[:, :, 0] + U[:, :, 1], -np.inf)
    idx = int(np.argmax(W))
    K = U.shape[0]
    return idx // K, idx % K


# ---------------------------------------------------------------------------
# LaTeX rendering
# ---------------------------------------------------------------------------
def format_poev_table_latex(c1_rows: list[dict],
                            eps_uni: float,
                            A_uni_sum: float) -> str:
    body_lines = []
    ne_rows = [r for r in c1_rows if r["label"].startswith("NE")]
    ne_rows_sorted = sorted(ne_rows, key=lambda r: (r["eps1"], r["eps2"]))
    for idx, r in enumerate(ne_rows_sorted, start=1):
        body_lines.append(
            fr"NE$_{{{idx}}}$ & $({r['eps1']:.2f}, {r['eps2']:.2f})$ "
            fr"& {r['A1']:.3f} & {r['A2']:.3f} & {r['A_sum']:.3f} "
            fr"& {r['poev']:+.3f} \\"
        )
    so = [r for r in c1_rows if r["label"] == "SO"][0]
    body_lines.append(
        fr"social opt & $({so['eps1']:.2f}, {so['eps2']:.2f})$ "
        fr"& {so['A1']:.3f} & {so['A2']:.3f} & {so['A_sum']:.3f} "
        fr"& {so['poev']:+.3f} \\"
    )
    body = "\n".join(body_lines)
    caption = (
        rf"Numerical Price-of-Externality-on-Values (\PoEV, "
        rf"\cref{{def:poev}}) at each pure Nash equilibrium of the "
        rf"symmetric 2-player PEG and at its social optimum. "
        rf"$K=200$ log-spaced action grid on $[0.5,64]$, $|G_1|=|G_2|=5000$, "
        rf"$E_b=128$, $5$ epochs, $\delta=10^{{-12}}$, $w_1=w_2=1$, "
        rf"$V(\sigma) = 1/(1+\sigma^2)$. Uniform baseline "
        rf"$\varepsilon^{{\mathrm{{uni}}}} = {eps_uni:.2f}$ (geometric midpoint "
        rf"of the log grid) with $\sum_i A_i(\varepsilon^{{\mathrm{{uni}}}}) "
        rf"= {A_uni_sum:.3f}$. Social optimum is the feasible action profile "
        rf"maximising $u_1 + u_2$."
    )
    return (
        r"\begin{table}[t]" "\n"
        r"\centering" "\n"
        rf"\caption{{{caption}}}" "\n"
        r"\label{tab:PoEV}" "\n"
        r"\begin{tabular}{lccccc}" "\n"
        r"\toprule" "\n"
        r" & $(\varepsilon_1^*, \varepsilon_2^*)$ & $A_1$ & $A_2$ "
        r"& $\sum_i A_i$ & $\PoEV$ \\" "\n"
        r"\midrule" "\n"
        + body + "\n" +
        r"\bottomrule" "\n"
        r"\end{tabular}" "\n"
        r"\end{table}" "\n"
    )


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main() -> None:
    print("[T10] loading K=200 mechanism grid...")
    grid = pickle.load(open(CACHE / "mech_grid_K200_g5000x5000_b128_e5.pkl", "rb"))
    K = len(grid["actions"])
    k_uni, eps_uni, (A1_uni, A2_uni) = uniform_baseline_from_grid(grid)
    A_uni_sum = A1_uni + A2_uni
    print(f"  K={K}  uniform baseline: k={k_uni} eps={eps_uni:.4f} "
          f"A1={A1_uni:.4f} A2={A2_uni:.4f} sum={A_uni_sum:.4f}")

    # Sanity: PoEV at the baseline itself is zero.
    assert abs(compute_poev(A_uni_sum, A_uni_sum)) < 1e-12

    # -----------------------------------------------------------------------
    # C1: pure NEs at w1 = w2 = 1  (recompute from mechanism grid at K=200 --
    # the paper's table_NE_v2.tex is at K=22 and would understate the number
    # of pure NEs on the finer grid; we use whichever K=200 NEs exist).
    # -----------------------------------------------------------------------
    print("[T10] deriving C1 sweep at K=200 ...")
    sweep_c1 = sweep_from_grid(grid, w1=1.0, w2=1.0)
    nes_c1 = find_pure_NE_2player(sweep_c1)
    A_c1 = sweep_c1["A"]; actions = sweep_c1["actions"]
    print(f"  |NE|_C1(K=200) = {len(nes_c1)}")

    ne_rows = poev_at_ne_list(A_c1, actions, nes_c1, A_uni_sum)
    so_i, so_j = social_optimum(sweep_c1)
    a1 = float(A_c1[so_i, so_j, 0]); a2 = float(A_c1[so_i, so_j, 1])
    so_row = dict(
        i=so_i, j=so_j,
        eps1=float(actions[so_i]), eps2=float(actions[so_j]),
        A1=a1, A2=a2, A_sum=a1 + a2,
        poev=compute_poev(a1 + a2, A_uni_sum),
    )
    print(f"  social opt: ({so_row['eps1']:.3f}, {so_row['eps2']:.3f}) "
          f"A_sum={so_row['A_sum']:.4f} PoEV={so_row['poev']:+.4f}")

    # For the LaTeX table we display a representative subset when the K=200
    # grid produces many NEs: we mirror the paper's K=22 canonical table by
    # picking the 8 pure NEs that are extremal in the reflection classes.
    # We select the NE with maximum |eps1 - eps2| in each of 4 "distance"
    # bins per side of the diagonal. If |NE|_C1 <= 8 we take all of them.
    def canonical_eight(rows: list[dict]) -> list[dict]:
        if len(rows) <= 8:
            return rows
        # Split by which side of the diagonal.
        below = [r for r in rows if r["eps1"] < r["eps2"]]
        above = [r for r in rows if r["eps1"] > r["eps2"]]
        below_sorted = sorted(below, key=lambda r: r["eps1"] / r["eps2"])
        above_sorted = sorted(above, key=lambda r: r["eps2"] / r["eps1"])
        # Pick 4 quantile-spaced representatives per side.
        def pick(seq: list[dict], k: int) -> list[dict]:
            if len(seq) <= k:
                return seq
            idxs = np.linspace(0, len(seq) - 1, k).round().astype(int)
            return [seq[i] for i in idxs]
        return pick(below_sorted, 4) + pick(above_sorted, 4)

    ne_display = canonical_eight(ne_rows)
    print(f"  displaying {len(ne_display)} representative NEs in LaTeX table")

    c1_rows = []
    for r in ne_display:
        c1_rows.append({**r, "label": "NE"})
    c1_rows.append({**so_row, "label": "SO"})

    tex = format_poev_table_latex(c1_rows, eps_uni, A_uni_sum)
    (OUT / "table_PoEV.tex").write_text(tex)
    print(f"  -> {OUT / 'table_PoEV.tex'}")

    # Full C1 dump (all NEs, not just the 8 canonical).
    with open(OUT / "poev_C1.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["label", "i", "j", "eps1", "eps2", "A1", "A2", "A_sum", "poev"])
        for r in ne_rows:
            w.writerow(["NE", r["i"], r["j"], r["eps1"], r["eps2"],
                        r["A1"], r["A2"], r["A_sum"], r["poev"]])
        w.writerow(["SO", so_row["i"], so_row["j"], so_row["eps1"],
                    so_row["eps2"], so_row["A1"], so_row["A2"],
                    so_row["A_sum"], so_row["poev"]])

    # -----------------------------------------------------------------------
    # C7 sweep
    # -----------------------------------------------------------------------
    print("[T10] loading C7 ...")
    c7 = pickle.load(open(CACHE / "C7_K200.pkl", "rb"))
    c7_records = []
    with open(OUT / "poev_C7.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["w", "ne_idx", "i", "j", "eps1", "eps2", "A1", "A2",
                    "A_sum", "poev"])
        for row in c7["rows"]:
            wv = row["w"]
            for k_ne, (i, j) in enumerate(row["nes"]):
                a1 = float(row["A"][i, j, 0]); a2 = float(row["A"][i, j, 1])
                poev = compute_poev(a1 + a2, A_uni_sum)
                w.writerow([wv, k_ne, i, j, float(row["actions"][i]),
                            float(row["actions"][j]), a1, a2,
                            a1 + a2, poev])
                c7_records.append(dict(w=wv, poev=poev))

    # -----------------------------------------------------------------------
    # C8 sweep
    # -----------------------------------------------------------------------
    print("[T10] loading C8 ...")
    c8 = pickle.load(open(CACHE / "C8_K200.pkl", "rb"))
    c8_records = []
    with open(OUT / "poev_C8.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["ratio", "ne_idx", "i", "j", "eps1", "eps2", "A1", "A2",
                    "A_sum", "poev"])
        for row in c8["rows"]:
            rv = row["ratio"]
            for k_ne, (i, j) in enumerate(row["nes"]):
                a1 = float(row["A"][i, j, 0]); a2 = float(row["A"][i, j, 1])
                poev = compute_poev(a1 + a2, A_uni_sum)
                w.writerow([rv, k_ne, i, j, float(row["actions"][i]),
                            float(row["actions"][j]), a1, a2,
                            a1 + a2, poev])
                c8_records.append(dict(ratio=rv, poev=poev))

    # -----------------------------------------------------------------------
    # Distributional summary
    # -----------------------------------------------------------------------
    all_poevs = [r["poev"] for r in ne_rows] + [so_row["poev"]] \
        + [r["poev"] for r in c7_records] + [r["poev"] for r in c8_records]
    all_poevs = np.array(all_poevs)

    def group_stats(records: list[dict], key: str) -> dict:
        out = {}
        by = {}
        for r in records:
            by.setdefault(r[key], []).append(r["poev"])
        for k, vals in sorted(by.items()):
            v = np.array(vals)
            out[str(k)] = dict(n=int(len(v)),
                                min=float(v.min()), max=float(v.max()),
                                mean=float(v.mean()), median=float(np.median(v)))
        return out

    c1_poevs = np.array([r["poev"] for r in ne_rows] + [so_row["poev"]])
    summary = dict(
        K=K,
        eps_uni=eps_uni,
        A_uni_sum=A_uni_sum,
        A_uni_per_player=(A1_uni, A2_uni),
        C1=dict(
            n_ne=len(nes_c1),
            ne_poev_min=float(np.min([r["poev"] for r in ne_rows])),
            ne_poev_max=float(np.max([r["poev"] for r in ne_rows])),
            ne_poev_median=float(np.median([r["poev"] for r in ne_rows])),
            so_poev=so_row["poev"],
            so_profile=(so_row["eps1"], so_row["eps2"]),
        ),
        C7=group_stats(c7_records, "w"),
        C8=group_stats(c8_records, "ratio"),
        overall=dict(
            min=float(all_poevs.min()),
            max=float(all_poevs.max()),
            median=float(np.median(all_poevs)),
            n=int(len(all_poevs)),
        ),
    )
    (OUT / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"  -> {OUT / 'summary.json'}")
    print(f"[T10] overall PoEV: min={summary['overall']['min']:+.4f} "
          f"max={summary['overall']['max']:+.4f} "
          f"median={summary['overall']['median']:+.4f} "
          f"(n={summary['overall']['n']})")


if __name__ == "__main__":
    main()
