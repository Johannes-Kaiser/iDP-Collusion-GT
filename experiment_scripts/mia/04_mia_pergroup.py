"""T24 -- per-group DP-SGD shadow-model trainer for the n=2 PEG LiRA sweep.

Fork of `experiment_scripts/mia/04_mia.py`, but instead
of resolving per-record `pp_budgets` via
`generate_pp_budgets -> assign_pp_values -> make_private_with_epsilon`, it
takes the mechanism-solver output `(sigma, p_g)` directly on the CLI and
feeds it into Opacus verbatim. This lets `tasks_findings/_scripts/24_lira_analysis.py`
pin the training-time noise/sample-rate to the exact (sigma, p_1, p_2) it
solved for each 5x5 grid profile.

Output layout (per profile pid, matched to
`24_lira_analysis.py::score_profile`):

    <savedir_result>/
        labels.npy         # (N,) int class labels for the LiRA membership pool
        silo_id.npy        # (N,) in {0, 1} silo assignment
        pool_idx.npy       # (N,) source-dataset indices; provenance
        shadow_0000/
            logits.npy     # (N, n_queries, n_classes) test-time logits
            keep.npy       # (N,) bool membership label
            test_acc.npy   # scalar
        shadow_0001/ ...

CLAUDE.md Rule 3: GPU only because DP-SGD trains an NN; the sanity-test path
uses `--n_shadows_override 2 --epochs_override 5` and stays under 2 min.
CLAUDE.md Rule 5: this trainer feeds paper figures via 24_lira_analysis.py;
do not perturb the mechanism-solver.
"""
from __future__ import annotations

import argparse
import gc
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, TensorDataset

_CVS = Path("/vol/miltank/users/kaiserj/Collusion_Gametheory")
sys.path.insert(0, str(_CVS / "experiment_scripts" / "mia"))
sys.path.insert(0, str(_CVS))

from utils.utils_general import (  # noqa: E402
    load_dataset,
    load_model,
    load_yaml_args,
)
from opacus_new import PrivacyEngine  # noqa: E402

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _parse_cli() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--exp_yaml", required=True,
                    help="YAML config; supplies dataset/model/epochs/n_shadows/pkeep/...")
    ap.add_argument("--pg_sigma", type=float, required=True,
                    help="DP-SGD noise multiplier (from mechanism grid).")
    ap.add_argument("--pg_sample_rates", required=True,
                    help="Comma-separated per-silo Poisson sample rates.")
    ap.add_argument("--group_sizes", required=True,
                    help="Comma-separated per-silo record counts.")
    ap.add_argument("--silo_split", default="modulo2",
                    choices=["modulo2", "block2"],
                    help="Silo assignment on the LiRA pool.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--savedir_result", required=True)
    ap.add_argument("--profile_idx", type=int, default=-1,
                    help="Grid-profile index (documentation only; not consumed).")
    # Sanity-test hooks. Override YAML values without editing the YAML.
    ap.add_argument("--n_shadows_override", type=int, default=None)
    ap.add_argument("--epochs_override", type=int, default=None)
    ap.add_argument("--dataset_override", default=None)
    return ap.parse_args()


def _seed_all(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _materialise_pool(train_ds, pool_idx: np.ndarray) -> TensorDataset:
    """Read the fixed n_total records into a TensorDataset once."""
    xs, ys = [], []
    for i in pool_idx:
        x, y = train_ds[int(i)]
        if not isinstance(x, torch.Tensor):
            x = torch.as_tensor(np.asarray(x))
        xs.append(x)
        ys.append(int(y) if isinstance(y, torch.Tensor) else int(y))
    pool_x = torch.stack(xs)
    pool_y = torch.tensor(ys, dtype=torch.long)
    return TensorDataset(pool_x, pool_y)


def _train_one_shadow(
    model: nn.Module,
    train_subset: Subset,
    pp_sr_subset: np.ndarray,
    pp_budgets_subset: np.ndarray,
    sigma: float,
    epochs: int,
    batchsize: int,
    lr: float,
    max_grad_norm: float,
    accountant: str,
) -> nn.Module:
    optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=5e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epochs)
    criterion = nn.CrossEntropyLoss()

    # If every silo shares the same sample rate (symmetric profile, e.g. the
    # diagonal of the 5x5 grid), fall back to plain DP-SGD (single-group). The
    # individualized code path assumes >=2 unique per-record sample rates and
    # errors out on np.unique(pp_weights).length < n_groups.
    unique_srs = np.unique(pp_sr_subset)
    if len(unique_srs) == 1:
        shared_sr = float(unique_srs[0])
        n_records = len(train_subset)
        base_batch = max(1, int(round(shared_sr * n_records)))
        base_dl = DataLoader(
            train_subset, batch_size=base_batch, shuffle=True, num_workers=0,
        )
        privacy_engine = PrivacyEngine(
            accountant=accountant,
            individualize=None,
        )
        dp_model, dp_optim, dp_dl = privacy_engine.make_private(
            module=model,
            optimizer=optim,
            data_loader=base_dl,
            noise_multiplier=sigma,
            max_grad_norm=max_grad_norm,
        )
    else:
        base_dl = DataLoader(
            train_subset, batch_size=batchsize, shuffle=True, num_workers=0,
        )
        privacy_engine = PrivacyEngine(
            accountant=accountant,
            individualize="sampling",
            pp_budgets=pp_budgets_subset,
        )
        dp_model, dp_optim, dp_dl = privacy_engine.make_private(
            module=model,
            optimizer=optim,
            data_loader=base_dl,
            noise_multiplier=sigma,
            max_grad_norm=max_grad_norm,
            pp_weights=pp_sr_subset,
        )

    dp_model.train()
    for _ep in range(epochs):
        for batch in dp_dl:
            if len(batch) == 2:
                x, y = batch
            else:
                x, y, _ = batch
            if x.numel() == 0:
                # Poisson sampling produced an empty batch; skip.
                dp_optim.zero_grad()
                continue
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            dp_optim.zero_grad()
            out = dp_model(x)
            loss = criterion(out, y)
            loss.backward()
            dp_optim.step()
        sched.step()

    return dp_model


def _eval_pool_logits(model: nn.Module, pool_ds: TensorDataset,
                      n_queries: int) -> np.ndarray:
    model.eval()
    dl = DataLoader(pool_ds, batch_size=512, shuffle=False, num_workers=0)
    all_q = []
    with torch.no_grad():
        for _q in range(n_queries):
            chunks = []
            for x, _y in dl:
                x = x.to(DEVICE)
                chunks.append(model(x).detach().cpu().numpy())
            all_q.append(np.concatenate(chunks))
    return np.stack(all_q, axis=1)  # (N, Q, C)


def _eval_test_acc(model: nn.Module, test_ds) -> float:
    model.eval()
    dl = DataLoader(test_ds, batch_size=512, shuffle=False, num_workers=0)
    correct = 0
    total = 0
    with torch.no_grad():
        for x, y in dl:
            x = x.to(DEVICE)
            y = y.to(DEVICE)
            pr = model(x).argmax(dim=1)
            correct += (pr == y).sum().item()
            total += y.numel()
    return correct / max(total, 1)


def main() -> None:
    cli = _parse_cli()

    cfg = load_yaml_args(cli.exp_yaml)
    # Apply CLI overrides.
    if cli.dataset_override:
        cfg["dataset"] = cli.dataset_override
    if cli.n_shadows_override is not None:
        cfg["n_shadows"] = cli.n_shadows_override
    if cli.epochs_override is not None:
        cfg["epochs"] = cli.epochs_override
    cfg.setdefault("n_queries", 1)
    cfg.setdefault("model", None)
    cfg.setdefault("pkeep", 0.5)
    cfg.setdefault("max_grad_norm", 1.0)
    cfg.setdefault("accountant", "rdp")
    cfg.setdefault("batchsize", 128)
    cfg.setdefault("lr", 0.01)
    cfg["target_delta"] = float(cfg.get("target_delta", 1e-12))

    dataset = cfg["dataset"]
    n_shadows = int(cfg["n_shadows"])
    pkeep = float(cfg["pkeep"])
    epochs = int(cfg["epochs"])
    batchsize = int(cfg["batchsize"])
    lr = float(cfg["lr"])
    max_grad_norm = float(cfg["max_grad_norm"])
    accountant = cfg["accountant"]
    n_queries = int(cfg["n_queries"])
    model_name = cfg["model"]

    sigma = cli.pg_sigma
    pg_sr = np.array([float(x) for x in cli.pg_sample_rates.split(",")], dtype=np.float64)
    group_sizes = np.array([int(x) for x in cli.group_sizes.split(",")], dtype=np.int64)
    n_silos = len(group_sizes)
    assert len(pg_sr) == n_silos, "pg_sample_rates and group_sizes must match in length"
    n_total = int(group_sizes.sum())

    outdir = Path(cli.savedir_result)
    outdir.mkdir(parents=True, exist_ok=True)

    # ==================================================================
    # Load dataset + assemble LiRA membership pool once.
    # ==================================================================
    train_ds = load_dataset(dataset, train=True)
    test_ds = load_dataset(dataset, train=False)
    if len(train_ds) < n_total:
        raise ValueError(
            f"Dataset {dataset} has only {len(train_ds)} train rows; "
            f"pool needs {n_total}."
        )

    # Deterministic pool of n_total rows (seed-scoped, independent of shadow_id).
    rng_pool = np.random.default_rng(cli.seed)
    pool_idx = rng_pool.permutation(len(train_ds))[:n_total]

    pool_ds = _materialise_pool(train_ds, pool_idx)
    labels_np = pool_ds.tensors[1].numpy().astype(np.int64)

    # Silo assignment on the pool.
    if cli.silo_split == "modulo2":
        assert n_silos == 2, "modulo2 requires exactly 2 silos"
        silo_id = (np.arange(n_total) % 2).astype(np.int64)
    else:  # block2
        assert n_silos == 2
        silo_id = np.zeros(n_total, dtype=np.int64)
        silo_id[group_sizes[0]:] = 1

    # Per-record sample rate + a synthetic per-record budget (only used by
    # PrivacyEngine to enumerate groups).
    pp_sr_full = pg_sr[silo_id]
    pp_budgets_full = (1.0 + silo_id).astype(np.float64)

    np.save(outdir / "labels.npy", labels_np)
    np.save(outdir / "silo_id.npy", silo_id)
    np.save(outdir / "pool_idx.npy", pool_idx.astype(np.int64))
    with open(outdir / "profile_meta.txt", "w") as f:
        f.write(
            f"dataset={dataset}\n"
            f"sigma={sigma}\n"
            f"pg_sample_rates={list(pg_sr)}\n"
            f"group_sizes={list(map(int, group_sizes))}\n"
            f"silo_split={cli.silo_split}\n"
            f"n_shadows={n_shadows}\n"
            f"epochs={epochs}\n"
            f"batchsize={batchsize}\n"
            f"seed={cli.seed}\n"
            f"profile_idx={cli.profile_idx}\n"
        )

    # ==================================================================
    # Train shadow models one by one.
    # ==================================================================
    for shadow_id in range(n_shadows):
        sdir = outdir / f"shadow_{shadow_id:04d}"
        done = (
            (sdir / "logits.npy").exists()
            and (sdir / "keep.npy").exists()
            and (sdir / "test_acc.npy").exists()
        )
        if done:
            continue
        sdir.mkdir(parents=True, exist_ok=True)

        seed = cli.seed * 10_000 + shadow_id
        _seed_all(seed)

        # Biregular pkeep membership label.
        rng = np.random.default_rng(seed)
        n_in = int(round(pkeep * n_total))
        perm = rng.permutation(n_total)
        keep = np.zeros(n_total, dtype=bool)
        keep[perm[:n_in]] = True
        subset_idx = np.where(keep)[0]

        train_subset = Subset(pool_ds, subset_idx.tolist())
        pp_sr_subset = pp_sr_full[subset_idx].astype(np.float64)
        pp_budgets_subset = pp_budgets_full[subset_idx].astype(np.float64)

        # If a silo is entirely absent from the subset (shouldn't happen with
        # pkeep=0.5 and n_total>>2, but guard anyway), pad by promoting the
        # first record's budget so PrivacyEngine still sees 2 unique groups.
        if len(np.unique(pp_budgets_subset)) < n_silos:
            # Fall back to a single-group run for this shadow -- rare.
            pp_budgets_subset = np.ones_like(pp_budgets_subset)
            pp_sr_subset = np.full_like(pp_sr_subset, pp_sr_subset.mean())

        model = load_model(dataset, model_name=model_name, num_classes=None).to(DEVICE)
        dp_model = _train_one_shadow(
            model=model,
            train_subset=train_subset,
            pp_sr_subset=pp_sr_subset,
            pp_budgets_subset=pp_budgets_subset,
            sigma=sigma,
            epochs=epochs,
            batchsize=batchsize,
            lr=lr,
            max_grad_norm=max_grad_norm,
            accountant=accountant,
        )

        logits = _eval_pool_logits(dp_model, pool_ds, n_queries=n_queries)
        test_acc = _eval_test_acc(dp_model, test_ds)

        np.save(sdir / "logits.npy", logits)
        np.save(sdir / "keep.npy", keep)
        np.save(sdir / "test_acc.npy", np.array(test_acc, dtype=np.float64))

        print(
            f"[shadow {shadow_id + 1}/{n_shadows}] "
            f"test_acc={test_acc:.4f} keep_sum={int(keep.sum())} "
            f"sigma={sigma:.4g} mean_sr={pp_sr_subset.mean():.4g}"
        )

        del dp_model, model
        torch.cuda.empty_cache()
        gc.collect()


if __name__ == "__main__":
    main()
