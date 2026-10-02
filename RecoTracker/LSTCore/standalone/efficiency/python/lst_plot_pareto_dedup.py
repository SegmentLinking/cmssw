#!/usr/bin/env python3
"""
lst_plot_pareto_dedup.py

Analyze the dedup-cut tuning sweep log (efficiency/tune_dedup_log.txt) as a
Pareto study: plot jet-core TC efficiency vs fake rate, highlight the
non-dominated frontier, and print a ranked table of frontier points with their
parameter deltas from baseline.

A point is Pareto-optimal if no other point has both >= efficiency AND
<= fake rate. Loosening the dedup cuts raises efficiency but also fakes, so the
frontier traces the achievable efficiency/purity trade-off; pick the knee.

Usage:
  python3 efficiency/python/lst_plot_pareto_dedup.py \
      [efficiency/tune_dedup_log.txt] \
      [--metric core010_eff] [--fake fakerate] [--out pareto_dedup.png]
"""

import argparse
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# The 7 in-scope tuning params as logged (log key -> short display name).
PARAM_KEYS = {
    "AB_nm": "AB_NM",
    "BTC_nm": "BTC_NM",
    "PT5_nm": "PT5_NM",
    "BTC_dr2t": "BTC_DR2T",
    "BTC_d2l": "BTC_D2L",
    "BTC_dr2l": "BTC_DR2L",
    "BTC_d2t": "BTC_D2T",
}


def parse_log(path, stages=None):
    """Return list of dict rows (one per completed run) from the sweep log.

    If `stages` is a set, only rows whose `stage=` is in it are kept (used to
    exclude legacy gate-param sweeps from the 7-parameter Pareto analysis).
    """
    rows = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "DONE" in line or "RUN_FAILED" in line:
                continue
            kv = {}
            for tok in line.split():
                if "=" in tok:
                    k, v = tok.split("=", 1)
                    kv[k] = v
            # A valid data row must carry the metric and the fake rate.
            if "core010_eff" not in kv or "fakerate" not in kv:
                continue
            if stages is not None and kv.get("stage") not in stages:
                continue
            rows.append(kv)
    return rows


def to_float(s):
    try:
        return float(s)
    except (TypeError, ValueError):
        return float("nan")


def pareto_mask(eff, fake):
    """Boolean mask of non-dominated points: maximize eff, minimize fake."""
    n = len(eff)
    keep = np.ones(n, dtype=bool)
    for i in range(n):
        if np.isnan(eff[i]) or np.isnan(fake[i]):
            keep[i] = False
            continue
        for j in range(n):
            if i == j or np.isnan(eff[j]) or np.isnan(fake[j]):
                continue
            # j dominates i if j is >= in eff and <= in fake, strictly better in one.
            if (eff[j] >= eff[i] and fake[j] <= fake[i] and
                    (eff[j] > eff[i] or fake[j] < fake[i])):
                keep[i] = False
                break
    return keep


def find_baseline(rows):
    for r in rows:
        if r.get("label") in ("base", "master"):
            return r
    return None


def deltas_from_baseline(row, base):
    """List 'KEY:base->val' for params that differ from baseline."""
    if base is None:
        return "(no baseline row)"
    out = []
    for key, disp in PARAM_KEYS.items():
        v, b = row.get(key), base.get(key)
        if v is not None and b is not None and v != b:
            out.append(f"{disp}:{b}->{v}")
    return ", ".join(out) if out else "(baseline)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log", nargs="?",
                    default="efficiency/tune_dedup_log.txt")
    ap.add_argument("--metric", default="core010_eff",
                    help="efficiency key to maximize (default core010_eff)")
    ap.add_argument("--fake", default="fakerate",
                    help="fake-rate key to minimize (default fakerate)")
    ap.add_argument("--out", default="pareto_dedup.png")
    ap.add_argument("--stage", default="baseline,oat7,frontier",
                    help="comma-separated stages to include (default the 7-param "
                         "study stages; pass 'all' to include legacy gate sweeps)")
    args = ap.parse_args()

    stages = None if args.stage.lower() == "all" else set(args.stage.split(","))
    rows = parse_log(args.log, stages)
    if not rows:
        sys.exit(f"No usable data rows in {args.log}")

    eff = np.array([to_float(r.get(args.metric)) for r in rows])
    fake = np.array([to_float(r.get(args.fake)) for r in rows])
    dupfake = np.array([to_float(r.get("dupfakerate")) for r in rows])

    base = find_baseline(rows)
    keep = pareto_mask(eff, fake)

    # ── ranked frontier table (best efficiency first) ─────────────────────────
    front = [(eff[i], fake[i], dupfake[i], rows[i]) for i in range(len(rows)) if keep[i]]
    front.sort(key=lambda t: -t[0])
    print(f"\nPareto frontier ({args.metric} vs {args.fake}) — {len(front)} points, best efficiency first:\n")
    be = to_float(base.get(args.metric)) if base else float("nan")
    bf = to_float(base.get(args.fake)) if base else float("nan")
    print(f"  {'#':>2}  {args.metric:>10}  {args.fake:>9}  dupfake   changes-from-baseline")
    print(f"  {'--':>2}  {'-'*10}  {'-'*9}  {'-'*7}   {'-'*40}")
    for i, (e, f, df, r) in enumerate(front):
        de = f"{e-be:+.4f}" if not np.isnan(be) else "  n/a "
        print(f"  {i:>2}  {e:>10.4f}  {f:>9.4f}  {df:>7.4f}   {deltas_from_baseline(r, base)}   (Δeff {de})")
    if base is not None:
        print(f"\n  baseline: {args.metric}={be:.4f}  {args.fake}={bf:.4f}")

    # ── two-panel scatter ─────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    for ax, xvals, xlabel in ((axes[0], fake, args.fake),
                              (axes[1], dupfake, "dupfakerate")):
        ax.scatter(xvals[~keep], eff[~keep], c="0.7", s=28, label="dominated")
        ax.scatter(xvals[keep], eff[keep], c="tab:red", s=52,
                   edgecolor="k", zorder=3, label="Pareto frontier")
        # frontier line (sort by x)
        fx = xvals[keep]
        fy = eff[keep]
        order = np.argsort(fx)
        ax.plot(fx[order], fy[order], "-", c="tab:red", alpha=0.5, zorder=2)
        if base is not None:
            ax.scatter([to_float(base.get(xlabel))], [be], marker="*",
                       s=320, c="tab:blue", edgecolor="k", zorder=4, label="baseline")
        ax.set_xlabel(xlabel)
        ax.set_ylabel(args.metric)
        ax.set_title(f"{args.metric} vs {xlabel}")
        ax.grid(alpha=0.3)
        ax.legend(loc="lower right")

    nev = rows[0].get("nevents", "?")
    fig.suptitle(f"Dedup-cut Pareto scan ({len(rows)} points, nevents={nev})")
    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
