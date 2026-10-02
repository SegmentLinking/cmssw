#!/usr/bin/env python3
"""
lst_pt5_pairs.py

Replay the RemoveDupPixelQuintupletsFromMap pair logic post-hoc.
For every pair of pT5s in the same event satisfying |Δη|<0.2 AND |Δφ|<0.2
(nMatched≥7 assumed always satisfied — confirmed empirically), record:
  deta, dphi, score_i, score_j, genuine_i, genuine_j, killed_i, killed_j

Saves pair data to a .npz file and produces two diagnostic plots.

Usage:
  python3 lst_pt5_pairs.py <ntuple.root> [--nevents N] [--out PREFIX]
"""

import argparse
import numpy as np
import uproot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


def delta_phi(phi1, phi2):
    dphi = phi1 - phi2
    dphi = np.where(dphi > np.pi,  dphi - 2 * np.pi, dphi)
    dphi = np.where(dphi < -np.pi, dphi + 2 * np.pi, dphi)
    return dphi


def process_event(eta, phi, score, is_dup, sim_idx, jet_dr_sim):
    N = len(eta)
    if N < 2:
        return None

    genuine = sim_idx >= 0
    # cat: 0=genuine-survived, 1=fake-survived, 2=genuine-killed, 3=fake-killed
    cat = np.where(genuine,
                   np.where(~is_dup, 0, 2),
                   np.where(~is_dup, 1, 3)).astype(np.uint8)

    # jet ΔR per pT5: look up via simIdx; -1 for fakes
    jet_dr = np.full(N, -1.0, dtype=np.float32)
    valid = (sim_idx >= 0) & (sim_idx < len(jet_dr_sim))
    jet_dr[valid] = jet_dr_sim[sim_idx[valid]].astype(np.float32)

    deta_mat = eta[:, None] - eta[None, :]            # (N, N)
    dphi_mat = delta_phi(phi[:, None], phi[None, :])  # (N, N)

    mask = (np.abs(deta_mat) < 0.2) & (np.abs(dphi_mat) < 0.2)
    np.fill_diagonal(mask, False)
    upper = np.triu(mask, k=1)
    ii, jj = np.where(upper)

    if len(ii) == 0:
        return None

    return {
        "deta":      deta_mat[ii, jj].astype(np.float32),
        "dphi":      dphi_mat[ii, jj].astype(np.float32),
        "score_i":   score[ii].astype(np.float32),
        "score_j":   score[jj].astype(np.float32),
        "genuine_i": genuine[ii],
        "genuine_j": genuine[jj],
        "killed_i":  is_dup[ii],
        "killed_j":  is_dup[jj],
        "cat_i":     cat[ii],
        "cat_j":     cat[jj],
        "jetdR_i":   jet_dr[ii],
        "jetdR_j":   jet_dr[jj],
    }


def plot_deta_dphi(data, prefix):
    pair_types = {
        "genuine–genuine": (data["genuine_i"] & data["genuine_j"]),
        "genuine–fake":    (data["genuine_i"] ^ data["genuine_j"]),
        "fake–fake":       (~data["genuine_i"] & ~data["genuine_j"]),
    }

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=True)
    fig.suptitle("|Δη| vs |Δφ| for pT5 competitor pairs", fontsize=12)

    bins = np.linspace(0, 0.2, 61)

    for ax, (label, mask) in zip(axes, pair_types.items()):
        deta_sel = np.abs(data["deta"][mask])
        dphi_sel = np.abs(data["dphi"][mask])
        n = int(mask.sum())
        if n == 0:
            ax.set_title(f"{label}\n(empty)")
            continue
        h, xe, ye, img = ax.hist2d(deta_sel, dphi_sel, bins=[bins, bins],
                                    norm=LogNorm(), cmap="viridis")
        fig.colorbar(img, ax=ax, label="pair count (log)")
        ax.set_title(f"{label}  (N={n:,})")
        ax.set_xlabel("|Δη|")
        ax.set_ylabel("|Δφ|")

    outpath = f"{prefix}_deta_dphi.png"
    plt.savefig(outpath, dpi=150)
    print(f"Saved → {outpath}")
    plt.close()


def plot_score_ratio(data, prefix):
    # Directed: identify winner (lower score) and loser (higher score) in each pair
    # Tiebreakers (equal score) are skipped since we can't determine direction without indices
    si = data["score_i"].astype(np.float64)
    sj = data["score_j"].astype(np.float64)

    not_tie = si != sj
    si, sj = si[not_tie], sj[not_tie]
    gi, gj = data["genuine_i"][not_tie], data["genuine_j"][not_tie]
    ki, kj = data["killed_i"][not_tie], data["killed_j"][not_tie]

    # winner = lower score
    i_wins = si < sj
    score_winner = np.where(i_wins, si, sj)
    score_loser  = np.where(i_wins, sj, si)
    genuine_winner = np.where(i_wins, gi, gj)
    genuine_loser  = np.where(i_wins, gj, gi)

    ratio = score_loser / np.maximum(score_winner, 1e-9)
    log_ratio = np.log10(np.maximum(ratio, 1e-6))

    pair_cats = [
        ("genuine wins vs genuine", genuine_winner & genuine_loser,  "#2ca02c"),
        ("genuine wins vs fake",    genuine_winner & ~genuine_loser, "#17becf"),
        ("fake wins vs genuine",    ~genuine_winner & genuine_loser, "#d62728"),
        ("fake wins vs fake",       ~genuine_winner & ~genuine_loser,"#ff7f0e"),
    ]

    fig, ax = plt.subplots(figsize=(10, 6))
    bins = np.linspace(-2, 6, 80)
    for label, mask, color in pair_cats:
        vals = log_ratio[mask]
        if len(vals) == 0:
            continue
        ax.hist(vals, bins=bins, histtype="stepfilled", alpha=0.45,
                color=color, edgecolor=color, linewidth=1.2,
                label=f"{label} (N={int(mask.sum()):,})")

    ax.set_xlabel("log₁₀(score_loser / score_winner)", fontsize=12)
    ax.set_ylabel("pair count", fontsize=12)
    ax.set_title("Score ratio for pT5 competitor pairs (non-tie only)", fontsize=12)
    ax.legend(fontsize=9)
    plt.tight_layout()

    outpath = f"{prefix}_score_ratio.png"
    plt.savefig(outpath, dpi=150)
    print(f"Saved → {outpath}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("ntuple", help="LSTNtuple ROOT file")
    parser.add_argument("--nevents", type=int, default=-1, help="First N events (default: all)")
    parser.add_argument("--out", default="pt5_pairs", help="Output prefix (default: pt5_pairs)")
    args = parser.parse_args()

    tree = uproot.open(args.ntuple)["tree"]
    entry_stop = None if args.nevents < 0 else args.nevents
    branches = ["pT5_eta", "pT5_phi", "pT5_score", "pT5_isDupReco", "pT5_simIdx",
                "sim_genjet_deltaR"]
    arrs = {b: tree[b].array(library="np", entry_stop=entry_stop) for b in branches}

    nevents = len(arrs["pT5_eta"])
    print(f"Processing {nevents} events from {args.ntuple} ...")

    accum = {k: [] for k in ("deta","dphi","score_i","score_j",
                              "genuine_i","genuine_j","killed_i","killed_j","cat_i","cat_j",
                              "jetdR_i","jetdR_j")}
    total_pairs = 0

    for iev in range(nevents):
        eta        = arrs["pT5_eta"][iev].astype(np.float32)
        phi        = arrs["pT5_phi"][iev].astype(np.float32)
        score      = arrs["pT5_score"][iev].astype(np.float32)
        is_dup     = arrs["pT5_isDupReco"][iev].astype(bool)
        simidx     = arrs["pT5_simIdx"][iev].astype(np.int32)
        jet_dr_sim = arrs["sim_genjet_deltaR"][iev].astype(np.float32)

        result = process_event(eta, phi, score, is_dup, simidx, jet_dr_sim)
        if result is None:
            continue
        for k in accum:
            accum[k].append(result[k])
        total_pairs += len(result["deta"])

        if (iev + 1) % 10 == 0:
            print(f"  event {iev+1}/{nevents}, pairs so far: {total_pairs:,}")

    print(f"Total pairs in window: {total_pairs:,}")

    data = {k: np.concatenate(v) for k, v in accum.items()}

    npz_path = f"{args.out}.npz"
    np.savez_compressed(npz_path, **data)
    print(f"Saved pair data → {npz_path}")

    plot_deta_dphi(data, args.out)
    plot_score_ratio(data, args.out)


if __name__ == "__main__":
    main()
