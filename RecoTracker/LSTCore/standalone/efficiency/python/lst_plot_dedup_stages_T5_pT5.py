#!/usr/bin/env python3
"""
lst_plot_dedup_stages_T5_pT5.py

Plot T5 and pT5 efficiency vs ΔR at each dedup stage, read directly from the
LST ntuple (no NumDen file required):

  T5_lower       — any built T5 with hit-purity > 0.75 (no dedup filter)
  T5_post_AB     — surviving RemoveDupQuintupletsAfterBuild  (isDupBits & 0x01 == 0)
  T5_post_dedup  — surviving all T5 dedup (isDupBits == 0) → standalone T5-type TCs
  pT5_lower      — any built pT5 with hit-purity > 0.75 (no dedup filter)
  pT5_post_dedup — surviving RemoveDupPixelQuintupletsFromMap (isDupReco == 0)

The gaps between curves quantify losses at each dedup stage.

Usage:
  python3 lst_plot_dedup_stages_T5_pT5.py \\
      Ntuple-files/LSTNtuple_idealpls_mastercuts_100evt_v2.root \\
      -o eff_vs_deltaR_dedup_stages_T5_pT5.png
"""

import argparse
import numpy as np
import uproot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import beta as beta_dist

# ── selection cuts (match createPerfNumDenHists -J base_0_0) ──────────────────
PT_CUT      = 0.8
ETA_CUT     = 4.5
VTX_Z_MAX   = 30.0
VTX_R_MAX   = 2.5
GJ_PT_MIN   = 1000.0
GJ_ETA_MAX  = 2.5
FRAC_CUT    = 0.75
N_BINS      = 50
DR_LO       = 0.0
DR_HI       = 0.1


def clopper_pearson(k, n, cl=0.683):
    lo = np.where(k > 0, beta_dist.ppf((1 - cl) / 2, k, n - k + 1), 0.0)
    hi = np.where(k < n, beta_dist.ppf(1 - (1 - cl) / 2, k + 1, n - k), 1.0)
    return lo, hi


def compute_curves(ntuple_path, nevents=None):
    """
    Returns a dict of {curve_name: (centers, eff, lo, hi)}.
    Curves: T5_lower, T5_post_AB, T5_post_dedup, pT5_lower, pT5_post_dedup.
    """
    tree = uproot.open(ntuple_path)["tree"]
    branches = [
        "sim_pt", "sim_eta", "sim_q", "sim_vx", "sim_vy", "sim_vz",
        "sim_genjet_deltaR", "sim_genjet_idx",
        "genjet_pt", "genjet_eta",
        "sim_t5IdxAll", "sim_t5IdxAllFrac",
        "t5_isDupBits",
        "sim_pt5IdxAll", "sim_pt5IdxAllFrac",
        "pT5_isDupReco",
    ]
    data = tree.arrays(branches, library="np")
    n_events = len(data["sim_pt"])
    if nevents is not None:
        n_events = min(n_events, nevents)

    edges   = np.linspace(DR_LO, DR_HI, N_BINS + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])

    curve_names = ["T5_lower", "T5_post_AB", "T5_post_dedup", "pT5_lower", "pT5_post_dedup"]
    num = {c: np.zeros(N_BINS, dtype=float) for c in curve_names}
    den = np.zeros(N_BINS, dtype=float)

    for ievt in range(n_events):
        sim_pt  = data["sim_pt"][ievt].astype(float)
        sim_eta = data["sim_eta"][ievt].astype(float)
        sim_q   = data["sim_q"][ievt].astype(np.int32)
        sim_vx  = data["sim_vx"][ievt].astype(float)
        sim_vy  = data["sim_vy"][ievt].astype(float)
        sim_vz  = data["sim_vz"][ievt].astype(float)
        sim_dr  = data["sim_genjet_deltaR"][ievt].astype(float)
        sim_gj  = data["sim_genjet_idx"][ievt].astype(np.int32)
        gj_pt   = data["genjet_pt"][ievt].astype(float)
        gj_eta  = data["genjet_eta"][ievt].astype(float)
        t5_bits = data["t5_isDupBits"][ievt].astype(np.int32)
        pt5_dup = data["pT5_isDupReco"][ievt].astype(np.int32)

        vtx_perp   = np.sqrt(sim_vx**2 + sim_vy**2)
        gj_idx_clp = np.clip(sim_gj, 0, len(gj_pt) - 1)
        gj_pt_sim  = gj_pt[gj_idx_clp]
        gj_eta_sim = gj_eta[gj_idx_clp]

        denom_mask = (
            (sim_q != 0) &
            (sim_pt > PT_CUT) &
            (np.abs(sim_eta) < ETA_CUT) &
            (np.abs(sim_vz) < VTX_Z_MAX) &
            (vtx_perp < VTX_R_MAX) &
            (sim_gj >= 0) &
            (gj_pt_sim > GJ_PT_MIN) &
            (np.abs(gj_eta_sim) < GJ_ETA_MAX)
        )

        t5_idxs_per_sim  = data["sim_t5IdxAll"][ievt]
        t5_fracs_per_sim = data["sim_t5IdxAllFrac"][ievt]
        pt5_idxs_per_sim  = data["sim_pt5IdxAll"][ievt]
        pt5_fracs_per_sim = data["sim_pt5IdxAllFrac"][ievt]

        for isim in np.where(denom_mask)[0]:
            dr = float(sim_dr[isim])
            if dr < DR_LO or dr >= DR_HI:
                continue
            ibin = min(int((dr - DR_LO) / (DR_HI - DR_LO) * N_BINS), N_BINS - 1)
            den[ibin] += 1

            # T5 curves
            t5_found_lower    = False
            t5_found_post_ab  = False
            t5_found_post_all = False
            for idx, frac in zip(t5_idxs_per_sim[isim], t5_fracs_per_sim[isim]):
                if frac <= FRAC_CUT:
                    continue
                bits = int(t5_bits[idx])
                t5_found_lower = True
                if (bits & 0x01) == 0:
                    t5_found_post_ab = True
                if bits == 0:
                    t5_found_post_all = True

            if t5_found_lower:
                num["T5_lower"][ibin] += 1
            if t5_found_post_ab:
                num["T5_post_AB"][ibin] += 1
            if t5_found_post_all:
                num["T5_post_dedup"][ibin] += 1

            # pT5 curves
            pt5_found_lower    = False
            pt5_found_post_dup = False
            for idx, frac in zip(pt5_idxs_per_sim[isim], pt5_fracs_per_sim[isim]):
                if frac <= FRAC_CUT:
                    continue
                pt5_found_lower = True
                if int(pt5_dup[idx]) == 0:
                    pt5_found_post_dup = True

            if pt5_found_lower:
                num["pT5_lower"][ibin] += 1
            if pt5_found_post_dup:
                num["pT5_post_dedup"][ibin] += 1

    results = {}
    for c in curve_names:
        k = num[c].astype(int)
        n = den.astype(int)
        eff = np.where(n > 0, num[c] / den, np.nan)
        lo, hi = clopper_pearson(k, n)
        results[c] = (centers, eff, lo, hi)
    return results


# ── visual style ───────────────────────────────────────────────────────────────
CURVE_STYLE = {
    "T5_lower":       dict(color="#ff7f0e", marker="^", ms=3, ls=":",  lw=0.8),
    "T5_post_AB":     dict(color="#ff7f0e", marker="^", ms=3, ls="--", lw=0.8),
    "T5_post_dedup":  dict(color="#ff7f0e", marker="^", ms=3, ls="-",  lw=1.2),
    "pT5_lower":      dict(color="#1f77b4", marker="s", ms=3, ls=":",  lw=0.8),
    "pT5_post_dedup": dict(color="#1f77b4", marker="s", ms=3, ls="-",  lw=1.2),
}

LEGEND_LABELS = {
    "T5_lower":       "T5 built (lower)",
    "T5_post_AB":     "T5 post-AfterBuild",
    "T5_post_dedup":  "T5 post-all-dedup",
    "pT5_lower":      "pT5 built (lower)",
    "pT5_post_dedup": "pT5 post-dedup",
}

DRAW_ORDER = ["T5_lower", "T5_post_AB", "T5_post_dedup", "pT5_lower", "pT5_post_dedup"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ntuple", help="LST ntuple ROOT file")
    parser.add_argument("-o", "--out", default="eff_vs_deltaR_dedup_stages_T5_pT5.png")
    parser.add_argument("--nevents", type=int, default=None, help="Max events to process")
    parser.add_argument("--zoom", type=float, default=0.05, help="ΔR axis upper limit")
    parser.add_argument("--tag", default="")
    args = parser.parse_args()

    print(f"Reading {args.ntuple} ...")
    results = compute_curves(args.ntuple, nevents=args.nevents)

    fig, ax = plt.subplots(figsize=(7, 5))

    for c in DRAW_ORDER:
        centers, eff, lo, hi = results[c]
        sty = CURVE_STYLE[c]
        mask = centers <= args.zoom
        ax.errorbar(
            centers[mask], eff[mask],
            yerr=[eff[mask] - lo[mask], hi[mask] - eff[mask]],
            fmt=sty["marker"],
            color=sty["color"],
            markersize=sty["ms"],
            capsize=0,
            linestyle=sty["ls"],
            linewidth=sty["lw"],
            label=LEGEND_LABELS[c],
        )

    ax.set_xlim(0, args.zoom)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("ΔR (track — nearest GenJet)")
    ax.set_ylabel("Efficiency")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    title = "T5 & pT5 Efficiency vs ΔR — Dedup Stages"
    if args.tag:
        title += f"\n{args.tag}"
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
