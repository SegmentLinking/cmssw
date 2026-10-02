#!/usr/bin/env python3
"""
lst_plot_eff_vs_deltaR_postAB.py

Reproduce eff_vs_deltaR_idealpls_fixed.png but with T5_lower replaced by a
"post-AfterBuild" T5 efficiency: a sim track is counted in the numerator only
if it has at least one T5 with hit-purity > 0.75 *and* t5_isDupBits & 0x01 == 0
(i.e., NOT killed by RemoveDupQuintupletsAfterBuild).

All other stages are read directly from the pre-computed LSTNumDen ROOT file
(via uproot, no PyROOT required).

Usage:
  python3 lst_plot_eff_vs_deltaR_postAB.py \\
      --numden LSTNumDen_idealpls_fixed.root \\
      --ntuple LSTNtuple_dedup_stages_v2.root \\
      --out eff_vs_deltaR_postAB.png
"""

import argparse
import numpy as np
import uproot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import beta as beta_dist

# ── selection cuts (must match createPerfNumDenHists with -J) ─────────────────
PT_CUT       = 0.8
ETA_CUT      = 4.5
VTX_Z_MAX    = 30.0
VTX_R_MAX    = 2.5
GJ_PT_MIN    = 1000.0
GJ_ETA_MAX   = 2.5
FRAC_CUT     = 0.75
N_BINS       = 50
DR_LO, DR_HI = 0.0, 0.1

# ── visual style (mirrors lst_plot_eff_vs_deltaR_stages.py) ───────────────────
STAGE_STYLE = {
    "TC":        dict(color="black",   marker="o", ms=4),
    "pT5_lower": dict(color="#1f77b4", marker="s", ms=3),
    "T5_lower":  dict(color="#ff7f0e", marker="^", ms=3),   # original (dashed)
    "T5_postAB": dict(color="#ff7f0e", marker="^", ms=3),   # post-AfterBuild (solid)
    "pT3_lower": dict(color="#2ca02c", marker="v", ms=3),
    "T3_lower":  dict(color="#d62728", marker="D", ms=3),
    "pLS_lower": dict(color="#9467bd", marker="P", ms=3),
    "LS_lower":  dict(color="#8c564b", marker="X", ms=3),
    "MD_lower":  dict(color="#7f7f7f", marker="*", ms=4),
}

NUMDEN_STAGES = [
    "MD_lower",
    "LS_lower",
    "pLS_lower",
    "T3_lower",
    "pT3_lower",
    "pT5_lower",
    "TC",
]

LEGEND_LABELS = {
    "MD_lower":  "MD",
    "LS_lower":  "LS",
    "pLS_lower": "pLS",
    "T3_lower":  "T3",
    "pT3_lower": "pT3",
    "pT5_lower": "pT5",
    "T5_postAB": "T5 (post-AfterBuild)",
    "TC":        "TC",
}


def clopper_pearson(k, n, cl=0.683):
    lo = np.where(k > 0, beta_dist.ppf((1 - cl) / 2, k, n - k + 1), 0.0)
    hi = np.where(k < n, beta_dist.ppf(1 - (1 - cl) / 2, k + 1, n - k), 1.0)
    return lo, hi


def read_numden_eff(numden_path, stage, sel="base", pdgid=0, charge=0):
    """Read numer/denom deltaR histograms and return (centers, eff, lo, hi)."""
    key_n = f"Root__{stage}_{sel}_{pdgid}_{charge}_ef_numer_deltaR"
    key_d = f"Root__{stage}_{sel}_{pdgid}_{charge}_ef_denom_deltaR"
    f = uproot.open(numden_path)
    if key_n not in f or key_d not in f:
        return None
    n_vals, edges = f[key_n].to_numpy()
    d_vals, _     = f[key_d].to_numpy()
    centers = 0.5 * (edges[:-1] + edges[1:])
    # Clopper-Pearson — treat fractional counts as rounded ints for the interval
    k = np.round(n_vals).astype(int)
    n = np.round(d_vals).astype(int)
    eff = np.where(n > 0, k / n, np.nan)
    lo, hi = clopper_pearson(k, n)
    return centers, eff, lo, hi


def compute_t5_postAB(ntuple_path):
    """
    Compute T5 efficiency vs ΔR counting only T5s that survived AfterBuild.
    A sim track passes if it has ≥1 T5 with:
      sim_t5IdxAllFrac > FRAC_CUT  AND  t5_isDupBits[t5_idx] & 0x01 == 0
    Denominator selection mirrors createPerfNumDenHists -J with base_0_0.
    """
    tree = uproot.open(ntuple_path)["tree"]
    branches = [
        "sim_pt", "sim_eta", "sim_q", "sim_vx", "sim_vy", "sim_vz",
        "sim_genjet_deltaR", "sim_genjet_idx",
        "genjet_pt", "genjet_eta",
        "sim_t5IdxAll", "sim_t5IdxAllFrac",
        "t5_isDupBits",
    ]
    data = tree.arrays(branches, library="np")
    n_events = len(data["sim_pt"])

    edges   = np.linspace(DR_LO, DR_HI, N_BINS + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    num = np.zeros(N_BINS, dtype=float)
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

        t5_idxs  = data["sim_t5IdxAll"][ievt]
        t5_fracs = data["sim_t5IdxAllFrac"][ievt]

        for isim in np.where(denom_mask)[0]:
            dr = float(sim_dr[isim])
            if dr < DR_LO or dr >= DR_HI:
                continue
            ibin = min(int((dr - DR_LO) / (DR_HI - DR_LO) * N_BINS), N_BINS - 1)
            den[ibin] += 1

            # numerator: any T5 surviving AfterBuild with frac > FRAC_CUT
            has_post_ab = False
            for idx, frac in zip(t5_idxs[isim], t5_fracs[isim]):
                if frac > FRAC_CUT and (int(t5_bits[idx]) & 0x01) == 0:
                    has_post_ab = True
                    break
            if has_post_ab:
                num[ibin] += 1

    eff = np.where(den > 0, num / den, np.nan)
    k = num.astype(int)
    n = den.astype(int)
    lo, hi = clopper_pearson(k, n)
    return centers, eff, lo, hi


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--numden", default="LSTNumDen_idealpls_fixed.root",
                        help="NumDen ROOT file for all stages except T5")
    parser.add_argument("--ntuple", default="LSTNtuple_dedup_stages_v2.root",
                        help="LST ntuple with t5_isDupBits (post-instrumentation run)")
    parser.add_argument("--out", default="eff_vs_deltaR_postAB.png")
    parser.add_argument("--zoom", type=float, default=0.05,
                        help="ΔR axis upper limit [default: 0.05]")
    parser.add_argument("--tag", default="820464a2b09")
    args = parser.parse_args()

    fig, ax = plt.subplots(figsize=(7, 5))

    # ── other stages from NumDen ───────────────────────────────────────────────
    for stage in NUMDEN_STAGES:
        result = read_numden_eff(args.numden, stage)
        if result is None:
            print(f"WARNING: {stage} not found in {args.numden}, skipping")
            continue
        centers, eff, lo, hi = result
        sty = STAGE_STYLE[stage]
        label = LEGEND_LABELS[stage]
        mask = centers <= args.zoom
        ax.errorbar(centers[mask], eff[mask],
                    yerr=[eff[mask] - lo[mask], hi[mask] - eff[mask]],
                    fmt=sty["marker"], color=sty["color"],
                    markersize=sty["ms"], capsize=0, linestyle="none", label=label)

    # ── T5 post-AfterBuild from ntuple ────────────────────────────────────────
    print(f"Computing T5 post-AfterBuild efficiency from {args.ntuple} ...")
    centers, eff, lo, hi = compute_t5_postAB(args.ntuple)
    sty = STAGE_STYLE["T5_postAB"]
    mask = centers <= args.zoom
    ax.errorbar(centers[mask], eff[mask],
                yerr=[eff[mask] - lo[mask], hi[mask] - eff[mask]],
                fmt=sty["marker"], color=sty["color"],
                markersize=sty["ms"], capsize=0, linestyle="none",
                label=LEGEND_LABELS["T5_postAB"])

    ax.set_xlim(0, args.zoom)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("ΔR (track — nearest GenJet)")
    ax.set_ylabel("Efficiency")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    ax.set_title(f"LST Efficiency vs ΔR by Stage ({args.tag})\n"
                 "T5 = post-RemoveDupQuintupletsAfterBuild")
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
