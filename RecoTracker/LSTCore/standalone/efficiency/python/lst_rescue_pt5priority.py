#!/usr/bin/env python3
"""
lst_rescue_pt5priority.py

Plots efficiency vs ΔR for all LST reconstruction stages, matching the style of
eff_vs_deltaR_idealpls_fixed.png, and overlays a "TC + rescued" series showing
what happens if isPT5-priority-killed T5s (isDupBits & 0x08, triedInPT5, ~partOfPT5)
are promoted to TCs.

Denominator matches createPerfNumDenHists / performance.cc:
  |q| != 0, pT > 0.8, |eta| < 4.5, |vtx_z| < 30 cm, vtx_perp < 2.5 cm,
  genJetPt > 1000 GeV, |genJetEta| < 2.5

Stage numerators (all use >75% hit-purity match, same as performance.cc):
  MD, LS, pLS, T3, pT3, T5, pT5 — any match in sim_*IdxAll (already filtered >0.75)
  TC                             — sim_tcIdx >= 0
  TC + rescued                   — TC OR rescued bit3 T5 in sim_t5IdxAll

Usage:
  python3 lst_rescue_pt5priority.py <ntuple.root> [--out rescued_eff.png] [--zoom 0.05]
Usage:
  python3 lst_rescue_pt5priority.py <ntuple.root> [--out rescued_eff.png] [--zoom 0.05]
  python3 lst_rescue_pt5priority.py <ntuple.root> --all-btc [--out rescued_eff_629.png]
    --all-btc: rescue ALL tried-but-failed BeforeTC kills (629 in jet core), not just isPT5-priority (421)
"""

import sys
import argparse
import numpy as np
import uproot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import beta as beta_dist

B_BTC_PT5 = 0x08          # bit3: isPT5-priority decisive kill (421 in jet core)
B_BTC_ANY = 0x06          # bit1|bit2: any geometric BeforeTC kill; bit3 always co-occurs with one of these

PT_CUT         = 0.8
ETA_CUT        = 4.5
VTX_Z_MAX      = 30.0
VTX_R_MAX      = 2.5
GENJET_PT_MIN  = 1000.0
GENJET_ETA_MAX = 2.5

N_BINS = 50
DR_LO  = 0.0
DR_HI  = 0.1

# Matches _STAGE_STYLE in lst_plot_eff_vs_deltaR_stages.py
STAGES = [
    ("MD",         "sim_mdIdxAll",  "#7f7f7f", "*", 4),
    ("LS",         "sim_lsIdxAll",  "#8c564b", "X", 3),
    ("pLS",        "sim_plsIdxAll", "#9467bd", "P", 3),
    ("T3",         "sim_t3IdxAll",  "#d62728", "D", 3),
    ("pT3",        "sim_pt3IdxAll", "#2ca02c", "v", 3),
    ("T5",         "sim_t5IdxAll",  "#ff7f0e", "^", 3),
    ("pT5",        "sim_pt5IdxAll", "#1f77b4", "s", 3),
    ("TC",         None,            "black",   "o", 4),
    ("TC+rescued", None,            "#e377c2", "o", 4),
]


def clopper_pearson(k, n, cl=0.683):
    alpha = 1.0 - cl
    lo = np.where(k > 0, beta_dist.ppf(alpha / 2,     k,     n - k + 1), 0.0)
    hi = np.where(k < n, beta_dist.ppf(1 - alpha / 2, k + 1, n - k),     1.0)
    return lo, hi


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("ntuple")
    parser.add_argument("--out",     default=None)
    parser.add_argument("--zoom",    type=float, default=0.05)
    parser.add_argument("--all-btc", action="store_true",
                        help="rescue all tried-but-failed BeforeTC kills (629), not just isPT5-priority (421)")
    args = parser.parse_args()
    if args.out is None:
        args.out = "rescued_eff_629.png" if args.all_btc else "rescued_eff.png"

    tree = uproot.open(args.ntuple)["tree"]
    branches = [
        "t5_isDupBits", "t5_triedInPT5", "t5_partOfPT5",
        "sim_tcIdx",
        "sim_mdIdxAll", "sim_lsIdxAll", "sim_t3IdxAll",
        "sim_plsIdxAll", "sim_pt3IdxAll", "sim_t5IdxAll", "sim_pt5IdxAll",
        "sim_pt", "sim_eta", "sim_q",
        "sim_vx", "sim_vy", "sim_vz",
        "sim_genjet_deltaR", "sim_genjet_idx",
        "genjet_pt", "genjet_eta",
    ]
    data = {b: tree[b].array(library="np") for b in branches}
    n_events = len(data["sim_pt"])
    print(f"Events: {n_events}")

    dr_edges   = np.linspace(DR_LO, DR_HI, N_BINS + 1)
    dr_centers = 0.5 * (dr_edges[:-1] + dr_edges[1:])

    # Accumulators: one row per stage
    n_stages = len(STAGES)
    num = np.zeros((n_stages, N_BINS))
    den = np.zeros(N_BINS)

    for ievt in range(n_events):
        sim_pt  = np.array(data["sim_pt"][ievt],  dtype=float)
        sim_eta = np.array(data["sim_eta"][ievt], dtype=float)
        sim_q   = np.array(data["sim_q"][ievt],   dtype=np.int32)
        sim_vx  = np.array(data["sim_vx"][ievt],  dtype=float)
        sim_vy  = np.array(data["sim_vy"][ievt],  dtype=float)
        sim_vz  = np.array(data["sim_vz"][ievt],  dtype=float)
        sim_dr  = np.array(data["sim_genjet_deltaR"][ievt], dtype=float)
        sim_gj  = np.array(data["sim_genjet_idx"][ievt],    dtype=np.int32)
        gj_pt   = np.array(data["genjet_pt"][ievt],  dtype=float)
        gj_eta  = np.array(data["genjet_eta"][ievt], dtype=float)

        vtx_perp   = np.sqrt(sim_vx**2 + sim_vy**2)
        gj_pt_sim  = gj_pt[np.clip(sim_gj, 0, len(gj_pt) - 1)]
        gj_eta_sim = gj_eta[np.clip(sim_gj, 0, len(gj_eta) - 1)]

        denom_mask = (
            (sim_q != 0) &
            (sim_pt > PT_CUT) &
            (np.abs(sim_eta) < ETA_CUT) &
            (np.abs(sim_vz) < VTX_Z_MAX) &
            (vtx_perp < VTX_R_MAX) &
            (gj_pt_sim > GENJET_PT_MIN) &
            (np.abs(gj_eta_sim) < GENJET_ETA_MAX)
        )

        # Baseline TC match (flat int per sim track)
        tc_matched = np.array(data["sim_tcIdx"][ievt], dtype=np.int32) >= 0

        # Rescued T5s: tried pT5 matching but failed, killed by BeforeTC
        bits  = np.array(data["t5_isDupBits"][ievt],  dtype=np.int32)
        tried = np.array(data["t5_triedInPT5"][ievt], dtype=bool)
        pt5   = np.array(data["t5_partOfPT5"][ievt],  dtype=bool)
        rescue_bits = B_BTC_ANY if args.all_btc else B_BTC_PT5
        rescued_set = set(np.where(((bits & rescue_bits) > 0) & tried & ~pt5)[0])

        # Per-stage match vectors (jagged branches: found if any entry in IdxAll)
        stage_matched = []
        for name, branch, *_ in STAGES:
            if branch is not None:
                stage_matched.append(
                    np.array([len(v) > 0 for v in data[branch][ievt]], dtype=bool)
                )
            elif name == "TC":
                stage_matched.append(tc_matched)
            else:  # TC+rescued
                t5_rescue = np.array([
                    bool(rescued_set.intersection(v)) for v in data["sim_t5IdxAll"][ievt]
                ], dtype=bool)
                stage_matched.append(tc_matched | t5_rescue)

        # Fill bins for denominator sim tracks
        denom_idx = np.where(denom_mask)[0]
        for isim in denom_idx:
            dr = float(sim_dr[isim])
            if dr < DR_LO or dr >= DR_HI:
                continue
            ibin = min(int((dr - DR_LO) / (DR_HI - DR_LO) * N_BINS), N_BINS - 1)
            den[ibin] += 1
            for s, matched in enumerate(stage_matched):
                if matched[isim]:
                    num[s, ibin] += 1

    rescued_label = "TC+629 tried-failed" if args.all_btc else "TC+421 pT5-priority"
    title_tag     = "all tried-failed BeforeTC T5s (629)" if args.all_btc else "isPT5-priority rescued T5s (421)"

    # Plot
    fig, ax = plt.subplots(figsize=(7, 5))

    for s, (name, _, color, marker, ms) in enumerate(STAGES):
        label = rescued_label if name == "TC+rescued" else name
        eff = np.where(den > 0, num[s] / den, np.nan)
        lo, hi = clopper_pearson(num[s], den)
        yerr = [eff - lo, hi - eff]
        ax.errorbar(dr_centers, eff, yerr=yerr,
                    fmt=marker, color=color, markersize=ms, capsize=0,
                    linestyle="none", label=label)

    ax.set_xlim(0, args.zoom)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("ΔR (track — nearest GenJet)")
    ax.set_ylabel("Efficiency")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    ax.set_title(f"LST Efficiency vs ΔR — + {title_tag}\n(100 events, idealpls)")
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"Saved → {args.out}")


if __name__ == "__main__":
    main()
