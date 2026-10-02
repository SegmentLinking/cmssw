#!/usr/bin/env python3
"""
lst_compare_nodnn_eff.py

Overlay efficiency vs ΔR for baseline vs DNN-disabled (--nopt5dnn) runs.
Plots the four most relevant stages — pLS, T5, pT5, TC — with:
  solid markers = baseline
  open markers  = DNN disabled

Usage:
  python3 lst_compare_nodnn_eff.py <baseline.root> <nodnn.root> [--out out.png] [--zoom 0.05]
"""

import argparse
import numpy as np
import uproot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import beta as beta_dist

PT_CUT         = 0.8
ETA_CUT        = 4.5
VTX_Z_MAX      = 30.0
VTX_R_MAX      = 2.5
GENJET_PT_MIN  = 1000.0
GENJET_ETA_MAX = 2.5
N_BINS = 50
DR_LO  = 0.0
DR_HI  = 0.1

STAGES = [
    ("pLS", "sim_plsIdxAll",  "#9467bd", "P", 5),
    ("T5",  "sim_t5IdxAll",   "#ff7f0e", "^", 5),
    ("pT5", "sim_pt5IdxAll",  "#1f77b4", "s", 5),
    ("TC",  None,             "black",   "o", 6),
]


def clopper_pearson(k, n, cl=0.683):
    alpha = 1.0 - cl
    lo = np.where(k > 0, beta_dist.ppf(alpha / 2,     k,     n - k + 1), 0.0)
    hi = np.where(k < n, beta_dist.ppf(1 - alpha / 2, k + 1, n - k),     1.0)
    return lo, hi


def compute_eff(ntuple_path):
    tree = uproot.open(ntuple_path)["tree"]
    branches = [
        "sim_pt", "sim_eta", "sim_q", "sim_vx", "sim_vy", "sim_vz",
        "sim_genjet_deltaR", "sim_genjet_idx",
        "genjet_pt", "genjet_eta",
        "sim_plsIdxAll", "sim_t5IdxAll", "sim_pt5IdxAll",
        "sim_tcIdx",
    ]
    data = {b: tree[b].array(library="np") for b in branches}
    n_events = len(data["sim_pt"])

    dr_edges   = np.linspace(DR_LO, DR_HI, N_BINS + 1)
    dr_centers = 0.5 * (dr_edges[:-1] + dr_edges[1:])
    num = np.zeros((len(STAGES), N_BINS))
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
        tc_matched = np.array(data["sim_tcIdx"][ievt], dtype=np.int32) >= 0

        vtx_perp   = np.sqrt(sim_vx**2 + sim_vy**2)
        gj_pt_sim  = gj_pt[np.clip(sim_gj, 0, len(gj_pt)   - 1)]
        gj_eta_sim = gj_eta[np.clip(sim_gj, 0, len(gj_eta) - 1)]

        denom_mask = (
            (sim_q != 0) &
            (sim_pt > PT_CUT) &
            (np.abs(sim_eta) < ETA_CUT) &
            (np.abs(sim_vz) < VTX_Z_MAX) &
            (vtx_perp < VTX_R_MAX) &
            (sim_gj >= 0) &
            (gj_pt_sim > GENJET_PT_MIN) &
            (np.abs(gj_eta_sim) < GENJET_ETA_MAX)
        )

        stage_matched = []
        for name, branch, *_ in STAGES:
            if branch is not None:
                stage_matched.append(
                    np.array([len(v) > 0 for v in data[branch][ievt]], dtype=bool)
                )
            else:
                stage_matched.append(tc_matched)

        for isim in np.where(denom_mask)[0]:
            dr = float(sim_dr[isim])
            if dr < DR_LO or dr >= DR_HI:
                continue
            ibin = min(int((dr - DR_LO) / (DR_HI - DR_LO) * N_BINS), N_BINS - 1)
            den[ibin] += 1
            for s, matched in enumerate(stage_matched):
                if matched[isim]:
                    num[s, ibin] += 1

    return dr_centers, num, den


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline", help="Baseline ntuple (DNN active)")
    parser.add_argument("nodnn",    help="DNN-disabled ntuple (--nopt5dnn)")
    parser.add_argument("--out",  default="eff_vs_deltaR_nodnn_compare.png")
    parser.add_argument("--zoom", type=float, default=0.05)
    args = parser.parse_args()

    print(f"Computing baseline: {args.baseline}")
    dr, num_base, den_base = compute_eff(args.baseline)
    print(f"Computing DNN-off:  {args.nodnn}")
    dr, num_nod,  den_nod  = compute_eff(args.nodnn)

    fig, ax = plt.subplots(figsize=(7, 5))

    PINK = "#e75480"

    for s, (name, _, color, marker, ms) in enumerate(STAGES):
        eff_n = np.where(den_nod > 0, num_nod[s] / den_nod, np.nan)
        lo_n, hi_n = clopper_pearson(num_nod[s], den_nod)

        ax.errorbar(dr, eff_n,
                    yerr=[eff_n - lo_n, hi_n - eff_n],
                    fmt=marker, color=PINK, markersize=ms,
                    capsize=0, linestyle="none", label=name)

    ax.legend(fontsize=8, loc="lower right")

    ax.set_xlim(0, args.zoom)
    ax.set_ylim(0, 1.05)
    ax.set_xlabel("ΔR (track — nearest GenJet)")
    ax.set_ylabel("Efficiency")
    ax.grid(True, alpha=0.3)
    ax.set_title("LST Efficiency vs ΔR — DNN disabled (--nopt5dnn)\n"
                 "(100 events, idealpls)")
    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"Saved → {args.out}")


if __name__ == "__main__":
    main()
