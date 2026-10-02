#!/usr/bin/env python3
"""
lst_check_pt5_matching_density.py

Diagnostic for whether pT5 (pLS+T5) matching itself breaks down more often in
dense regions (small deltaR to nearest GenJet), independent of the already-understood
upstream pLS-supply deficit. For each selected sim track where an eligible pLS (quad,
not flagged duplicate) AND a true T5 both exist ('ingredients present'), checks whether a
true pT5 actually formed. Plots, vs deltaR:
  (1) fraction of 'ingredients present' sim tracks where no pT5 formed
  (2) pT5_isFake rate among pT5s that did form
  (3) mean nCompetingT5 = number of pT5s sharing the same pLS as the truth-matched one
      (distinguishes 'pLS pairs with NO T5' from 'pLS pairs with wrong T5')

Reads the flat LSTNtuple tree produced with '--allobj --jet --idealpls'.
Denominator mirrors efficiency/src/performance.cc deltaR plot exactly.

Usage:
  python3 lst_check_pt5_matching_density.py <ntuple.root> [options]
"""

import argparse
from collections import Counter

import numpy as np
import uproot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import beta as beta_dist


def clopper_pearson(k, n, cl=0.683):
    alpha = 1.0 - cl
    lo = np.where(k > 0, beta_dist.ppf(alpha / 2,     k,     n - k + 1), 0.0)
    hi = np.where(k < n, beta_dist.ppf(1 - alpha / 2, k + 1, n - k),     1.0)
    return lo, hi


def main():
    parser = argparse.ArgumentParser(
        description="pT5 matching-density diagnostic (uproot version, no PyROOT)"
    )
    parser.add_argument("input", help="LSTNtuple*.root (must have --allobj --jet branches)")
    parser.add_argument("--pt-cut",        type=float, default=0.8,
                        help="sim track pt cut [DEFAULT=0.8]")
    parser.add_argument("--eta-cut",       type=float, default=4.5,
                        help="sim track |eta| cut [DEFAULT=4.5]")
    parser.add_argument("--genjetpt-cut",  type=float, default=1000.0,
                        help="GenJet pt cut [DEFAULT=1000]")
    parser.add_argument("--genjeteta-cut", type=float, default=2.5,
                        help="GenJet |eta| cut [DEFAULT=2.5]")
    parser.add_argument("--matchfrac",     type=float, default=0.75,
                        help="hit-purity match fraction threshold [DEFAULT=0.75]")
    parser.add_argument("--zoom",          type=float, default=0.05,
                        help="deltaR cutoff for the zoomed left panel [DEFAULT=0.05]")
    parser.add_argument("--nbins",         type=int,   default=50,
                        help="number of deltaR bins [DEFAULT=50]")
    parser.add_argument("--max-deltar",    type=float, default=0.1,
                        help="max deltaR histogrammed [DEFAULT=0.1]")
    parser.add_argument("-o", "--output",  default="pt5_matching_density_idealpls.png",
                        help="output image path [DEFAULT=pt5_matching_density_idealpls.png]")
    args = parser.parse_args()

    VTX_Z_MAX = 30.0
    VTX_R_MAX = 2.5
    NBINS     = args.nbins
    DR_LO     = 0.0
    DR_HI     = args.max_deltar

    tree = uproot.open(args.input)["tree"]
    branches = [
        "sim_pt", "sim_eta", "sim_q", "sim_vx", "sim_vy", "sim_vz",
        "sim_genjet_deltaR", "sim_genjet_idx",
        "genjet_pt", "genjet_eta",
        "sim_plsIdxAll", "sim_plsIdxAllFrac",
        "sim_t5IdxAll", "sim_t5IdxAllFrac",
        "sim_pt5IdxAll", "sim_pt5IdxAllFrac",
        "pLS_isQuad", "pLS_isDuplicate",
        "pT5_isFake", "pT5_plsIdx",
        "t5_isDupBits", "t5_triedInPT5",
    ]
    data = {b: tree[b].array(library="np") for b in branches}
    n_events = len(data["sim_pt"])
    print(f"Events: {n_events}")

    dr_edges   = np.linspace(DR_LO, DR_HI, NBINS + 1)
    dr_centers = 0.5 * (dr_edges[:-1] + dr_edges[1:])

    h_denom_ing   = np.zeros(NBINS)
    h_numer_nopt5 = np.zeros(NBINS)
    h_denom_pt5   = np.zeros(NBINS)
    h_numer_fake  = np.zeros(NBINS)
    sum_ncomp     = np.zeros(NBINS)
    cnt_ncomp     = np.zeros(NBINS)

    n_ingredients       = 0
    n_ingredients_nopt5 = 0
    bucket_counts = {0:   {"denom": 0, "nopt5": 0},
                     "1+":{"denom": 0, "nopt5": 0}}
    # For nCompetingT5==0 failures: connectivity exclusion vs. cut failure
    n_nc0_tried_cut_fail   = 0  # truth T5 tried (triedInPT5=True) but no pT5 → cut failed
    n_nc0_not_tried_connex = 0  # truth T5 NOT tried (triedInPT5=False) → connectivity excluded

    for ievt in range(n_events):
        sim_pt  = np.array(data["sim_pt"][ievt],  dtype=float)
        sim_eta = np.array(data["sim_eta"][ievt], dtype=float)
        sim_q   = np.array(data["sim_q"][ievt],   dtype=np.int32)
        sim_vx  = np.array(data["sim_vx"][ievt],  dtype=float)
        sim_vy  = np.array(data["sim_vy"][ievt],  dtype=float)
        sim_vz  = np.array(data["sim_vz"][ievt],  dtype=float)
        dr_arr  = np.array(data["sim_genjet_deltaR"][ievt], dtype=float)
        gj_arr  = np.array(data["sim_genjet_idx"][ievt],    dtype=np.int32)
        gj_pt   = np.array(data["genjet_pt"][ievt],  dtype=float)
        gj_eta  = np.array(data["genjet_eta"][ievt], dtype=float)

        pls_isquad = np.array(data["pLS_isQuad"][ievt],      dtype=bool)
        pls_isdup  = np.array(data["pLS_isDuplicate"][ievt],  dtype=bool)
        pt5_isfake = np.array(data["pT5_isFake"][ievt],       dtype=np.int32)
        pt5_plsidx = np.array(data["pT5_plsIdx"][ievt],       dtype=np.int32)
        t5_ab_dead = (np.array(data["t5_isDupBits"][ievt], dtype=np.int32) & 0x01) > 0
        t5_tried   = np.array(data["t5_triedInPT5"][ievt], dtype=bool)

        competitors_per_pls = Counter(pt5_plsidx.tolist())

        vtx_perp   = np.sqrt(sim_vx**2 + sim_vy**2)
        gj_pt_sim  = gj_pt[np.clip(gj_arr, 0, len(gj_pt)   - 1)]
        gj_eta_sim = gj_eta[np.clip(gj_arr, 0, len(gj_eta) - 1)]

        denom_mask = (
            (sim_q != 0) &
            (sim_pt > args.pt_cut) &
            (np.abs(sim_eta) < args.eta_cut) &
            (np.abs(sim_vz) < VTX_Z_MAX) &
            (vtx_perp < VTX_R_MAX) &
            (gj_arr >= 0) &
            (gj_pt_sim > args.genjetpt_cut) &
            (np.abs(gj_eta_sim) < args.genjeteta_cut)
        )

        for isim in np.where(denom_mask)[0]:
            dr = float(dr_arr[isim])
            if dr < DR_LO or dr >= DR_HI:
                continue
            ibin = min(int((dr - DR_LO) / (DR_HI - DR_LO) * NBINS), NBINS - 1)

            # Eligible pLS: first entry with frac>threshold, isQuad, not isDuplicate
            eligible_pls = False
            eligible_pls_idx = -1
            for i, frac in zip(data["sim_plsIdxAll"][ievt][isim],
                               data["sim_plsIdxAllFrac"][ievt][isim]):
                if frac > args.matchfrac and pls_isquad[i] and not pls_isdup[i]:
                    eligible_pls = True
                    eligible_pls_idx = int(i)
                    break

            has_t5 = any(
                f > args.matchfrac and not t5_ab_dead[i]
                for i, f in zip(data["sim_t5IdxAll"][ievt][isim],
                                data["sim_t5IdxAllFrac"][ievt][isim])
            )

            # Has pT5: first entry with frac>threshold
            has_pt5 = False
            pt5_idx_matched = -1
            for i, frac in zip(data["sim_pt5IdxAll"][ievt][isim],
                               data["sim_pt5IdxAllFrac"][ievt][isim]):
                if frac > args.matchfrac:
                    has_pt5 = True
                    pt5_idx_matched = int(i)
                    break

            if eligible_pls and has_t5:
                h_denom_ing[ibin] += 1
                n_ingredients += 1
                n_competing = competitors_per_pls.get(eligible_pls_idx, 0)
                sum_ncomp[ibin] += n_competing
                cnt_ncomp[ibin] += 1
                bucket = 0 if n_competing == 0 else "1+"
                bucket_counts[bucket]["denom"] += 1
                if not has_pt5:
                    h_numer_nopt5[ibin] += 1
                    n_ingredients_nopt5 += 1
                    bucket_counts[bucket]["nopt5"] += 1
                    if bucket == 0:
                        # Check whether truth T5 was tried or excluded by connectivity
                        truth_t5_tried = any(
                            f > args.matchfrac and not t5_ab_dead[i] and t5_tried[i]
                            for i, f in zip(data["sim_t5IdxAll"][ievt][isim],
                                            data["sim_t5IdxAllFrac"][ievt][isim])
                        )
                        if truth_t5_tried:
                            n_nc0_tried_cut_fail += 1
                        else:
                            n_nc0_not_tried_connex += 1

            if has_pt5:
                h_denom_pt5[ibin] += 1
                if pt5_isfake[pt5_idx_matched] > 0:
                    h_numer_fake[ibin] += 1

    print(f"Sim tracks with eligible pLS + T5 ('ingredients present'): {n_ingredients}")
    if n_ingredients:
        print(f"  of which no pT5 formed: {n_ingredients_nopt5} "
              f"({n_ingredients_nopt5 / n_ingredients:.1%})")
    for bucket in (0, "1+"):
        d = bucket_counts[bucket]["denom"]
        n = bucket_counts[bucket]["nopt5"]
        label = ("nCompetingT5==0 (pLS pairs with NO T5 at all)" if bucket == 0
                 else "nCompetingT5>=1 (pLS pairs with some other T5)")
        frac = n / d if d else float("nan")
        print(f"  {label}: {d} tracks, {n} no-pT5 ({frac:.1%})")
    nc0_fail = bucket_counts[0]["nopt5"]
    if nc0_fail:
        print(f"\nnCompetingT5==0 failure breakdown ({nc0_fail} tracks):")
        print(f"  truth T5 tried (triedInPT5=True) → geometric cut failed: "
              f"{n_nc0_tried_cut_fail} ({n_nc0_tried_cut_fail/nc0_fail:.1%})")
        print(f"  truth T5 NOT tried (triedInPT5=False) → connectivity excluded: "
              f"{n_nc0_not_tried_connex} ({n_nc0_not_tried_connex/nc0_fail:.1%})")

    # Clopper-Pearson efficiency curves
    fr_nopt5 = np.where(h_denom_ing > 0, h_numer_nopt5 / h_denom_ing, np.nan)
    lo1, hi1 = clopper_pearson(h_numer_nopt5, h_denom_ing)
    yerr1 = [np.where(np.isnan(fr_nopt5), 0, fr_nopt5 - lo1),
             np.where(np.isnan(fr_nopt5), 0, hi1 - fr_nopt5)]

    fr_fake = np.where(h_denom_pt5 > 0, h_numer_fake / h_denom_pt5, np.nan)
    lo2, hi2 = clopper_pearson(h_numer_fake, h_denom_pt5)
    yerr2 = [np.where(np.isnan(fr_fake), 0, fr_fake - lo2),
             np.where(np.isnan(fr_fake), 0, hi2 - fr_fake)]

    mean_ncomp = np.where(cnt_ncomp > 0, sum_ncomp / cnt_ncomp, np.nan)

    fig, (ax_zoom, ax_full, ax_ncomp) = plt.subplots(1, 3, figsize=(17, 5))

    for ax in (ax_zoom, ax_full):
        ax.errorbar(dr_centers, fr_nopt5, yerr=yerr1, fmt="o", markersize=3,
                    capsize=0, linestyle="none", label="ingredients present, no pT5 formed")
        ax.errorbar(dr_centers, fr_fake,  yerr=yerr2, fmt="s", markersize=3,
                    capsize=0, linestyle="none", label="pT5_isFake rate (of formed pT5)")
        ax.set_xlabel("ΔR (track — nearest GenJet)")
        ax.set_ylabel("Fraction")
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    ax_zoom.set_xlim(0, args.zoom)
    ax_zoom.set_title(f"Zoom: ΔR < {args.zoom}")
    ax_full.set_xlim(0, DR_HI)
    ax_full.set_title(f"Full range ΔR < {DR_HI}")

    valid = cnt_ncomp > 0
    ax_ncomp.plot(dr_centers[valid], mean_ncomp[valid], "-o", markersize=3,
                  linewidth=1, color="tab:green")
    ax_ncomp.set_xlabel("ΔR (track — nearest GenJet)")
    ax_ncomp.set_ylabel("mean nCompetingT5 per eligible pLS")
    ax_ncomp.set_xlim(0, DR_HI)
    ax_ncomp.set_title("Candidate competition vs ΔR")
    ax_ncomp.grid(True, alpha=0.3)

    fig.suptitle("pT5 matching-density diagnostic (post-fix idealpls, 100 evt)")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(args.output, dpi=150)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
