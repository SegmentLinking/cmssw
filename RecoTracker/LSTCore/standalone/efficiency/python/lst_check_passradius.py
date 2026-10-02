#!/bin/env python
"""
Offline check of passRadiusCriterion for truth-matched (pLS, T5) pairs that fail
to form a pT5 in dense regions (small deltaR to nearest GenJet).

Background: betaOutCut and dBeta loosening experiments (4x each) had zero effect on
pT5_lower, implying the dense-region pT5 matching loss occurs BEFORE runTripletDefaultAlgoPPBB/EE
is entered. passRadiusCriterion is the first gate in runPixelTripletDefaultAlgo (called by
runPixelQuintupletDefaultAlgo), checking that the pLS pixel radius agrees with the inner T3's
fitted radius within ~16% (BBB, low-pT) to ~66% (BBB, high-pT).

For each "failing" sim track (eligible pLS + truth-matched T5, but no pT5 formed), this script
recomputes passRadiusCriterion offline using:
  - pixelRadius      = pLS_pt  * kR1GeVf   (same as LST's pixelData.ptIn * kR1GeVf)
  - pixelRadiusError = pLS_ptErr * kR1GeVf  (same as pixelData.ptErr * kR1GeVf)
  - tripletRadius    = t5_innerRadius        (inner T3 fitted radius, stored as quintuplets.innerRadius())

Reports, vs deltaR:
  - Fraction of failing cases where NO truth-matched T5 passes passRadiusCriterion
    (radius is the blocker for this sim track)
  - Fraction where at least one T5 passes (radius is NOT the blocker; loss is elsewhere)
  - Distribution of radius ratio (t5_innerRadius / pixelRadius) for passing vs failing pairs
"""

import argparse
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import ROOT as r

r.gROOT.SetBatch(True)

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("input", help="LSTNtuple*.root file produced with --allobj --jet --idealpls")
parser.add_argument("--pt-cut",        type=float, default=0.9,    help="sim track pt cut [0.9]")
parser.add_argument("--eta-cut",       type=float, default=4.5,    help="sim track |eta| cut [4.5]")
parser.add_argument("--genjetpt-cut",  type=float, default=1000.0, help="GenJet pt cut [1000]")
parser.add_argument("--genjeteta-cut", type=float, default=2.5,    help="GenJet |eta| cut [2.5]")
parser.add_argument("--matchfrac",     type=float, default=0.75,   help="hit-purity threshold [0.75]")
parser.add_argument("--nbins",         type=int,   default=50,     help="deltaR bins [50]")
parser.add_argument("--max-deltar",    type=float, default=0.10,   help="max deltaR [0.10]")
parser.add_argument("--zoom",          type=float, default=0.05,   help="zoom cutoff [0.05]")
parser.add_argument("-o", "--output",  default="passradius_check.png")

# From interface/Common.h
_k2Rinv1GeVf = (2.99792458e-3 * 3.8) / 2.0   # ≈ 0.005696 cm^-1 GeV^-1
kR1GeVf = 1.0 / (2.0 * _k2Rinv1GeVf)          # ≈ 87.8 cm / GeV


def _interval_overlap(lo1, hi1, lo2, hi2):
    return lo1 <= hi2 and lo2 <= hi1


def pass_radius_criterion(pixel_radius, pixel_radius_error, triplet_radius, abs_eta):
    """Replicates passRadiusCriterion{BBB,BBE,BEE,EEE} from PixelTriplet.h.
    Category is approximated from the T5 |eta|:
      < 1.7  -> BBB  (all barrel)
      < 2.0  -> BBE  (upper module endcap)
      < 2.5  -> BEE  (middle+upper endcap)
      >= 2.5 -> EEE  (all endcap)
    """
    high_pt = pixel_radius > 2.0 * kR1GeVf

    if abs_eta < 1.7:          # BBB
        pix_b = 0.6375  if high_pt else 0.17235
        tri_b = 0.6588  if high_pt else 0.15624
    elif abs_eta < 2.0:        # BBE
        pix_b = 0.6805  if high_pt else 0.19644
        tri_b = 0.8557  if high_pt else 0.45972
    elif abs_eta < 2.5:        # BEE
        pix_b = 2.2091  if high_pt else 0.255181
        tri_b = 2.3548  if high_pt else 1.59294
    else:                      # EEE
        pix_b = 2.286   if high_pt else 0.26367
        tri_b = 2.436   if high_pt else 1.7006

    tri_inv_max = (1.0 + tri_b) / triplet_radius
    tri_inv_min = max((1.0 - tri_b) / triplet_radius, 0.0)

    pix_inv_max = max((1.0 + pix_b) / pixel_radius,
                       1.0 / (pixel_radius - pixel_radius_error))
    pix_inv_min = min((1.0 - pix_b) / pixel_radius,
                       1.0 / (pixel_radius + pixel_radius_error))

    return _interval_overlap(tri_inv_min, tri_inv_max, pix_inv_min, pix_inv_max)


def main():
    args = parser.parse_args()

    f = r.TFile.Open(args.input)
    if not f or f.IsZombie():
        parser.error("Cannot open: {}".format(args.input))
    tree = f.Get("tree")
    if not tree:
        parser.error("No 'tree' in {}".format(args.input))

    vtx_z_thresh   = 30.0
    vtx_perp_thresh = 2.5

    h_fail_denom  = r.TH1F("h_fail_denom",  "", args.nbins, 0, args.max_deltar)  # failing (ingred. present, no pT5)
    h_rad_blocks  = r.TH1F("h_rad_blocks",  "", args.nbins, 0, args.max_deltar)  # radius blocks ALL T5s
    h_rad_ok      = r.TH1F("h_rad_ok",      "", args.nbins, 0, args.max_deltar)  # at least one T5 passes radius
    for h in (h_fail_denom, h_rad_blocks, h_rad_ok):
        h.Sumw2()

    # Collect radius ratios (innerRadius / pixelRadius) for failing pairs
    ratios_failing = []   # pairs from sim tracks that never form a pT5
    ratios_passing = []   # pairs from sim tracks that DO form a pT5 (for comparison)

    n_fail = 0
    n_fail_rad_blocks = 0

    for ievt in range(tree.GetEntries()):
        tree.GetEntry(ievt)

        sim_pt       = tree.sim_pt
        sim_eta      = tree.sim_eta
        sim_q        = tree.sim_q
        sim_vx       = tree.sim_vx
        sim_vy       = tree.sim_vy
        sim_vz       = tree.sim_vz
        deltaR       = tree.sim_genjet_deltaR
        genjet_idx   = tree.sim_genjet_idx
        genjet_pt    = tree.genjet_pt
        genjet_eta   = tree.genjet_eta

        sim_plsIdxAll     = tree.sim_plsIdxAll
        sim_plsIdxAllFrac = tree.sim_plsIdxAllFrac
        sim_t5IdxAll      = tree.sim_t5IdxAll
        sim_t5IdxAllFrac  = tree.sim_t5IdxAllFrac
        sim_pt5IdxAll     = tree.sim_pt5IdxAll
        sim_pt5IdxAllFrac = tree.sim_pt5IdxAllFrac

        pLS_isQuad     = tree.pLS_isQuad
        pLS_isDuplicate = tree.pLS_isDuplicate
        pLS_pt         = tree.pLS_pt
        pLS_ptErr      = tree.pLS_ptErr

        t5_innerRadius = tree.t5_innerRadius
        t5_eta         = tree.t5_eta
        t5_isDuplicate = tree.t5_isDuplicate

        for isim in range(len(sim_pt)):
            if sim_q[isim] == 0:
                continue
            gj = genjet_idx[isim]
            if gj < 0 or genjet_pt[gj] <= args.genjetpt_cut or abs(genjet_eta[gj]) >= args.genjeteta_cut:
                continue
            vperp = math.hypot(sim_vx[isim], sim_vy[isim])
            if not (sim_pt[isim] > args.pt_cut and abs(sim_eta[isim]) < args.eta_cut
                    and abs(sim_vz[isim]) < vtx_z_thresh and vperp < vtx_perp_thresh):
                continue

            dr = deltaR[isim]

            # Find best eligible pLS (quad, not dup, frac > threshold)
            eligible_pls_idx = -1
            for i, frac in zip(sim_plsIdxAll[isim], sim_plsIdxAllFrac[isim]):
                if frac > args.matchfrac and pLS_isQuad[i] and not pLS_isDuplicate[i]:
                    eligible_pls_idx = i
                    break

            # Collect truth-matched T5s (not dup)
            matched_t5_idxs = [i for i, frac in zip(sim_t5IdxAll[isim], sim_t5IdxAllFrac[isim])
                                if frac > args.matchfrac and not t5_isDuplicate[i]]

            if eligible_pls_idx < 0 or not matched_t5_idxs:
                continue

            has_pt5 = any(frac > args.matchfrac for frac in sim_pt5IdxAllFrac[isim])

            pR  = pLS_pt[eligible_pls_idx]  * kR1GeVf
            pRe = pLS_ptErr[eligible_pls_idx] * kR1GeVf

            # Compute passRadiusCriterion for each matched T5
            t5_pass_flags = []
            for it5 in matched_t5_idxs:
                tR       = t5_innerRadius[it5]
                abs_eta  = abs(t5_eta[it5])
                passed   = pass_radius_criterion(pR, pRe, tR, abs_eta)
                t5_pass_flags.append(passed)
                ratio = tR / pR if pR > 0 else float("nan")
                if has_pt5:
                    ratios_passing.append(ratio)
                else:
                    ratios_failing.append(ratio)

            if not has_pt5:
                n_fail += 1
                h_fail_denom.Fill(dr)
                any_pass = any(t5_pass_flags)
                if not any_pass:
                    n_fail_rad_blocks += 1
                    h_rad_blocks.Fill(dr)
                else:
                    h_rad_ok.Fill(dr)

    print("Failing sim tracks (eligible pLS + T5, no pT5): {}".format(n_fail))
    if n_fail:
        pct = 100.0 * n_fail_rad_blocks / n_fail
        print("  passRadiusCriterion blocks ALL T5s: {} ({:.1f}%)".format(n_fail_rad_blocks, pct))
        print("  at least one T5 passes radius:      {} ({:.1f}%)".format(n_fail - n_fail_rad_blocks,
                                                                            100.0 - pct))

    # --- Build TEfficiency graphs ---
    def eff_graph(numer, denom):
        te = r.TEfficiency(numer, denom)
        g  = te.CreateGraph()
        n  = g.GetN()
        x   = [g.GetPointX(i)     for i in range(n)]
        y   = [g.GetPointY(i)     for i in range(n)]
        eyl = [g.GetErrorYlow(i)  for i in range(n)]
        eyh = [g.GetErrorYhigh(i) for i in range(n)]
        return x, y, [eyl, eyh]

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    ax_zoom, ax_full, ax_ratio = axes

    xb, yb, eb = eff_graph(h_rad_blocks, h_fail_denom)
    xo, yo, eo = eff_graph(h_rad_ok,     h_fail_denom)

    for ax in (ax_zoom, ax_full):
        ax.errorbar(xb, yb, yerr=eb, fmt="o", markersize=3, capsize=0,
                    label="passRadius blocks ALL T5s (radius is culprit)")
        ax.errorbar(xo, yo, yerr=eo, fmt="s", markersize=3, capsize=0,
                    label="≥1 T5 passes radius (loss elsewhere)")
        ax.set_xlabel("ΔR (track – nearest GenJet)")
        ax.set_ylabel("Fraction of failing sim tracks")
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    ax_zoom.set_xlim(0, args.zoom)
    ax_zoom.set_title("Zoom: ΔR < {}".format(args.zoom))
    ax_full.set_xlim(0, args.max_deltar)
    ax_full.set_title("Full range: ΔR < {}".format(args.max_deltar))

    # Ratio distribution
    bins = np.linspace(0.3, 2.0, 60)
    if ratios_passing:
        ax_ratio.hist(ratios_passing, bins=bins, density=True, alpha=0.5,
                      label="pT5 formed (passing sim tracks)", color="tab:blue")
    if ratios_failing:
        ax_ratio.hist(ratios_failing, bins=bins, density=True, alpha=0.5,
                      label="no pT5 formed (failing sim tracks)", color="tab:orange")
    ax_ratio.axvline(1.0, color="k", linestyle="--", linewidth=0.8, label="ratio = 1")
    ax_ratio.axvline(1.0 + 0.17235, color="gray", linestyle=":", linewidth=0.8, label="BBB low-pT bound (±17%)")
    ax_ratio.axvline(1.0 - 0.17235, color="gray", linestyle=":", linewidth=0.8)
    ax_ratio.set_xlabel("t5_innerRadius / pixelRadius")
    ax_ratio.set_ylabel("Density")
    ax_ratio.set_title("Radius ratio: passing vs failing sim tracks")
    ax_ratio.legend(fontsize=8)
    ax_ratio.grid(True, alpha=0.3)

    fig.suptitle("passRadiusCriterion offline check (idealpls)")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(args.output, dpi=150)
    print("Wrote {}".format(args.output))


if __name__ == "__main__":
    main()
