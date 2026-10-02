#!/usr/bin/env python3
"""
s46_core_funnel.py

Where do jet-core TC failures stop? Self-contained re-derivation of the Session 44 top-level
buckets that still works on post-rebase (new T5 DNN) --allobj ntuples. It needs neither
t5_tightCutFlag nor t5_score. Denominator = s44_eval_fixes.py; genuine object = match fraction >= 0.75.

Each failing (no matched TC) core track is put in the FURTHEST bucket it reaches:
  pT5_survived   a genuine pT5 is not dedup-killed, yet no TC matches (downstream / extension dilution)
  pT5_all_killed genuine pT5(s) built, all killed by RemoveDupPixelQuintupletsFromMap (pT5_isDupReco)
  pair_failed    genuine unflagged pLS and genuine T5 both exist, but no genuine pT5 was built
  no_T5          genuine unflagged pLS, but no genuine T5
  pLS_flagged    genuine pLS exist, but all are flagged by CheckHitspLS pass 1 (pLS_isDup bit0)
  no_pLS         no genuine pLS
Columns with genuine T5 / pT3 / T4 / T3 are listed as extra information.

Usage:
  python3 s46_core_funnel.py <LSTNtuple.root> [label] [dR_max=0.02]
"""

import sys
import collections
import numpy as np
import uproot

PT_CUT, ETA_CUT, VTX_Z_MAX, VTX_R_MAX, GJ_PT_MIN, GJ_ETA_MAX = 0.8, 4.5, 30.0, 2.5, 1000.0, 2.5
GEN = 0.75
ORDER = ["pT5_survived", "pT5_all_killed", "pair_failed", "no_T5", "pLS_flagged", "no_pLS"]


def genuine(idx, frac):
    idx = np.asarray(idx)
    return idx[np.asarray(frac) >= GEN].astype(np.int64)


def main():
    fn = sys.argv[1]
    label = sys.argv[2] if len(sys.argv) > 2 else fn.split("/")[-1]
    drmax = float(sys.argv[3]) if len(sys.argv) > 3 else 0.02

    branches = ["sim_pt", "sim_eta", "sim_q", "sim_vx", "sim_vy", "sim_vz", "sim_genjet_deltaR",
                "sim_genjet_idx", "genjet_pt", "genjet_eta", "sim_tcIdx"]
    for o in ["pls", "t3", "t5", "pt3", "pt5", "t4"]:
        branches += [f"sim_{o}IdxAll", f"sim_{o}IdxAllFrac"]
    branches += ["pLS_isDup", "pT5_isDupReco"]
    tree = uproot.open(fn)["tree"]

    buckets = collections.Counter()
    pt_hi = collections.Counter()  # same, sim pT > 100 GeV
    extra = collections.Counter()
    n_den = n_pass = n_den_hi = n_pass_hi = 0

    for A in tree.iterate(branches, step_size=20, library="np"):
        for ie in range(len(A["sim_pt"])):
            sim_pt = A["sim_pt"][ie].astype(float)
            gj = A["sim_genjet_idx"][ie].astype(np.int64)
            gpt, geta = A["genjet_pt"][ie].astype(float), A["genjet_eta"][ie].astype(float)
            gc = np.clip(gj, 0, max(len(gpt) - 1, 0))
            gpt_s = gpt[gc] if len(gpt) else np.zeros_like(sim_pt)
            geta_s = geta[gc] if len(geta) else np.zeros_like(sim_pt)
            dr = A["sim_genjet_deltaR"][ie].astype(float)
            sel = ((A["sim_q"][ie] != 0) & (sim_pt > PT_CUT) & (np.abs(A["sim_eta"][ie]) < ETA_CUT) &
                   (np.abs(A["sim_vz"][ie]) < VTX_Z_MAX) &
                   (np.hypot(A["sim_vx"][ie].astype(float), A["sim_vy"][ie].astype(float)) < VTX_R_MAX) &
                   (gj >= 0) & (gpt_s > GJ_PT_MIN) & (np.abs(geta_s) < GJ_ETA_MAX) & (dr >= 0) & (dr < drmax))
            pls_dup = np.asarray(A["pLS_isDup"][ie]).astype(np.int64)
            pt5_dup = np.asarray(A["pT5_isDupReco"][ie]).astype(np.int64)
            tcidx = A["sim_tcIdx"][ie]

            for s in np.nonzero(sel)[0]:
                hi = sim_pt[s] > 100.0
                n_den += 1
                n_den_hi += hi
                if tcidx[s] >= 0:
                    n_pass += 1
                    n_pass_hi += hi
                    continue
                g = {o: genuine(A[f"sim_{o}IdxAll"][ie][s], A[f"sim_{o}IdxAllFrac"][ie][s])
                     for o in ["pls", "t3", "t5", "pt3", "pt5", "t4"]}
                pls_ok = [p for p in g["pls"] if not (pls_dup[p] & 1)]
                if len(g["pt5"]):
                    b = "pT5_survived" if np.any(pt5_dup[g["pt5"]] == 0) else "pT5_all_killed"
                elif not len(g["pls"]):
                    b = "no_pLS"
                elif not pls_ok:
                    b = "pLS_flagged"
                elif not len(g["t5"]):
                    b = "no_T5"
                else:
                    b = "pair_failed"
                buckets[b] += 1
                if hi:
                    pt_hi[b] += 1
                for o in ["t3", "t5", "pt3", "t4"]:
                    if len(g[o]):
                        extra[(b, o)] += 1

    nf, nf_hi = n_den - n_pass, n_den_hi - n_pass_hi
    print(f"== {label}   core dR<{drmax}: den {n_den}, eff {n_pass / n_den:.4f}, failures {nf}   |   "
          f"pT>100: den {n_den_hi}, eff {n_pass_hi / max(n_den_hi, 1):.4f}, failures {nf_hi}")
    print(f"{'bucket':<16}{'fails':>7}{'% of den':>10}{'pT>100':>8}   genuine: {'T3':>5}{'T5':>6}{'pT3':>6}{'T4':>6}")
    for b in ORDER:
        print(f"{b:<16}{buckets[b]:>7}{100 * buckets[b] / n_den:>9.1f}%{pt_hi[b]:>8}            "
              f"{extra[(b, 't3')]:>5}{extra[(b, 't5')]:>6}{extra[(b, 'pt3')]:>6}{extra[(b, 't4')]:>6}")


if __name__ == "__main__":
    main()
