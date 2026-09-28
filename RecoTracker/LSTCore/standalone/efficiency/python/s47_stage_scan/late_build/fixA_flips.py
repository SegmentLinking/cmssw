#!/usr/bin/env python3
"""Fix A flip analysis on 1000-evt TC-level ntuples: which core sims flip matched<->unmatched,
their TC type in base vs fixA, and what fixA TC now holds their hits.
Usage: fixA_flips.py base.root fixA.root [dRmax]"""
import sys, collections, numpy as np, uproot
PT_CUT, ETA_CUT, VZ, VR, GJPT, GJETA = 0.8, 4.5, 30., 2.5, 1000., 2.5
TYPES = {4: "T5", 5: "pT3", 7: "pT5", 8: "pLS", 9: "T4"}
B = ["sim_pt", "sim_eta", "sim_q", "sim_vx", "sim_vy", "sim_vz", "sim_genjet_deltaR", "sim_genjet_idx",
     "genjet_pt", "genjet_eta", "sim_tcIdx", "sim_tcIdxAll", "sim_tcIdxAllFrac", "tc_type", "tc_isFake",
     "tc_simIdx", "tc_pt", "tc_simIdxAll", "tc_simIdxAllFrac", "tc_nhits"]

def sel(A, ie, drmax):
    pt = A["sim_pt"][ie].astype(float); gj = A["sim_genjet_idx"][ie].astype(np.int64)
    gpt, geta = A["genjet_pt"][ie], A["genjet_eta"][ie]
    gc = np.clip(gj, 0, max(len(gpt) - 1, 0))
    gp = gpt[gc] if len(gpt) else np.zeros_like(pt); ge = geta[gc] if len(geta) else np.zeros_like(pt)
    dr = A["sim_genjet_deltaR"][ie]
    return ((A["sim_q"][ie] != 0) & (pt > PT_CUT) & (np.abs(A["sim_eta"][ie]) < ETA_CUT) & (np.abs(A["sim_vz"][ie]) < VZ)
            & (np.hypot(A["sim_vx"][ie], A["sim_vy"][ie]) < VR) & (gj >= 0) & (gp > GJPT) & (np.abs(ge) < GJETA)
            & (dr >= 0) & (dr < drmax))

def main():
    fb, fa = sys.argv[1], sys.argv[2]; drmax = float(sys.argv[3]) if len(sys.argv) > 3 else 0.02
    tb, ta = uproot.open(fb)["tree"], uproot.open(fa)["tree"]
    c = collections.Counter(); ntc = collections.Counter()
    Ab0, Aa0 = tb.arrays(B, library="np"), ta.arrays(B, library="np")
    key = lambda x: (len(x), round(float(np.sum(x)), 2))
    amap = {key(x): j for j, x in enumerate(Aa0["sim_pt"])}
    order = [amap[key(x)] for x in Ab0["sim_pt"]]
    Aa0 = {k: v[order] for k, v in Aa0.items()}
    for Ab, Aa in [(Ab0, Aa0)]:
        for ie in range(len(Ab["sim_pt"])):
            assert len(Ab["sim_pt"][ie]) == len(Aa["sim_pt"][ie])
            for name, A in (("base", Ab), ("fixA", Aa)):
                for t in A["tc_type"][ie]: ntc[(name, TYPES.get(int(t), t))] += 1
                ft = A["tc_isFake"][ie]
                for t, f in zip(A["tc_type"][ie], ft): ntc[(name, TYPES.get(int(t), t), "fake")] += int(f)
            s = sel(Ab, ie, drmax)
            for i in np.nonzero(s)[0]:
                hi = Ab["sim_pt"][ie][i] > 100
                mb, ma = Ab["sim_tcIdx"][ie][i] >= 0, Aa["sim_tcIdx"][ie][i] >= 0
                c[("den", hi)] += 1
                if mb == ma: continue
                if mb and not ma:
                    tbt = TYPES[int(Ab["tc_type"][ie][Ab["sim_tcIdx"][ie][i]])]
                    # best fixA TC containing this sim's hits
                    idx = np.asarray(Aa["sim_tcIdxAll"][ie][i]); fr = np.asarray(Aa["sim_tcIdxAllFrac"][ie][i])
                    if len(idx):
                        k = int(np.argmax(fr)); j = int(idx[k]); f = fr[k]
                        rt = TYPES[int(Aa["tc_type"][ie][j])]
                        fake = int(Aa["tc_isFake"][ie][j]); other = int(Aa["tc_simIdx"][ie][j])
                        fb_ = "0.5-0.75" if f >= 0.5 else "<0.5"
                        rdesc = f"{rt} frac{fb_} {'FAKE' if fake else ('matchesOther' if other != i else 'same?')}"
                    else:
                        rdesc = "no TC with its hits"
                    c[("lost", hi)] += 1
                    c[("lost_by", tbt, rdesc)] += 1
                    c[("lost_type", tbt, hi)] += 1
                else:
                    tat = TYPES[int(Aa["tc_type"][ie][Aa["sim_tcIdx"][ie][i]])]
                    c[("gain", hi)] += 1; c[("gain_type", tat, hi)] += 1
    print(f"core dR<{drmax}: den {c[('den',False)]+c[('den',True)]} (pT>100 {c[('den',True)]})")
    print(f"lost {c[('lost',False)]+c[('lost',True)]} (pT>100 {c[('lost',True)]}), gained {c[('gain',False)]+c[('gain',True)]} (pT>100 {c[('gain',True)]})")
    print("lost by base TC type (all / pT>100):")
    for t in TYPES.values():
        n = c[("lost_type", t, False)] + c[("lost_type", t, True)]
        if n: print(f"  {t:4s} {n:5d} {c[('lost_type', t, True)]:5d}")
    print("gained by fixA TC type (all / pT>100):")
    for t in TYPES.values():
        n = c[("gain_type", t, False)] + c[("gain_type", t, True)]
        if n: print(f"  {t:4s} {n:5d} {c[('gain_type', t, True)]:5d}")
    print("lost: base type -> fixA TC holding most of its hits:")
    for k, v in sorted(((k, v) for k, v in c.items() if k[0] == "lost_by"), key=lambda x: -x[1]):
        print(f"  {k[1]:4s} -> {k[2]:35s} {v}")
    print("n_TC by type (base / fixA), fakes:")
    for t in TYPES.values():
        print(f"  {t:4s} {ntc[('base',t)]:7d} {ntc[('fixA',t)]:7d}   fake {ntc[('base',t,'fake')]:6d} {ntc[('fixA',t,'fake')]:6d}")

main()
