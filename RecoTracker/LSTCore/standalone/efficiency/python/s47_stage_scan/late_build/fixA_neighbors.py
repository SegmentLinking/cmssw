#!/usr/bin/env python3
"""For core sims lost by Fix A (1000 evt), list nearby (dEta,dPhi<W) TCs in base and fixA by type/fake status.
TC-level ntuples keep only >=0.75 matches, so the 'replacement' TC is found kinematically."""
import sys, collections, numpy as np, uproot
sys.path.insert(0, __file__.rsplit('/', 1)[0])
PT_CUT, ETA_CUT, VZ, VR, GJPT, GJETA = 0.8, 4.5, 30., 2.5, 1000., 2.5
TYPES = {4: "T5", 5: "pT3", 7: "pT5", 8: "pLS", 9: "T4"}
B = ["sim_pt", "sim_eta", "sim_phi", "sim_q", "sim_vx", "sim_vy", "sim_vz", "sim_genjet_deltaR", "sim_genjet_idx",
     "genjet_pt", "genjet_eta", "sim_tcIdx", "tc_type", "tc_isFake", "tc_simIdx", "tc_pt", "tc_eta", "tc_phi", "tc_nhits", "tc_nhitOT", "tc_pMatched"]
W = float(sys.argv[3]) if len(sys.argv) > 3 else 0.01

def sel(A, ie):
    pt = A["sim_pt"][ie].astype(float); gj = A["sim_genjet_idx"][ie].astype(np.int64)
    gpt, geta = A["genjet_pt"][ie], A["genjet_eta"][ie]
    gc = np.clip(gj, 0, max(len(gpt) - 1, 0))
    gp = gpt[gc] if len(gpt) else np.zeros_like(pt); ge = geta[gc] if len(geta) else np.zeros_like(pt)
    dr = A["sim_genjet_deltaR"][ie]
    return ((A["sim_q"][ie] != 0) & (pt > PT_CUT) & (np.abs(A["sim_eta"][ie]) < ETA_CUT) & (np.abs(A["sim_vz"][ie]) < VZ)
            & (np.hypot(A["sim_vx"][ie], A["sim_vy"][ie]) < VR) & (gj >= 0) & (gp > GJPT) & (np.abs(ge) < GJETA)
            & (dr >= 0) & (dr < 0.02))

def near(A, ie, eta, phi):
    de = np.abs(A["tc_eta"][ie] - eta); dp = np.abs((A["tc_phi"][ie] - phi + np.pi) % (2 * np.pi) - np.pi)
    return np.nonzero((de < W) & (dp < W))[0]

tb, ta = uproot.open(sys.argv[1])["tree"], uproot.open(sys.argv[2])["tree"]
Ab, Aa = tb.arrays(B, library="np"), ta.arrays(B, library="np")
key = lambda x: (len(x), round(float(np.sum(x)), 2))
amap = {key(x): j for j, x in enumerate(Aa["sim_pt"])}
order = [amap[key(x)] for x in Ab["sim_pt"]]
Aa = {k: v[order] for k, v in Aa.items()}
c = collections.Counter(); pmb = []
for ie in range(len(Ab["sim_pt"])):
    for i in np.nonzero(sel(Ab, ie))[0]:
        mb, ma = Ab["sim_tcIdx"][ie][i] >= 0, Aa["sim_tcIdx"][ie][i] >= 0
        if not (mb and not ma): continue
        eta, phi = Ab["sim_eta"][ie][i], Ab["sim_phi"][ie][i]
        c["lost"] += 1
        for nm, A in (("base", Ab), ("fixA", Aa)):
            idx = near(A, ie, eta, phi)
            for j in idx:
                t = TYPES[int(A["tc_type"][ie][j])]; f = int(A["tc_isFake"][ie][j])
                c[(nm, t, "fake" if f else "real")] += 1
        # fake pT5 near in fixA: pMatched distribution
        idx = near(Aa, ie, eta, phi)
        fp = [j for j in idx if Aa["tc_type"][ie][j] == 7 and Aa["tc_isFake"][ie][j]]
        c["has_fake_pT5_near_fixA"] += bool(fp)
        idxb = near(Ab, ie, eta, phi)
        c["has_fake_pT5_near_base"] += any(Ab["tc_type"][ie][j] == 7 and Ab["tc_isFake"][ie][j] for j in idxb)
        for j in fp: pmb.append((Aa["tc_pMatched"][ie][j], Aa["tc_nhits"][ie][j], Aa["tc_nhitOT"][ie][j]))
print(f"window |dEta|,|dPhi|<{W}; lost sims {c['lost']}")
print(f"lost sims with a FAKE pT5 TC nearby: base {c['has_fake_pT5_near_base']}  fixA {c['has_fake_pT5_near_fixA']}")
print("TCs near lost sims (type, fake/real): base  fixA")
for t in TYPES.values():
    for f in ("real", "fake"):
        print(f"  {t:4s} {f}: {c[('base',t,f)]:5d} {c[('fixA',t,f)]:5d}")
if pmb:
    p = np.array(pmb)
    print("fake pT5 near lost sims in fixA: pMatched quantiles", np.round(np.quantile(p[:, 0], [.1, .25, .5, .75, .9]), 3))
    vals, cnt = np.unique(p[:, 1].astype(int), return_counts=True); print("  nhits:", dict(zip(vals, cnt)))
    vals, cnt = np.unique(np.round(p[:, 0], 3), return_counts=True)
    print("  top pMatched values:", sorted(zip(cnt, vals))[-6:])
