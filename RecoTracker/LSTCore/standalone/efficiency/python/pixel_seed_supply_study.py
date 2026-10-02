#!/usr/bin/env python3
"""
Per-sim-track PIXEL-SEED-SUPPLY study for LST.

Goal: for genuine sim tracks in jet cores that FAIL to reconstruct (sim_tcIdx<0),
classify by pixel line-segment (pLS) seed status:
  A: no genuine pLS at all
  B: genuine pLS exists but ALL non-quad (isQuad==False) -> cannot seed pT5
  C: genuine quad pLS exists but still no TC
     C sub-split: did it build a genuine pT5/pT3 (frac>=0.75)?
        C_postbuild: yes -> loss is downstream (dedup/crossclean)
        C_nobuild  : no  -> pT5/pT3 building failed despite good seed

Also real-vs-ideal recovery: core sim tracks failing in REAL that reconstruct in IDEAL.

Genuine object match: fraction >= 0.75.
Run: python3 pixel_seed_supply_study.py
"""
import uproot, awkward as ak, numpy as np

REAL = "Ntuple-files/LSTNtuple_realpls_mastercuts_100evt_v2.root"
IDEAL = "LSTNtuple_ABon_oracle_100evt.root"
FRAC = 0.75

BR = ["sim_q","sim_pt","sim_eta","sim_vx","sim_vy","sim_vz",
      "sim_genjet_idx","sim_genjet_deltaR","genjet_pt","genjet_eta",
      "sim_tcIdx","sim_plsIdxAll","sim_plsIdxAllFrac",
      "sim_pt5IdxAll","sim_pt3IdxAll","sim_pt5IdxAllFrac","sim_pt3IdxAllFrac",
      "pLS_isQuad"]

def load(fn):
    t = uproot.open(fn+":tree")
    keys = set(t.keys())
    br = [b for b in BR if b in keys]
    return t.arrays(br, library="ak"), keys

def denom_mask(a):
    """Base denominator + jet-core matched-genjet requirement, per event jagged mask."""
    gj_pt  = a["genjet_pt"][a["sim_genjet_idx"]]      # broadcast index per sim track
    gj_eta = a["genjet_eta"][a["sim_genjet_idx"]]
    m = (a["sim_q"]!=0) & (a["sim_pt"]>0.9) & (abs(a["sim_eta"])<4.5) \
        & (abs(a["sim_vz"])<30) & (np.sqrt(a["sim_vx"]**2+a["sim_vy"]**2)<2.5) \
        & (a["sim_genjet_idx"]>=0) & (gj_pt>1000) & (abs(gj_eta)<2.5)
    return m

def any_genuine(idxlist, fraclist):
    """per sim track: is there any object with frac>=FRAC (and valid idx>=0)?"""
    good = (fraclist>=FRAC) & (idxlist>=0)
    return ak.any(good, axis=-1)

a_real, _ = load(REAL)
a_ideal, _ = load(IDEAL)

# --- denominator consistency check ---
print("=== Denominator consistency (REAL vs IDEAL) ===")
for label,a in [("REAL",a_real),("IDEAL",a_ideal)]:
    m = denom_mask(a)
    print(f"  {label}: nevents={len(a['sim_pt'])}  denom sim tracks (all jets)={int(ak.sum(m))}")

mR = denom_mask(a_real)
mI = denom_mask(a_ideal)

# Events are PERMUTED between files. Build REAL->IDEAL event map by matching the
# full per-event sim_pt vector (an exact fingerprint; within a matched event the
# per-track row order is identical, verified: max abs sim_pt diff == 0).
def evkey(ptvec):
    return tuple(np.asarray(ptvec).tobytes())  # exact float32 fingerprint
ideal_map = {}
for ev in range(len(a_ideal["sim_pt"])):
    ideal_map.setdefault(evkey(a_ideal["sim_pt"][ev]), ev)
real2ideal = {}
nmatched = 0
for ev in range(len(a_real["sim_pt"])):
    k = evkey(a_real["sim_pt"][ev])
    if k in ideal_map:
        real2ideal[ev] = ideal_map[k]; nmatched += 1
print(f"  REAL events matched to a unique IDEAL event by full sim_pt fingerprint: {nmatched}/{len(a_real['sim_pt'])}")

# --- ideal reconstruction: per matched IDEAL event, reco flag per track row index ---
tcI = a_ideal["sim_tcIdx"]
ideal_reco_rows = {}  # ideal_ev -> np.array(bool) reconstructed per row
for iev in set(real2ideal.values()):
    ideal_reco_rows[iev] = (np.asarray(tcI[iev]) >= 0)

# --- bins ---
BINS = [("dR<0.02", 0.02), ("dR<0.05", 0.05), ("dR<0.10", 0.10)]

# precompute genuine flags per event for real
dR_real = a_real["sim_genjet_deltaR"]
tc_real = a_real["sim_tcIdx"]
pt_real = a_real["sim_pt"]
plsIdx = a_real["sim_plsIdxAll"]; plsFrac = a_real["sim_plsIdxAllFrac"]
isQuad = a_real["pLS_isQuad"]
pt5Idx = a_real["sim_pt5IdxAll"]; pt5Frac = a_real["sim_pt5IdxAllFrac"]
pt3Idx = a_real["sim_pt3IdxAll"]; pt3Frac = a_real["sim_pt3IdxAllFrac"]

def classify_track(ev, si):
    """Return dict of booleans for one sim track (already in denom)."""
    idxs = plsIdx[ev][si]; fracs = plsFrac[ev][si]
    genuine_mask = (fracs>=FRAC) & (idxs>=0)
    gen_idxs = idxs[genuine_mask]
    n_gen = len(gen_idxs)
    if n_gen==0:
        return "A", None
    # any genuine pLS that is a quad?
    quads = isQuad[ev][gen_idxs]
    has_quad = bool(ak.any(quads))
    if not has_quad:
        return "B", None
    # C: has genuine quad pLS
    # did it build a genuine pT5 or pT3?
    p5 = any_genuine(pt5Idx[ev][si:si+1], pt5Frac[ev][si:si+1])[0]
    p3 = any_genuine(pt3Idx[ev][si:si+1], pt3Frac[ev][si:si+1])[0]
    built = bool(p5 or p3)
    return "C", ("postbuild" if built else "nobuild")

results = {b[0]: {"fail":{"A":0,"B":0,"C_post":0,"C_nobuild":0},
                  "succ":{"A":0,"B":0,"C_post":0,"C_nobuild":0},
                  "recovered":0, "nfail":0, "nsucc":0} for b in BINS}

for ev in range(len(pt_real)):
    dm = mR[ev]
    si_all = np.where(np.asarray(dm))[0]
    iev = real2ideal.get(ev, None)
    for si in si_all:
        dr = float(dR_real[ev][si])
        failed = bool(tc_real[ev][si] < 0)
        bucket, sub = classify_track(ev, int(si))
        # same row index in matched ideal event (row order identical within event)
        rec_ideal = bool(ideal_reco_rows[iev][int(si)]) if iev is not None else False
        for bname, thr in BINS:
            if dr < thr:
                R = results[bname]
                grp = "fail" if failed else "succ"
                R["n"+("fail" if failed else "succ")] += 1
                if bucket=="A": R[grp]["A"]+=1
                elif bucket=="B": R[grp]["B"]+=1
                elif bucket=="C":
                    R[grp]["C_post" if sub=="postbuild" else "C_nobuild"]+=1
                if failed and rec_ideal:
                    R["recovered"]+=1

def pct(n,d): return f"{100.0*n/d:5.1f}%" if d else "  n/a"

print("\n=== CORE FAILING population decomposition (REAL-pLS, sim_tcIdx<0) ===")
for bname,_ in BINS:
    R = results[bname]; nf=R["nfail"]; f=R["fail"]
    A,B,Cp,Cn = f["A"],f["B"],f["C_post"],f["C_nobuild"]
    supply = A+B
    print(f"\n[{bname}] core failing tracks = {nf}")
    print(f"  A no genuine pLS        : {A:4d} ({pct(A,nf)})")
    print(f"  B pLS all non-quad      : {B:4d} ({pct(B,nf)})")
    print(f"  C genuine quad pLS      : {Cp+Cn:4d} ({pct(Cp+Cn,nf)})")
    print(f"      C-postbuild (dnstrm): {Cp:4d} ({pct(Cp,nf)})")
    print(f"      C-nobuild (bld fail): {Cn:4d} ({pct(Cn,nf)})")
    print(f"  --> SEED-SUPPLY (A+B)   : {supply:4d} ({pct(supply,nf)})")
    print(f"  real->ideal RECOVERED   : {R['recovered']:4d} ({pct(R['recovered'],nf)})")

print("\n=== CORE SUCCEEDING population (reference, sim_tcIdx>=0) ===")
for bname,_ in BINS:
    R = results[bname]; ns=R["nsucc"]; s=R["succ"]
    A,B,Cp,Cn = s["A"],s["B"],s["C_post"],s["C_nobuild"]
    print(f"[{bname}] succ={ns:4d}  A={A}({pct(A,ns)}) B={B}({pct(B,ns)}) "
          f"C_post={Cp}({pct(Cp,ns)}) C_nobuild={Cn}({pct(Cn,ns)})")
