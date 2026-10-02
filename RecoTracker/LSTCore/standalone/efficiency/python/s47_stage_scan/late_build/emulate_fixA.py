#!/usr/bin/env python3
"""Offline emulation of the Fix A pairing fallback (PixelQuintuplet.h:520-557) on the gate-OFF 100-evt --allobj ntuple.
For every pLS (pT>=MINPT, CheckHitspLS bit0 clear) x T5 (AfterBuild bit0 clear) with |dEta|<0.05,|dPhi|<0.1 that is NOT
already a built pT5, compute the RMS distance of pixel hits 0 and 2 (the InLo/InUp anchors) to a circle fitted to the
T5's 10 hits (stand-in for the T5 regression circle). Pairs with RMS<=MAXRES are Fix-A candidates (upper bound: the
tracklet pointing cuts are not emulated). Classify genuine (pLS sim == T5 sim) vs foreign and what the T5 was in base.
Also: r-z residual of the pixel hits to the T5 inner-T3 z(r) line, and a proxy of the pT5 dedup score
(pixel circle vs T5 hits, PixelQuintuplet.h:643) to estimate how often a foreign pairing wins pT5 dedup.
Env: MINPT (50), MAXRES (0.03 cm), MAXZ (cm, default none).
"""
import os, pickle, collections, numpy as np, uproot
F = "/mnt/data1/kk829/CMSSW_16_1_1/src/RecoTracker/LSTCore/standalone/Ntuple-files/LSTNtuple_s45_rebased_100evt.root"
SCR = "/tmp/claude-63526/-mnt-data1-kk829-CMSSW-16-1-1-src-RecoTracker-LSTCore-standalone/dbd9af1c-87ff-4b4e-9196-c80ebb15a786/scratchpad/s47/late_build/"
CACHE = SCR + "emu_cache_v2.pkl"
MINPT = float(os.environ.get("MINPT", 50)); MAXRES = float(os.environ.get("MAXRES", 0.03))
MAXZ = float(os.environ.get("MAXZ", 1e9))
BR = ["pLS_pt", "pLS_eta", "pLS_phi", "pLS_isDup", "pLS_simIdx", "pLS_hit0_x", "pLS_hit0_y", "pLS_hit2_x", "pLS_hit2_y",
      "pLS_hit0_z", "pLS_hit2_z", "pLS_circleCenterX", "pLS_circleCenterY", "pLS_circleRadius", "pLS_isQuad",
      "t5_eta", "t5_phi", "t5_pt", "t5_isDupBits", "t5_simIdx", "t5_t3Idx0", "t5_t3Idx1", "t5_partOfPT5", "t5_tc_idx",
      "t5_partOfTC", "t5_nLayers", "pT5_plsIdx", "pT5_t5Idx", "pT5_isDupReco", "pT5_score", "pT5_simIdx",
      "sim_pt", "sim_eta", "sim_q", "sim_vx", "sim_vy", "sim_vz", "sim_genjet_deltaR", "sim_genjet_idx", "genjet_pt",
      "genjet_eta", "sim_tcIdx"]
BR += [f"t5_t3_{i}_{c}" for i in range(6) for c in ("x", "y", "z", "r")]


def load():
    if os.path.exists(CACHE):
        return pickle.load(open(CACHE, "rb"))
    A = uproot.open(F)["tree"].arrays(BR, library="np")
    os.makedirs(SCR, exist_ok=True)
    pickle.dump(A, open(CACHE, "wb"))
    return A


def core_mask(A, ie):
    pt = A["sim_pt"][ie].astype(float); gj = A["sim_genjet_idx"][ie].astype(np.int64)
    gpt, geta = A["genjet_pt"][ie], A["genjet_eta"][ie]
    gc = np.clip(gj, 0, max(len(gpt) - 1, 0))
    gp = gpt[gc] if len(gpt) else np.zeros_like(pt); ge = geta[gc] if len(geta) else np.zeros_like(pt)
    dr = A["sim_genjet_deltaR"][ie]
    return ((A["sim_q"][ie] != 0) & (pt > 0.8) & (np.abs(A["sim_eta"][ie]) < 4.5) & (np.abs(A["sim_vz"][ie]) < 30)
            & (np.hypot(A["sim_vx"][ie], A["sim_vy"][ie]) < 2.5) & (gj >= 0) & (gp > 1000) & (np.abs(ge) < 2.5)
            & (dr >= 0) & (dr < 0.02))


def t5_hits(A, ie, k, c):
    i0, i1 = A["t5_t3Idx0"][ie][k], A["t5_t3Idx1"][ie][k]
    return np.array([A[f"t5_t3_{h}_{c}"][ie][i0] for h in range(6)] + [A[f"t5_t3_{h}_{c}"][ie][i1] for h in range(2, 6)], float)


def fit_circle(x, y):  # Kasa fit
    M = np.stack([x, y, np.ones_like(x)], 1); b = -(x * x + y * y)
    (D, E, Fc), *_ = np.linalg.lstsq(M, b, rcond=None)
    xc, yc = -D / 2, -E / 2
    return xc, yc, np.sqrt(max(xc * xc + yc * yc - Fc, 0))


def zline_res(A, ie, p, k):
    """RMS perpendicular distance (cm) in the (r,z) plane of the T5 inner-T3 anchor hits (MD1, MD2) to the straight
    line through pixel hits 0 and 2 (high-pT tracks are straight in r-z; pixel z/r are precise)."""
    i0 = A["t5_t3Idx0"][ie][k]
    r1 = np.hypot(A["pLS_hit0_x"][ie][p], A["pLS_hit0_y"][ie][p]); z1 = A["pLS_hit0_z"][ie][p]
    r2 = np.hypot(A["pLS_hit2_x"][ie][p], A["pLS_hit2_y"][ie][p]); z2 = A["pLS_hit2_z"][ie][p]
    ur, uz = r2 - r1, z2 - z1; n = np.hypot(ur, uz)
    ur, uz = ur / n, uz / n
    d = []
    for h in (0, 2):
        rr, zz = A[f"t5_t3_{h}_r"][ie][i0] - r1, A[f"t5_t3_{h}_z"][ie][i0] - z1
        d.append(rr * uz - zz * ur)
    d = np.array(d)
    return float(np.sqrt(np.mean(d * d)))


def main():
    A = load()
    c = collections.Counter(); resG, resF, resBuilt, zBuilt, cands, allpairs = [], [], [], [], [], []
    chiBuilt = {}
    for ie in range(len(A["pLS_pt"])):
        coreset = set(np.nonzero(core_mask(A, ie))[0])
        built = set(zip(A["pT5_plsIdx"][ie], A["pT5_t5Idx"][ie]))
        t5ok = np.nonzero((A["t5_isDupBits"][ie] & 1) == 0)[0]
        geo = {}
        tcidx = A["sim_tcIdx"][ie]
        for p in np.nonzero((A["pLS_pt"][ie] >= MINPT) & ((A["pLS_isDup"][ie] & 1) == 0))[0]:
            pe, pp = A["pLS_eta"][ie][p], A["pLS_phi"][ie][p]
            de = np.abs(A["t5_eta"][ie][t5ok] - pe); dp = np.abs((A["t5_phi"][ie][t5ok] - pp + np.pi) % (2 * np.pi) - np.pi)
            px = np.array([A["pLS_hit0_x"][ie][p], A["pLS_hit2_x"][ie][p]], float)
            py = np.array([A["pLS_hit0_y"][ie][p], A["pLS_hit2_y"][ie][p]], float)
            pz = np.array([A["pLS_hit0_z"][ie][p], A["pLS_hit2_z"][ie][p]], float)
            for k in t5ok[(de < 0.05) & (dp < 0.1)]:
                if k not in geo:
                    x, y, z, r = (t5_hits(A, ie, k, q) for q in "xyzr")
                    b, a = np.polyfit(r[:6], z[:6], 1)  # inner T3 z(r) line
                    geo[k] = (fit_circle(x, y), (a, b), x[[0, 2, 4, 7, 9]], y[[0, 2, 4, 7, 9]])
                (xc, yc, R), (a, b), ax, ay = geo[k]
                d = np.hypot(px - xc, py - yc) - R
                rms = np.sqrt(np.mean(d * d))
                dz = pz - (a + b * np.hypot(px, py)); zres = np.sqrt(np.mean(dz * dz))
                dc = np.hypot(ax - A["pLS_circleCenterX"][ie][p], ay - A["pLS_circleCenterY"][ie][p]) - A["pLS_circleRadius"][ie][p]
                chi = float(np.mean(dc * dc))
                sp, st = A["pLS_simIdx"][ie][p], A["t5_simIdx"][ie][k]
                gen = sp >= 0 and sp == st
                dzl = zline_res(A, ie, p, k)
                allpairs.append((ie, p, k, gen, st, sp, rms, dzl, (p, k) in built, int(A["t5_partOfPT5"][ie][k]),
                                 int(A["t5_partOfTC"][ie][k]), st in coreset, tcidx[st] >= 0 if st >= 0 else False,
                                 sp in coreset, tcidx[sp] >= 0 if sp >= 0 else False, float(A["pLS_pt"][ie][p])))
                if (p, k) in built:
                    if gen:
                        resBuilt.append(rms); zBuilt.append(zres); chiBuilt[(ie, k)] = chi
                    continue
                cands.append((ie, p, k, gen, st, sp, rms, zres, chi, st in coreset, tcidx[st] >= 0 if st >= 0 else False,
                              sp in coreset, tcidx[sp] >= 0 if sp >= 0 else False))
                (resG if gen else resF).append(rms)
                if rms > MAXRES or zres > MAXZ:
                    continue
                kind = "genuine" if gen else ("foreign(T5 sim)" if st >= 0 else "fake T5")
                c[("cand", kind)] += 1
                if not gen and st >= 0:
                    t5tc = A["t5_partOfTC"][ie][k]; inpt5 = A["t5_partOfPT5"][ie][k]
                    c[("foreign: T5 base status", "T5 is a T5-TC" if t5tc and not inpt5 else ("T5 in a pT5" if inpt5 else "T5 not TC"))] += 1
                    if st in coreset:
                        c[("foreign: T5 owner is core sim", "matched in base" if tcidx[st] >= 0 else "unmatched in base")] += 1
                    if sp >= 0 and sp in coreset:
                        c[("foreign: pLS owner is core sim", "matched in base" if tcidx[sp] >= 0 else "unmatched in base")] += 1
                if gen and st in coreset:
                    c[("genuine: owner core sim", "matched in base" if tcidx[st] >= 0 else "UNMATCHED in base (gain)")] += 1
    pickle.dump(dict(allpairs=allpairs, cands=cands, chiBuilt=chiBuilt, zBuilt=zBuilt), open(SCR + "emu_cands.pkl", "wb"))
    print(f"MINPT={MINPT} MAXRES={MAXRES} MAXZ={MAXZ}  (100 evt; upper bound: tracklet pointing cuts not emulated)")
    for k in sorted(c):
        print(f"  {k}: {c[k]}")
    for nm, r in (("built genuine pT5", resBuilt), ("unbuilt genuine pair", resG), ("unbuilt foreign pair", resF)):
        r = np.array(r) * 1e4
        if len(r):
            print(f"  r-phi RMS residual [um] {nm}: n={len(r)} q10/50/90 = {np.round(np.quantile(r, [.1, .5, .9]), 0)}"
                  f"  frac<300um={np.mean(r < 300):.2f} <100um={np.mean(r < 100):.2f}")
    zB = np.array(zBuilt) * 1e4
    print(f"  r-z RMS residual [um] (pixel hits vs T5 inner-T3 z(r) line), built genuine pT5: q50/90/99 {np.round(np.quantile(zB, [.5, .9, .99]), 0)}")
    cz = np.array([x[7] for x in cands]) * 1e4; cg = np.array([x[3] for x in cands]); crms = np.array([x[6] for x in cands])
    for nm, m in (("genuine", cg), ("foreign/fake", ~cg)):
        mm = m & (crms <= MAXRES)
        if mm.any():
            print(f"  r-z residual [um] unbuilt {nm} pairs passing r-phi res: n={mm.sum()} q10/50/90 {np.round(np.quantile(cz[mm], [.1, .5, .9]), 0)}")
    fw = [x[8] < chiBuilt[(x[0], x[2])] for x in cands if (not x[3]) and x[6] <= MAXRES and x[7] <= MAXZ and (x[0], x[2]) in chiBuilt]
    print(f"  foreign Fix-A pairs on a T5 that has a genuine built pT5: {len(fw)}; foreign pixel-circle score lower (wins dedup): {np.mean(fw) if fw else 0:.2f}")


main()
