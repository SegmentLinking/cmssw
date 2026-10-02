#!/usr/bin/env python3
"""For denominator sim tracks WITHOUT a TC but WITH a truth-matched T5 (frac>=0.75),
check whether every matched T5 has a pT3 within dR<sqrt(1e-3) (CrossCleanT5 vs-pT3
kill cone). Compare jet-core (dR<0.01) vs control (dR>0.05) populations.
"""
import math
import sys
import ROOT

FNAME = sys.argv[1] if len(sys.argv) > 1 else "LSTNtuple_noT5pT5dedup.root"
FRAC = 0.75
KILL_DR = math.sqrt(1e-3)  # 0.0316

f = ROOT.TFile(FNAME)
t = f.Get("tree")


def dphi(a, b):
    d = a - b
    while d > math.pi:
        d -= 2 * math.pi
    while d < -math.pi:
        d += 2 * math.pi
    return d


stats = {}  # region -> [n_failing_withT5, n_allT5_killcone, n_someT5_free]
mindr_fail = {"core": [], "far": []}

for ev in t:
    dr = ev.sim_genjet_deltaR
    jidx = ev.sim_genjet_idx
    gpt = ev.genjet_pt
    geta = ev.genjet_eta
    t5eta = ev.t5_eta
    t5phi = ev.t5_phi
    p3eta = list(ev.pT3_eta)
    p3phi = list(ev.pT3_phi)
    nsim = len(ev.sim_pt)
    for i in range(nsim):
        if ev.sim_q[i] != -1:
            continue
        if not (ev.sim_pt[i] > 0.9 and abs(ev.sim_eta[i]) < 4.5):
            continue
        if not (abs(ev.sim_vz[i]) < 30 and (ev.sim_vx[i] ** 2 + ev.sim_vy[i] ** 2) ** 0.5 < 2.5):
            continue
        ji = jidx[i]
        if ji < 0 or not (gpt[ji] > 1000 and abs(geta[ji]) < 2.5):
            continue
        if dr[i] < 0.01:
            region = "core"
        elif dr[i] > 0.05:
            region = "far"
        else:
            continue
        if ev.sim_tcIdx[i] >= 0:
            continue  # only failing tracks
        t5s = [
            idx for idx, fr in zip(ev.sim_t5IdxAll[i], ev.sim_t5IdxAllFrac[i]) if fr >= FRAC
        ]
        if not t5s:
            continue
        s = stats.setdefault(region, [0, 0, 0])
        s[0] += 1
        # per matched T5: min dR to any pT3
        per_t5_mindr = []
        for j in t5s:
            e1, p1 = t5eta[j], t5phi[j]
            md = min(
                (
                    math.sqrt((e1 - e2) ** 2 + dphi(p1, p2) ** 2)
                    for e2, p2 in zip(p3eta, p3phi)
                ),
                default=999.0,
            )
            per_t5_mindr.append(md)
        best = min(per_t5_mindr)
        mindr_fail[region].append(best)
        if all(md < KILL_DR for md in per_t5_mindr):
            s[1] += 1
        else:
            s[2] += 1

for region in ("core", "far"):
    s = stats.get(region, [0, 0, 0])
    print(f"[{region}] failing tracks with matched T5: {s[0]}")
    print(f"    ALL matched T5s have pT3 within dR<{KILL_DR:.4f}: {s[1]}")
    print(f"    at least one T5 outside kill cone:            {s[2]}")
    v = sorted(mindr_fail[region])
    if v:
        n = len(v)
        print(f"    best-T5 min-dR-to-pT3: median {v[n//2]:.4f}  "
              f"q25 {v[n//4]:.4f}  q75 {v[3*n//4]:.4f}")
