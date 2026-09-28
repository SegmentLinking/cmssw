#!/usr/bin/env python3
"""Fix A variants on the emulated pLS x T5 pair list (emulate_fixA.py cache). Per variant: admitted unbuilt pairs,
genuine vs foreign, predicted core gains (genuine pair for a base-unmatched core sim) and core-risk (foreign pair whose
T5 owner is a base-matched core sim whose T5 is a T5-TC or in a pT5 -> absorbed; or whose pLS owner is matched core)."""
import pickle, collections, numpy as np
SCR = "/tmp/claude-63526/-mnt-data1-kk829-CMSSW-16-1-1-src-RecoTracker-LSTCore-standalone/dbd9af1c-87ff-4b4e-9196-c80ebb15a786/scratchpad/s47/late_build/"
D = pickle.load(open(SCR + "emu_cands.pkl", "rb"))
P = D["allpairs"]
# fields
ie, p, k, gen, st, sp, rms, dzl, built, t5pt5, t5tc, stcore, stm, spcore, spm, plspt = map(np.array, zip(*P))
gen = gen.astype(bool); built = built.astype(bool)
print(f"pairs in window (pLS pT>=50, unflagged; T5 AB-clean): {len(P)}; built {built.sum()} (genuine {np.sum(built & gen)})")
q = lambda x: np.round(np.quantile(x, [.1, .5, .9, .99]) * 1e4, 0)
print("r-z line residual [um] (T5 MD1/MD2 to pixel line): built genuine", q(dzl[built & gen]),
      " unbuilt genuine", q(dzl[~built & gen]), " unbuilt foreign", q(dzl[~built & ~gen]))
# mutual best by rphi residual among ALL pairs (built or not) in the window
bestT5 = {}; bestP = {}
for i in range(len(P)):
    a = (ie[i], p[i]); b = (ie[i], k[i])
    if a not in bestT5 or rms[i] < rms[bestT5[a]]: bestT5[a] = i
    if b not in bestP or rms[i] < rms[bestP[b]]: bestP[b] = i
mutual = np.array([bestT5[(ie[i], p[i])] == i and bestP[(ie[i], k[i])] == i for i in range(len(P))])
# pLS already in a built pT5?
plsBuilt = set((ie[i], p[i]) for i in np.nonzero(built)[0])
plsFree = np.array([(ie[i], p[i]) not in plsBuilt for i in range(len(P))])
V = {
    "A as run (rphi<=300um)": rms <= 0.03,
    "rphi<=100um": rms <= 0.01,
    "rphi<=50um": rms <= 0.005,
    "A + rz-line<=1mm": (rms <= 0.03) & (dzl <= 0.1),
    "A + rz-line<=0.5mm": (rms <= 0.03) & (dzl <= 0.05),
    "A + T5 not partOfPT5 + pLS not in pT5 (2nd pass)": (rms <= 0.03) & (t5pt5 == 0) & plsFree,
    "A + mutual-best": (rms <= 0.03) & mutual,
    "A + mutual-best + rz<=1mm": (rms <= 0.03) & mutual & (dzl <= 0.1),
    "A + 2nd pass + mutual + rz<=1mm": (rms <= 0.03) & (t5pt5 == 0) & plsFree & mutual & (dzl <= 0.1),
    "rphi<=100um + 2nd pass + mutual + rz<=0.5mm": (rms <= 0.01) & (t5pt5 == 0) & plsFree & mutual & (dzl <= 0.05),
}
print(f"{'variant':52s}{'admit':>6}{'gen':>5}{'forgn':>6}{'gain':>6}{'riskT5':>7}{'riskPLS':>8}")
for nm, m in V.items():
    m = m & ~built
    g = m & gen; f = m & ~gen
    gain = len(set((ie[i], st[i]) for i in np.nonzero(g & stcore & ~stm)[0]))
    # foreign pair: T5 owner loses if its T5 was its TC (T5-TC) -> T5 now partOfPT5 in a fake pT5
    riskT5 = len(set((ie[i], st[i]) for i in np.nonzero(f & (st >= 0) & stcore & stm & (t5tc == 1) & (t5pt5 == 0))[0]))
    riskP = len(set((ie[i], sp[i]) for i in np.nonzero(f & (sp >= 0) & spcore & spm)[0]))
    print(f"{nm:52s}{m.sum():6d}{g.sum():5d}{f.sum():6d}{gain:6d}{riskT5:7d}{riskP:8d}")
print("gain = distinct base-unmatched core sims with an admitted genuine pair (upper bound, tracklet cuts not emulated)")
print("riskT5 = distinct base-matched core sims whose T5-TC is absorbed by an admitted foreign pair")
print("riskPLS = distinct base-matched core sims whose genuine pLS is consumed by a foreign pair (loss only if its own TC used that pLS)")
