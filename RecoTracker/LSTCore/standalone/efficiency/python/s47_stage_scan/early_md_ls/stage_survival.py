#!/usr/bin/env python3
"""
stage_survival.py  (S47 early_md_ls)

Per denominator sim track (s44_eval_fixes.py / s46_core_funnel.py definition), jet core (dR<0.02)
vs outside, stage-by-stage survival hits -> MD -> LS -> T3 -> T5-pair, plus the s46 funnel bucket.

Stage definitions (per sim track):
  MD   : "MD-able module" = OT module (detId>>2) with reco hits of the sim on BOTH sensors.
         ok if a genuine MD (sim_mdIdxAllFrac >= 0.75, i.e. 2/2 hits) sits on that module.
         (identical to s44 md_ls_survival.py)
  LS   : consecutive genuine-MD logical layers that are adjacent (s44 definition); ok if an LS joins
         exactly one genuine MD of each layer (ls_mdIdx0/1).
  T3   : pairs of genuine LSs (sim_lsIdxAllFrac >= 0.75) with lsA.md1 == lsB.md0 ("T3-able");
         ok if a T3 exists with exactly (lsA, lsB) (t3_lsIdx0/1).
  T5   : pairs of genuine T3s (frac>=0.75) with t3A.md2 == t3B.md0 and t3A starting in the T5 region
         (barrel L1/L2 or endcap L1, Quintuplet.h isValidQuintRegion) = "T5-tried" pair
         (CreateQuintuplets loops exactly over these, Quintuplet.h:1950-1972); ok if a T5 exists with
         exactly (t3A, t3B) (t5_t3Idx0/1).
Output: per-track CSV (scratch) + printed tables.
Usage: stage_survival.py <ntuple> <tag>
"""
import sys
import collections
import numpy as np
import pickle
import uproot

PT_CUT, ETA_CUT, VTX_Z_MAX, VTX_R_MAX, GJ_PT_MIN, GJ_ETA_MAX = 0.8, 4.5, 30.0, 2.5, 1000.0, 2.5
GEN = 0.75
SCR = "/tmp/claude-63526/-mnt-data1-kk829-CMSSW-16-1-1-src-RecoTracker-LSTCore-standalone/dbd9af1c-87ff-4b4e-9196-c80ebb15a786/scratchpad/s47/early_md_ls/"

fn, tag = sys.argv[1], sys.argv[2]
tree = uproot.open(fn)["tree"]
keys = set(tree.keys())
BR = ["sim_pt", "sim_eta", "sim_q", "sim_vx", "sim_vy", "sim_vz", "sim_genjet_deltaR", "sim_genjet_idx",
      "genjet_pt", "genjet_eta", "sim_tcIdx", "sim_recoHitDetId", "sim_simHitDetId", "sim_simHitLayer", "pLS_isDup", "pT5_isDupReco",
      "md_detId", "md_layer", "md_isPLS", "md_anchor_x", "md_anchor_y",
      "ls_mdIdx0", "ls_mdIdx1", "t3_lsIdx0", "t3_lsIdx1", "t5_t3Idx0", "t5_t3Idx1"]
for o in ["md", "ls", "t3", "t5", "pls", "pt5", "t4", "pt3"]:
    BR += [f"sim_{o}IdxAll", f"sim_{o}IdxAllFrac"]
BR = [b for b in BR if b in keys]


def otsub(d):
    return (d >> 25) & 7


def logical_layer(d):
    s = otsub(d)
    if s == 5:
        return (d >> 20) & 0xF
    if s == 4:
        return 6 + ((d >> 18) & 0xF)
    return 0


def adjacent(l1, l2):
    if l1 == 0 or l2 == 0:
        return False
    b1, b2 = l1 <= 6, l2 <= 6
    if b1 == b2:
        return abs(l1 - l2) == 1
    return True


def gen(idx, frac):
    idx = np.asarray(idx); frac = np.asarray(frac)
    return idx[frac >= GEN].astype(np.int64)


rows = []
for A in tree.iterate(BR, step_size=10, library="np"):
    for ie in range(len(A["sim_pt"])):
        pt = A["sim_pt"][ie].astype(float)
        gj = A["sim_genjet_idx"][ie].astype(np.int64)
        gpt, geta = A["genjet_pt"][ie].astype(float), A["genjet_eta"][ie].astype(float)
        gc = np.clip(gj, 0, max(len(gpt) - 1, 0))
        gpt_s = gpt[gc] if len(gpt) else np.zeros_like(pt)
        geta_s = geta[gc] if len(geta) else np.zeros_like(pt)
        dr = A["sim_genjet_deltaR"][ie].astype(float)
        sel = ((A["sim_q"][ie] != 0) & (pt > PT_CUT) & (np.abs(A["sim_eta"][ie]) < ETA_CUT) &
               (np.abs(A["sim_vz"][ie]) < VTX_Z_MAX) &
               (np.hypot(A["sim_vx"][ie].astype(float), A["sim_vy"][ie].astype(float)) < VTX_R_MAX) &
               (gj >= 0) & (gpt_s > GJ_PT_MIN) & (np.abs(geta_s) < GJ_ETA_MAX))
        idxs = np.nonzero(sel)[0]
        if not len(idxs):
            continue
        md_det = np.asarray(A["md_detId"][ie]).astype(np.int64)
        md_lay = np.asarray(A["md_layer"][ie]).astype(np.int64)
        md_pls = np.asarray(A["md_isPLS"][ie]).astype(bool)
        md_r = np.hypot(A["md_anchor_x"][ie].astype(float), A["md_anchor_y"][ie].astype(float))
        ls0 = np.asarray(A["ls_mdIdx0"][ie]).astype(np.int64); ls1 = np.asarray(A["ls_mdIdx1"][ie]).astype(np.int64)
        t3l0 = np.asarray(A["t3_lsIdx0"][ie]).astype(np.int64); t3l1 = np.asarray(A["t3_lsIdx1"][ie]).astype(np.int64)
        t5a = np.asarray(A["t5_t3Idx0"][ie]).astype(np.int64); t5b = np.asarray(A["t5_t3Idx1"][ie]).astype(np.int64)
        lsset = set(zip(ls0.tolist(), ls1.tolist()))
        t5set = set(zip(t5a.tolist(), t5b.tolist()))
        pls_dup = np.asarray(A["pLS_isDup"][ie]).astype(np.int64)
        pt5_dup = np.asarray(A["pT5_isDupReco"][ie]).astype(np.int64)
        tcidx = A["sim_tcIdx"][ie]
        for s in idxs:
            g = {o: gen(A[f"sim_{o}IdxAll"][ie][s], A[f"sim_{o}IdxAllFrac"][ie][s])
                 for o in ["md", "ls", "t3", "t5", "pls", "pt5", "t4", "pt3"]}
            # ---- MD stage
            rh = np.asarray(A["sim_recoHitDetId"][ie][s]).astype(np.int64)
            reco_sens = collections.defaultdict(set)
            for d in rh:
                if otsub(int(d)) in (4, 5):
                    reco_sens[int(d) >> 2].add(int(d) & 3)
            mdable = {m for m, ss in reco_sens.items() if ss >= {1, 2}}
            nlay_mdable = len({logical_layer((m << 2) | 1) for m in mdable})
            nlay_reco = len({logical_layer(m << 2) for m in reco_sens})
            sh = np.asarray(A["sim_simHitDetId"][ie][s]).astype(np.int64)
            shl = np.asarray(A["sim_simHitLayer"][ie][s]).astype(np.int64)
            sim_sens = collections.defaultdict(set)
            for d in sh[shl > 0]:
                if otsub(int(d)) in (4, 5):
                    sim_sens[int(d) >> 2].add(int(d) & 3)
            nlay_sim2 = len({logical_layer((m << 2) | 1) for m, ss in sim_sens.items() if ss >= {1, 2}})
            gmd = [int(m) for m in g["md"] if not md_pls[m]]
            gmd_mods = {int(md_det[m]) >> 2 for m in gmd}
            n_mdable, n_md_ok = len(mdable), len(mdable & gmd_mods)
            # ---- LS stage (s44 def)
            bylay = collections.defaultdict(list)
            for m in gmd:
                bylay[int(md_lay[m])].append(m)
            lays = sorted(bylay, key=lambda L: np.mean([md_r[m] for m in bylay[L]]) if L <= 6 else 100 + L)
            n_lsp = n_lsp_ok = 0
            for L1, L2 in zip(lays[:-1], lays[1:]):
                if not adjacent(L1, L2):
                    continue
                n_lsp += 1
                n_lsp_ok += any((m1, m2) in lsset for m1 in bylay[L1] for m2 in bylay[L2])
            # ---- T3 stage
            gls = [int(l) for l in g["ls"] if l < len(ls0) and not md_pls[ls0[l]]]
            by_in = collections.defaultdict(list)
            for l in gls:
                by_in[int(ls0[l])].append(l)
            gt3 = [int(t) for t in g["t3"]]
            t3pairs_have = {(int(t3l0[t]), int(t3l1[t])) for t in gt3}
            # all built T3s with (lsA, lsB) among genuine LS: need global set restricted to genuine
            gls_set = set(gls)
            n_t3able = n_t3able_ok = 0
            t3able_pairs = []
            for la in gls:
                for lb in by_in.get(int(ls1[la]), []):
                    n_t3able += 1
                    t3able_pairs.append((la, lb))
            if t3able_pairs:
                # check built T3s among ALL T3s (not only genuine-matched ones): use t3 pair set of the event lazily
                pass
            # ---- T5 stage
            t3md = {t: (int(ls0[t3l0[t]]), int(ls1[t3l0[t]]), int(ls1[t3l1[t]])) for t in gt3}
            gt3_by_md0 = collections.defaultdict(list)
            for t, mm in t3md.items():
                gt3_by_md0[mm[0]].append(t)
            n_t5try = n_t5try_ok = 0
            n_gt3_region = 0
            for t, mm in t3md.items():
                inreg = md_lay[mm[0]] in (1, 2, 7)
                n_gt3_region += inreg
                if not inreg:
                    continue
                for u in gt3_by_md0.get(mm[2], []):
                    n_t5try += 1
                    n_t5try_ok += (t, u) in t5set
            # ---- funnel bucket (s46)
            core = 0 <= dr[s] < 0.02
            matched = tcidx[s] >= 0
            if matched:
                b = "matched"
            else:
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
            rows.append(dict(tag=tag, core=core, dr=dr[s], pt=pt[s], eta=float(A["sim_eta"][ie][s]), matched=matched,
                             bucket=b, nlay_sim2=nlay_sim2, nlay_reco=nlay_reco, nlay_mdable=nlay_mdable, n_mdable=n_mdable, n_md_ok=n_md_ok, nlay_md=len(bylay),
                             n_lsp=n_lsp, n_lsp_ok=n_lsp_ok,
                             n_gls=len(gls), n_t3able=n_t3able, t3able=repr(t3able_pairs),
                             n_gt3=len(gt3), n_gt3_region=n_gt3_region, n_t5try=n_t5try, n_t5try_ok=n_t5try_ok,
                             n_gt5=len(g["t5"]), n_gt4=len(g["t4"]), n_gpt3=len(g["pt3"]), n_gpls=len(g["pls"]),
                             evt=A["sim_pt"][ie][:3].sum()))
        # resolve T3-able pairs against all T3s of the event
        t3set = set(zip(t3l0.tolist(), t3l1.tolist()))
        for r in rows[-len(idxs):]:
            prs = eval(r["t3able"])
            r["n_t3able_ok"] = sum(p in t3set for p in prs)
            r["t3able"] = ""

class DF:
    """minimal column store (pandas unavailable)"""
    def __init__(self, rows):
        self.c = {k: np.array([r[k] for r in rows]) for k in rows[0]} if rows else {}
    def __getattr__(self, k):
        return self.__dict__["c"][k]
    def __getitem__(self, m):
        o = DF([]); o.c = {k: v[m] for k, v in self.c.items()}; return o
    def __len__(self):
        return len(next(iter(self.c.values()))) if self.c else 0


df = DF(rows)
with open(SCR + f"stage_survival_{tag}.pkl", "wb") as f:
    pickle.dump(df.c, f)


def summ(d, name):
    n = len(d)
    if n == 0:
        return
    mda, mok = d.n_mdable.sum(), d.n_md_ok.sum()
    lp, lpo = d.n_lsp.sum(), d.n_lsp_ok.sum()
    ta, tao = d.n_t3able.sum(), d.n_t3able_ok.sum()
    fa, fao = d.n_t5try.sum(), d.n_t5try_ok.sum()
    print(f"{name:18s} N={n:5d} eff={d.matched.mean():.3f} | MD/MDable={mok / mda:.4f} | LS/adjMDpair={lpo / lp:.4f} "
          f"| T3/T3-able={tao / max(ta, 1):.4f} ({ta}) | T5/T5-tried={fao / max(fa, 1):.4f} ({fa}) "
          f"| trk>=1: gLS {(d.n_gls > 0).mean():.3f} gT3 {(d.n_gt3 > 0).mean():.3f} gT3reg {(d.n_gt3_region > 0).mean():.3f} "
          f"T5try {(d.n_t5try > 0).mean():.3f} gT5 {(d.n_gt5 > 0).mean():.3f}")


print(f"== {tag}: {fn}")
core = df.core.astype(bool)
summ(df[core], "core dR<0.02")
summ(df[(~core) & (df.dr >= 0.02) & (df.dr < 0.1)], "0.02<=dR<0.10")
summ(df[(~core) & (df.dr >= 0.1)], "dR>=0.10")
summ(df[~core], "all non-core")
print("-- pT>50 control")
summ(df[core & (df.pt > 50)], "core pt>50")
summ(df[(~core) & (df.pt > 50)], "non-core pt>50")
print("-- core funnel buckets:", dict(collections.Counter(df[core].bucket.tolist())))
