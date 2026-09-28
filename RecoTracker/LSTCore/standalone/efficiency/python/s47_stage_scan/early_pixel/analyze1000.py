#!/usr/bin/env python3
"""Fix B per-track attribution (1000 evt) + no-genuine-pLS input attribution at 1000 evt."""
import pickle, collections
CACHE = "/tmp/claude-63526/-mnt-data1-kk829-CMSSW-16-1-1-src-RecoTracker-LSTCore-standalone/dbd9af1c-87ff-4b4e-9196-c80ebb15a786/scratchpad/s47/early_pixel/"
D = pickle.load(open(CACHE + "core1000.pkl", "rb")); R = D["rows"]
print("den", len(R), {k: sum(r["succ"][k] for r in R) for k in R[0]["succ"]})

def cat(r):
    if r["n_gen"] == 0:
        return "no_gen_pLS"
    if not all(r["gen_m"]):
        return "gen_pLS_alive_pass1"
    if not all(r["gen_b"]):
        return "all_pass1_killed__fixB_frees"
    return "all_pass1_killed__fixB_no_help"

tab = collections.Counter()
for r in R:
    s = r["succ"]; base = [s["base"], s["base_rep1"], s["base_rep2"]]
    if all(base) and s["fixB"]: st = "stable_pass"
    elif not any(base) and not s["fixB"]: st = "stable_fail"
    elif not any(base) and s["fixB"]: st = "GAIN(fixB only)"
    elif all(base) and not s["fixB"]: st = "LOSS(fixB only)"
    else: st = "noisy(base reps disagree)"
    tab[(cat(r), st)] += 1
    r["st"] = st
cats = ["no_gen_pLS", "gen_pLS_alive_pass1", "all_pass1_killed__fixB_frees", "all_pass1_killed__fixB_no_help"]
sts = ["stable_pass", "stable_fail", "GAIN(fixB only)", "LOSS(fixB only)", "noisy(base reps disagree)"]
print(f"{'':34s}" + "".join(f"{s:>14s}" for s in ["pass", "fail", "GAIN", "LOSS", "noisy"]))
for c in cats:
    print(f"{c:34s}" + "".join(f"{tab[(c, s)]:>14d}" for s in sts))
print("net fixB-only gains - losses:", sum(tab[(c, sts[2])] for c in cats) - sum(tab[(c, sts[3])] for c in cats))
fr = [r for r in R if cat(r) == "all_pass1_killed__fixB_frees"]
print("fixB_frees tracks, pT>100:", sum(r["pt"] > 100 for r in fr), " GAIN pT>100:", sum(r["pt"] > 100 and r["st"].startswith("GAIN") for r in fr))
print("LOSS by pT>100:", collections.Counter((cat(r), r["pt"] > 100) for r in R if r["st"].startswith("LOSS")))
print("GAIN by pT>100:", collections.Counter((cat(r), r["pt"] > 100) for r in R if r["st"].startswith("GAIN")))

# killer composition for all-pass1-killed failures in base
kc = collections.Counter()
for r in R:
    if cat(r).startswith("all_pass1") and not r["succ"]["base"]:
        ks = r["kills"]
        cls = set()
        for (w, npm, nd, wq, wfr) in ks:
            cls.add(("same_gen" if wfr > 0.75 else ("3/4" if abs(wfr - 0.75) < 1e-6 else ("2/3" if abs(wfr - 2 / 3) < 1e-6 else "other"))) + ("Q" if wq else "T") + f"nd{nd}")
        kc[tuple(sorted(cls))] += 1
print("\nkiller sets (base-failing, all genuine pass1-killed):")
for k, v in kc.most_common(12):
    print("  ", v, k)

# no-genuine-pLS attribution at 1000 evt (base failures)
ac = collections.Counter(); hi = collections.Counter()
for r in R:
    if r["n_gen"] or r["succ"]["base"]:
        continue
    ga = r["gen_in_algos"]
    if r["gen_in_lstalgo_ptfail"]: c = "B: genuine LST-algo seed fails pT cut"
    elif ga: c = "A: genuine seed only in non-LST algo " + str(tuple(ga))
    elif r["best_lst"] >= 0.75 - 1e-6: c = "C: best LST pLS exactly 3/4"
    elif max(r["best_by_algo"].values(), default=0) >= 0.75 - 1e-6: c = "C2: 3/4 only in non-LST algo"
    elif r["best_lst"] > 0: c = "D: best LST pLS <=2/3 (partial/contaminated)"
    elif r["n_in_seeds"]: c = "D2: only non-LST-algo partial seeds"
    else: c = "N: no seed contains any hit of this sim"
    ac[c] += 1; hi[c] += r["pt"] > 100
nfail = sum(1 for r in R if not r["succ"]["base"])
print(f"\nno-genuine-pLS base failures at 1000 evt: {sum(ac.values())} of {nfail} failures")
for k, v in ac.most_common():
    print(f"  {v:5d}  (pT>100: {hi[k]:4d})  {k}")
