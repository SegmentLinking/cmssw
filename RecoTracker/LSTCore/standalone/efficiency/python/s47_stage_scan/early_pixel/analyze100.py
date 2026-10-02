#!/usr/bin/env python3
"""Summaries from core_rows.pkl (scan100.py). Read-only."""
import os, sys, pickle, collections
SD = os.path.dirname(os.path.abspath(__file__))
D = pickle.load(open(sys.argv[1] if len(sys.argv) > 1 else os.path.join(SD, "core_rows.pkl"), "rb"))
R = D["rows"]
F = [r for r in R if r["fail"]]
print(f"core den {len(R)}  fail {len(F)}")
C = collections.Counter(r["bucket"] for r in F)
print("funnel:", dict(C))

# ---------------- pLS_flagged: pass1 vs CrossCleanpLS ----------------
print("\n=== pLS_flagged bucket: which flag killed the genuine pLS ===")
fl = [r for r in F if r["bucket"] == "pLS_flagged"]
sub = collections.Counter()
for r in fl:
    p1 = [gd["pass1"] for gd in r["gdet"]]
    if all(p1):
        k = "all_gen_pass1"
    elif not any(p1):
        k = "none_pass1(all CrossCleanpLS)"
    else:
        k = "mixed(some gen survive pass1, then CrossCleanpLS)"
    sub[k] += 1
    r["flag_sub"] = k
print(dict(sub))
# for tracks with a pass1-surviving genuine pLS: did those pLS make pT5/pT3? genuine T5?
for r in fl:
    if r["flag_sub"] != "all_gen_pass1":
        s = [gd for gd in r["gdet"] if not gd["pass1"]]
        print(f"  CCpLS: le{r['le']} t{r['t']} pt{r['pt']:.0f} hasT5={r['has']['t5']} hasT3={r['has']['t3']} "
              f"hasT4={r['has']['t4']} survivors: " +
              " ".join(f"[q{int(g['quad'])} dup{g['isdup']} npt5={g['npt5']} npt3={g['npt3']}]" for g in s))

# ---------------- killers of all-pass1 tracks ----------------
print("\n=== all_gen_pass1 tracks: fate under variants ===")
A = [r for r in fl if r["flag_sub"] == "all_gen_pass1"]
var = collections.Counter()
for r in A:
    for v in ["fixB", "nms", "nmsB"]:
        if any(not gd[v] for gd in r["gdet"]):
            var[v] += 1
    # "winner must be same/purer" oracle: genuine pLS survives if every killer is not better for this sim
print(f"n={len(A)}; genuine pLS freed (>=1 gen pLS unflagged) by:", dict(var))
print("  with genuine T5 present:", sum(1 for r in A if r["has"]["t5"]),
      " with genuine T3:", sum(1 for r in A if r["has"]["t3"]), " genuine T4:", sum(1 for r in A if r["has"]["t4"]))
for v in ["fixB", "nms", "nmsB"]:
    freed = [r for r in A if any(not gd[v] for gd in r["gdet"])]
    print(f"  {v}: freed {len(freed)}, of which genuine T5 {sum(1 for r in freed if r['has']['t5'])}, "
          f"freed pLS quad {sum(1 for r in freed if any(not gd[v] and gd['quad'] for gd in r['gdet']))}")

kc = collections.Counter(); kq = collections.Counter(); kn = collections.Counter(); kw = collections.Counter()
kst = collections.Counter()
for r in A:
    # per track: characterize killers of all genuine pLS (union)
    ks = [k for gd in r["gdet"] for k in gd["killers"]]
    for k in ks:
        kc[k["wcls"]] += 1
    for gd in r["gdet"]:
        for k in gd["killers"]:
            kq[("V" + ("quad" if gd["quad"] else "trip"), "W" + ("quad" if k["wquad"] else "trip"),
                f"npm{k['npm']}", f"nd{k['nd']}")] += 1
            kw[("w_pass1" if k["w_pass1"] else "w_alive", "w_pT5surv" if k["w_npt5surv"] else
                ("w_pT5killed" if k["w_npt5"] else ("w_pT3" if k["w_npt3"] else "w_noPixTrk")),
                "w_inTC" if k["w_intc"] else "w_notTC")] += 1
print("\nkiller classes (all killer edges):", dict(kc))
print("killer edges by (victim, winner, npMatched, nDistinct):")
for k, v in sorted(kq.items(), key=lambda x: -x[1]):
    if v:
        print("   ", k, v)
print("winner fate:")
for k, v in sorted(kw.items(), key=lambda x: -x[1]):
    print("   ", k, v)
# per track: best killer class
pt = collections.Counter()
for r in A:
    cls = set(k["wcls"] for gd in r["gdet"] for k in gd["killers"])
    nd = max(k["nd"] for gd in r["gdet"] for k in gd["killers"])
    anyalive = any(not k["w_pass1"] for gd in r["gdet"] for k in gd["killers"])
    pt[(tuple(sorted(cls)), f"maxnd{nd}", "someWinnerAlive" if anyalive else "allWinnersFlagged")] += 1
print("per-track killer summary:")
for k, v in sorted(pt.items(), key=lambda x: -x[1]):
    print("   ", k, v)

# ---------------- no_pLS attribution ----------------
print("\n=== no_pLS attribution ===")
N = [r for r in F if r["bucket"] == "no_pLS"]
ac = collections.Counter()
for r in N:
    ga = r["gen_in_algos"]
    if r["gen_in_lstalgo_ptfail"]:
        c = "B: genuine LST-algo seed failed pT cut"
    elif ga:
        c = f"A: genuine seed only in non-LST algo {tuple(ga)}"
    elif r["best_lst"] >= 0.75 - 1e-6:
        c = "C: best LST pLS exactly 3/4 (frac 0.75, not >0.75)"
    elif max(r["best_by_algo"].values(), default=0) >= 0.75 - 1e-6:
        c = "C2: 3/4 seed only in non-LST algo"
    elif r["n_in_seeds"] > 0 and r["best_lst"] > 0:
        c = f"D: best LST pLS {r['best_lst']:.2f} (contaminated/partial)"
        c = "D: best LST pLS <=2/3 (contaminated/partial)"
    elif r["n_in_seeds"] > 0:
        c = "D2: only non-LST-algo partial seeds"
    else:
        c = "N: no input seed with any hit of this sim"
    r["np_cls"] = c
    ac[c] += 1
for k, v in sorted(ac.items(), key=lambda x: -x[1]):
    print(f"   {v:4d}  {k}")
print("  A split by has genuine T5:", collections.Counter((r["np_cls"][:1], bool(r["has"]["t5"])) for r in N))
print("  pT>100 per class:", collections.Counter(r["np_cls"][:2] for r in N if r["pt"] > 100))
# 3/4 pLS fate
print("  C: 3/4 pLS fate:")
for r in N:
    if r["np_cls"].startswith("C:"):
        print(f"     le{r['le']} t{r['t']} pt{r['pt']:.0f} hasT5={r['has']['t5']} p34=" +
              " ".join(f"[q{int(x['quad'])} dup{x['isdup']} p1{int(x['pass1'])} pt5={x['npt5']}/{x['npt5surv']} tc{x['intc']}]" for x in r["p34"]))
print("  A: seed algos detail:", collections.Counter(tuple(sorted(r["gen_in_nraw"].items())) for r in N if r["np_cls"].startswith("A")))

# ---------------- bit1 (TC-stage pass 2) ----------------
print("\n=== bit1 (CheckHitspLS pass 2) / CrossCleanpLS on core failures ===")
b1 = collections.Counter()
for r in F:
    for gd in r["gdet"]:
        if gd["isdup"] & 2:
            b1[(r["bucket"], "quad" if gd["quad"] else "trip")] += 1
print("genuine pLS of failing core tracks with bit1:", dict(b1))
# failing tracks where a genuine quad pLS exists, unflagged by pass1: would it become pLS-TC? flagged by what?
qq = collections.Counter()
for r in F:
    qs = [gd for gd in r["gdet"] if gd["quad"] and not gd["pass1"]]
    if not qs:
        continue
    st = "has_bit1" if any(g["isdup"] & 2 for g in qs) else ("CCpLS" if all(g["isdup"] & 1 for g in qs) else "unflagged?")
    qq[(r["bucket"], st)] += 1
print("failing tracks with genuine pass1-surviving quad pLS:", dict(qq))
