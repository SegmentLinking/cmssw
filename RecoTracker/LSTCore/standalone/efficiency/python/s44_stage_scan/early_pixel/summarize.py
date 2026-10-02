#!/usr/bin/env python3
import pickle, collections, numpy as np, os
SD = os.path.dirname(os.path.abspath(__file__))
rows = pickle.load(open(os.path.join(SD, "core_rows.pkl"), "rb"))
fails = [r for r in rows if r["fail"]]
succ = [r for r in rows if not r["fail"]]
print(f"core denom {len(rows)}, fails {len(fails)}, eff {len(succ)/len(rows):.3f}")

A = [r for r in fails if not r["has_gen_pls"] and not r["has_pt5"]]
A_all = [r for r in fails if not r["has_gen_pls"]]
print(f"fails with no genuine pLS (LST ntuple): {len(A_all)}; of which no genuine pT5 either: {len(A)}")

def classify(r):
    if r["n_gen_in_lst"] > 0:
        return "X: genuine seed passes LST filter (inconsistency)"
    if any(r["gen_in_pass"]):
        pass
    gp = r["gen_in_pass"]
    if len(gp) > 0:
        if any(ga and not gpt for ga, gpt in gp):
            return "D1: genuine input seed, dropped by LST pT cut"
        return "D2: genuine input seed, dropped by LST algo filter (algo %s)" % r["gen_in_algos"]
    # no perfect seed in input
    if r["best_lst"] >= 0.75 - 1e-6:
        return "M1: LST HAS a 3/4 pLS (frac=0.75, not >0.75 -> matching artifact)"
    if r["best_all"] >= 0.75 - 1e-6:
        return "M2: 3/4 seed in input but only in non-LST algo"
    if r["best_lst"] >= 0.5:
        return "C1: best LST pLS 2/3 or 2/4 (contaminated seed)"
    if r["best_all"] > 0:
        return "C2: only partial (<=1/2) seed hits in input"
    if r["npixlayers"] < 3:
        return "N1: no seed; <3 pixel layers with reco hits (physics/pixel ineff.)"
    return "N2: no seed at all despite >=3 pixel layers with hits (CMSSW seeding failure)"

for label, pop in [("no-gen-pLS (all)", A_all), ("no-gen-pLS & no-gen-pT5", A)]:
    c = collections.Counter(classify(r) for r in pop)
    print(f"\n=== {label}: {len(pop)} ===")
    for k, v in sorted(c.items()):
        print(f"  {v:4d} ({100*v/len(pop):5.1f}%)  {k}")

# detail on M1: best seed hits
m1 = [r for r in A_all if classify(r).startswith("M1")]
print("\nM1 best_lst_n distribution:", collections.Counter(tuple(r["best_lst_n"]) for r in m1))
c1 = [r for r in A_all if classify(r).startswith("C1")]
print("C1 best_lst_n distribution:", collections.Counter(tuple(r["best_lst_n"]) for r in c1))
n2 = [r for r in A_all if classify(r).startswith("N")]
print("N* shared-pix fraction: median nshared/npix",
      np.median([r["nshared_pix"] / max(r["npixhits"], 1) for r in n2]) if n2 else None,
      " npixlayers dist", collections.Counter(r["npixlayers"] for r in n2))
print("N* CMSSW sim_seedIdx non-empty:", sum(1 for r in n2 if len(r["cmssw_seedIdx"]) > 0))
# how many pixel hits of M1/C1 tracks are shared with other sims
for lab, pop in [("M1", m1), ("C1", c1)]:
    if pop:
        print(f"{lab}: mean shared pix hits {np.mean([r['nshared_pix'] for r in pop]):.2f} / {np.mean([r['npixhits'] for r in pop]):.2f}; any-shared frac {np.mean([r['nshared_pix']>0 for r in pop]):.2f}")
print("Successful core tracks any-shared pix frac:", np.mean([r['nshared_pix'] > 0 for r in succ]).round(3),
      " failing A:", np.mean([r['nshared_pix'] > 0 for r in A_all]).round(3))

# --- bucket B: genuine pLS but no pT5 ---
B = [r for r in fails if r["has_gen_pls"] and not r["has_pt5"]]
print(f"\n=== fails with genuine pLS but no genuine pT5: {len(B)} ===")
print("  all genuine pLS triplets (non-quad):", sum(1 for r in B if not any(r["gen_pls_quad"])))
print("  all genuine pLS dedup-killed by CheckHitspLS (emulated):", sum(1 for r in B if r["all_gen_killed"]))
kk = collections.Counter()
for r in B:
    if r["all_gen_killed"]:
        labs = set(k[0] for k in r["killer_info"])
        kk["winner=" + ("/".join(sorted(labs)))] += 1
        if all(k[2] < 3 for k in r["killer_info"]):
            kk["killed ONLY via triplet double-count (<3 distinct shared)"] += 1
print("  killers:", dict(kk))
print("  has genuine pT3:", sum(1 for r in B if r["has_pt3"]))
# succ reference
print("\nsuccesses: has gen pLS", sum(r["has_gen_pls"] for r in succ), "/", len(succ),
      " all-gen-killed:", sum(r["all_gen_killed"] for r in succ))
print("fails bucket (pT5 built, killed):", sum(1 for r in fails if r["has_pt5"]))
# per-track dump for A_all
with open(os.path.join(SD, "no_pls_attribution_table.tsv"), "w") as f:
    f.write("le\tie\tsimIdx\tpt\teta\tdR\tpdg\tclass\tbest_all\tbest_lst\tbest_lst_n\tbest_by_algo\tnpixhits\tnpixlayers\tnshared_pix\tcmssw_seedIdx\n")
    for r in A_all:
        f.write("\t".join(str(x) for x in [r["le"], r["ie"], r["simIdx"], round(r["pt"], 2), round(r["eta"], 3), round(r["dR"], 4), r["pdg"],
                                           classify(r), round(r["best_all"], 3), round(r["best_lst"], 3), r["best_lst_n"],
                                           {k: round(v, 2) for k, v in r["best_by_algo"].items()}, r["npixhits"], r["npixlayers"],
                                           r["nshared_pix"], r["cmssw_seedIdx"]]) + "\n")
print("wrote no_pls_attribution_table.tsv")
