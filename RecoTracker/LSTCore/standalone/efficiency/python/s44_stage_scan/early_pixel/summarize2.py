import pickle, collections, numpy as np, os
SD=os.path.dirname(os.path.abspath(__file__))
rows=pickle.load(open(os.path.join(SD,"core_rows.pkl"),"rb"))
fails=[r for r in rows if r["fail"]]; succ=[r for r in rows if not r["fail"]]
A=[r for r in fails if not r["has_gen_pls"]]
B=[r for r in fails if r["has_gen_pls"] and not r["has_pt5"]]
# Q: other-algo input seeds for no-pLS tracks (any frac)
def algos_with(r,thr): return sorted(al for al,f in r["best_by_algo"].items() if f>=thr-1e-6 and al not in (4,22))
print("no-pLS tracks:",len(A))
for thr in (1.0,0.75,0.5):
    c=[r for r in A if algos_with(r,thr)]
    print(f" with an other-algo seed frac>={thr}: {len(c)}  algos:",collections.Counter(tuple(algos_with(r,thr)) for r in c))
# of those with perfect other-algo seed, do they also have ANY usable (>=0.5) LST pLS?
c=[r for r in A if algos_with(r,1.0)]
print(" perfect-other-algo & usable LST pLS(>=0.5):",sum(1 for r in c if r["usable"]),
      " & usable surviving dedup:",sum(1 for r in c if any(not u["isdup"] for u in r["usable"])),
      " has genuine T5:",sum(r["has_t5"] for r in c), " npixlayers:",collections.Counter(r["npixlayers"] for r in c))
# usable-seed fate for all fails without genuine pT5
def fate(r):
    if not r["usable"]: return "0 no usable LST pLS (frac>=0.5)"
    if all(u["isdup"] for u in r["usable"]):
        return "1 all usable pLS killed by CheckHitspLS pass1"
    if any(u["n_pt5"]>0 for u in r["usable"] if not u["isdup"]): return "3 surviving usable pLS built pT5 (non-genuine match)"
    return "2 usable pLS survives dedup, no pT5 built"
for lab,pop in [("no-gen-pLS (259)",A),("gen-pLS no-pT5 (205)",B),("no genuine pT5 total",[r for r in fails if not r["has_pt5"]])]:
    print(f"\n== {lab}: {len(pop)}")
    for k,v in sorted(collections.Counter(fate(r) for r in pop).items()): print(f"  {v:4d} {k}")
    k1=[r for r in pop if fate(r).startswith("1")]
    if k1:
        dc=sum(1 for r in k1 if all(all(k[3]<3 for k in u["killers"]) for u in r["usable"]))
        w=collections.Counter()
        for r in k1:
            for u in r["usable"]:
                for k in u["killers"]:
                    w[("win_frac_to_same=%.2f"%k[0], "win_genuine_other" if (k[1] and k[0]<=0.75) else "", "winQuad" if k[4] else "winTrip","winAlsoDup" if k[5] else "winSurv")]+=1
        print("   killed only by double-count:",dc, "; has genuine T5:",sum(r["has_t5"] for r in k1))
        for kk,v in w.most_common(8): print("   killer",kk,v)
        print("   victims quad?",collections.Counter(u["quad"] for r in k1 for u in r["usable"]))
    k2=[r for r in pop if fate(r).startswith("2")]
    if k2: print("   fate2: has genuine T5:",sum(r["has_t5"] for r in k2), " only-triplet usable:",sum(1 for r in k2 if not any(u["quad"] for u in r["usable"] if not u["isdup"])))
# triplet seeds and pT5: among B, genuine pLS triplet-only
tri=[r for r in B if not any(r["gen_pls_quad"])]
print("\nB triplet-only:",len(tri)," has genuine T5:",sum(r["has_t5"] for r in tri), " any surviving usable seed built pT5:",sum(1 for r in tri if any(u["n_pt5"]>0 for u in r["usable"] if not u["isdup"])))
qd=[r for r in B if any(r["gen_pls_quad"])]
print("B with quad genuine:",len(qd)," has genuine T5:",sum(r["has_t5"] for r in qd))
# successes reference: fraction triplet-only
print("successes whose genuine pLS are triplet-only:",sum(1 for r in succ if r["has_gen_pls"] and not any(r["gen_pls_quad"])),"/",len(succ))
# pT5 per usable surviving pLS by quad in fails without pT5
