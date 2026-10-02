import pickle, collections, numpy as np, os
SD=os.path.dirname(os.path.abspath(__file__))
rows=pickle.load(open(os.path.join(SD,"core_rows.pkl"),"rb"))
# conversion: core tracks with genuine pLS surviving dedup & genuine T5 -> reco rate
g=[r for r in rows if r["has_gen_pls"] and r["has_t5"]]
gs=[r for r in g if not all(r["gen_isdup"]) ]
print("core w/ genuine pLS & genuine T5:",len(g)," eff %.3f"%np.mean([not r["fail"] for r in g]))
print("  ... with >=1 genuine pLS surviving dedup:",len(gs)," eff %.3f"%np.mean([not r["fail"] for r in gs]))
for q in (True,False):
    s=[r for r in gs if any(r["gen_pls_quad"])==q]
    print(f"  quad-genuine={q}: n={len(s)} eff {np.mean([not r['fail'] for r in s]):.3f}")
k=[r for r in g if all(r["gen_isdup"])]
print("  ... ALL genuine pLS dedup-killed:",len(k)," eff %.3f"%np.mean([not r["fail"] for r in k]))
fails=[r for r in rows if r["fail"]]
B=[r for r in fails if r["has_gen_pls"] and not r["has_pt5"]]
kb=[r for r in B if r["all_gen_killed"]]
print("B all-perfect-killed:",len(kb)," with genuine T5:",sum(r["has_t5"] for r in kb),
      " killed only via doublecount:",sum(1 for r in kb if all(k[2]<3 for k in r["killer_info"])),
      " dc-only & T5:",sum(1 for r in kb if r["has_t5"] and all(k[2]<3 for k in r["killer_info"])))
print("victim quad in kb:",collections.Counter(q for r in kb for q in r["gen_pls_quad"]))
# pair seed tracks
A=[r for r in fails if not r["has_gen_pls"]]
p=[r for r in A if r["best_by_algo"].get(6,0)>=1.0]
print("pair-perfect no-pLS fails:",len(p),"genuine T5:",sum(r["has_t5"] for r in p),"pt median %.0f"%np.median([r["pt"] for r in p]),
      "no usable LST pLS:",sum(1 for r in p if not r["usable"]))
# M1 3/4 tracks
m1=[r for r in A if r["best_lst"]>=0.75-1e-6 and r["best_all"]<1]
print("M1:",len(m1),"has genuine pT5 (built+killed):",sum(r["has_pt5"] for r in m1),"T5:",sum(r["has_t5"] for r in m1))
