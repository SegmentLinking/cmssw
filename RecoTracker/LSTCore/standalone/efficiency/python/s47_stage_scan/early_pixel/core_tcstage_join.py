#!/usr/bin/env python3
"""Per failing jet-core track: what stops each genuine pLS from becoming a TC (quad) / a pixel TC seed."""
import os, pickle, collections
SD = os.path.dirname(os.path.abspath(__file__))
R = pickle.load(open(os.path.join(SD, "core_rows.pkl"), "rb"))["rows"]
C = {(r["le"], r["p"]): r for r in pickle.load(open(os.path.join(SD, "tcstage.pkl"), "rb"))}
RANK = ["pLS_TC?", "CC:T5_embed", "CC:dR", "CC:overlap1", "CC:overlap2", "CC:overlap3+", "bit1(pass2)",
        "CC:self_in_fake_pT5/pT3_TC", "pass1_killed_quad", "triplet_only(pass1-alive)", "triplet_only(pass1-killed)"]

def status(r, gd):
    if not gd["quad"]:
        return "triplet_only(pass1-killed)" if gd["pass1"] else "triplet_only(pass1-alive)"
    if gd["pass1"]:
        return "pass1_killed_quad"
    c = C[(r["le"], gd["p"])]
    if c["is_tc"]:
        return "pLS_TC?"
    if c["bit1"]:
        return "bit1(pass2)"
    if any(o["pl"] == gd["p"] for o in c["ov"]):
        return "CC:self_in_fake_pT5/pT3_TC"
    ov = [o for o in c["ov"] if o["pl"] != gd["p"]]
    if not ov:
        return "CC:T5_embed"
    m = max(o["nsh"] for o in ov)
    return "CC:dR" if m == 0 else ("CC:overlap%d" % m if m < 3 else "CC:overlap3+")

tab = collections.Counter(); hi = collections.Counter()
for r in R:
    if not r["fail"] or not r["gdet"]:
        continue
    st = [status(r, gd) for gd in r["gdet"]]
    best = min(st, key=RANK.index)
    tab[(r["bucket"], best)] += 1
    if r["pt"] > 100: hi[best] += 1
bk = ["pT5_all_killed", "pair_failed", "no_T5", "pLS_flagged"]
print(f"{'least-blocked genuine pLS status':32s}" + "".join(f"{b:>16s}" for b in bk) + f"{'total':>7s}{'pT>100':>8s}")
for s in RANK:
    row = [tab[(b, s)] for b in bk]
    if sum(row):
        print(f"{s:32s}" + "".join(f"{x:>16d}" for x in row) + f"{sum(row):>7d}{hi[s]:>8d}")
