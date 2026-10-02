#!/usr/bin/env python3
"""Summaries of tcstage.pkl: quad pLS that survive CheckHitspLS pass 1 and what stops them becoming pLS TCs."""
import os, pickle, collections
SD = os.path.dirname(os.path.abspath(__file__))
C = pickle.load(open(os.path.join(SD, "tcstage.pkl"), "rb"))


def cls(r):
    if r["is_tc"]:
        return "pLS_TC"
    if r["bit1"]:
        return "bit1(pass2)"
    ov = [o for o in r["ov"] if o["pl"] != r["p"]]
    self_in = any(o["pl"] == r["p"] for o in r["ov"])
    if self_in:
        return "CC:self_in_pT5/pT3_TC"
    if not ov:
        return "CC:T5_embed"
    m = max(o["nsh"] for o in ov)
    if m == 0:
        return "CC:dR2<1e-6_only"
    return f"CC:overlap_max{m}hits"


def fate(r):
    if not r["sims"]:
        return "fake"
    if any(r["sim_hastc"]):
        return "sim_has_TC"
    return "sim_noTC" + ("_core" if any(r["sim_core"]) else ("_den" if any(r["sim_den"]) else "_nonden"))


tab = collections.Counter()
for r in C:
    tab[(cls(r), fate(r))] += 1
cats = sorted(set(k[0] for k in tab))
fates = ["fake", "sim_has_TC", "sim_noTC_core", "sim_noTC_den", "sim_noTC_nonden"]
print(f"{'quad pLS surviving pass1 (100 evt)':32s}" + "".join(f"{f:>16s}" for f in fates) + f"{'total':>8s}")
for c in cats:
    print(f"{c:32s}" + "".join(f"{tab[(c, f)]:>16d}" for f in fates) + f"{sum(tab[(c, f)] for f in fates):>8d}")

# unique sims rescued per rule (a sim counts once)
def rescued(rule):
    sims_core, sims_den, dup, fake = set(), set(), 0, 0
    for r in C:
        if not rule(r):
            continue
        if not r["sims"]:
            fake += 1; continue
        if any(r["sim_hastc"]):
            dup += 1; continue
        for s, co, de in zip(r["sims"], r["sim_core"], r["sim_den"]):
            if co: sims_core.add((r["le"], s))
            if de: sims_den.add((r["le"], s))
    return len(sims_core), len(sims_den), dup, fake

rules = {
    "R1 no pass2 (bit1 pLS w/o CC overlap->TC)": lambda r: (not r["is_tc"]) and r["bit1"] and not [o for o in r["ov"]],
    "R1' no pass2, upper (all bit1)": lambda r: r["bit1"],
    "R4 pass2 as NMS (bit1 & NMS-kept & no CC overlap)": lambda r: r["bit1"] and not r["nms2"] and not r["ov"],
    "R4' pass2 as NMS upper (bit1 & NMS-kept)": lambda r: r["bit1"] and not r["nms2"],
    "R2 CC overlap needs >=2 hits": lambda r: cls(r) == "CC:overlap_max1hits",
    "R2' CC overlap needs >=3 hits": lambda r: cls(r) in ("CC:overlap_max1hits", "CC:overlap_max2hits"),
    "R3 no CC T5-embed cut": lambda r: cls(r) == "CC:T5_embed",
}
print("\nrule -> extra pLS TCs: new core sims, new den sims, dup-TCs (sim already has TC), fake-TCs")
for k, f in rules.items():
    print(f"  {k:45s}", rescued(f))

# bit1 killers
print("\nbit1 victims: killer relation")
kk = collections.Counter()
for r in C:
    if r["bit1"] and not r["is_tc"]:
        for k in r["k2"]:
            rel = ("w_same_sim" if set(k["w_sims"]) & set(r["sims"]) else ("w_fake" if not k["w_sims"] else "w_other_sim"))
            kk[(fate(r), rel, "shared" if k["shared"] else "dR", f"nd{k['nd']}", "w_TC" if k["w_isdup"] == 0 else f"w_dup{k['w_isdup']}")] += 1
for k, v in sorted(kk.items(), key=lambda x: -x[1])[:25]:
    print("  ", k, v)

# CC overlap: TC that caused it
print("\nCC-overlap victims whose sim has no TC: overlapping TC type/fake/nsh")
oo = collections.Counter()
for r in C:
    if cls(r).startswith("CC:overlap") and fate(r).startswith("sim_noTC"):
        for o in r["ov"]:
            if o["pl"] != r["p"]:
                oo[(fate(r), "pT5" if o["ty"] == 7 else "pT3", "TCfake" if o["fake"] else "TCmatched", f"nsh{o['nsh']}")] += 1
for k, v in sorted(oo.items(), key=lambda x: -x[1]):
    print("  ", k, v)
