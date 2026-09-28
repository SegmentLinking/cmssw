#!/usr/bin/env python3
"""Combined TC-stage pLS rules (pass2 mode x CrossCleanpLS overlap threshold), counted per extra pLS TC.
Embed part of CrossCleanpLS is only known for pLS that reached it; 'opt' assumes rescued pLS pass it."""
import os, pickle, collections
SD = os.path.dirname(os.path.abspath(__file__))
C = pickle.load(open(os.path.join(SD, "tcstage.pkl"), "rb"))

def becomes_tc(r, p2, k, embed_cut=True):
    if r["is_tc"]:
        return False  # already a TC
    if p2 == "master" and r["bit1"]: return False
    if p2 == "nms" and r["nms2"]: return False
    if any(o["pl"] == r["p"] for o in r["ov"]): return False  # its own pT5/pT3 is a TC
    ov = [o for o in r["ov"] if o["pl"] != r["p"]]
    if any(o["nsh"] >= k for o in ov) or any(o["dr2"] < 1e-6 for o in ov): return False
    if embed_cut and (r["isdup"] & 1) and not ov:  # was CC-flagged with no overlap -> embed flagged
        return False
    return True

print(f"{'pass2':8s}{'CC>=k':>6s}{'embed':>7s}{'extraTC':>9s}{'fake':>6s}{'dup':>6s}{'newSim':>7s}{'newDen':>7s}{'newCore':>8s}{'core>100':>9s}")
for p2 in ["master", "nms", "off"]:
    for k in [1, 2, 3, 99]:
        for emb in [True, False]:
            n = fake = 0
            new = set(); newden = set(); newcore = set(); hi = set(); cnt = collections.Counter()
            for r in C:
                if not becomes_tc(r, p2, k, emb):
                    continue
                n += 1
                if not r["sims"]:
                    fake += 1; continue
                if any(r["sim_hastc"]):
                    continue
                for s, co, de, pt in zip(r["sims"], r["sim_core"], r["sim_den"], r["sim_pt"]):
                    new.add((r["le"], s))
                    if de: newden.add((r["le"], s))
                    if co: newcore.add((r["le"], s))
                    if co and pt > 100: hi.add((r["le"], s))
            dup = n - fake - len(new)
            print(f"{p2:8s}{k:>6d}{str(emb):>7s}{n:>9d}{fake:>6d}{dup:>6d}{len(new):>7d}{len(newden):>7d}{len(newcore):>8d}{len(hi):>9d}")
