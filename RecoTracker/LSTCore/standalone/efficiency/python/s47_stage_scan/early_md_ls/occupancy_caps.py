#!/usr/bin/env python3
"""Check whether any per-module cap / truncation bites (current code).

For each event compare totOccupancy* (attempted adds, written in write_lst_ntuple.cc:2626-2632)
with the number of objects actually stored. Any excess == "excess alert" truncation.
Also report the max per-module T3 occupancy vs kNTripletThreshold=1000 (dense T5 path,
interface/alpaka/Common.h:46) and max per-module T5 occupancy vs kNQuintupletThreshold=100000.
Usage: occupancy_caps.py <ntuple>
"""
import sys
import numpy as np
import uproot

fn = sys.argv[1]
t = uproot.open(fn)["tree"]
br = ["md_occupancies", "sg_occupancies", "t3_occupancies", "t5_occupancies", "t4_occupancies",
      "md_isPLS", "ls_isPLS", "t3_pt", "t5_pt", "module_subdets", "module_layers", "module_rings",
      "md_detId", "t3_hit_0_detId"]
br = [b for b in br if b in t.keys()]
A = t.arrays(br, library="np")
n = len(A["md_occupancies"])
tot = dict(md=[0, 0], ls=[0, 0], t3=[0, 0], t5=[0, 0])
mx = dict(md=[], ls=[], t3=[], t5=[])
top_t3 = []
for e in range(n):
    mdo = np.asarray(A["md_occupancies"][e]); sgo = np.asarray(A["sg_occupancies"][e])
    t3o = np.asarray(A["t3_occupancies"][e]); t5o = np.asarray(A["t5_occupancies"][e])
    md_ot = int((~np.asarray(A["md_isPLS"][e]).astype(bool)).sum())
    ls_ot = int((~np.asarray(A["ls_isPLS"][e]).astype(bool)).sum())
    # last entry of md/sg occupancies is the pixel module
    tot["md"][0] += mdo[:-1].sum(); tot["md"][1] += md_ot
    tot["ls"][0] += sgo[:-1].sum(); tot["ls"][1] += ls_ot
    tot["t3"][0] += t3o.sum(); tot["t3"][1] += len(A["t3_pt"][e])
    tot["t5"][0] += t5o.sum(); tot["t5"][1] += len(A["t5_pt"][e])
    mx["md"].append(mdo[:-1].max()); mx["ls"].append(sgo[:-1].max()); mx["t3"].append(t3o.max()); mx["t5"].append(t5o.max())
    i = int(np.argmax(t3o))
    top_t3.append((t3o[i], e, i, A["module_subdets"][e][i], A["module_layers"][e][i]))
print(f"file {fn}  events {n}")
print(f"{'obj':4s} {'sum totOccupancy':>18s} {'stored':>10s} {'excess(truncated)':>18s}   per-module max over events (max, median)")
for k in ["md", "ls", "t3", "t5"]:
    a, s = tot[k]
    print(f"{k:4s} {a:>18d} {s:>10d} {a - s:>18d}   {max(mx[k]):>6d} {int(np.median(mx[k])):>6d}")
print("kNTripletThreshold=1000 (dense T5/T4 path) -> events with some module >=1000 T3:",
      sum(1 for m in mx["t3"] if m >= 1000))
print("kNQuintupletThreshold=100000 -> max T5 per module:", max(mx["t5"]))
top_t3.sort(reverse=True)
print("top-5 per-module T3 occupancies (occ, evt, lowerModIdx, subdet, layer):", top_t3[:5])
