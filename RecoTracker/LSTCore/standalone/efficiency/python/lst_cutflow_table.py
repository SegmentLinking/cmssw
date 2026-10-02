#!/usr/bin/env python3
"""
lst_cutflow_table.py

T5 and pT5 dedup cut-flow table across five configurations.

isDupBits encoding:
  bit0 (0x01) = RemoveDupQuintupletsAfterBuild
  bit1 (0x02) = BeforeTC condA
  bit2 (0x04) = BeforeTC condB
  bit3 (0x08) = BeforeTC isPT5-priority
  bit4 (0x10) = CrossCleanT5

pT5_isDupReco: 1 if killed by RemoveDupPixelQuintupletsFromMap
"""

import numpy as np
import uproot

STANDALONE = "/mnt/data1/kk829/CMSSW_16_1_1/src/RecoTracker/LSTCore/standalone"

COLUMNS = [
    ("ttbar\nABon",           f"{STANDALONE}/LSTNtuple-ttbar-50-ABon2.root",                       50, None),
    ("ttbar ABon\npT>10 GeV", f"{STANDALONE}/LSTNtuple-ttbar-50-ABon2.root",                       50, 10.0),
    ("QCD\nABon",             f"{STANDALONE}/LSTNtuple_allobj_ABon_ideal.root",                    50, None),
    ("QCD ABoff\npairwise",   f"{STANDALONE}/LSTNtuple_allobj3.root",                              50, None),
    ("QCD score\n<12",        f"{STANDALONE}/LSTNtuple_allobj5.root",                              50, None),
    ("QCD removepixel\noff",  f"{STANDALONE}/LSTNtuple_allobj4.root",                              50, None),
]

B_AB      = 0x01
B_BTC     = 0x0E  # bits 1-3: BeforeTC condA | condB | isPT5-priority
B_PT5PRIO = 0x08  # bit3: isPT5-priority kill
B_CC      = 0x10  # CrossClean


def _build_pt_mask(simidx_arr, sim_trkntupidx_arr, sim_pt_arr, pt_min):
    """Build per-event boolean masks selecting objects whose matched sim track has pT > pt_min.

    t5_simIdx / pT5_simIdx store tracking-ntuple sim indices, while sim_pt is indexed over the
    LST-ntuple's selected sim track subset.  sim_trkNtupIdx maps LST sim index → tracking-ntuple
    sim index, so we invert it to resolve the pT for each object.
    """
    masks = []
    for ev_simidx, ev_trkntupidx, ev_simpt in zip(simidx_arr, sim_trkntupidx_arr, sim_pt_arr):
        rev_map = {int(tni): i for i, tni in enumerate(ev_trkntupidx)}
        sidx = np.array(ev_simidx, dtype=np.int64)
        spt  = np.array(ev_simpt,  dtype=np.float32)
        mask = np.zeros(len(sidx), dtype=bool)
        for j in np.where(sidx >= 0)[0]:
            lst_i = rev_map.get(int(sidx[j]), -1)
            if lst_i >= 0:
                mask[j] = spt[lst_i] > pt_min
        masks.append(mask)
    return masks


def load(fname, nevents, pt_min=None):
    tree = uproot.open(fname)["tree"]
    stop = nevents if nevents > 0 else None
    t5        = tree["t5_isDupBits"].array(library="np",  entry_stop=stop)
    partofpt5 = tree["t5_partOfPT5"].array(library="np",  entry_stop=stop)
    tried     = tree["t5_triedInPT5"].array(library="np", entry_stop=stop)
    pt5       = tree["pT5_isDupReco"].array(library="np", entry_stop=stop)

    if pt_min is not None:
        t5_simidx      = tree["t5_simIdx"].array(library="np",       entry_stop=stop)
        pt5_simidx     = tree["pT5_simIdx"].array(library="np",      entry_stop=stop)
        sim_trkntupidx = tree["sim_trkNtupIdx"].array(library="np",  entry_stop=stop)
        sim_pt         = tree["sim_pt"].array(library="np",           entry_stop=stop)
        t5_ptmask  = _build_pt_mask(t5_simidx,  sim_trkntupidx, sim_pt, pt_min)
        pt5_ptmask = _build_pt_mask(pt5_simidx, sim_trkntupidx, sim_pt, pt_min)
    else:
        t5_ptmask  = None
        pt5_ptmask = None

    return t5, partofpt5, tried, pt5, t5_ptmask, pt5_ptmask


def pct(n, d):
    if d == 0:
        return "—"
    return f"{100.0 * n / d:.1f}%"


def fmt(n):
    return f"{int(n):,}"


def compute(t5_arr, pop_arr, tried_arr, pt5_arr, t5_ptmask=None, pt5_ptmask=None):
    ab_surv = [(np.array(ev, dtype=np.int32) & B_AB) == 0 for ev in t5_arr]

    if t5_ptmask is not None:
        t5_formed      = sum(int(np.sum(pm)) for pm in t5_ptmask)
        killed_ab      = sum(int(np.sum(pm & ~m)) for pm, m in zip(t5_ptmask, ab_surv))
        ab_surv_f      = [pm & m for pm, m in zip(t5_ptmask, ab_surv)]
        surv_btc_masks = [pm & (np.array(ev, dtype=np.int32) == 0)
                          for pm, ev in zip(t5_ptmask, t5_arr)]
    else:
        t5_formed      = sum(len(ev) for ev in t5_arr)
        killed_ab      = sum(int(np.sum(~m)) for m in ab_surv)
        ab_surv_f      = ab_surv
        surv_btc_masks = [(np.array(ev, dtype=np.int32) == 0) for ev in t5_arr]

    surv_ab = t5_formed - killed_ab

    # pT5-matched and tried: among T5s that survived AfterBuild (and pass pT cut)
    pt5_matched = sum(int(np.sum(m & (np.array(p, dtype=np.int32) != 0)))
                      for m, p in zip(ab_surv_f, pop_arr))
    tried_pt5   = sum(int(np.sum(m & (np.array(t, dtype=np.int32) != 0)))
                      for m, t in zip(ab_surv_f, tried_arr))

    # BeforeTC: bit0 clear AND any of bits 1-3 set
    killed_btc  = sum(int(np.sum(m & ((np.array(ev, dtype=np.int32) & B_BTC) != 0)))
                      for m, ev in zip(ab_surv_f, t5_arr))
    # isPT5-priority sub-count (bit3, among AB survivors)
    pt5prio     = sum(int(np.sum(m & ((np.array(ev, dtype=np.int32) & B_PT5PRIO) != 0)))
                      for m, ev in zip(ab_surv_f, t5_arr))
    # CrossClean (bit4, among AB survivors)
    crossclean  = sum(int(np.sum(m & ((np.array(ev, dtype=np.int32) & B_CC) != 0)))
                      for m, ev in zip(ab_surv_f, t5_arr))
    surv_btc    = sum(int(np.sum(m)) for m in surv_btc_masks)

    if pt5_ptmask is not None:
        pt5_formed = sum(int(np.sum(pm)) for pm in pt5_ptmask)
        killed_pt5 = sum(int(np.sum(pm & (np.array(ev, dtype=np.int32) != 0)))
                         for pm, ev in zip(pt5_ptmask, pt5_arr))
    else:
        pt5_formed = sum(len(ev) for ev in pt5_arr)
        killed_pt5 = sum(int(np.sum(np.array(ev, dtype=np.int32) != 0)) for ev in pt5_arr)

    surv_pt5 = pt5_formed - killed_pt5

    return dict(
        t5_formed   = t5_formed,
        killed_ab   = killed_ab,
        surv_ab     = surv_ab,
        pt5_matched = pt5_matched,
        tried_pt5   = tried_pt5,
        pt5_formed  = pt5_formed,
        killed_btc  = killed_btc,
        pt5prio     = pt5prio,
        crossclean  = crossclean,
        surv_btc    = surv_btc,
        killed_pt5  = killed_pt5,
        surv_pt5    = surv_pt5,
    )


def main():
    results = []
    for label, fname, nevents, pt_min in COLUMNS:
        tag = fname.split('/')[-1]
        suffix = f" (pT>{pt_min} GeV)" if pt_min is not None else ""
        print(f"Loading {tag}{suffix} ...", flush=True)
        t5_arr, pop_arr, tried_arr, pt5_arr, t5_ptmask, pt5_ptmask = load(fname, nevents, pt_min)
        results.append((label, compute(t5_arr, pop_arr, tried_arr, pt5_arr, t5_ptmask, pt5_ptmask)))

    # ── table ─────────────────────────────────────────────────────────────────
    col_labels = [label for label, _ in results]
    data       = [d for _, d in results]

    # column widths
    col_w = 22
    row_w = 38

    hdr_lines = max(lbl.count("\n") + 1 for lbl in col_labels)

    def pad_header(lbl):
        lines = lbl.split("\n")
        while len(lines) < hdr_lines:
            lines.insert(0, "")
        return lines

    # Print header
    print()
    sep = "+" + ("-" * row_w) + "+" + (("+" + "-" * col_w) * len(results)) + "+"
    print(sep)
    for i in range(hdr_lines):
        row = "|" + " " * row_w + "|"
        for lbl, _ in results:
            lines = pad_header(lbl)
            row += lines[i].center(col_w) + "|"
        print(row)
    print(sep)

    def print_row(label, cells):
        row = "| " + label.ljust(row_w - 1) + "|"
        for cell in cells:
            row += cell.center(col_w) + "|"
        print(row)

    # T5s formed
    print_row("T5s formed",
              [fmt(d["t5_formed"]) for d in data])

    # Killed by AfterBuild
    print_row("  Killed by AfterBuild",
              [f"{fmt(d['killed_ab'])} ({pct(d['killed_ab'], d['t5_formed'])})" for d in data])

    # T5s surviving AfterBuild
    print_row("  T5s surviving AfterBuild",
              [f"{fmt(d['surv_ab'])} ({pct(d['surv_ab'], d['t5_formed'])})" for d in data])

    # T5s tried in pT5 building (among AB survivors)
    print_row("  T5s tried in pT5 building",
              [f"{fmt(d['tried_pt5'])} ({pct(d['tried_pt5'], d['surv_ab'])})" for d in data])

    # T5s pT5-matched (partOfPT5, among AB survivors)
    print_row("  T5s pT5-matched (partOfPT5)",
              [f"{fmt(d['pt5_matched'])} ({pct(d['pt5_matched'], d['surv_ab'])})" for d in data])

    print(sep)

    # pT5s formed
    print_row("pT5s formed",
              [fmt(d["pt5_formed"]) for d in data])

    print(sep)

    # BeforeTC (T5s)
    print_row("  Killed by BeforeTC (T5s)",
              [f"{fmt(d['killed_btc'])} ({pct(d['killed_btc'], d['surv_ab'])})" for d in data])

    # isPT5-priority sub-count
    print_row("    of which isPT5-priority (bit3)",
              [f"{fmt(d['pt5prio'])} ({pct(d['pt5prio'], d['killed_btc'])})" for d in data])

    # CrossClean
    print_row("  Killed by CrossClean (bit4)",
              [f"{fmt(d['crossclean'])} ({pct(d['crossclean'], d['surv_ab'])})" for d in data])

    # T5s surviving BeforeTC
    print_row("  T5s surviving BeforeTC",
              [f"{fmt(d['surv_btc'])} ({pct(d['surv_btc'], d['surv_ab'])})" for d in data])

    print(sep)

    # pT5 dedup
    print_row("  Killed by pT5 dedup",
              [f"{fmt(d['killed_pt5'])} ({pct(d['killed_pt5'], d['pt5_formed'])})" for d in data])

    # pT5s surviving → TC
    print_row("  pT5s surviving (→ TC)",
              [f"{fmt(d['surv_pt5'])} ({pct(d['surv_pt5'], d['pt5_formed'])})" for d in data])

    print(sep)
    print()


if __name__ == "__main__":
    main()
