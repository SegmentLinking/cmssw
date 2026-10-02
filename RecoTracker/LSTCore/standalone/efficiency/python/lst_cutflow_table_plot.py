#!/usr/bin/env python3
"""
lst_cutflow_table_plot.py

Renders the T5/pT5 cut-flow table as a PNG.
"""

import numpy as np
import uproot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── structural palette (no categorical hues — this is a table) ────────────────
C_HDR_BG   = "#1e293b"   # column header band
C_HDR_FG   = "#f8fafc"   # column header text
C_SEP_BG   = "#dde3ed"   # section-break row
C_ROW_A    = "#ffffff"   # row background A
C_ROW_B    = "#f4f6fb"   # row background B (alternating)
C_LBL      = "#0f172a"   # row label — primary ink
C_LBL_SUB  = "#334155"   # sub-row label
C_NUM      = "#0f172a"   # count text
C_PCT      = "#64748b"   # percentage text (muted)
C_GRID     = "#cbd5e1"   # cell borders / grid

STANDALONE = "/mnt/data1/kk829/CMSSW_16_1_1/src/RecoTracker/LSTCore/standalone"

# Tuples: (column label, ntuple path, nevents, pt_min_GeV or None)
COLUMNS = [
    ("ttbar\nABon",           f"{STANDALONE}/LSTNtuple-ttbar-50-ABon2.root",                      50, None),
    ("ttbar ABon\npT>10 GeV", f"{STANDALONE}/LSTNtuple-ttbar-50-ABon2.root",                      50, 10.0),
    ("QCD\nABon",             f"{STANDALONE}/LSTNtuple_allobj_ABon_ideal.root",                   50, None),
    ("QCD ABoff\npairwise",   f"{STANDALONE}/LSTNtuple_allobj3.root",                             50, None),
    ("QCD score\n<12",        f"{STANDALONE}/LSTNtuple_allobj5.root",                             50, None),
    ("QCD remove-\npixel off", f"{STANDALONE}/LSTNtuple_allobj4.root",                            50, None),
]

B_AB      = 0x01
B_BTC     = 0x0E
B_PT5PRIO = 0x08
B_CC      = 0x10


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
        t5_simidx     = tree["t5_simIdx"].array(library="np",       entry_stop=stop)
        pt5_simidx    = tree["pT5_simIdx"].array(library="np",      entry_stop=stop)
        sim_trkntupidx = tree["sim_trkNtupIdx"].array(library="np", entry_stop=stop)
        sim_pt        = tree["sim_pt"].array(library="np",           entry_stop=stop)
        t5_ptmask  = _build_pt_mask(t5_simidx,  sim_trkntupidx, sim_pt, pt_min)
        pt5_ptmask = _build_pt_mask(pt5_simidx, sim_trkntupidx, sim_pt, pt_min)
    else:
        t5_ptmask  = None
        pt5_ptmask = None

    return t5, partofpt5, tried, pt5, t5_ptmask, pt5_ptmask


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

    surv_ab     = t5_formed - killed_ab

    pt5_matched = sum(int(np.sum(m & (np.array(p, dtype=np.int32) != 0)))
                      for m, p in zip(ab_surv_f, pop_arr))
    tried_pt5   = sum(int(np.sum(m & (np.array(t, dtype=np.int32) != 0)))
                      for m, t in zip(ab_surv_f, tried_arr))
    killed_btc  = sum(int(np.sum(m & ((np.array(ev, dtype=np.int32) & B_BTC) != 0)))
                      for m, ev in zip(ab_surv_f, t5_arr))
    pt5prio     = sum(int(np.sum(m & ((np.array(ev, dtype=np.int32) & B_PT5PRIO) != 0)))
                      for m, ev in zip(ab_surv_f, t5_arr))
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

    surv_pt5    = pt5_formed - killed_pt5

    return dict(t5_formed=t5_formed, killed_ab=killed_ab, surv_ab=surv_ab,
                tried_pt5=tried_pt5, pt5_matched=pt5_matched,
                pt5_formed=pt5_formed, killed_btc=killed_btc,
                pt5prio=pt5prio, crossclean=crossclean, surv_btc=surv_btc,
                killed_pt5=killed_pt5, surv_pt5=surv_pt5)


# ── row spec ──────────────────────────────────────────────────────────────────
# (label, is_sep, key, pct_denom_key, indent)
#   is_sep=True  → thin grey separator band (no data)
#   indent 0=bold toplevel, 1=sub, 2=sub-sub
ROWS = [
    ("T5s formed",                    False, "t5_formed",   None,         0),
    ("Killed by AfterBuild",          False, "killed_ab",   "t5_formed",  1),
    ("T5s surviving AfterBuild",      False, "surv_ab",     "t5_formed",  1),
    ("T5s tried in pT5 building",     False, "tried_pt5",   "surv_ab",    1),
    ("T5s pT5-matched (partOfPT5)",   False, "pt5_matched", "surv_ab",    1),
    ("",                              True,  None,          None,         0),
    ("pT5s formed",                   False, "pt5_formed",  None,         0),
    ("",                              True,  None,          None,         0),
    ("Killed by BeforeTC (T5s)",      False, "killed_btc",  "surv_ab",    1),
    ("  of which isPT5-priority (bit3)", False, "pt5prio",  "killed_btc", 2),
    ("Killed by CrossClean (bit4)",   False, "crossclean",  "surv_ab",    1),
    ("T5s surviving BeforeTC",        False, "surv_btc",    "surv_ab",    1),
    ("",                              True,  None,          None,         0),
    ("Killed by pT5 dedup",           False, "killed_pt5",  "pt5_formed", 1),
    ("pT5s surviving (→ TC)",    False, "surv_pt5",    "pt5_formed", 1),
]


def fmt_count(n):
    return f"{int(n):,}"


def fmt_pct(n, d):
    if not d:
        return "—"
    return f"({100.0 * n / d:.1f}%)"


def draw_table(data_list, col_labels, out_path):
    n_cols   = len(col_labels)
    lbl_w    = 3.1    # inches — label column
    dat_w    = 2.45   # inches — each data column
    fig_w    = lbl_w + n_cols * dat_w

    ROW_H  = 0.54   # normal data row height (inches)
    HDR_H  = 0.68   # column header row height
    SEP_H  = 0.14   # separator band height
    PAD    = 0.08   # top/bottom padding

    total_h = PAD + HDR_H + sum(SEP_H if r[1] else ROW_H for r in ROWS) + PAD

    fig, ax = plt.subplots(figsize=(fig_w, total_h))
    fig.patch.set_facecolor("white")
    ax.set_xlim(0, fig_w)
    ax.set_ylim(0, total_h)
    ax.axis("off")

    def rect(x, y, w, h, fc, ec="none", lw=0.5, zorder=1):
        ax.add_patch(mpatches.FancyBboxPatch(
            (x, y), w, h, boxstyle="square,pad=0",
            facecolor=fc, edgecolor=ec, linewidth=lw, zorder=zorder))

    def hline(y, x0=0, x1=None, lw=0.4, color=C_GRID, zorder=4):
        ax.plot([x0, x1 if x1 else fig_w], [y, y], color=color, lw=lw, zorder=zorder)

    def vline(x, y0=0, y1=None, lw=0.4, color=C_GRID, zorder=4):
        ax.plot([x, x], [y0, y1 if y1 else total_h], color=color, lw=lw, zorder=zorder)

    # ── header row ────────────────────────────────────────────────────────────
    y = total_h - PAD
    rect(0, y - HDR_H, fig_w, HDR_H, fc=C_HDR_BG)
    ax.text(0.12, y - HDR_H / 2, "Step",
            va="center", ha="left", fontsize=9, fontweight="bold",
            color=C_HDR_FG, zorder=5)

    for j, lbl in enumerate(col_labels):
        cx = lbl_w + j * dat_w + dat_w / 2
        lines = lbl.split("\n")
        # vertically centre two-line headers
        for li, line in enumerate(lines):
            dy = (0.5 - li) * 0.19 if len(lines) > 1 else 0
            ax.text(cx, y - HDR_H / 2 + dy, line,
                    va="center", ha="center", fontsize=8.5, fontweight="bold",
                    color=C_HDR_FG, zorder=5)

    hline(y, lw=0.8)
    hline(y - HDR_H, lw=0.8)
    y -= HDR_H

    # ── data rows ────────────────────────────────────────────────────────────
    toggle = False
    for label, is_sep, key, pct_key, indent in ROWS:
        if is_sep:
            rect(0, y - SEP_H, fig_w, SEP_H, fc=C_SEP_BG)
            hline(y,         lw=0.6, color=C_GRID)
            hline(y - SEP_H, lw=0.6, color=C_GRID)
            y -= SEP_H
            continue

        h   = ROW_H
        bg  = C_ROW_A if toggle else C_ROW_B
        toggle = not toggle
        rect(0, y - h, fig_w, h, fc=bg)

        # label
        x_lbl = 0.12 + indent * 0.28
        fw    = "bold"  if indent == 0 else "normal"
        fs    = 8.5     if indent < 2  else 8.0
        fc    = C_LBL   if indent == 0 else C_LBL_SUB
        ax.text(x_lbl, y - h / 2, label,
                va="center", ha="left", fontsize=fs, fontweight=fw,
                color=fc, zorder=5)

        # data cells
        for j, d in enumerate(data_list):
            cx = lbl_w + j * dat_w + dat_w / 2

            n   = d[key]
            cnt = fmt_count(n)
            pct = fmt_pct(n, d[pct_key]) if pct_key else None

            if pct:
                ax.text(cx, y - h * 0.36, cnt,
                        va="center", ha="center",
                        fontsize=8.5, fontweight="semibold", color=C_NUM, zorder=5)
                ax.text(cx, y - h * 0.68, pct,
                        va="center", ha="center",
                        fontsize=7.5, color=C_PCT, zorder=5)
            else:
                ax.text(cx, y - h / 2, cnt,
                        va="center", ha="center",
                        fontsize=8.5, fontweight="bold", color=C_NUM, zorder=5)

        hline(y,     lw=0.3)
        hline(y - h, lw=0.3)
        y -= h

    # ── vertical dividers ─────────────────────────────────────────────────────
    vline(lbl_w,  y0=PAD, y1=total_h - PAD, lw=0.8)
    for j in range(1, n_cols):
        vline(lbl_w + j * dat_w, y0=PAD, y1=total_h - PAD, lw=0.4)

    # ── outer border ─────────────────────────────────────────────────────────
    rect(0, PAD, fig_w, total_h - 2 * PAD, fc="none", ec=C_GRID, lw=0.8, zorder=6)

    plt.savefig(out_path, dpi=180, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close()
    print(f"Saved → {out_path}")


def main():
    data_list  = []
    col_labels = []
    for lbl, fname, nevents, pt_min in COLUMNS:
        tag = fname.split('/')[-1]
        suffix = f" (pT>{pt_min} GeV)" if pt_min is not None else ""
        print(f"Loading {tag}{suffix} ...", flush=True)
        t5_arr, pop_arr, tried_arr, pt5_arr, t5_ptmask, pt5_ptmask = load(fname, nevents, pt_min)
        data_list.append(compute(t5_arr, pop_arr, tried_arr, pt5_arr, t5_ptmask, pt5_ptmask))
        col_labels.append(lbl)

    out = f"{STANDALONE}/cutflow_table.png"
    draw_table(data_list, col_labels, out)


if __name__ == "__main__":
    main()
