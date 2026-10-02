#!/usr/bin/env python3
"""
lst_cutflow_table_plot_scorefix.py

Cut-flow table for the three scorefix CPU runs (pT5 dedup window 0.2→0.01).
"""

import numpy as np
import uproot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

C_HDR_BG   = "#1e293b"
C_HDR_FG   = "#f8fafc"
C_SEP_BG   = "#dde3ed"
C_ROW_A    = "#ffffff"
C_ROW_B    = "#f4f6fb"
C_LBL      = "#0f172a"
C_LBL_SUB  = "#334155"
C_NUM      = "#0f172a"
C_PCT      = "#64748b"
C_GRID     = "#cbd5e1"

STANDALONE = "/mnt/data1/kk829/CMSSW_16_1_1/src/RecoTracker/LSTCore/standalone"

COLUMNS = [
    ("ttbar\nABon",        f"{STANDALONE}/LSTNtuple-ttbar-50-ABon_scorefix.root",          50),
    ("QCD ABon\nideal pLS", f"{STANDALONE}/LSTNtuple_ABon_scorefix_idealpls_50evt.root",   50),
    ("QCD ABoff\nideal pLS", f"{STANDALONE}/LSTNtuple_ABoff_scorefix_idealpls_50evt.root", 50),
]

B_AB      = 0x01
B_BTC     = 0x0E
B_PT5PRIO = 0x08
B_CC      = 0x10


def load(fname, nevents):
    tree = uproot.open(fname)["tree"]
    stop = nevents if nevents > 0 else None
    t5        = tree["t5_isDupBits"].array(library="np",  entry_stop=stop)
    partofpt5 = tree["t5_partOfPT5"].array(library="np",  entry_stop=stop)
    tried     = tree["t5_triedInPT5"].array(library="np", entry_stop=stop)
    pt5       = tree["pT5_isDupReco"].array(library="np", entry_stop=stop)
    return t5, partofpt5, tried, pt5


def compute(t5_arr, pop_arr, tried_arr, pt5_arr):
    ab_surv = [(np.array(ev, dtype=np.int32) & B_AB) == 0 for ev in t5_arr]
    t5_formed   = sum(len(ev) for ev in t5_arr)
    killed_ab   = sum(int(np.sum(~m)) for m in ab_surv)
    surv_ab     = t5_formed - killed_ab
    pt5_matched = sum(int(np.sum(m & (np.array(p, dtype=np.int32) != 0)))
                      for m, p in zip(ab_surv, pop_arr))
    tried_pt5   = sum(int(np.sum(m & (np.array(t, dtype=np.int32) != 0)))
                      for m, t in zip(ab_surv, tried_arr))
    killed_btc  = sum(int(np.sum(m & ((np.array(ev, dtype=np.int32) & B_BTC) != 0)))
                      for m, ev in zip(ab_surv, t5_arr))
    pt5prio     = sum(int(np.sum(m & ((np.array(ev, dtype=np.int32) & B_PT5PRIO) != 0)))
                      for m, ev in zip(ab_surv, t5_arr))
    crossclean  = sum(int(np.sum(m & ((np.array(ev, dtype=np.int32) & B_CC) != 0)))
                      for m, ev in zip(ab_surv, t5_arr))
    surv_btc    = sum(int(np.sum(np.array(ev, dtype=np.int32) == 0)) for ev in t5_arr)
    pt5_formed  = sum(len(ev) for ev in pt5_arr)
    killed_pt5  = sum(int(np.sum(np.array(ev, dtype=np.int32) != 0)) for ev in pt5_arr)
    surv_pt5    = pt5_formed - killed_pt5
    return dict(t5_formed=t5_formed, killed_ab=killed_ab, surv_ab=surv_ab,
                tried_pt5=tried_pt5, pt5_matched=pt5_matched,
                pt5_formed=pt5_formed, killed_btc=killed_btc,
                pt5prio=pt5prio, crossclean=crossclean, surv_btc=surv_btc,
                killed_pt5=killed_pt5, surv_pt5=surv_pt5)


ROWS = [
    ("T5s formed",                       False, "t5_formed",   None,         0),
    ("Killed by AfterBuild",             False, "killed_ab",   "t5_formed",  1),
    ("T5s surviving AfterBuild",         False, "surv_ab",     "t5_formed",  1),
    ("T5s tried in pT5 building",        False, "tried_pt5",   "surv_ab",    1),
    ("T5s pT5-matched (partOfPT5)",      False, "pt5_matched", "surv_ab",    1),
    ("",                                 True,  None,          None,         0),
    ("pT5s formed",                      False, "pt5_formed",  None,         0),
    ("",                                 True,  None,          None,         0),
    ("Killed by BeforeTC (T5s)",         False, "killed_btc",  "surv_ab",    1),
    ("  of which isPT5-priority (bit3)", False, "pt5prio",     "killed_btc", 2),
    ("Killed by CrossClean (bit4)",      False, "crossclean",  "surv_ab",    1),
    ("T5s surviving BeforeTC",           False, "surv_btc",    "surv_ab",    1),
    ("",                                 True,  None,          None,         0),
    ("Killed by pT5 dedup",              False, "killed_pt5",  "pt5_formed", 1),
    ("pT5s surviving (→ TC)",            False, "surv_pt5",    "pt5_formed", 1),
]


def fmt_count(n):
    return f"{int(n):,}"


def fmt_pct(n, d):
    if not d:
        return "—"
    return f"({100.0 * n / d:.1f}%)"


def draw_table(data_list, col_labels, out_path):
    n_cols = len(col_labels)
    lbl_w  = 3.1
    dat_w  = 2.45
    fig_w  = lbl_w + n_cols * dat_w

    ROW_H = 0.54
    HDR_H = 0.68
    SEP_H = 0.14
    PAD   = 0.08

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

    y = total_h - PAD
    rect(0, y - HDR_H, fig_w, HDR_H, fc=C_HDR_BG)
    ax.text(0.12, y - HDR_H / 2, "Step",
            va="center", ha="left", fontsize=9, fontweight="bold",
            color=C_HDR_FG, zorder=5)
    for j, lbl in enumerate(col_labels):
        cx = lbl_w + j * dat_w + dat_w / 2
        lines = lbl.split("\n")
        for li, line in enumerate(lines):
            dy = (0.5 - li) * 0.19 if len(lines) > 1 else 0
            ax.text(cx, y - HDR_H / 2 + dy, line,
                    va="center", ha="center", fontsize=8.5, fontweight="bold",
                    color=C_HDR_FG, zorder=5)
    hline(y, lw=0.8)
    hline(y - HDR_H, lw=0.8)
    y -= HDR_H

    toggle = False
    for label, is_sep, key, pct_key, indent in ROWS:
        if is_sep:
            rect(0, y - SEP_H, fig_w, SEP_H, fc=C_SEP_BG)
            hline(y,         lw=0.6, color=C_GRID)
            hline(y - SEP_H, lw=0.6, color=C_GRID)
            y -= SEP_H
            continue

        h  = ROW_H
        bg = C_ROW_A if toggle else C_ROW_B
        toggle = not toggle
        rect(0, y - h, fig_w, h, fc=bg)

        x_lbl = 0.12 + indent * 0.28
        fw    = "bold"   if indent == 0 else "normal"
        fs    = 8.5      if indent < 2  else 8.0
        fc    = C_LBL    if indent == 0 else C_LBL_SUB
        ax.text(x_lbl, y - h / 2, label,
                va="center", ha="left", fontsize=fs, fontweight=fw,
                color=fc, zorder=5)

        for j, d in enumerate(data_list):
            cx  = lbl_w + j * dat_w + dat_w / 2
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

    vline(lbl_w, y0=PAD, y1=total_h - PAD, lw=0.8)
    for j in range(1, n_cols):
        vline(lbl_w + j * dat_w, y0=PAD, y1=total_h - PAD, lw=0.4)

    rect(0, PAD, fig_w, total_h - 2 * PAD, fc="none", ec=C_GRID, lw=0.8, zorder=6)

    plt.savefig(out_path, dpi=180, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close()
    print(f"Saved → {out_path}")


def main():
    data_list  = []
    col_labels = []
    for lbl, fname, nevents in COLUMNS:
        print(f"Loading {fname.split('/')[-1]} ...", flush=True)
        t5_arr, pop_arr, tried_arr, pt5_arr = load(fname, nevents)
        data_list.append(compute(t5_arr, pop_arr, tried_arr, pt5_arr))
        col_labels.append(lbl)

    out = f"{STANDALONE}/cutflow_table_scorefix.png"
    draw_table(data_list, col_labels, out)


if __name__ == "__main__":
    main()
