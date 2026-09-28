#!/usr/bin/env python3
"""Plot the per-object rescue-threshold distributions written by s48_dedup_threshold_hists.py.

For each scanned dedup cut: objects killed at the master default, binned by the cut value at which
they would be kept (loosening runs left -> right), stacked by what keeping them would add:
  recovers a missed track  - first copy admitted for an efficiency-denominator sim with no TC
  extra copy               - genuine, but its sim already has a TC (or an earlier-admitted copy)
  genuine, outside denom.  - genuine for a sim outside the efficiency denominator
  fake                     - no sim with > 75% matched hits
  not TC-eligible          - (BeforeTC only) T5 is part of a pT5 or fails tightCutFlag
Bottom panel: cumulative objects admitted vs cut value, with the oat7 scan points marked.

Usage: s48_plot_dedup_thresholds.py [records.pkl] [outdir]
"""
import sys, os, pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PKL = sys.argv[1] if len(sys.argv) > 1 else 'plots-files/s48_dedup_thresholds/records.pkl'
OUT = sys.argv[2] if len(sys.argv) > 2 else os.path.dirname(PKL)
sys.argv = sys.argv[:1]  # the replay module parses argv at import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from s48_dedup_threshold_hists import DEF, OAT7, KIND
CORE = 0.02

CATS = ['recovers a missed track', 'extra copy of a found track', 'genuine, outside eff. denominator', 'fake',
        'not TC-eligible (in pT5 / tight cut)']
COL = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4']
TITLE = dict(PT5_NM='pT5 dedup (FromMap): min. shared hits to call a duplicate',
             AB_NM='T5 AfterBuild dedup: min. shared hits to call a duplicate',
             BTC_NM='T5 BeforeTC dedup: min. shared hits (with DNN d² < D2L)',
             BTC_DR2T='T5 BeforeTC dedup: tight ΔR² limit (with DNN d² < D2L)',
             BTC_D2L='T5 BeforeTC dedup: loose DNN-embedding d² limit',
             BTC_DR2L='T5 BeforeTC dedup: loose ΔR² limit (with DNN d² < D2T)',
             BTC_D2T='T5 BeforeTC dedup: tight DNN-embedding d² limit')
OBJ = dict(PT5_NM='pT5s', AB_NM='T5s', BTC_NM='T5s', BTC_DR2T='T5s', BTC_D2L='T5s', BTC_DR2L='T5s', BTC_D2T='T5s')
RECKEY = dict(PT5_NM='PT5_NM', AB_NM='T5_AB', BTC_NM='BTC_NM', BTC_DR2T='BTC_DR2T', BTC_D2L='BTC_D2L',
              BTC_DR2L='BTC_DR2L', BTC_D2T='BTC_D2T')

plt.rcParams.update({'font.size': 10, 'axes.edgecolor': '#52514e', 'axes.labelcolor': '#0b0b0b',
                     'xtick.color': '#52514e', 'ytick.color': '#52514e', 'axes.spines.top': False,
                     'axes.spines.right': False, 'figure.facecolor': '#fcfcfb', 'axes.facecolor': '#fcfcfb',
                     'grid.color': '#e4e3df', 'grid.linewidth': 0.6})


def classify(param, recs, sims, core):
    """Returns per-object arrays (x, cat, killed, rescuable, kept) over all events."""
    kind, d = KIND[param], DEF[param]
    xs, cats, killed_l, resc_l = [], [], [], []
    for r, s in zip(recs, sims):
        m = r['mask'] & ((r['dr'] < CORE) if core else True)
        v, sim, elig = r['v'][m], r['sim'][m], r['elig'][m]
        if kind == 'int_up':
            killed = v >= d
            resc = np.isfinite(v)
            x = v + 1                    # smallest cut value that keeps the object
            order = x
        else:
            killed = v < d
            resc = np.isfinite(v) & (v > -np.inf)
            x = v                        # largest cut value that keeps the object
            order = -x
        gen = sim >= 0
        simc = np.where(gen, sim, 0)
        in_den = gen & ((s['den'][simc] & (s['dr'][simc] < CORE)) if core else s['den_all'][simc])
        found = gen & s['found'][simc]
        cat = np.full(len(v), 1)
        cat[gen & ~in_den] = 2
        cat[~gen] = 3
        cat[gen & ~elig] = 4
        # first admitted (in loosening order) killed, rescuable, eligible copy of each missed denominator sim
        cand = np.where(in_den & ~found & killed & resc & elig)[0]
        if len(cand):
            o = cand[np.lexsort((order[cand], sim[cand]))]
            first = np.r_[True, sim[o][1:] != sim[o][:-1]]
            cat[o[first]] = 0
        xs.append(x); cats.append(cat); killed_l.append(killed); resc_l.append(resc)
    return (np.concatenate(xs), np.concatenate(cats), np.concatenate(killed_l), np.concatenate(resc_l))


def plot_param(param, recs, sims, core, nev, table):
    x, cat, killed, resc = classify(param, recs, sims, core)
    kind, d = KIND[param], DEF[param]
    sel = killed & resc
    kept = ~killed
    n_unresc = int((killed & ~resc).sum())
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True, gridspec_kw=dict(height_ratios=[1.1, 1]))
    present = [c for c in range(5) if ((cat == c) & sel).any() or c < 4]
    if kind == 'int_up':
        hi = int(np.nanmax(x[sel])) if sel.any() else d + 1
        edges = np.arange(d + 0.5, hi + 1.5, 1.0)
        centers = np.arange(d + 1, hi + 1)
        grid = centers.astype(float)
        adm = lambda c, g: ((cat == c) & sel & (x <= g)).sum()
        ax2.set_xlabel(f'{param} cut value (default {d}); loosening →')
        width = 0.8
    else:
        xv = x[sel]
        lo = max(np.min(xv[xv > 0]) * 0.95 if (xv > 0).any() else d * 1e-3, d * 1e-3)  # below: pooled at 0
        has0 = bool((xv < lo).any())
        edges = np.geomspace(lo, d, 25)
        edges = np.r_[0.0, edges] if has0 else edges
        centers = None
        grid = np.r_[np.geomspace(d, lo, 60), [0.0] if has0 else []]
        adm = lambda c, g: ((cat == c) & sel & (x >= g)).sum()
        ax2.set_xlabel(f'{param} cut value (default {d:g}); loosening →')
    bottom = None
    hist = {}
    for c in range(5):
        if c == 4 and not param.startswith('BTC'):
            continue
        xc = np.clip(x[sel & (cat == c)], edges[0], edges[-1])
        h, _ = np.histogram(xc, bins=edges)
        hist[c] = h
    # stacked bars (2 px surface gap via white edge)
    base = np.zeros(len(edges) - 1)
    for c, h in hist.items():
        if kind == 'int_up':
            ax1.bar(centers, h, width=0.8, bottom=base, color=COL[c], edgecolor='#fcfcfb', linewidth=1.0,
                    label=CATS[c])
        else:
            ax1.bar(edges[:-1], h, width=np.diff(edges), align='edge', bottom=base, color=COL[c],
                    edgecolor='#fcfcfb', linewidth=1.0, label=CATS[c])
        base = base + h
    # direct labels: recovered tracks per bin (the sliver that matters)
    if 0 in hist and kind == 'int_up':
        xc_bins = centers if kind == 'int_up' else np.sqrt(np.maximum(edges[:-1], edges[1] * 1e-3) * edges[1:])
        for xb, h0, tot in zip(xc_bins, hist[0], base):
            if h0 > 0:
                ax1.annotate(f'{h0} rec.', (xb, tot), xytext=(0, 3), textcoords='offset points', ha='center',
                             va='bottom', fontsize=7.5, color='#0b0b0b')
    ax1.set_ylabel(f'{OBJ[param]} killed at default\n(per bin of the cut value that keeps them)')
    ax1.grid(axis='y')
    # cumulative panel
    for c in hist:
        cum = np.array([adm(c, g) for g in grid])
        ax2.step(grid, np.maximum(cum, 0.8), where='post' if kind == 'int_up' else 'pre', color=COL[c],
                 linewidth=2, label=CATS[c])
    if kind == 'int_up':
        ax2.set_xticks(centers)
    ax2.set_yscale('log')
    ax2.set_ylim(0.8, None)
    ax2.set_ylabel(f'cumulative {OBJ[param]} admitted\nwhen loosened to this value')
    ax2.grid(axis='y', which='major')
    if kind != 'int_up':
        ax2.set_xscale('symlog', linthresh=edges[1] if edges[0] == 0 else edges[0])
        ax2.set_xlim(d * 1.05, edges[0])
    for v in OAT7[param]:
        if (kind == 'int_up' and v > d) or (kind != 'int_up' and v < d):
            for ax in (ax1, ax2):
                ax.axvline(v, color='#52514e', linestyle=':', linewidth=1)
            row = [param, 'core' if core else 'all', v] + [int(adm(c, v)) for c in range(5)]
            ax2.text(v, 1.0, f' oat7 {v:g}\n {row[3]} : {sum(row[4:])}', va='bottom', ha='left',
                     color='#52514e', fontsize=7.5, transform=ax2.get_xaxis_transform(), clip_on=False)
            table.append(row)
    n_kept = int(kept.sum())
    n_kept_g = int((kept & (cat != 3)).sum())
    scope = f'jet core (object ΔR < {CORE} to a jet)' if core else 'all objects'
    fig.suptitle(f'{TITLE[param]}\nS44 base, {nev} evt, {scope}', fontsize=11, x=0.02, ha='left')
    ax1.text(0.01, 0.97, f'kept at default: {n_kept:,} ({n_kept_g:,} genuine)\n'
                         f'killed, not rescuable by this cut alone: {n_unresc:,}',
             transform=ax1.transAxes, ha='left', va='top', fontsize=8.5, color='#52514e')
    h_, l_ = ax1.get_legend_handles_labels()
    fig.legend(h_, l_, loc='upper right', ncol=2, fontsize=8.5, frameon=False, bbox_to_anchor=(0.99, 0.995))
    ax2.text(0.99, 0.03, 'oat7 labels: recovered : all other admitted', transform=ax2.transAxes, ha='right',
             va='bottom', fontsize=7.5, color='#52514e')
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fn = os.path.join(OUT, f'{param}_{"core" if core else "all"}.png')
    fig.savefig(fn, dpi=130)
    plt.close(fig)
    return fn


def main():
    D = pickle.load(open(PKL, 'rb'))
    rec, nev = D['rec'], D['nev']
    sims = rec['sims']
    print('VALIDATION', D['val'])
    table = []
    for param in ['PT5_NM', 'AB_NM', 'BTC_NM', 'BTC_DR2T', 'BTC_D2L', 'BTC_DR2L', 'BTC_D2T']:
        for core in (False, True):
            print(plot_param(param, rec[RECKEY[param]], sims, core, nev, table))
    hdr = ['param', 'scope', 'cut'] + ['recovered', 'extra_copy', 'gen_outside_den', 'fake', 'not_TC_elig']
    lines = ['\t'.join(hdr)] + ['\t'.join(str(v) for v in r) for r in table]
    open(os.path.join(OUT, 'oat7_admitted_table.tsv'), 'w').write('\n'.join(lines) + '\n')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
