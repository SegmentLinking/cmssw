#!/usr/bin/env python3
"""Jet-core figures arguing that tuning the S44 dedup cuts is unproductive (from the S48 replay).

Inputs: plots-files/s48_dedup_thresholds/records.pkl (s48_dedup_threshold_hists.py), the S44 base
--allobj ntuple (base core TC counts and the core-failure funnel) and efficiency/tune_dedup_log.txt
(measured oat7 LST runs). Outputs in plots-files/s48_dedup_argument/:
  tradeoff_core.png            efficiency gained vs junk admitted, all 7 cuts (+ measured oat7 panel)
  junk_per_recovered_core.png  junk admitted per recovered track, per cut
  projected_rates_core.png     projected core efficiency / fake rate / duplicate rate vs cut value
  ceiling_core.png             S44 core failures by funnel bucket vs what any dedup loosening recovers
  numbers_core.tsv             every plotted number

Object classes (jet core = object within dR < 0.02 of a selected jet):
  recovered  first admitted genuine copy of a core-denominator sim that has no TC
  copy       genuine copy of a sim that already has a TC (or an earlier-admitted copy)
  other      genuine, sim outside the core denominator and without a TC
  fake       no sim with >= 75% of the hits
  inelig     (BeforeTC) T5 in a pT5 or failing tightCutFlag: can never become a TC
"junk" = copy + fake.

Usage: s48_plot_dedup_argument.py [records.pkl] [ntuple] [outdir]
"""
import sys, os, re, pickle, collections
import numpy as np
import uproot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

PKL = sys.argv[1] if len(sys.argv) > 1 else 'plots-files/s48_dedup_thresholds/records.pkl'
NTUP = sys.argv[2] if len(sys.argv) > 2 else 'Ntuple-files/LSTNtuple_s44_base_100evt.root'
CORE = float(os.environ.get('CORE_DR', '0.02'))  # jet-core window (object and sim dR to the jet axis)
OUT = sys.argv[3] if len(sys.argv) > 3 else ('plots-files/s48_dedup_argument' +
                                             ('' if CORE == 0.02 else f'_dr{CORE:g}'.replace('.', 'p')))
LOG = 'efficiency/tune_dedup_log.txt'
sys.argv = sys.argv[:1]  # the replay module parses argv at import
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from s48_dedup_threshold_hists import DEF, OAT7, KIND

PARAMS = ['PT5_NM', 'AB_NM', 'BTC_NM', 'BTC_DR2T', 'BTC_D2L', 'BTC_DR2L', 'BTC_D2T']
RECKEY = dict(PT5_NM='PT5_NM', AB_NM='T5_AB', BTC_NM='BTC_NM', BTC_DR2T='BTC_DR2T', BTC_D2L='BTC_D2L',
              BTC_DR2L='BTC_DR2L', BTC_D2T='BTC_D2T')
LABEL = dict(PT5_NM='pT5 dedup: shared hits (PT5_NM)', AB_NM='T5 AfterBuild: shared hits (AB_NM)',
             BTC_NM='T5 BeforeTC: shared hits (BTC_NM)', BTC_DR2T='T5 BeforeTC: tight ΔR² (BTC_DR2T)',
             BTC_D2L='T5 BeforeTC: loose DNN d² (BTC_D2L)', BTC_DR2L='T5 BeforeTC: loose ΔR² (BTC_DR2L)',
             BTC_D2T='T5 BeforeTC: tight DNN d² (BTC_D2T)')
SHORT = dict(PT5_NM='PT5_NM', AB_NM='AB_NM', BTC_NM='BTC_NM', BTC_DR2T='BTC_DR2T', BTC_D2L='BTC_D2L',
             BTC_DR2L='BTC_DR2L', BTC_D2T='BTC_D2T')
PCOL = dict(zip(PARAMS, ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#4a3aa7']))
PMRK = dict(zip(PARAMS, ['o', 's', '^', 'D', 'v', 'P', 'X']))
OFF = dict(PT5_NM=15, AB_NM=11, BTC_NM=11)  # NM above the max possible shared hits = cut off
INK, INK2, GRID, BG = '#0b0b0b', '#52514e', '#e4e3df', '#fcfcfb'
CAPTION = (f'S44 base (real pLS, AfterBuild on), 99 events (event 89 skipped). Jet core: object within ΔR < {CORE:g} of a jet, or matched to a core sim track. '
           'Counts at the dedup stage; "recovered" is an upper bound (later cleaning not replayed); '
           'AB_NM counted at T5 level.')

plt.rcParams.update({'font.size': 10, 'axes.edgecolor': INK2, 'axes.labelcolor': INK, 'xtick.color': INK2,
                     'ytick.color': INK2, 'axes.spines.top': False, 'axes.spines.right': False,
                     'figure.facecolor': BG, 'axes.facecolor': BG, 'grid.color': GRID, 'grid.linewidth': 0.6,
                     'text.color': INK})


def classify(recs, sims):
    """Per object (core only): x (cut value that keeps it), cls, killed, rescuable, (ev, sim)."""
    out = collections.defaultdict(list)
    for r, s in zip(recs, sims):
        # object near the jet axis, or a genuine copy of a core-denominator sim (its own direction can be further out)
        simc0 = np.where(r['sim'] >= 0, r['sim'], 0)
        sim_core = (r['sim'] >= 0) & s['den'][simc0] & (s['dr'][simc0] < CORE)
        m = r['mask'] & ((r['dr'] < CORE) | sim_core)
        v, sim, elig = r['v'][m], r['sim'][m], r['elig'][m]
        out['ev'].append(np.full(len(v), r['ev']))
        out['sim'].append(sim)
        out['v'].append(v)
        gen = sim >= 0
        simc = np.where(gen, sim, 0)
        in_den = gen & s['den'][simc] & (s['dr'][simc] < CORE)
        found = gen & s['found'][simc]
        cls = np.full(len(v), 'copy', dtype=object)
        cls[gen & ~in_den & ~found] = 'other'
        cls[~gen] = 'fake'
        cls[~elig] = 'inelig'
        out['cls'].append(cls)
        out['cand'].append(in_den & ~found & elig)
    return {k: np.concatenate(v) for k, v in out.items()}


def admitted(param, C, cut):
    """Objects killed at default and kept at `cut` (one-parameter loosening). Also returns recovered (ev, sim)."""
    kind, d, v = KIND[param], DEF[param], C['v']
    if kind == 'int_up':
        adm = (v >= d) & np.isfinite(v) & (v < cut)
    else:
        adm = (v < d) & (v > -np.inf) & (v >= cut)
    cls = C['cls'].copy()
    # recovered: first admitted copy of each missed denominator sim; the others count as copies
    rec = set()
    for i in np.where(adm & C['cand'])[0]:
        key = (int(C['ev'][i]), int(C['sim'][i]))
        if key in rec:
            cls[i] = 'copy'
        else:
            rec.add(key)
            cls[i] = 'recovered'
    cls[C['cand'] & ~adm] = 'copy'
    n = collections.Counter(cls[adm])
    return {k: n.get(k, 0) for k in ('recovered', 'copy', 'other', 'fake', 'inelig')}, rec


def grid_for(param):
    d = DEF[param]
    if KIND[param] == 'int_up':
        return list(range(d + 1, OFF[param] + 1))
    g = set(np.geomspace(d, d * 1e-3, 31)[1:]) | {o for o in OAT7[param] if o < d} | {0.0}
    return sorted(g, reverse=True)


def base_core_tcs(skip):
    """Base core TC counts (TC within dR < 0.02 of a selected jet): total, fake, duplicate."""
    t = uproot.open(NTUP)['tree']
    n = collections.Counter()
    for ie, A in enumerate(t.iterate(['tc_eta', 'tc_phi', 'tc_isFake', 'tc_isDuplicate', 'genjet_pt', 'genjet_eta',
                                      'genjet_phi'], step_size=1, library='np')):
        if ie in skip:
            continue
        gp, ge, gph = A['genjet_pt'][0], A['genjet_eta'][0], A['genjet_phi'][0]
        g = (gp > 1000) & (np.abs(ge) < 2.5)
        te, tp = A['tc_eta'][0], A['tc_phi'][0]
        if not g.any() or len(te) == 0:
            continue
        dphi = (tp[:, None] - gph[g][None] + np.pi) % (2 * np.pi) - np.pi
        dr = np.sqrt((te[:, None] - ge[g][None]) ** 2 + dphi ** 2).min(1)
        c = dr < CORE
        n['tc'] += int(c.sum())
        n['fake'] += int((A['tc_isFake'][0][c] != 0).sum())
        n['dup'] += int((A['tc_isDuplicate'][0][c] != 0).sum())
    return n


def funnel(skip):
    """Per failing core sim: s46_core_funnel.py bucket (same logic and denominator, event-indexed)."""
    sys.argv = ['x', NTUP]
    import s46_core_funnel as F
    br = ["sim_pt", "sim_eta", "sim_q", "sim_vx", "sim_vy", "sim_vz", "sim_genjet_deltaR", "sim_genjet_idx",
          "genjet_pt", "genjet_eta", "sim_tcIdx", "sim_plsIdxAll", "sim_plsIdxAllFrac", "sim_t5IdxAll",
          "sim_t5IdxAllFrac", "sim_pt5IdxAll", "sim_pt5IdxAllFrac", "pLS_isDup", "pT5_isDupReco"]
    t = uproot.open(NTUP)['tree']
    fails, n_den = {}, 0
    for ie, A in enumerate(t.iterate(br, step_size=1, library='np')):
        if ie in skip:
            continue
        pt = A['sim_pt'][0].astype(float)
        gj = A['sim_genjet_idx'][0].astype(np.int64)
        gpt, geta = A['genjet_pt'][0].astype(float), A['genjet_eta'][0].astype(float)
        gc = np.clip(gj, 0, max(len(gpt) - 1, 0))
        gpt_s = gpt[gc] if len(gpt) else np.zeros_like(pt)
        geta_s = geta[gc] if len(geta) else np.zeros_like(pt)
        dr = A['sim_genjet_deltaR'][0].astype(float)
        sel = ((A['sim_q'][0] != 0) & (pt > F.PT_CUT) & (np.abs(A['sim_eta'][0]) < F.ETA_CUT)
               & (np.abs(A['sim_vz'][0]) < F.VTX_Z_MAX)
               & (np.hypot(A['sim_vx'][0].astype(float), A['sim_vy'][0].astype(float)) < F.VTX_R_MAX)
               & (gj >= 0) & (gpt_s > F.GJ_PT_MIN) & (np.abs(geta_s) < F.GJ_ETA_MAX) & (dr >= 0) & (dr < CORE))
        plsdup = np.asarray(A['pLS_isDup'][0]).astype(np.int64)
        pt5dup = np.asarray(A['pT5_isDupReco'][0]).astype(np.int64)
        for s in np.nonzero(sel)[0]:
            n_den += 1
            if A['sim_tcIdx'][0][s] >= 0:
                continue
            g = {o: F.genuine(A[f'sim_{o}IdxAll'][0][s], A[f'sim_{o}IdxAllFrac'][0][s]) for o in ('pls', 't5', 'pt5')}
            pls_ok = [p for p in g['pls'] if not (plsdup[p] & 1)]
            if len(g['pt5']):
                b = 'pT5_survived' if np.any(pt5dup[g['pt5']] == 0) else 'pT5_all_killed'
            elif not len(g['pls']):
                b = 'no_pLS'
            elif not pls_ok:
                b = 'pLS_flagged'
            elif not len(g['t5']):
                b = 'no_T5'
            else:
                b = 'pair_failed'
            fails[(ie, int(s))] = b
    return fails, n_den


def caption(fig, y=0.005):
    fig.text(0.01, y, CAPTION, fontsize=7.5, color=INK2, ha='left', va='bottom', wrap=True)


def main():
    os.makedirs(OUT, exist_ok=True)
    D = pickle.load(open(PKL, 'rb'))
    rec, sims = D['rec'], D['rec']['sims']
    skip = set(D.get('skipped', []))
    nev = D['nev']
    den = sum(int((s['den'] & (s['dr'] < CORE)).sum()) for s in sims)
    num0 = sum(int((s['den'] & (s['dr'] < CORE) & s['found']).sum()) for s in sims)
    eff0 = num0 / den
    print(f'core denominator {den}, base found {num0}, eff {eff0:.4f}')
    C = {p: classify(rec[RECKEY[p]], sims) for p in PARAMS}
    rows = []
    curves = {}
    for p in PARAMS:
        pts = []
        for g in grid_for(p):
            n, rs = admitted(p, C[p], g)
            pts.append((g, n, rs))
            rows.append([p, f'{g:g}', n['recovered'], n['copy'], n['other'], n['fake'], n['inelig']])
        curves[p] = pts
    with open(os.path.join(OUT, 'numbers_core.tsv'), 'w') as f:
        f.write(f'# core den {den}, base num {num0}, eff {eff0:.4f}, {nev} events\n')
        f.write('param\tcut\trecovered\tcopy\tother_genuine\tfake\tnot_TC_eligible\n')
        for r in rows:
            f.write('\t'.join(str(x) for x in r) + '\n')

    junk = lambda n: n['copy'] + n['fake']

    # ---------------- Figure 1: trade-off
    fig, (ax, axm) = plt.subplots(1, 2, figsize=(13, 7), gridspec_kw=dict(width_ratios=[1.35, 1]))
    xs_all = []
    btc_max = 0.0
    for p in PARAMS:
        pts = [(junk(n) / nev, 100 * n['recovered'] / den, g) for g, n, _ in curves[p]]
        oat = [(junk(n) / nev, 100 * n['recovered'] / den) for g, n, _ in curves[p]
               if any(np.isclose(g, o) for o in OAT7[p])]
        x = np.array([a for a, _, _ in pts]); y = np.array([b for _, b, _ in pts])
        keep = x > 0
        xs_all += list(x[keep])
        ax.plot(np.maximum(x, 1e-2), y, color=PCOL[p], linewidth=2, zorder=3)
        if oat:
            ax.scatter([max(a, 1e-2) for a, _ in oat], [b for _, b in oat], marker=PMRK[p], s=55, color=PCOL[p],
                       edgecolor=BG, linewidth=1.5, zorder=4, label=LABEL[p])
        else:
            ax.scatter([], [], marker=PMRK[p], s=55, color=PCOL[p], label=LABEL[p])
        ax.scatter([max(x[-1], 1e-2)], [y[-1]], marker=PMRK[p], s=70, facecolor=BG, edgecolor=PCOL[p],
                   linewidth=1.8, zorder=4)
        if not p.startswith('BTC'):
            ax.annotate(f'{SHORT[p]} fully off: +{y[-1]:.1f} pp\n({x[-1]:,.0f} junk / event)',
                        (max(x[-1], 1e-2), y[-1]), xytext=(8, 0), textcoords='offset points', fontsize=8,
                        color=INK, va='center')
        else:
            btc_max = max(btc_max, y[-1])
    ax.set_xscale('log')
    ytop = max(100 * pts[-1][1]['recovered'] / den for pts in curves.values()) * 1.4
    ax.set_ylim(0, ytop)
    ax.annotate(f'5 BeforeTC cuts: ≤ +{btc_max:.1f} pp even fully off', (4, 0.01 * ytop), xytext=(15, 0.05 * ytop),
                fontsize=8, color=INK, arrowprops=dict(arrowstyle='-', color=INK2, linewidth=0.8))
    xr = np.geomspace(1e-2, max(xs_all) * 3, 50)
    ideal = 100 * xr * nev / den
    ax.plot(xr, ideal, color=INK2, linestyle='--', linewidth=1)
    iy = 0.85 * ytop
    ax.annotate('ideal: every admitted object\nrecovers a missed track', (den * iy / 100 / nev, iy), xytext=(-8, 0),
                textcoords='offset points', ha='right', fontsize=8, color=INK2, rotation=0)
    ax.set_xlim(1e-2, max(xs_all) * 20)
    ax.set_xlabel('junk admitted per event (fakes + extra copies of found tracks)')
    ax.set_ylabel(f'Δ jet-core (ΔR < {CORE:g}) efficiency (pp), base {100 * eff0:.1f}%')
    ax.set_title('Replay: loosening one dedup cut at a time\n(filled = oat7 scan values, hollow = cut fully off)',
                 fontsize=10, loc='left')
    ax.grid(True, which='major')
    ax.legend(fontsize=7.5, frameon=False, loc='upper left', bbox_to_anchor=(0.0, -0.13), ncol=2)
    # measured oat7 (ideal pLS, core010 den 712)
    rows_m = []
    for line in open(LOG):
        if 'stage=oat7' not in line or 'label=' not in line:
            continue
        lab = re.search(r'label=(\S+)', line).group(1)
        kv = dict(re.findall(r'(\w+)=([\d.]+)', line))
        rows_m.append((lab, int(kv['core010_num']), int(kv['core010_den']), int(kv['n_tc']), int(kv['nevents'])))
    base = [r for r in rows_m if r[0] == 'base'][0]
    for p in PARAMS:
        pts = []
        for lab, nm, dn, ntc, ne in rows_m:
            if not lab.startswith(p + '='):
                continue
            v = float(lab.split('=')[1])
            loos = v > DEF[p] if KIND[p] == 'int_up' else v < DEF[p]
            if loos:
                pts.append(((ntc - base[3]) / ne, 100 * (nm - base[1]) / dn, v))
        if not pts:
            continue
        pts.sort(key=lambda t: t[0])
        axm.plot([a for a, _, _ in pts], [b for _, b, _ in pts], color=PCOL[p], marker=PMRK[p], markersize=7,
                 linewidth=1.5, markeredgecolor=BG)
        a, b, v = pts[-1]
        if not p.startswith('BTC'):
            axm.annotate(f'{SHORT[p]}={v:g}', (a, b), xytext=(5, 4), textcoords='offset points', fontsize=8)
    axm.annotate('5 BeforeTC cuts: within ±0.3 pp (±2 tracks)', (12, 0.2), xytext=(20, 0.9), fontsize=8,
                 arrowprops=dict(arrowstyle='-', color=INK2, linewidth=0.8))
    axm.axhline(0, color=INK2, linewidth=0.8)
    axm.set_xlabel('Δ track candidates per event')
    axm.set_ylabel('Δ core efficiency (pp, ΔR < 0.1)')
    axm.set_title(f'Measured in LST (Sept-15 oat7 scan)\nideal pLS, 100 evt, ΔR<0.1 den {base[2]}; '
                  f'1 track = {100 / base[2]:.2f} pp', fontsize=10, loc='left')
    axm.grid(True)
    fig.suptitle('Loosening dedup cuts buys almost no jet-core efficiency for a large amount of junk',
                 x=0.01, ha='left', fontsize=12, fontweight='bold')
    caption(fig)
    fig.tight_layout(rect=(0, 0.04, 1, 0.95))
    fig.savefig(os.path.join(OUT, 'tradeoff_core.png'), dpi=140)
    plt.close(fig)

    # ---------------- Figure 2: junk per recovered track
    fig, ax = plt.subplots(figsize=(10, 5.2))
    ylab, vals_o, vals_f, txt = [], [], [], []
    for p in PARAMS:
        loosest = max(OAT7[p]) if KIND[p] == 'int_up' else min(OAT7[p])
        n_o = [n for g, n, _ in curves[p] if np.isclose(g, loosest)][0]
        n_f = curves[p][-1][1]
        ylab.append(f'{LABEL[p]}\n→ {loosest:g} (oat7 loosest)')
        vals_o.append((junk(n_o), n_o['recovered']))
        vals_f.append((junk(n_f), n_f['recovered']))
    yy = np.arange(len(PARAMS))[::-1]
    XMAX = 2e4
    for y, (j, r), (jf, rf), p in zip(yy, vals_o, vals_f, PARAMS):
        ratio = j / r if r else np.inf
        w = min(ratio, XMAX) if np.isfinite(ratio) else XMAX
        rf_ratio = jf / rf if rf else np.inf
        if r:
            ax.barh(y, w, height=0.55, color='#eda100', edgecolor=BG)
            lab = f'{ratio:,.0f} : 1  ({j:,} junk / {r} recovered)'
            tx = max(w, rf_ratio if np.isfinite(rf_ratio) else 0) * 1.4
        else:
            lab = f'recovers nothing: {j:,} junk, 0 recovered'
            tx = 0.7
        ax.text(tx, y, lab, va='center', fontsize=8.5, color=INK if r else INK2)
        if np.isfinite(rf_ratio):
            ax.scatter([rf_ratio], [y], marker='|', s=260, color=INK, zorder=4, linewidths=2)
    ax.scatter([], [], marker='|', s=160, color=INK, linewidths=2, label='same, with the cut fully off')
    ax.axvline(1, color=INK2, linestyle='--', linewidth=1)
    ax.text(1.05, yy[0] + 0.45, '1 : 1', fontsize=8, color=INK2)
    ax.set_yticks(yy)
    ax.set_yticklabels(ylab, fontsize=8.5)
    ax.set_xscale('log')
    ax.set_xlim(0.5, 3e7)
    ax.set_ylim(-0.6, len(PARAMS) - 0.4)
    ax.set_xlabel('junk admitted per missed jet-core track recovered (fakes + extra copies), log scale')
    ax.grid(True, axis='x')
    ax.legend(frameon=False, fontsize=8, loc='lower right', bbox_to_anchor=(1.0, 1.0))
    fig.suptitle('Every missed jet-core track a looser dedup cut recovers costs tens to thousands of junk objects',
                 x=0.01, ha='left', fontsize=12, fontweight='bold')
    caption(fig)
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    fig.savefig(os.path.join(OUT, 'junk_per_recovered_core.png'), dpi=140)
    plt.close(fig)

    # ---------------- Figure 3: projected efficiency / fake rate / dup rate
    B = base_core_tcs(skip)
    print('base core TCs', dict(B), f'fake {B["fake"] / B["tc"]:.3f} dup {B["dup"] / B["tc"]:.3f}')
    fig, axs = plt.subplots(2, 7, figsize=(17, 6.2), sharey='row')
    for k, p in enumerate(PARAMS):
        g = np.array([DEF[p]] + [c for c, _, _ in curves[p]], float)
        ns = [dict(recovered=0, copy=0, other=0, fake=0, inelig=0)] + [n for _, n, _ in curves[p]]
        eff = np.array([100 * (num0 + n['recovered']) / den for n in ns])
        added = np.array([n['recovered'] + n['copy'] + n['other'] + n['fake'] for n in ns], float)
        fr = np.array([100 * (B['fake'] + n['fake']) / (B['tc'] + a) for n, a in zip(ns, added)])
        dr = np.array([100 * (B['dup'] + n['copy']) / (B['tc'] + a) for n, a in zip(ns, added)])
        xpos = np.arange(len(g))
        a1, a2 = axs[0, k], axs[1, k]
        a1.plot(xpos, eff, color='#2a78d6', linewidth=2)
        a1.annotate(f'{eff[-1]:.1f}%', (xpos[-1], eff[-1]), xytext=(0, 5), textcoords='offset points', ha='right',
                    fontsize=8)
        a2.plot(xpos, fr, color='#eda100', linewidth=2)
        a2.plot(xpos, dr, color='#eb6834', linewidth=2)
        a2.annotate(f'{fr[-1]:.0f}%', (xpos[-1], fr[-1]), xytext=(0, 5), textcoords='offset points', ha='right',
                    fontsize=8)
        if dr[-1] >= 1:
            a2.annotate(f'{dr[-1]:.0f}%', (xpos[-1], dr[-1]), xytext=(0, 5), textcoords='offset points',
                        ha='right', fontsize=8)
        a1.set_title(LABEL[p].replace(': ', ':\n'), fontsize=8.5, loc='left')
        loosest = max(OAT7[p]) if KIND[p] == 'int_up' else min(OAT7[p])
        il = [i for i, c in enumerate(g) if i > 0 and np.isclose(c, loosest)]
        ticks = sorted({0, len(g) - 1} | {i for i in il if 2 < i < len(g) - 2})
        tl = [f'{g[i]:g}' if i != len(g) - 1 else f'{g[i]:g}\n(off)' for i in ticks]
        for a in (a1, a2):
            a.set_xticks(ticks)
            a.set_xticklabels(tl, fontsize=7.5)
            a.grid(True, axis='y')
        a2.set_xlabel('cut value, loosening →', fontsize=8)
    axs[0, 0].set_ylabel('projected jet-core\nefficiency (%)')
    axs[1, 0].set_ylabel('projected jet-core rate (%)')
    axs[0, 0].set_ylim(0, 100)
    axs[1, 0].set_ylim(0, 100)
    axs[1, 0].text(0.05, 0.93, 'fake rate', color=INK, fontsize=8, transform=axs[1, 0].transAxes)
    axs[1, 0].plot([0.02], [0.955], marker='s', color='#eda100', transform=axs[1, 0].transAxes, clip_on=False)
    axs[1, 0].text(0.05, 0.83, 'duplicate rate', color=INK, fontsize=8, transform=axs[1, 0].transAxes)
    axs[1, 0].plot([0.02], [0.855], marker='s', color='#eb6834', transform=axs[1, 0].transAxes, clip_on=False)
    fig.suptitle(f'Projected jet-core rates as each cut is loosened: efficiency stays near {100 * eff0:.0f}% '
                 f'while fake and duplicate rates climb', x=0.01, ha='left', fontsize=12, fontweight='bold')
    fig.text(0.01, 0.035, f'Base (99 evt): {B["tc"]:,} core TCs, fake {100 * B["fake"] / B["tc"]:.1f}%, duplicate '
             f'{100 * B["dup"] / B["tc"]:.1f}%. Every admitted object is counted as a new TC (no later cleaning); '
             f'"not TC-eligible" BeforeTC T5s are not added.', fontsize=7.5, color=INK2)
    caption(fig)
    fig.tight_layout(rect=(0, 0.06, 1, 0.94))
    fig.savefig(os.path.join(OUT, 'projected_rates_core.png'), dpi=140)
    plt.close(fig)

    # ---------------- Figure 4: ceiling vs the gap
    fails, fden = funnel(skip)
    union = set()
    for p in PARAMS:
        union |= curves[p][-1][2]
    rec_f = {k for k in union if k in fails}
    print(f'funnel den {fden}, failures {len(fails)}, union recovered {len(union)}, of which in funnel fails {len(rec_f)}')
    order = ['no_pLS', 'pLS_flagged', 'no_T5', 'pair_failed', 'pT5_all_killed', 'pT5_survived']
    names = dict(no_pLS='no genuine pixel seed', pLS_flagged='pixel seed flagged (CheckHitspLS)',
                 no_T5='seed, but no genuine T5', pair_failed='seed + T5, never paired into a pT5',
                 pT5_all_killed='genuine pT5s all killed by pT5 dedup', pT5_survived='genuine pT5 alive, no TC')
    cnt = collections.Counter(fails.values())
    rcnt = collections.Counter(fails[k] for k in rec_f)
    fig, ax = plt.subplots(figsize=(11, 5))
    yy = np.arange(len(order))[::-1]
    for y, b in zip(yy, order):
        ax.barh(y, cnt[b], height=0.6, color='#d9d8d3', edgecolor=BG)
        ax.barh(y, rcnt[b], height=0.6, color='#2a78d6', edgecolor=BG)
        ax.text(cnt[b] + 3, y, f'{rcnt[b]} of {cnt[b]} recoverable', va='center', fontsize=8.5)
    ax.set_yticks(yy)
    ax.set_yticklabels([names[b] for b in order], fontsize=9)
    ax.set_xlabel(f'missed jet-core tracks at S44 base (total {len(fails)} of {fden})')
    ax.grid(True, axis='x')
    ax.set_xlim(0, max(cnt.values()) * 1.35)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color='#2a78d6', label='recovered by turning at least one dedup cut fully off'),
                       Patch(color='#d9d8d3', label='not recoverable by any dedup cut')],
              frameon=False, fontsize=8.5, loc='lower right')
    ceil = 100 * len(rec_f) / fden
    fig.suptitle(f'Even with a dedup cut fully off, at most {len(rec_f)} of {len(fails)} missed jet-core tracks '
                 f'come back (≤ +{ceil:.1f} pp)', x=0.01, ha='left', fontsize=12, fontweight='bold')
    fig.text(0.01, 0.905, 'For scale: two LST-internal fixes later made on the rebased code (S47 pT5 demotion + '
             'stale-flag reset) gained +4.55 pp core at 1000 evt, with a lower fake rate (different code base).',
             fontsize=8.5, color=INK2)
    caption(fig)
    fig.tight_layout(rect=(0, 0.05, 1, 0.89))
    fig.savefig(os.path.join(OUT, 'ceiling_core.png'), dpi=140)
    plt.close(fig)
    with open(os.path.join(OUT, 'numbers_core.tsv'), 'a') as f:
        f.write(f'# funnel: den {fden}, failures {len(fails)}, union recovered {len(rec_f)}\n')
        for b in order:
            f.write(f'# bucket {b}\t{cnt[b]}\trecoverable {rcnt[b]}\n')
    # ---------------- Figure 5: efficiency vs dR inside the core, base vs cuts fully off
    by_ev = {s['ev']: s for s in sims}
    edges = np.linspace(0, CORE, 9)
    dr_d = np.concatenate([s['dr'][s['den'] & (s['dr'] < CORE)] for s in sims])
    f_d = np.concatenate([s['found'][s['den'] & (s['dr'] < CORE)] for s in sims])
    nb = np.histogram(dr_d, edges)[0]
    base_b = np.histogram(dr_d[f_d], edges)[0]
    btc_union = set().union(*[curves[p][-1][2] for p in PARAMS if p.startswith('BTC')])
    scen = [('PT5_NM fully off', curves['PT5_NM'][-1][2], PCOL['PT5_NM'], PMRK['PT5_NM'],
             junk(curves['PT5_NM'][-1][1]) / nev),
            ('AB_NM fully off (T5 level)', curves['AB_NM'][-1][2], PCOL['AB_NM'], PMRK['AB_NM'],
             junk(curves['AB_NM'][-1][1]) / nev),
            ('any of 5 BeforeTC cuts fully off', btc_union, PCOL['BTC_DR2T'], PMRK['BTC_DR2T'], None)]
    fig, ax = plt.subplots(figsize=(10, 5.8))
    xc = 0.5 * (edges[1:] + edges[:-1])
    ax.plot(xc, 100 * base_b / nb, color=INK, marker='o', linewidth=2.2, markersize=7, label='S44 base', zorder=5)
    tsv = [['dR_lo', 'dR_hi', 'den', 'base'] + [n for n, *_ in scen]]
    cols = []
    for name, rs, col, mk, j in scen:
        drs = np.array([by_ev[e]['dr'][sm] for e, sm in rs if by_ev[e]['dr'][sm] < CORE])
        add = np.histogram(drs, edges)[0] if len(drs) else np.zeros(len(nb), int)
        cols.append(add)
        lab = name + (f'  ({j:,.0f} junk / event)' if j is not None else '')
        ax.plot(xc, 100 * (base_b + add) / nb, color=col, marker=mk, linewidth=1.8, markersize=7,
                markeredgecolor=BG, label=lab)
    for i, x in enumerate(xc):
        ax.annotate(f'{nb[i]}', (x, 2), ha='center', fontsize=7.5, color=INK2)
        tsv.append([f'{edges[i]:g}', f'{edges[i + 1]:g}', nb[i], base_b[i]] + [int(c[i]) for c in cols])
    ax.text(0.0, 5.5, 'tracks per bin:', fontsize=7.5, color=INK2)
    ax.set_ylim(0, 100)
    ax.set_xlim(0, CORE)
    ax.set_xlabel('ΔR between sim track and jet axis')
    ax.set_ylabel('jet-core efficiency (%)')
    ax.grid(True, axis='y')
    ax.legend(frameon=False, fontsize=8.5, loc='upper left')
    fig.suptitle('Turning a dedup cut fully off barely moves efficiency at the centre of the jet',
                 x=0.01, ha='left', fontsize=12, fontweight='bold')
    caption(fig)
    fig.tight_layout(rect=(0, 0.05, 1, 0.95))
    fig.savefig(os.path.join(OUT, 'eff_vs_dR_core.png'), dpi=140)
    plt.close(fig)
    with open(os.path.join(OUT, 'eff_vs_dR_core.tsv'), 'w') as f:
        f.write('\n'.join('\t'.join(str(x) for x in r) for r in tsv) + '\n')
    print('wrote', OUT)


if __name__ == '__main__':
    main()
