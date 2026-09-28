"""Shared helpers for the S47 late_tc scans (read-only analysis of the cached --allobj ntuple)."""
import glob, pickle, math
import numpy as np

CACHE = '/tmp/claude-63526/-mnt-data1-kk829-CMSSW-16-1-1-src-RecoTracker-LSTCore-standalone/' \
        'dbd9af1c-87ff-4b4e-9196-c80ebb15a786/scratchpad/s47/late_tc/cache'
GEN = 0.75           # genuine object: match fraction >= 0.75 (funnel convention)
TCMATCH = 0.75       # TC match: frac > 0.75 (sim_tcIdx convention)
EMPTY = 4294967295   # lst::kTCEmptyHitIdx as stored (cast to int it becomes -1)


def events(first=0, last=10**9):
    import os
    side = CACHE + '/../genjet_phi.pkl'
    if not os.path.exists(side):
        import uproot
        t = uproot.open('/mnt/data1/kk829/CMSSW_16_1_1/src/RecoTracker/LSTCore/standalone/Ntuple-files/'
                        'LSTNtuple_s45_rebased_100evt.root')['tree']
        gphi = [np.asarray(x) for x in t['genjet_phi'].array(library='np')]
        with open(side, 'wb') as f:
            pickle.dump(gphi, f)
    with open(side, 'rb') as f:
        gphi = pickle.load(f)
    ie = 0
    for fn in sorted(glob.glob(CACHE + '/ev_*.pkl')):
        if ie + 10 <= first:
            ie += 10
            continue
        with open(fn, 'rb') as f:
            for ev in pickle.load(f):
                if first <= ie < last:
                    ev['genjet_phi'] = gphi[ie]
                    yield ie, ev
                ie += 1
        if ie >= last:
            return


def denominator(ev, drmax=0.02, drmin=0.0):
    sim_pt = ev['sim_pt'].astype(float)
    gj = ev['sim_genjet_idx'].astype(np.int64)
    gpt, geta = ev['genjet_pt'].astype(float), ev['genjet_eta'].astype(float)
    gc = np.clip(gj, 0, max(len(gpt) - 1, 0))
    gpt_s = gpt[gc] if len(gpt) else np.zeros_like(sim_pt)
    geta_s = geta[gc] if len(geta) else np.zeros_like(sim_pt)
    dr = ev['sim_genjet_deltaR'].astype(float)
    return ((ev['sim_q'] != 0) & (sim_pt > 0.8) & (np.abs(ev['sim_eta']) < 4.5) & (np.abs(ev['sim_vz']) < 30) &
            (np.hypot(ev['sim_vx'].astype(float), ev['sim_vy'].astype(float)) < 2.5) & (gj >= 0) &
            (gpt_s > 1000) & (np.abs(geta_s) < 2.5) & (dr >= drmin) & (dr < drmax))


def genuine(ev, obj, s, thr=GEN):
    idx = np.asarray(ev[f'sim_{obj}IdxAll'][s]).astype(np.int64)
    fr = np.asarray(ev[f'sim_{obj}IdxAllFrac'][s])
    return idx[fr >= thr]


def dphi(a, b):
    d = a - b
    return (d + np.pi) % (2 * np.pi) - np.pi


def f16(x):
    return np.asarray(x, dtype=np.float32).astype(np.float16).astype(np.float32)


class Hits:
    """Hit-key builder.  OT hits keyed by the ntuple hit index where known (T5 hitIndices / pT3 otHitIndices),
    otherwise by rounded MD coordinates; pixel hits keyed by coordinates (pixel and OT never collide)."""

    def __init__(self, ev):
        self.ev = ev
        self.ax, self.ay, self.az = ev['md_anchor_x'], ev['md_anchor_y'], ev['md_anchor_z']
        self.ox, self.oy, self.oz = ev['md_other_x'], ev['md_other_y'], ev['md_other_z']
        self.ls0, self.ls1 = ev['ls_mdIdx0'].astype(int), ev['ls_mdIdx1'].astype(int)
        self.t3l0, self.t3l1 = ev['t3_lsIdx0'].astype(int), ev['t3_lsIdx1'].astype(int)
        self.t5a, self.t5b = ev['t5_t3Idx0'].astype(int), ev['t5_t3Idx1'].astype(int)
        self._c = {}
        # coordinate key -> hit index map, built from every T5's base slots (slot 2i = anchor(MD i), 2i+1 = outer)
        self.h2k = {}
        self.k2h = {}
        for t in range(len(self.t5a)):
            mds = self.t5_mds(t)
            hi = ev['t5_hitIndices'][t]
            for i, m in enumerate(mds):
                for j, k in enumerate(self.md_keys(m)):
                    h = int(hi[2 * i + j])
                    self.h2k[h] = k
                    self.k2h[k] = h

    def md_keys(self, m):
        return (('c', round(float(self.ax[m]), 4), round(float(self.ay[m]), 4), round(float(self.az[m]), 4)),
                ('c', round(float(self.ox[m]), 4), round(float(self.oy[m]), 4), round(float(self.oz[m]), 4)))

    def ls_mds(self, l):
        return [int(self.ls0[l]), int(self.ls1[l])]

    def t3_mds(self, t3):
        a = self.ls_mds(self.t3l0[t3])
        b = self.ls_mds(self.t3l1[t3])
        return [a[0], a[1], b[1]]

    def t5_mds(self, t5):
        a = self.t3_mds(self.t5a[t5])
        b = self.t3_mds(self.t5b[t5])
        return a + b[1:]

    def key_of_hidx(self, h):
        return self.h2k.get(int(h), ('h', int(h)))

    def pls_hits(self, p):
        """4 pixel hits in the order of addPixelQuintupletToMemory (anchor/outer of inner MD, then outer MD)."""
        l = int(self.ev['pLS_lsIdx'][p])
        out = []
        for m in self.ls_mds(l):
            out += list(self.md_keys(m))
        return out

    def t5_hits(self, t5, extended=True):
        """T5 hits: SoA hitIndices (incl. ExtendT5FromDupT5ByMD slots) mapped to coordinate keys."""
        hi = self.ev['t5_hitIndices'][t5]
        n = len(hi) if extended else 10
        return [self.key_of_hidx(h) for h in hi[:n] if int(h) != EMPTY and int(h) != -1]

    def t3_hits(self, t3):
        out = []
        for m in self.t3_mds(t3):
            out += list(self.md_keys(m))
        return out

    def t4_hits(self, t4):
        a = self.t3_mds(int(self.ev['t4_t3_idx0'][t4]))
        b = self.t3_mds(int(self.ev['t4_t3_idx1'][t4]))
        mds = a + [m for m in b if m not in a]
        out = []
        for m in mds:
            out += list(self.md_keys(m))
        return out

    def pt5_hits(self, i):
        return self.pls_hits(int(self.ev['pT5_plsIdx'][i])) + self.t5_hits(int(self.ev['pT5_t5Idx'][i]))


def count_in(h1, s2):
    """checkHitspT5 semantics: number of entries of h1 (with repeats) present in set s2."""
    return sum(1 for h in h1 if h in s2)


def replay_pt5_dedup(ev, H, deta=0.2, dphicut=0.2, nm=7):
    """Exact replay of RemoveDupPixelQuintupletsFromMap (Kernels.h:735-768).
    Returns (hits, neighbours) where neighbours[i] = list of j with nMatched(i in j) >= nm inside the window
    (the kernel's i-row test), and the replayed isDup array under the master key."""
    n = len(ev['pT5_score'])
    t5 = ev['pT5_t5Idx'].astype(int)
    eta = f16(ev['t5_eta'][t5]) if n else np.zeros(0)
    phi = f16(ev['t5_phi'][t5]) if n else np.zeros(0)
    hits = [H.pt5_hits(i) for i in range(n)]
    sets = [set(h) for h in hits]
    nbr = [[] for _ in range(n)]
    nmat = {}
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if abs(eta[i] - eta[j]) > deta:
                continue
            if abs(dphi(phi[i], phi[j])) > dphicut:
                continue
            c = count_in(hits[i], sets[j])
            nmat[(i, j)] = c
            if c >= nm:
                nbr[i].append(j)
    return hits, nbr, nmat
