#!/usr/bin/env python3
"""Per-sim-track difference between the real and synthetic reconstructed pLS.

For every accepted sim track that has BOTH a genuine real pLS (LSTNtuple_fix2.root) and a
genuine synthetic pLS (LSTNtuple_idealpls_fixed.root), take the BEST-matched (highest hit-frac)
pLS of each type and compare kinematics track-by-track. The two files are index-aligned by
sim track (verified: identical sim_pt/sim_eta per (event, index)).

Synthetic pLS momentum/direction is truth-derived, so these differences read as the real
CMSSW pixel-seed resolution relative to truth:
  A. relative pT:  (pT_real - pT_synth) / pT_synth
  B. eta:          eta_real - eta_synth
  C. phi:          deltaPhi(phi_real, phi_synth)

Usage: python3 lst_plot_pls_matched_diff.py [out.png]
"""
import sys
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import ROOT

REAL_F = "LSTNtuple_fix2.root"
SYNTH_F = "LSTNtuple_idealpls_fixed.root"
OUT = sys.argv[1] if len(sys.argv) > 1 else "pls_matched_diff.png"


def dphi(a, b):
    d = a - b
    while d > np.pi:
        d -= 2 * np.pi
    while d < -np.pi:
        d += 2 * np.pi
    return d


def best_pls(t, i):
    """Return (pt, eta, phi) of the highest-frac genuine pLS matched to accepted sim track i,
    or None if the track has no matched pLS in this file."""
    idxs = t.sim_plsIdxAll[i]
    fracs = t.sim_plsIdxAllFrac[i]
    if len(idxs) == 0:
        return None
    best = max(range(len(idxs)), key=lambda k: fracs[k])
    j = idxs[best]
    return t.pLS_pt[j], t.pLS_eta[j], t.pLS_phi[j]


fr = ROOT.TFile(REAL_F); tr = fr.Get("tree")
fs = ROOT.TFile(SYNTH_F); ts = fs.Get("tree")

d_relpt, d_eta, d_phi = [], [], []
n_both = 0
for e in range(tr.GetEntries()):
    tr.GetEntry(e); ts.GetEntry(e)
    n = len(tr.sim_pt)
    for i in range(n):
        # standard denominator acceptance
        if abs(tr.sim_q[i]) != 1:
            continue
        if not (tr.sim_pt[i] > 0.9 and abs(tr.sim_eta[i]) < 4.5):
            continue
        if not (abs(tr.sim_vz[i]) < 30 and (tr.sim_vx[i] ** 2 + tr.sim_vy[i] ** 2) ** 0.5 < 2.5):
            continue
        R = best_pls(tr, i)
        S = best_pls(ts, i)
        if R is None or S is None:
            continue
        n_both += 1
        d_relpt.append((R[0] - S[0]) / S[0])
        d_eta.append(R[1] - S[1])
        d_phi.append(dphi(R[2], S[2]))

d_relpt = np.array(d_relpt); d_eta = np.array(d_eta); d_phi = np.array(d_phi)
print(f"sim tracks with BOTH a real and synthetic pLS: {n_both}")


def summ(name, a, unit=""):
    med = np.median(a)
    lo, hi = np.percentile(a, [16, 84])
    print(f"  {name:10s} median={med:+.4f}{unit}  68% [{lo:+.4f}, {hi:+.4f}]  "
          f"half-width={(hi - lo) / 2:.4f}")


summ("rel pT", d_relpt)
summ("eta", d_eta)
summ("phi", d_phi, " rad")

# ---- data-driven symmetric ranges (robust to tails)
def rng(a, q=99.0, hardcap=None):
    r = np.percentile(np.abs(a), q)
    if hardcap is not None:
        r = min(r, hardcap)
    return r


r_pt = rng(d_relpt, 98, hardcap=1.0)
r_eta = rng(d_eta, 99)
r_phi = rng(d_phi, 99)

INK, MUTED, GRID = "#333333", "#767676", "#DDDDDD"
C = "#009E73"  # green: real-minus-synthetic per-track difference

fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
fig.suptitle("Per-track real − synthetic pLS difference (best-matched pLS per sim track; "
             f"{n_both} tracks with both)", fontsize=12, color=INK, y=0.99)

specs = [
    (d_relpt, r_pt, "(pT$_{real}$ − pT$_{synth}$) / pT$_{synth}$", "A — relative pT"),
    (d_eta, r_eta, "$\\eta_{real}$ − $\\eta_{synth}$", "B — $\\eta$"),
    (d_phi, r_phi, "$\\Delta\\phi(\\phi_{real}, \\phi_{synth})$ [rad]", "C — $\\phi$"),
]

for ax, (data, r, xlabel, title) in zip(axes, specs):
    bins = np.linspace(-r, r, 61)
    clipped = np.clip(data, bins[0], bins[-1] - 1e-12)
    ax.hist(clipped, bins=bins, color=C, edgecolor="white", linewidth=0.3)
    ax.axvline(0, color="#D55E00", linestyle="--", linewidth=1.1)
    med = np.median(data)
    ax.axvline(med, color=INK, linestyle=":", linewidth=1.1)
    ax.text(0.03, 0.95, f"median={med:+.4f}\nN={len(data)}", transform=ax.transAxes,
            fontsize=9, color=INK, va="top")
    ax.set_title(title, fontsize=10.5, color=INK, loc="left")
    ax.set_xlabel(xlabel, fontsize=10, color=INK)
    ax.set_ylabel("sim tracks", fontsize=10, color=INK)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)

fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(OUT, dpi=160)
print("plot written:", OUT)
