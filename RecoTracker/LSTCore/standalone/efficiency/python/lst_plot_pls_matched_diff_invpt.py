#!/usr/bin/env python3
"""Per-sim-track difference between real and synthetic reconstructed pLS, with 1/pT.

Same as lst_plot_pls_matched_diff.py but panel A uses the curvature variable 1/pT:
  A. 1/pT:  1/pT_real - 1/pT_synth   [GeV^-1]   (absolute, the well-measured resolution variable)
  B. eta:   eta_real - eta_synth
  C. phi:   deltaPhi(phi_real, phi_synth)

Real:      LSTNtuple_fix2.root           (real CMSSW pixel seeds)
Synthetic: LSTNtuple_idealpls_fixed.root (truth-derived --idealpls seeds)
Files are index-aligned by sim track (verified: identical sim_pt/sim_eta per (event, index)).

Usage: python3 lst_plot_pls_matched_diff_invpt.py [out.png]
"""
import sys
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import ROOT

REAL_F = "LSTNtuple_fix2.root"
SYNTH_F = "LSTNtuple_idealpls_fixed.root"
OUT = sys.argv[1] if len(sys.argv) > 1 else "pls_matched_diff_invpt.png"


def dphi(a, b):
    d = a - b
    while d > np.pi:
        d -= 2 * np.pi
    while d < -np.pi:
        d += 2 * np.pi
    return d


def best_pls(t, i):
    """(pt, eta, phi) of the highest-frac genuine pLS matched to accepted sim track i, or None."""
    idxs = t.sim_plsIdxAll[i]
    fracs = t.sim_plsIdxAllFrac[i]
    if len(idxs) == 0:
        return None
    best = max(range(len(idxs)), key=lambda k: fracs[k])
    j = idxs[best]
    return t.pLS_pt[j], t.pLS_eta[j], t.pLS_phi[j]


fr = ROOT.TFile(REAL_F); tr = fr.Get("tree")
fs = ROOT.TFile(SYNTH_F); ts = fs.Get("tree")

d_invpt, d_eta, d_phi = [], [], []
n_both = 0
for e in range(tr.GetEntries()):
    tr.GetEntry(e); ts.GetEntry(e)
    n = len(tr.sim_pt)
    for i in range(n):
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
        d_invpt.append(1.0 / R[0] - 1.0 / S[0])
        d_eta.append(R[1] - S[1])
        d_phi.append(dphi(R[2], S[2]))

d_invpt = np.array(d_invpt); d_eta = np.array(d_eta); d_phi = np.array(d_phi)
print(f"sim tracks with BOTH a real and synthetic pLS: {n_both}")


def summ(name, a, unit=""):
    med = np.median(a)
    lo, hi = np.percentile(a, [16, 84])
    print(f"  {name:10s} median={med:+.5f}{unit}  68% [{lo:+.5f}, {hi:+.5f}]  "
          f"half-width={(hi - lo) / 2:.5f}")


summ("d(1/pT)", d_invpt, " /GeV")
summ("eta", d_eta)
summ("phi", d_phi, " rad")


def rng(a, q=99.0):
    return np.percentile(np.abs(a), q)


r_invpt = rng(d_invpt, 98)
r_eta = rng(d_eta, 99)
r_phi = rng(d_phi, 99)

INK, MUTED, GRID = "#333333", "#767676", "#DDDDDD"
C = "#009E73"  # green: real-minus-synthetic per-track difference

fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
fig.suptitle("Per-track real - synthetic pLS difference (best-matched pLS per sim track; "
             f"{n_both} tracks with both)", fontsize=12, color=INK, y=0.99)

specs = [
    (d_invpt, r_invpt, "1/pT$_{real}$ - 1/pT$_{synth}$  [GeV$^{-1}$]", "A - curvature (1/pT)"),
    (d_eta, r_eta, "$\\eta_{real}$ - $\\eta_{synth}$", "B - $\\eta$"),
    (d_phi, r_phi, "$\\Delta\\phi(\\phi_{real}, \\phi_{synth})$ [rad]", "C - $\\phi$"),
]

for ax, (data, r, xlabel, title) in zip(axes, specs):
    bins = np.linspace(-r, r, 61)
    clipped = np.clip(data, bins[0], bins[-1] - 1e-12)
    ax.hist(clipped, bins=bins, color=C, edgecolor="white", linewidth=0.3)
    ax.axvline(0, color="#D55E00", linestyle="--", linewidth=1.1)
    med = np.median(data)
    ax.axvline(med, color=INK, linestyle=":", linewidth=1.1)
    ax.text(0.03, 0.95, f"median={med:+.5f}\nN={len(data)}", transform=ax.transAxes,
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
