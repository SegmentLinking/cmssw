#!/usr/bin/env python3
"""Compare kinematics (pT, eta, phi) of real vs synthetic (--idealpls) reconstructed pLS.

Real seeds:      LSTNtuple_fix2.root          (real CMSSW pixel seeds)
Synthetic seeds: LSTNtuple_idealpls_fixed.root (truth-derived --idealpls seeds)

Restricted to GENUINE pLS (pLS_isFake==0, i.e. >75% hit-matched to a sim track), all events.
Histograms are unit-area normalized so shapes compare despite real seeding producing ~4x more
(duplicate) genuine pLS per track.
"""
import sys
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import ROOT

REAL_F = "LSTNtuple_fix2.root"
SYNTH_F = "LSTNtuple_idealpls_fixed.root"
OUT = sys.argv[1] if len(sys.argv) > 1 else "pls_real_vs_synth_kinematics.png"


def collect(fn):
    f = ROOT.TFile(fn)
    t = f.Get("tree")
    pt, eta, phi = [], [], []
    for ev in t:
        isfake = ev.pLS_isFake
        p, e, ph = ev.pLS_pt, ev.pLS_eta, ev.pLS_phi
        for j in range(len(p)):
            if isfake[j] == 0:
                pt.append(p[j])
                eta.append(e[j])
                phi.append(ph[j])
    return np.array(pt), np.array(eta), np.array(phi)


real = collect(REAL_F)
synth = collect(SYNTH_F)
print(f"genuine pLS: real={len(real[0])}  synth={len(synth[0])}")

# ---- binning
pt_bins = np.logspace(np.log10(0.8), np.log10(2000.0), 45)  # log; overflow clipped into last bin
eta_bins = np.linspace(-4.5, 4.5, 46)
phi_bins = np.linspace(-np.pi, np.pi, 37)


def clip_hi(a, bins):
    return np.clip(a, bins[0], bins[-1] - 1e-9)


# ---- style (Okabe-Ito, colorblind-safe; consistent with project house style)
INK, MUTED, GRID = "#333333", "#767676", "#DDDDDD"
C_REAL, C_SYNTH = "#0072B2", "#E69F00"  # blue = real, orange = synthetic

fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
fig.suptitle("Reconstructed pLS kinematics: real CMSSW seeds vs synthetic (--idealpls) truth seeds"
             "   [genuine pLS, unit-area normalized]",
             fontsize=12, color=INK, y=0.99)

specs = [
    ("pT [GeV]", 0, pt_bins, True, "A — transverse momentum"),
    ("η", 1, eta_bins, False, "B — pseudorapidity"),
    ("φ [rad]", 2, phi_bins, False, "C — azimuth"),
]

for xlabel, idx, bins, logx, title in specs:
    ax = axes[idx]
    for data, color, label in [(real, C_REAL, f"real ({len(real[0])})"),
                               (synth, C_SYNTH, f"synthetic ({len(synth[0])})")]:
        ax.hist(clip_hi(data[idx], bins), bins=bins, density=True, histtype="step",
                color=color, linewidth=1.7, label=label)
    if logx:
        ax.set_xscale("log")
    ax.set_title(title, fontsize=10.5, color=INK, loc="left")
    ax.set_xlabel(xlabel, fontsize=10, color=INK)
    ax.set_ylabel("normalized density", fontsize=10, color=INK)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)

axes[0].legend(fontsize=9, frameon=False, loc="upper right", title="genuine pLS")

fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(OUT, dpi=160)
print("plot written:", OUT)
