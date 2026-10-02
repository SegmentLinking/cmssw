#!/usr/bin/env python3
"""Efficiency and fake rate vs deltaR for the Experiment-H runs vs baseline.

Panels: (A) pT5_lower object efficiency, (B) pT5-type TC efficiency,
(C) pT5-type TC fake rate — all vs sim-track deltaR to nearest GenJet.
Series: baseline (idealpls_fixed) / H1 (pT5 sees isDup T5s) / H2 (+ isDup pLS).
Reads existing NumDen files only.
"""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import ROOT

OUT = "eff_fake_vs_deltaR_expH.png"
REBIN = 2  # 50 bins of 0.002 -> 25 bins of 0.004

FILES = [
    ("baseline", "LSTNumDen_idealpls_fixed.root", "#767676"),
    ("H1: pairing sees isDup T5s", "LSTNumDen_expH1.root", "#0072B2"),
    ("H2: + isDup pLS", "LSTNumDen_expH2.root", "#E69F00"),
]

PANELS = [
    ("A — genuine pT5 objects (pT5_lower eff)", "Root__pT5_lower_base_0_-1_ef_numer_deltaR",
     "Root__pT5_lower_base_0_-1_ef_denom_deltaR"),
    ("B — pT5-type TC efficiency", "Root__pT5_base_0_-1_ef_numer_deltaR",
     "Root__pT5_base_0_-1_ef_denom_deltaR"),
    ("C — pT5-type TC fake rate", "Root__pT5_fr_numer_deltaR", "Root__pT5_fr_denom_deltaR"),
]


def ratio_points(f, num_name, den_name):
    num, den = f.Get(num_name), f.Get(den_name)
    xs, ys, es = [], [], []
    nb = num.GetNbinsX()
    for b0 in range(1, nb + 1, REBIN):
        n = sum(num.GetBinContent(b) for b in range(b0, min(b0 + REBIN, nb + 1)))
        d = sum(den.GetBinContent(b) for b in range(b0, min(b0 + REBIN, nb + 1)))
        if d < 1:
            continue
        lo = num.GetXaxis().GetBinLowEdge(b0)
        hi = num.GetXaxis().GetBinUpEdge(min(b0 + REBIN - 1, nb))
        r = n / d
        xs.append(0.5 * (lo + hi))
        ys.append(r)
        es.append((r * (1 - r) / d) ** 0.5)
    return xs, ys, es


INK, MUTED, GRID = "#333333", "#767676", "#DDDDDD"
fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), sharex=True)
fig.suptitle("Experiment H (pT5-building visibility gates opened): efficiency and fake rate vs $\\Delta$R",
             fontsize=12.5, color=INK, y=0.99)

tfiles = [(label, ROOT.TFile(fn), color) for label, fn, color in FILES]
for ax, (title, num_name, den_name) in zip(axes, PANELS):
    for label, f, color in tfiles:
        xs, ys, es = ratio_points(f, num_name, den_name)
        ax.errorbar(xs, ys, yerr=es, fmt="o", ms=4, color=color, label=label,
                    linewidth=0, elinewidth=1, capsize=0)
    ax.set_title(title, fontsize=10.5, color=INK, loc="left")
    ax.set_xlabel("sim-track $\\Delta$R to nearest GenJet", fontsize=10, color=INK)
    ax.set_xlim(0, 0.1)
    ax.set_ylim(0, 1.05)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)

axes[0].set_ylabel("efficiency / fake rate", fontsize=10, color=INK)
axes[0].legend(fontsize=8.5, frameon=False, loc="lower right")

fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig(OUT, dpi=160)
print("plot written:", OUT)
