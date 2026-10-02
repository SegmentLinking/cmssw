#!/usr/bin/env python3
"""
Plot etaErr distributions for real vs synthetic pLS.
One best-matched pLS per accepted sim track (highest sim_plsIdxAllFrac).
"""
import numpy as np
import uproot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REAL_F  = "LSTNtuple_fix2.root"
SYNTH_F = "LSTNtuple_idealpls_fixed.root"
OUT     = "pls_etaerr_distributions.png"

PT_CUT    = 0.9
ETA_CUT   = 4.5
VTX_Z_MAX = 30.0
VTX_R_MAX = 2.5

BRANCHES = [
    "sim_pt", "sim_eta", "sim_q", "sim_vx", "sim_vy", "sim_vz",
    "sim_plsIdxAll", "sim_plsIdxAllFrac", "pLS_etaErr",
]

def collect_etaerr(path):
    data = uproot.open(path)["tree"].arrays(BRANCHES, library="np")
    vals = []
    for ievt in range(len(data["sim_pt"])):
        sim_pt  = data["sim_pt"][ievt].astype(float)
        sim_eta = data["sim_eta"][ievt].astype(float)
        sim_q   = data["sim_q"][ievt].astype(np.int32)
        sim_vx  = data["sim_vx"][ievt].astype(float)
        sim_vy  = data["sim_vy"][ievt].astype(float)
        sim_vz  = data["sim_vz"][ievt].astype(float)
        accept  = (
            (np.abs(sim_q) == 1) &
            (sim_pt > PT_CUT) &
            (np.abs(sim_eta) < ETA_CUT) &
            (np.abs(sim_vz) < VTX_Z_MAX) &
            (np.hypot(sim_vx, sim_vy) < VTX_R_MAX)
        )
        etaErr_arr = data["pLS_etaErr"][ievt]
        for isim in np.where(accept)[0]:
            idxs  = data["sim_plsIdxAll"][ievt][isim]
            fracs = data["sim_plsIdxAllFrac"][ievt][isim]
            if len(idxs) == 0:
                continue
            best_j = int(idxs[np.argmax(fracs)])
            vals.append(float(etaErr_arr[best_j]))
    return np.array(vals)

print(f"Loading {REAL_F}")
real_vals  = collect_etaerr(REAL_F)
print(f"Loading {SYNTH_F}")
synth_vals = collect_etaerr(SYNTH_F)

print(f"Real  pLS: N={len(real_vals)}, median={np.median(real_vals):.5f}, "
      f"mean={np.mean(real_vals):.5f}")
print(f"Synth pLS: N={len(synth_vals)}, median={np.median(synth_vals):.5f}, "
      f"mean={np.mean(synth_vals):.5f}")

# axis range: 99th percentile of the real distribution (synth may be narrower)
xmax = float(np.percentile(real_vals, 99.5))
bins = np.linspace(0, xmax, 80)

INK, MUTED, GRID = "#333333", "#767676", "#DDDDDD"
C_REAL  = "#1f77b4"   # blue
C_SYNTH = "#ff7f0e"   # orange

fig, ax = plt.subplots(figsize=(7, 4.5))

ax.hist(real_vals,  bins=bins, color=C_REAL,  alpha=0.6, label="Real pLS",      density=False)
ax.hist(synth_vals, bins=bins, color=C_SYNTH, alpha=0.6, label="Synthetic pLS", density=False)

ax.axvline(np.median(real_vals),  color=C_REAL,  linestyle="--", linewidth=1.2,
           label=f"Real median = {np.median(real_vals):.4f}")
ax.axvline(np.median(synth_vals), color=C_SYNTH, linestyle="--", linewidth=1.2,
           label=f"Synth median = {np.median(synth_vals):.4f}")

ax.set_xlabel("pLS etaErr", fontsize=11, color=INK)
ax.set_ylabel("Counts per bin", fontsize=11, color=INK)
ax.set_title("pLS etaErr: real vs synthetic\n"
             "(best-matched pLS per accepted sim track)", fontsize=11, color=INK)
ax.legend(fontsize=9)
ax.set_xlim(0, xmax)
ax.spines[["top", "right"]].set_visible(False)
ax.spines[["left", "bottom"]].set_color(MUTED)
ax.tick_params(colors=MUTED, labelsize=9)
ax.grid(axis="y", color=GRID, linewidth=0.6)
ax.set_axisbelow(True)

fig.tight_layout()
fig.savefig(OUT, dpi=150)
print(f"Wrote {OUT}")
