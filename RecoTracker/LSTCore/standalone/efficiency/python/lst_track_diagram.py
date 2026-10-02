#!/usr/bin/env python3
"""
Draw a hit-by-hit diagram of one sim track and its matched pT5.
Two panels: r-z (longitudinal) and x-y (transverse).

Usage:
  python3 lst_track_diagram.py [ntuple.root] [event_idx] [sim_idx]

Defaults: LSTNtuple_idealpls_fixed.root, event 0, sim track 29
"""

import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import uproot

# ── arguments ──────────────────────────────────────────────────────────────────
ntuple  = sys.argv[1] if len(sys.argv) > 1 else "LSTNtuple_idealpls_fixed.root"
evt_i   = int(sys.argv[2]) if len(sys.argv) > 2 else 0
sim_i   = int(sys.argv[3]) if len(sys.argv) > 3 else 29

# ── load one event ─────────────────────────────────────────────────────────────
BRANCHES = [
    # sim track kinematics
    "sim_pt", "sim_phi", "sim_eta", "sim_q",
    "sim_vx",  "sim_vy",  "sim_vz",
    "sim_pca_dxy", "sim_pca_dz",
    # sim hits
    "sim_simHitX", "sim_simHitY", "sim_simHitZ", "sim_simHitLayer",
    "sim_recoHitX", "sim_recoHitY", "sim_recoHitZ",
    # pT5 → pLS → pixel hits
    "pT5_t5Idx", "pT5_plsIdx", "pT5_simIdx", "pT5_simIdxAll", "pT5_simIdxAllFrac",
    "pT5_pt", "pT5_eta", "pT5_phi",
    "pLS_hit0_x","pLS_hit0_y","pLS_hit0_z",
    "pLS_hit1_x","pLS_hit1_y","pLS_hit1_z",
    "pLS_hit2_x","pLS_hit2_y","pLS_hit2_z",
    "pLS_hit3_x","pLS_hit3_y","pLS_hit3_z",
    "pLS_nhit",
    # t5 → T3 bridge
    "t5_t3Idx0", "t5_t3Idx1",
    # T3 hit-detail table (6 hits per T3 entry)
    "t5_t3_0_x","t5_t3_0_y","t5_t3_0_z","t5_t3_0_layer",
    "t5_t3_1_x","t5_t3_1_y","t5_t3_1_z","t5_t3_1_layer",
    "t5_t3_2_x","t5_t3_2_y","t5_t3_2_z","t5_t3_2_layer",
    "t5_t3_3_x","t5_t3_3_y","t5_t3_3_z","t5_t3_3_layer",
    "t5_t3_4_x","t5_t3_4_y","t5_t3_4_z","t5_t3_4_layer",
    "t5_t3_5_x","t5_t3_5_y","t5_t3_5_z","t5_t3_5_layer",
]

f    = uproot.open(ntuple)
tree = f["tree"]
ev   = tree.arrays(BRANCHES, entry_start=evt_i, entry_stop=evt_i+1, library="np")
# unwrap the length-1 outer dimension
ev   = {k: v[0] for k, v in ev.items()}

# ── sim track ──────────────────────────────────────────────────────────────────
pt   = ev["sim_pt"][sim_i]
phi0 = ev["sim_phi"][sim_i]
eta  = ev["sim_eta"][sim_i]
q    = ev["sim_q"][sim_i]         # +1 / -1
vx   = ev["sim_vx"][sim_i]       # mm
vy   = ev["sim_vy"][sim_i]
vz   = ev["sim_vz"][sim_i]

print(f"Sim track {sim_i}: pt={pt:.2f} GeV  phi={phi0:.3f}  eta={eta:.3f}  q={int(q)}")
print(f"  vertex: ({vx:.2f}, {vy:.2f}, {vz:.2f}) mm")

# sim hits
sh_x = np.array(ev["sim_simHitX"][sim_i])    # mm
sh_y = np.array(ev["sim_simHitY"][sim_i])
sh_z = np.array(ev["sim_simHitZ"][sim_i])
sh_l = np.array(ev["sim_simHitLayer"][sim_i])

# reco hits matched to this sim track (cluster-level, in tracker ntuple)
rh_x = np.array(ev["sim_recoHitX"][sim_i])
rh_y = np.array(ev["sim_recoHitY"][sim_i])
rh_z = np.array(ev["sim_recoHitZ"][sim_i])

print(f"  sim hits: {len(sh_x)}   reco hits (cluster): {len(rh_x)}")

# ── find the best-matched pT5 ──────────────────────────────────────────────────
pt5_best = None
best_frac = -1.0
for i, (si, frac) in enumerate(zip(ev["pT5_simIdxAll"], ev["pT5_simIdxAllFrac"])):
    si   = np.asarray(si)
    frac = np.asarray(frac)
    mask = si == sim_i
    if mask.any():
        f_val = float(frac[mask].max())
        if f_val > best_frac:
            best_frac = f_val
            pt5_best  = i

if pt5_best is None:
    print("No pT5 matched to this sim track. Check sim_i.")
    sys.exit(1)

print(f"Matched pT5 index {pt5_best}  (match fraction {best_frac:.2f})")
print(f"  pT5: pt={ev['pT5_pt'][pt5_best]:.2f} eta={ev['pT5_eta'][pt5_best]:.3f} phi={ev['pT5_phi'][pt5_best]:.3f}")

# ── pLS pixel hits ─────────────────────────────────────────────────────────────
pls_i = ev["pT5_plsIdx"][pt5_best]
nhit  = int(ev["pLS_nhit"][pls_i])
pix_x = np.array([ev[f"pLS_hit{h}_x"][pls_i] for h in range(nhit)])
pix_y = np.array([ev[f"pLS_hit{h}_y"][pls_i] for h in range(nhit)])
pix_z = np.array([ev[f"pLS_hit{h}_z"][pls_i] for h in range(nhit)])
print(f"  pLS index {pls_i}: {nhit} pixel hits")

# ── T5 OT hits (index-trap-safe) ──────────────────────────────────────────────
t5_i   = ev["pT5_t5Idx"][pt5_best]
t3_0   = ev["t5_t3Idx0"][t5_i]   # index into hit-detail table, inner T3
t3_1   = ev["t5_t3Idx1"][t5_i]   # index into hit-detail table, outer T3

ot_x, ot_y, ot_z, ot_layer = [], [], [], []
for t3_idx, label in [(t3_0, "inner"), (t3_1, "outer")]:
    xs = [ev[f"t5_t3_{h}_x"][t3_idx] for h in range(6)]
    ys = [ev[f"t5_t3_{h}_y"][t3_idx] for h in range(6)]
    zs = [ev[f"t5_t3_{h}_z"][t3_idx] for h in range(6)]
    ls = [ev[f"t5_t3_{h}_layer"][t3_idx] for h in range(6)]
    print(f"  T3 {label} (idx={t3_idx}): layers={ls}  r={[f'{np.sqrt(x**2+y**2):.0f}' for x,y in zip(xs,ys)]} mm")
    ot_x.extend(xs); ot_y.extend(ys); ot_z.extend(zs); ot_layer.extend(ls)

ot_x = np.array(ot_x); ot_y = np.array(ot_y)
ot_z = np.array(ot_z); ot_layer = np.array(ot_layer)

# deduplicate shared middle MD (identical positions)
unique_mask = np.ones(len(ot_x), dtype=bool)
for i in range(len(ot_x)):
    for j in range(i):
        if abs(ot_x[i]-ot_x[j])<0.01 and abs(ot_y[i]-ot_y[j])<0.01:
            unique_mask[i] = False
            break
ot_x_u = ot_x[unique_mask]; ot_y_u = ot_y[unique_mask]
ot_z_u = ot_z[unique_mask]; ot_layer_u = ot_layer[unique_mask]
print(f"  OT hits (unique): {unique_mask.sum()} / {len(ot_x)} total")

# ── helix parametrization ──────────────────────────────────────────────────────
# CMS: B = 3.8 T along +z.  pT [GeV], R [mm]
B   = 3.8   # T
R   = pt / (0.3 * B) * 1e3  # mm
# cotTheta = pz/pT = sinh(eta)
cot = np.sinh(eta)

# helix parameter t: in transverse plane, phi(t) = phi0 + q*t
# x(t) = vx + R*(sin(phi0 + q*t) - sin(phi0))
# y(t) = vy - R*(cos(phi0 + q*t) - cos(phi0))  [sign matches CMS convention]
# z(t) = vz + R*cot*t
# r(t) = sqrt(x^2+y^2)

# integrate to r_max = 800 mm (just past OT layer 6)
# find t_max: at large r, r ≈ 2*R gives t_max ~ pi (half-circle) for low-pT
r_max_helix = 830.0   # mm
t_vals = np.linspace(0, 2*np.pi, 4000)
phi_t  = phi0 + q * t_vals
hx = vx + R * (np.sin(phi_t) - np.sin(phi0))
hy = vy - R * (np.cos(phi_t) - np.cos(phi0))
hz = vz + R * cot * t_vals
hr = np.sqrt(hx**2 + hy**2)

# keep only t where r < r_max and z within ±3000 mm
keep = (hr < r_max_helix) & (np.abs(hz) < 3000)
# stop at first exit
if keep.any():
    last_good = np.where(keep)[0][-1]
    keep[last_good+1:] = False
hx_p = hx[keep]; hy_p = hy[keep]
hz_p = hz[keep]; hr_p = hr[keep]

print(f"  Helix: R={R:.0f} mm  cot(theta)={cot:.3f}")

# ── derive layer guide radii from the actual hit data ─────────────────────────
# Use the unique r values of the T5 OT hits, rounded to nearest 5 mm
ot_r_all = np.sqrt(ot_x**2 + ot_y**2)
# each pair of hits at nearly same r = one MD sensor layer
guide_r_ot = sorted(set(round(r/5)*5 for r in ot_r_all))   # OT (T5) layers
# pLS pixel hit radii
pix_r_all = np.sqrt(pix_x**2 + pix_y**2)
guide_r_pix = sorted(set(round(r/2)*2 for r in pix_r_all))  # pixel layers

# ── compute useful plot ranges ─────────────────────────────────────────────────
sh_r = np.sqrt(sh_x**2 + sh_y**2)
rh_r = np.sqrt(rh_x**2 + rh_y**2)
pix_r = pix_r_all
ot_r_u = np.sqrt(ot_x_u**2 + ot_y_u**2)

all_r  = np.concatenate([sh_r, rh_r, pix_r, ot_r_u])
all_x  = np.concatenate([sh_x, rh_x, pix_x, ot_x_u])
all_y  = np.concatenate([sh_y, rh_y, pix_y, ot_y_u])
all_z  = np.concatenate([sh_z, rh_z, pix_z, ot_z_u])

r_pad  = max(all_r) * 1.3
z_lo   = min(all_z) - 20
z_hi   = max(all_z) + 20
r_hi   = max(all_r) + 15

# ── PLOT ───────────────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 8))
fig.suptitle(
    f"Event {evt_i} · Sim track {sim_i} · pT={pt:.1f} GeV · η={eta:.2f} · φ={phi0:.2f} · q={int(q)}"
    f"\nMatched pT5 index {pt5_best} (frac {best_frac:.2f})   "
    f"pLS idx {pls_i} ({nhit} pixel hits) + T5 idx {t5_i} (10 OT hits)",
    fontsize=11
)

# three panels: big r-z on left, x-y zoomed top-right, x-y full bottom-right
gs = fig.add_gridspec(2, 2, width_ratios=[1.1, 1], hspace=0.35, wspace=0.3)
ax_rz  = fig.add_subplot(gs[:, 0])     # full-height longitudinal
ax_xy  = fig.add_subplot(gs[0, 1])     # transverse zoomed
ax_xy2 = fig.add_subplot(gs[1, 1])     # transverse full (helix context)

# colour palette
C_HELIX = "#2196F3"   # blue   – truth helix
C_SIM   = "#4CAF50"   # green  – sim hits
C_RECO  = "#F44336"   # red    – reco cluster hits
C_PIX   = "#FF9800"   # orange – pLS pixel hits
C_OT    = "#9C27B0"   # purple – T5 hits

# ── panel 1: r-z (zoomed to hits) ─────────────────────────────────────────────
ax = ax_rz
ax.set_title("Longitudinal view  (r – z)", fontsize=10)

# layer guide lines
for r in guide_r_pix:
    ax.axhline(r, color='#FFF3E0', lw=6, zorder=0, alpha=0.8)   # orange tint for pixel
for r in guide_r_ot:
    ax.axhline(r, color='#EDE7F6', lw=6, zorder=0, alpha=0.8)   # purple tint for OT

# helix (clipped to zoom range)
hz_clip = hz_p[(hz_p >= z_lo) & (hz_p <= z_hi) & (hr_p <= r_hi)]
hr_clip = hr_p[(hz_p >= z_lo) & (hz_p <= z_hi) & (hr_p <= r_hi)]
ax.plot(hz_clip, hr_clip, color=C_HELIX, lw=2.0, label="Truth helix", zorder=2)

# sim hits
ax.scatter(sh_z, sh_r, color=C_SIM,  s=60, zorder=4, label=f"Sim hits ({len(sh_x)})", marker='s', edgecolors='darkgreen', linewidths=0.8)
# reco cluster hits
ax.scatter(rh_z, rh_r, color=C_RECO, s=65, zorder=5, label=f"Reco cluster hits ({len(rh_x)})", marker='D', edgecolors='darkred', linewidths=0.8)
# pLS pixel hits
ax.scatter(pix_z, pix_r, color=C_PIX,  s=120, zorder=6, label=f"pLS pixel hits ({nhit})", marker='*', edgecolors='darkorange', linewidths=0.8)
# T5 OT hits
ax.scatter(ot_z_u, ot_r_u, c=C_OT, s=100, zorder=6, label=f"T5 hits ({len(ot_x_u)})", marker='o', edgecolors='indigo', linewidths=0.8)

# annotate T5 layer numbers
for xi, ri, li in zip(ot_z_u, ot_r_u, ot_layer_u):
    ax.annotate(f"L{int(li)}", (xi, ri), textcoords="offset points",
                xytext=(6, 3), fontsize=7, color='indigo', fontweight='bold')

# annotate pLS hit numbers
for hi, (xi, ri) in enumerate(zip(pix_z, pix_r)):
    ax.annotate(f"px{hi}", (xi, ri), textcoords="offset points",
                xytext=(6, -8), fontsize=7, color='darkorange', fontweight='bold')

ax.set_xlabel("z  [mm]", fontsize=9)
ax.set_ylabel("r  [mm]", fontsize=9)
ax.set_xlim(z_lo, z_hi)
ax.set_ylim(0, r_hi)
ax.legend(fontsize=7.5, loc='upper left')

# label detector regions
for r, name in zip(guide_r_pix[-1:] + guide_r_ot[:1],
                   ["pixel", "T5"]):
    ax.text(z_hi - 5, r + 1.5, name, fontsize=7, color='grey', ha='right')

# ── panel 2: x-y zoomed ───────────────────────────────────────────────────────
ax = ax_xy
ax.set_title("Transverse view  (x – y, zoomed)", fontsize=9)
ax.set_aspect('equal')

# guide circles – pixel layers
for r in guide_r_pix:
    circ = plt.Circle((0,0), r, fill=False, color='#FF9800', lw=0.8, ls='--', zorder=0, alpha=0.5)
    ax.add_patch(circ)
# guide circles – T5 OT layers
for r in guide_r_ot:
    circ = plt.Circle((0,0), r, fill=False, color='#9C27B0', lw=0.8, ls=':', zorder=0, alpha=0.5)
    ax.add_patch(circ)

ax.plot(hx_p, hy_p, color=C_HELIX, lw=2.0, zorder=2)
ax.scatter(sh_x, sh_y, color=C_SIM,  s=60, zorder=4, marker='s', edgecolors='darkgreen', linewidths=0.8)
ax.scatter(rh_x, rh_y, color=C_RECO, s=65, zorder=5, marker='D', edgecolors='darkred', linewidths=0.8)
ax.scatter(pix_x, pix_y, color=C_PIX, s=120, zorder=6, marker='*', edgecolors='darkorange', linewidths=0.8)
ax.scatter(ot_x_u, ot_y_u, c=C_OT,   s=100, zorder=6, marker='o', edgecolors='indigo', linewidths=0.8)
for xi, yi, li in zip(ot_x_u, ot_y_u, ot_layer_u):
    ax.annotate(f"L{int(li)}", (xi, yi), textcoords="offset points",
                xytext=(5, 3), fontsize=7, color='indigo', fontweight='bold')
ax.scatter([vx], [vy], color='black', s=60, marker='+', zorder=7, label="Sim vertex")

ax.set_xlabel("x  [mm]", fontsize=9)
ax.set_ylabel("y  [mm]", fontsize=9)
ax.set_xlim(-r_pad, r_pad)
ax.set_ylim(-r_pad, r_pad)

# ── panel 3: x-y wide (helix context) ─────────────────────────────────────────
ax = ax_xy2
ax.set_title("Transverse view  (x – y, full scale)", fontsize=9)
ax.set_aspect('equal')

ax.plot(hx_p, hy_p, color=C_HELIX, lw=1.5, label="Truth helix", zorder=2)
# draw a box showing the zoomed region
rect = mpatches.Rectangle((-r_pad, -r_pad), 2*r_pad, 2*r_pad,
                           linewidth=1.5, edgecolor='gray', facecolor='none',
                           linestyle='--', zorder=5)
ax.add_patch(rect)
ax.scatter(sh_x, sh_y, color=C_SIM,  s=40, zorder=4, label="Sim hits", marker='s')
ax.scatter(pix_x, pix_y, color=C_PIX, s=80, zorder=6, label="pLS hits", marker='*')
ax.scatter(ot_x_u, ot_y_u, c=C_OT,   s=60, zorder=6, label="T5 hits", marker='o')
ax.scatter([vx], [vy], color='black', s=60, marker='+', zorder=7, label="Vertex")
ax.set_xlabel("x  [mm]", fontsize=9)
ax.set_ylabel("y  [mm]", fontsize=9)
r_wide = 820
ax.set_xlim(-r_wide, r_wide)
ax.set_ylim(-r_wide, r_wide)
ax.legend(fontsize=7, loc='upper right')

plt.tight_layout()
outname = f"track_diagram_evt{evt_i}_sim{sim_i}.png"
plt.savefig(outname, dpi=150, bbox_inches='tight')
print(f"\nSaved: {outname}")
