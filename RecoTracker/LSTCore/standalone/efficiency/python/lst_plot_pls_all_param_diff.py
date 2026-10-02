#!/usr/bin/env python3
"""
lst_plot_pls_all_param_diff.py

For every accepted sim track that has BOTH a real pLS and a synthetic pLS
(best-matched by highest sim_plsIdxAllFrac), compare all computed pLS parameters.

Three output figures:
  <prefix>_kinematics.png   — pt, eta, phi, px, py, pz, ptErr, etaErr,
                               deltaPhi, circleRadius, circleCenterX, circleCenterY
  <prefix>_hits.png         — hit0-3 x/y/z positions (hit3==hit2 for triplet pLS)
  <prefix>_categorical.png  — charge, nhit, isQuad

Usage:
  python3 lst_plot_pls_all_param_diff.py \\
      [--real LSTNtuple_fix2.root] \\
      [--synth LSTNtuple_idealpls_fixed.root] \\
      [--out-prefix pls_all_param_diff]
"""

import argparse
import numpy as np
import uproot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── acceptance cuts (match original lst_plot_pls_matched_diff.py) ─────────────
PT_CUT    = 0.9
ETA_CUT   = 4.5
VTX_Z_MAX = 30.0
VTX_R_MAX = 2.5

# ── plot style ─────────────────────────────────────────────────────────────────
INK   = "#333333"
MUTED = "#767676"
GRID  = "#DDDDDD"
C     = "#009E73"   # same green as original comparison plots


def dphi_scalar(a, b):
    d = a - b
    while d >  np.pi: d -= 2 * np.pi
    while d < -np.pi: d += 2 * np.pi
    return d


PLS_FIELDS = [
    "pLS_pt", "pLS_eta", "pLS_phi", "pLS_px", "pLS_py", "pLS_pz",
    "pLS_ptErr", "pLS_etaErr",
    "pLS_deltaPhi", "pLS_circleRadius", "pLS_circleCenterX", "pLS_circleCenterY",
    "pLS_hit0_x", "pLS_hit0_y", "pLS_hit0_z",
    "pLS_hit1_x", "pLS_hit1_y", "pLS_hit1_z",
    "pLS_hit2_x", "pLS_hit2_y", "pLS_hit2_z",
    "pLS_hit3_x", "pLS_hit3_y", "pLS_hit3_z",
    "pLS_charge", "pLS_nhit", "pLS_isQuad",
]

SIM_FIELDS = [
    "sim_pt", "sim_eta", "sim_q", "sim_vx", "sim_vy", "sim_vz",
    "sim_plsIdxAll", "sim_plsIdxAllFrac",
]


def load_data(path):
    return uproot.open(path)["tree"].arrays(SIM_FIELDS + PLS_FIELDS, library="np")


def best_pls_params(data, ievt, isim):
    """Return dict of pLS values for the best-matched pLS, or None."""
    idxs  = data["sim_plsIdxAll"][ievt][isim]
    fracs = data["sim_plsIdxAllFrac"][ievt][isim]
    if len(idxs) == 0:
        return None
    best_j = int(idxs[np.argmax(fracs)])
    return {f: float(data[f][ievt][best_j]) for f in PLS_FIELDS}


def compute_diff(r_val, s_val, field):
    """Return the scalar difference value to accumulate."""
    if field == "pLS_phi":
        return dphi_scalar(r_val, s_val)
    elif field in ("pLS_pt", "pLS_ptErr", "pLS_circleRadius"):
        # always positive → relative difference
        return (r_val - s_val) / abs(s_val) if abs(s_val) > 1e-9 else float("nan")
    else:
        return r_val - s_val


def collect_diffs(dr, ds):
    n_events = len(dr["sim_pt"])
    assert n_events == len(ds["sim_pt"]), "Event count mismatch"

    diffs = {f: [] for f in PLS_FIELDS}
    n_both = 0

    for ievt in range(n_events):
        sim_pt  = dr["sim_pt"][ievt].astype(float)
        sim_eta = dr["sim_eta"][ievt].astype(float)
        sim_q   = dr["sim_q"][ievt].astype(np.int32)
        sim_vx  = dr["sim_vx"][ievt].astype(float)
        sim_vy  = dr["sim_vy"][ievt].astype(float)
        sim_vz  = dr["sim_vz"][ievt].astype(float)
        vtx_perp = np.hypot(sim_vx, sim_vy)

        accept = (
            (np.abs(sim_q) == 1) &
            (sim_pt > PT_CUT) &
            (np.abs(sim_eta) < ETA_CUT) &
            (np.abs(sim_vz) < VTX_Z_MAX) &
            (vtx_perp < VTX_R_MAX)
        )

        for isim in np.where(accept)[0]:
            R = best_pls_params(dr, ievt, isim)
            S = best_pls_params(ds, ievt, isim)
            if R is None or S is None:
                continue
            n_both += 1
            for f in PLS_FIELDS:
                diffs[f].append(compute_diff(R[f], S[f], f))

    print(f"Sim tracks with BOTH real and synthetic pLS: {n_both}")
    return {f: np.array(v, dtype=float) for f, v in diffs.items()}, n_both


def rng99(a):
    """Symmetric 99th-percentile range; minimum 1e-10 to avoid degenerate axes."""
    finite = a[np.isfinite(a)]
    if len(finite) == 0:
        return 1.0
    return max(float(np.percentile(np.abs(finite), 99)), 1e-10)


def plot_panel(ax, data, xlabel, title):
    a = data[np.isfinite(data)]
    if len(a) == 0:
        ax.text(0.5, 0.5, "no data", ha="center", va="center",
                transform=ax.transAxes, fontsize=9)
        ax.set_title(title, fontsize=9, color=INK, loc="left")
        return
    r = rng99(a)
    bins = np.linspace(-r, r, 61)
    clipped = np.clip(a, bins[0], bins[-1] - 1e-12)
    ax.hist(clipped, bins=bins, color=C, edgecolor="white", linewidth=0.3)
    ax.axvline(0, color="#D55E00", linestyle="--", linewidth=1.1)
    med = np.median(a)
    ax.axvline(med, color=INK, linestyle=":", linewidth=1.1)
    p16, p84 = np.percentile(a, [16, 84])
    ax.text(0.03, 0.96,
            f"med={med:+.4g}\n68%=[{p16:+.3g},{p84:+.3g}]",
            transform=ax.transAxes, fontsize=7, color=INK, va="top")
    ax.set_title(title, fontsize=9, color=INK, loc="left")
    ax.set_xlabel(xlabel, fontsize=8, color=INK)
    ax.set_ylabel("sim tracks", fontsize=8, color=INK)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines[["left", "bottom"]].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=7)
    ax.grid(axis="y", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real",       default="LSTNtuple_fix2.root")
    parser.add_argument("--synth",      default="LSTNtuple_idealpls_fixed.root")
    parser.add_argument("--out-prefix", default="pls_all_param_diff")
    args = parser.parse_args()

    print(f"Loading real  pLS: {args.real}")
    dr = load_data(args.real)
    print(f"Loading synth pLS: {args.synth}")
    ds = load_data(args.synth)

    diffs, n_both = collect_diffs(dr, ds)
    title_stem = f"real − synthetic pLS (best-matched per sim track, N={n_both})"

    # ── Figure 1: kinematic + derived (3 rows × 4 cols) ───────────────────────
    panels_kin = [
        ("pLS_pt",           "(pT_r − pT_s)/pT_s",            "A — pT (rel)"),
        ("pLS_eta",          "η_r − η_s",                       "B — η"),
        ("pLS_phi",          "Δφ(φ_r, φ_s) [rad]",             "C — φ"),
        ("pLS_deltaPhi",     "δφ_r − δφ_s [rad]",              "D — deltaPhi"),
        ("pLS_px",           "px_r − px_s [GeV/c]",            "E — px"),
        ("pLS_py",           "py_r − py_s [GeV/c]",            "F — py"),
        ("pLS_pz",           "pz_r − pz_s [GeV/c]",            "G — pz"),
        ("pLS_circleRadius", "(R_r − R_s)/R_s",                 "H — circleRadius (rel)"),
        ("pLS_ptErr",        "(σpT_r − σpT_s)/σpT_s",          "I — ptErr (rel)"),
        ("pLS_etaErr",       "σ(η)_r − σ(η)_s",                "J — etaErr"),
        ("pLS_circleCenterX","cx_r − cx_s [cm]",                "K — circleCenterX"),
        ("pLS_circleCenterY","cy_r − cy_s [cm]",                "L — circleCenterY"),
    ]

    fig1, axes1 = plt.subplots(3, 4, figsize=(18, 11))
    fig1.suptitle(
        f"pLS parameter differences: {title_stem}\nKinematic + derived parameters",
        fontsize=11, color=INK
    )
    for ax, (field, xlabel, title) in zip(axes1.flat, panels_kin):
        plot_panel(ax, diffs[field], xlabel, title)
    fig1.tight_layout(rect=[0, 0, 1, 0.96])
    out1 = f"{args.out_prefix}_kinematics.png"
    fig1.savefig(out1, dpi=150)
    print(f"Wrote {out1}")

    # ── Figure 2: hit positions (4 rows × 3 cols) ─────────────────────────────
    panels_hits = []
    labels = ["hit0 (innermost)", "hit1", "hit2", "hit3 (=hit2 for triplet pLS)"]
    for hit_n, label in enumerate(labels):
        for coord in ["x", "y", "z"]:
            field = f"pLS_hit{hit_n}_{coord}"
            panels_hits.append((field, f"{coord}_r − {coord}_s [cm]",
                                 f"hit{hit_n} {coord}  [{label}]"))

    fig2, axes2 = plt.subplots(4, 3, figsize=(13, 14))
    fig2.suptitle(
        f"pLS parameter differences: {title_stem}\nHit positions",
        fontsize=11, color=INK
    )
    for ax, (field, xlabel, title) in zip(axes2.flat, panels_hits):
        plot_panel(ax, diffs[field], xlabel, title)
    fig2.tight_layout(rect=[0, 0, 1, 0.97])
    out2 = f"{args.out_prefix}_hits.png"
    fig2.savefig(out2, dpi=150)
    print(f"Wrote {out2}")

    # ── Figure 3: categorical (1 row × 3 cols) ────────────────────────────────
    panels_cat = [
        ("pLS_charge", "charge_r − charge_s",  "charge"),
        ("pLS_nhit",   "nhit_r − nhit_s",        "nhit"),
        ("pLS_isQuad", "isQuad_r − isQuad_s",    "isQuad"),
    ]

    fig3, axes3 = plt.subplots(1, 3, figsize=(11, 3.8))
    fig3.suptitle(
        f"pLS parameter differences: {title_stem}\nDiscrete / categorical parameters",
        fontsize=11, color=INK
    )
    for ax, (field, xlabel, title) in zip(axes3.flat, panels_cat):
        a = diffs[field][np.isfinite(diffs[field])].astype(int)
        vals = np.sort(np.unique(a))
        counts = np.array([np.sum(a == v) for v in vals])
        ax.bar(vals, counts, color=C, edgecolor="white", width=0.6)
        for v, c in zip(vals, counts):
            ax.text(v, c + max(counts) * 0.01, str(c),
                    ha="center", va="bottom", fontsize=8, color=INK)
        ax.set_title(title, fontsize=10, color=INK)
        ax.set_xlabel(xlabel, fontsize=9, color=INK)
        ax.set_ylabel("sim tracks", fontsize=9, color=INK)
        ax.set_xticks(vals)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["left", "bottom"]].set_color(MUTED)
        ax.tick_params(colors=MUTED, labelsize=8)
        ax.grid(axis="y", color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)

    fig3.tight_layout()
    out3 = f"{args.out_prefix}_categorical.png"
    fig3.savefig(out3, dpi=150)
    print(f"Wrote {out3}")


if __name__ == "__main__":
    main()
