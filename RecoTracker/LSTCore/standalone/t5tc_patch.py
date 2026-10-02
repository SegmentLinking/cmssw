#!/usr/bin/env python3
"""Apply or restore code changes for T5-TC diagnostic experiments.
Usage: python3 t5tc_patch.py <apply|restore> <A|B|D>
"""
import sys
import os

SRCDIR = os.path.join(os.path.dirname(__file__),
                      "../src/alpaka")
TC_H      = os.path.join(SRCDIR, "TrackCandidate.h")
KERNELS_H = os.path.join(SRCDIR, "Kernels.h")
PQ_H      = os.path.join(SRCDIR, "PixelQuintuplet.h")


def patch(fname, old, new):
    with open(fname) as f:
        code = f.read()
    if old not in code:
        raise RuntimeError(f"Pattern not found in {fname}:\n{old!r}")
    with open(fname, "w") as f:
        f.write(code.replace(old, new, 1))


# ---------------------------------------------------------------------------
# Experiment A: disable CrossCleanT5 vs-pT5 isDup assignment
# ---------------------------------------------------------------------------
A_OLD = (
    "              if ((dR2 < 0.02f && d2 < 0.1f) || (dR2 < 1e-3f && d2 < 1.0f)) {\n"
    "                quintuplets.isDup()[iT5] = true;\n"
    "              }"
)
A_NEW = (
    "              // DISABLED experiment A: CrossCleanT5 vs pT5\n"
    "              // if ((dR2 < 0.02f && d2 < 0.1f) || (dR2 < 1e-3f && d2 < 1.0f)) {\n"
    "              //   quintuplets.isDup()[iT5] = true;\n"
    "              // }"
)

# ---------------------------------------------------------------------------
# Experiment B: remove isPT5_jx/ix unconditional priority in RemoveDupQuintupletsBeforeTC
# ---------------------------------------------------------------------------
B_OLD = (
    "                if (isPT5_jx || score_rphisum1 > score_rphisum2) {\n"
    "                  rmQuintupletFromMemory(quintuplets, ix, true);\n"
    "                } else if (isPT5_ix || score_rphisum1 < score_rphisum2) {\n"
    "                  rmQuintupletFromMemory(quintuplets, jx, true);\n"
    "                } else {\n"
    "                  rmQuintupletFromMemory(quintuplets, (ix < jx ? ix : jx), true);\n"
    "                }"
)
B_NEW = (
    "                // MODIFIED experiment B: score-only, no pT5 priority\n"
    "                if (score_rphisum1 > score_rphisum2) {\n"
    "                  rmQuintupletFromMemory(quintuplets, ix, true);\n"
    "                } else if (score_rphisum1 < score_rphisum2) {\n"
    "                  rmQuintupletFromMemory(quintuplets, jx, true);\n"
    "                } else {\n"
    "                  rmQuintupletFromMemory(quintuplets, (ix < jx ? ix : jx), true);\n"
    "                }"
)

# ---------------------------------------------------------------------------
# Experiment D: disable tightCutFlag requirement in AddT5asTrackCandidate
# ---------------------------------------------------------------------------
D_OLD = (
    "          if (!(quintuplets.tightCutFlag()[quintupletIndex]))\n"
    "            continue;"
)
D_NEW = (
    "          // DISABLED experiment D: tightCutFlag\n"
    "          // if (!(quintuplets.tightCutFlag()[quintupletIndex]))\n"
    "          //   continue;"
)


# ---------------------------------------------------------------------------
# Experiment F1: skip dedup between standalone T5 and pT5-embedded T5
#   Change && → || in RemoveDupQuintupletsBeforeTC so the skip fires when
#   EITHER T5 is part of a pT5, not only when both are.
# ---------------------------------------------------------------------------
F1_OLD = (
    "              if (isPT5_ix && isPT5_jx)\n"
    "                continue;"
)
F1_NEW = (
    "              if (isPT5_ix || isPT5_jx)  // MODIFIED experiment F1: skip standalone-vs-pT5\n"
    "                continue;"
)

# F2 is the same code change as experiment A (CrossCleanT5 vs-pT5 disabled).
# It's defined as a separate key so F1+F2 can be applied/restored together without
# reusing A's entry (which might be in a different patched state).
F2_OLD = A_OLD
F2_NEW = A_NEW


# ---------------------------------------------------------------------------
# Experiment G: RemoveDupQuintupletsBeforeTC — require hit overlap for the kill,
#   dropping the pure-geometric arms (dR2<0.001 && d2<1.0) and (dR2<0.02 && d2<0.1).
#   Geometric proximity is not evidence of duplication in jet cores.
# ---------------------------------------------------------------------------
G_OLD = (
    "              if (((dR2 < 0.001f || nMatched >= minNHitsForDup_T5) && d2 < 1.0f) || (dR2 < 0.02f && d2 < 0.1f)) {"
)
G_NEW = (
    "              if (nMatched >= minNHitsForDup_T5 && d2 < 1.0f) {  // MODIFIED experiment G: hit-overlap-only dedup"
)


# ---------------------------------------------------------------------------
# Experiment H1: pT5 building sees dedup-flagged T5s (remove isDup skip)
# ---------------------------------------------------------------------------
H1_OLD = (
    "            if (quintuplets.isDup()[quintupletIndex])\n"
    "              continue;"
)
H1_NEW = (
    "            // DISABLED experiment H1: pT5 building sees isDup T5s\n"
    "            // if (quintuplets.isDup()[quintupletIndex])\n"
    "            //   continue;"
)

# ---------------------------------------------------------------------------
# Experiment H2: pT5 building sees dedup-flagged pLS too (remove isDup skip)
# ---------------------------------------------------------------------------
H2_OLD = (
    "          if (pixelSegments.isDup()[i_pLS])\n"
    "            continue;"
)
H2_NEW = (
    "          // DISABLED experiment H2: pT5 building sees isDup pLS\n"
    "          // if (pixelSegments.isDup()[i_pLS])\n"
    "          //   continue;"
)


EXPERIMENTS = {
    "A":  (TC_H,      A_OLD,  A_NEW),
    "B":  (KERNELS_H, B_OLD,  B_NEW),
    "D":  (TC_H,      D_OLD,  D_NEW),
    "F1": (KERNELS_H, F1_OLD, F1_NEW),
    "F2": (TC_H,      F2_OLD, F2_NEW),
    "G":  (KERNELS_H, G_OLD,  G_NEW),
    "H1": (PQ_H,      H1_OLD, H1_NEW),
    "H2": (PQ_H,      H2_OLD, H2_NEW),
}


def apply(name):
    fname, old, new = EXPERIMENTS[name]
    patch(fname, old, new)
    print(f"Applied experiment {name} to {os.path.basename(fname)}")


def restore(name):
    fname, old, new = EXPERIMENTS[name]
    patch(fname, new, old)
    print(f"Restored experiment {name} in {os.path.basename(fname)}")


if __name__ == "__main__":
    action, exp = sys.argv[1], sys.argv[2]
    if action == "apply":
        apply(exp)
    elif action == "restore":
        restore(exp)
    else:
        raise ValueError(f"Unknown action: {action}")
