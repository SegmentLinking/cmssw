#!/bin/bash
# Register / spill CI check. Runs test/replay_kernel_registers.sh, whose gate
# is ptxas's "Registers are spilled to local memory" lines for our functions (re-run on the PTX embedded in the
# package's CUDA objects, per architecture; or a build log with -l LOG). cuobjdump's LOCAL is always 0 under the CUDA
# ABI and can not see a spill: the REG / STACK table is information only. Waivers: test/register_spill_waivers.txt.
# Used by 'scram b runtests' (test testMkFitAlpakaKernelRegisters). Exit 0 = no unwaived spill (PASS, or
# KNOWN-SPILLS: test/register_spill_known.txt); 1 = unwaived spills;
# 2 = nothing to check (no CUDA objects / library: skipped).
here=$(dirname $(readlink -f $0))
bash $here/replay_kernel_registers.sh -q "$@"
rc=$?
[ $rc -eq 0 ] && echo PASS
[ $rc -eq 1 ] && echo "FAIL: unwaived register spills (list above)"
# known spills: reported and counted, not a PASS; the test does not fail on them (decision taken)
[ $rc -eq 3 ] && { echo "KNOWN-SPILLS: no unwaived spill, but kernels of the menu spill (list above); NOT a PASS"; exit 0; }
exit $rc
