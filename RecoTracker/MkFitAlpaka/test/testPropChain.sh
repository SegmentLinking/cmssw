#!/bin/bash
# Self-contained propagation unit test: the MkFitCore functions on synthetic Phase-2-like tracks, then MkFitAlpaka on
# every available Alpaka backend (a backend without a device prints a skip line and passes).
# No pipe on the comparators: under "set -e" their exit code (FAIL or a crash) is the test result.
set -e
workdir=$(mktemp -d "${TMPDIR:-/tmp}/proptest.XXXXXX")
trap 'rm -rf "$workdir"' EXIT
testMkFitAlpakaPropMkFitCoreRef "$workdir/mkfitcore_ref.bin" 5000 4242
for backend in SerialSync CudaAsync ROCmAsync; do
  if command -v "testMkFitAlpakaProp$backend" > /dev/null; then
    "testMkFitAlpakaProp$backend" "$workdir/mkfitcore_ref.bin"
  fi
done
