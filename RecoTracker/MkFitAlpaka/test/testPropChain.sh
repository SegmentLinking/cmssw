#!/bin/bash
# Self-contained propagation unit test: the MkFitCore functions on synthetic Phase-2-like tracks, then MkFitAlpaka on
# every available Alpaka backend (a backend without a device prints a skip line and passes).
set -e
workdir=$(mktemp -d "${TMPDIR:-/tmp}/proptest.XXXXXX")
trap 'rm -rf "$workdir"' EXIT
testMkFitAlpakaPropMkFitCoreRef "$workdir/mkfitcore_ref.bin" 5000 4242
for backend in SerialSync CudaAsync ROCmAsync; do
  if command -v "testMkFitAlpakaProp$backend" > /dev/null; then
    "testMkFitAlpakaProp$backend" "$workdir/mkfitcore_ref.bin" | tail -n 1
  fi
done
