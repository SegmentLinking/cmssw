#!/bin/bash
# Propagation self-contained unit test: MkFitCore reference on synthetic Phase-2-like tracks, then the port on every
# available Alpaka backend (a backend without a device prints a skip line and passes).
set -e
WORK=$(mktemp -d "${TMPDIR:-/tmp}/proptest.XXXXXX")
trap 'rm -rf "$WORK"' EXIT
testMkFitAlpakaPropMkFitCoreRef "$WORK/mkfitcore_ref.bin" 5000 4242
for b in SerialSync CudaAsync ROCmAsync; do
  if command -v "testMkFitAlpakaProp$b" > /dev/null; then
    "testMkFitAlpakaProp$b" "$WORK/mkfitcore_ref.bin" | tail -n 1
  fi
done
