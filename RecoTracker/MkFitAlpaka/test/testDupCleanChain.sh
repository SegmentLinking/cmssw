#!/bin/bash
# Self-contained duplicate-cleaner unit test: the MkFitCore cleaners on synthetic events, then the MkFitAlpaka
# cleaner on every available Alpaka backend (a backend without a device prints a skip line and passes).
set -e
workdir=$(mktemp -d "${TMPDIR:-/tmp}/dupcleantest.XXXXXX")
trap 'rm -rf "$workdir"' EXIT
testMkFitAlpakaDupCleanMkFitCoreRef "$workdir/mkfitcore_ref.bin"
for backend in SerialSync CudaAsync ROCmAsync; do
  if command -v "testMkFitAlpakaDupClean$backend" > /dev/null; then
    "testMkFitAlpakaDupClean$backend" "$workdir/mkfitcore_ref.bin"
  fi
done
