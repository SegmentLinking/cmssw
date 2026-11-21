#!/bin/bash
# Self-contained EventOfHits unit test: MkFitCore's LayerOfHits on synthetic hits, then the MkFitAlpaka EventOfHits
# build on every available Alpaka backend (a backend without a device prints a skip line and passes).
set -e
workdir=$(mktemp -d "${TMPDIR:-/tmp}/eventofhitstest.XXXXXX")
trap 'rm -rf "$workdir"' EXIT
testMkFitAlpakaEventOfHitsMkFitCoreRef "$workdir/mkfitcore_ref.bin"
for backend in SerialSync CudaAsync ROCmAsync; do
  if command -v "testMkFitAlpakaEventOfHits$backend" > /dev/null; then
    "testMkFitAlpakaEventOfHits$backend" "$workdir/mkfitcore_ref.bin"
  fi
done
