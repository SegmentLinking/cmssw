#!/bin/bash
# Self-contained unit test of the v2p2 finder pieces: the MkFitCore code on synthetic inputs, then MkFitAlpaka on every
# available Alpaka backend (a backend without a device prints a skip line and passes).
# No pipe on the comparators: under "set -e" their exit code (FAIL or a crash) is the test result.
set -e
workdir=$(mktemp -d "${TMPDIR:-/tmp}/v2p2test.XXXXXX")
trap 'rm -rf "$workdir"' EXIT
testMkFitAlpakaV2p2MkFitCoreRef "$workdir/mkfitcore_ref.bin" 4000 4242
for backend in SerialSync CudaAsync ROCmAsync; do
  if command -v "testMkFitAlpakaV2p2$backend" > /dev/null; then
    "testMkFitAlpakaV2p2$backend" "$workdir/mkfitcore_ref.bin"
  fi
done
