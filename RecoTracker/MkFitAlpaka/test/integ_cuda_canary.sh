#!/bin/bash
# CUDA canary for plugin-library kernels: MkFitAlpakaProductsTest@alpaka writes known values into the event products
# on the device and MkFitAlpakaProductsCheck recomputes them on the host copy, so a kernel that fails to launch cannot
# pass by comparing zeros. The serial backend runs as a control; without a CUDA device the CUDA part is skipped.
cfg=${LOCALTOP:-$CMSSW_BASE}/src/RecoTracker/MkFitAlpaka/test/integ_products_cfg.py
cmsRun $cfg backend=serial_sync maxEvents=2 streams=1 || { echo "FAIL: serial control"; exit 1; }
if ! cudaIsEnabled; then
  echo "no CUDA device: CUDA canary skipped"
  exit 0
fi
cmsRun $cfg backend=cuda_async maxEvents=4 streams=2 || { echo "FAIL: CUDA plugin-kernel canary"; exit 1; }
echo PASS
