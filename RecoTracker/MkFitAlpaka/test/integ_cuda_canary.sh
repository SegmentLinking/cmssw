#!/bin/bash
# CUDA canary for kernels of the portable PLUGIN library.
# MkFitAlpakaProductsTest@alpaka (plugins/alpaka/MkFitProductsTestKernels.dev.cc, a plugin-library kernel) writes known
# values into the event products on the device; MkFitAlpakaProductsCheck recomputes every value on the host copy and
# throws on any mismatch. Catches the failure mode where plugin kernels fail to launch
# (cudaErrorInvalidResourceHandle) and tests silently compare zeros. Also runs the serial backend as a control.
# Exit 0 = pass (or no CUDA device: CUDA part skipped), non-zero = the canary died.
cfg=${LOCALTOP:-$CMSSW_BASE}/src/RecoTracker/MkFitAlpaka/test/integ_products_cfg.py
cmsRun $cfg backend=serial_sync maxEvents=2 streams=1 || { echo "FAIL: serial control"; exit 1; }
if ! cudaIsEnabled; then
  echo "no CUDA device: CUDA canary skipped"
  exit 0
fi
cmsRun $cfg backend=cuda_async maxEvents=4 streams=2 || { echo "FAIL: CUDA plugin-kernel canary"; exit 1; }
echo PASS
