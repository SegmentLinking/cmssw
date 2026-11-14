RecoTracker/MkFitAlpakaFormats: CUDA/ROCm dictionaries of the RecoTracker/MkFitAlpaka event products.
No device code here, on purpose (as RecoTracker/LST vs RecoTracker/LSTCore). Host-type dictionaries stay in
RecoTracker/MkFitAlpaka/src/classes_def.xml (host library, no device code).

Why a second package: with src/alpaka/classes_cuda_def.xml inside RecoTracker/MkFitAlpaka, the CUDA dictionary lands in
libRecoTrackerMkFitAlpakaCudaAsync.so (which holds the package kernels), and then every kernel compiled in the portable
plugin library (plugins/alpaka/*.dev.cc) failed to launch: cudaErrorInvalidResourceHandle at cudaLaunchKernel
(compute-sanitizer: cuKernelGetName invalid handle). The same tree without the CUDA dictionary, or with the dictionary
moved here, works. Root cause not traced further.

A product added to MkFitAlpaka needs its device lines here (classes_cuda_def.xml, classes_rocm_def.xml) and its host
lines in MkFitAlpaka/src/classes_def.xml.
