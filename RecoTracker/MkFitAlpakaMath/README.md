# RecoTracker/MkFitAlpakaMath

Header-only portable (Alpaka) mkFit math, used by `RecoTracker/MkFitAlpaka` and available to other device packages:

| headers | content |
|---|---|
| `interface/matriplex/` | portable Matriplex (`MPlex*<N>`, `kNN` slots: 1 on a GPU, 8 on a CPU backend) and its generated products |
| `interface/math/` | vdt transliteration (`vdtMath.h`), CMSSW TkBfield (`TkBfield.h`), beam-line PCA and curvilinear Jacobian (`PcaToBeamLine.h`), helix plane crossings, navigation search, constants |
| `interface/alpaka/prop/` | plane propagation and Kalman update on Matriplex (`PropagationMPlex.h`, `KalmanUtilsMPlex.h`, `PropagationFlags.h`) |
| `interface/es/MaterialView.h` | the material-map view the propagation flags carry |

The package has no library and no device code, so a `<use name="RecoTracker/MkFitAlpakaMath"/>` adds include paths only.

## Why a separate package

`RecoTracker/MkFitAlpaka` is a library with kernels (`src/alpaka/*.dev.cc`). For the CUDA backend SCRAM builds such a
library with relocatable device code and exports its device objects as a static `lib<name>_nv.a`; every plugin or
library that `<use>`s the package device-links that archive (`nvcc -dlink ... -l<name>_nv`), so the package's kernels
are linked again into the dependent's own fatbin. In a process that also loads MkFitAlpaka's plugins, the same kernels
are then registered from several device-linked images, and the dependent's own first kernel launch can fail with
`cudaErrorInvalidResourceHandle` ("invalid resource handle"; seen with an alpaka seeder plugin that included only the
propagation headers; the serial backend is not affected).

So:
- a package that needs only the math / propagation headers: `<use name="RecoTracker/MkFitAlpakaMath"/>`;
- a package that needs other MkFitAlpaka headers (data formats, ES products) but no MkFitAlpaka symbols from its
  device code: `<use name="RecoTracker/MkFitAlpaka" source_only="1"/>` (include paths only, no link);
- a normal `<use name="RecoTracker/MkFitAlpaka"/>` only where the MkFitAlpaka library itself is called.

Test: `testMkFitAlpakaMath` (every backend) evaluates the field and the vdt functions on the device
and on the host and compares them.
