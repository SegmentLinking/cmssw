# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

The **Line Segment Tracking (LST)** standalone — a heterogeneous CMS track reconstruction algorithm that builds tracks bottom-up from detector hits. It lives at `RecoTracker/LSTCore/standalone` within a CMSSW area and can be compiled/run independently of the full CMSSW framework.

## Project Overview

LST (Line Segment Tracking) is a high-performance particle track reconstruction algorithm for the CMS detector at the High Luminosity LHC. It uses heterogeneous computing (CPU/CUDA/ROCm) via the Alpaka abstraction layer.

The algorithm reconstructs tracks hierarchically:
- **MiniDoublets (MD)**: Pairs of adjacent hits (2 hits)
- **Segments (LS)**: Pairs of MiniDoublets (2 MDs, 4 hits)
- **Triplets (T3)**: Pairs of Segments sharing a middle MD (3 MDs, 6 hits)
- **Quadruplets (T4)**: Pairs of Triplets sharing a middle Segment (4 MDs, 8 hits)
- **Quintuplets (T5)**: Pairs of Triplets sharing a middle MD (5 MDs, 10 hits)
- **Pixel Line Segments (pLS)**: Inner Tracker (pixel) track seeds
- **Pixel Triplets (pT3)**: pLS matched with a T3 in the Outer Tracker
- **Pixel Quintuplets (pT5)**: pLS matched with a T5 in the Outer Tracker
- **Track Candidates (TC)**: Final output collection

## Standalone Directory
**Important**: All standalone builds and execution must be run from:

```
/mnt/data1/kk829/CMSSW_16_1_1/src/RecoTracker/LSTCore/standalone
```
Always `cd` to this directory before running setup, build, or execution commands.

Note for Claude Code: Each Bash tool invocation starts a fresh shell at `/mnt/data1/kk829`. Use `pushd` (NOT `cd` — it gets silently dropped). Prefix ALL commands with:

```bash
pushd /mnt/data1/kk829/CMSSW_16_1_1/src/RecoTracker/LSTCore/standalone && source setup.sh && cmsenv && source setup.sh && <your_command>
```

## Environment setup (required before every session)

```bash
cd RecoTracker/LSTCore/standalone
source setup.sh
```

`setup.sh` sources the CMSSW runtime to export `ALPAKA_ROOT`, `BOOST_ROOT`, `CUDA_HOME`, `FMT_ROOT`, `ROCM_ROOT`, `ROOT_ROOT`, and sets `TRACKLOOPERDIR`, `PATH`, and `LD_LIBRARY_PATH`.

## Build Commands

### CMSSW Build (Primary)
```bash
cd CMSSW_16_1_1/src
cmsenv
scram b -j 12
```

### Standalone Build
```bash
cd /mnt/data1/kk829/CMSSW_16_1_1/src/RecoTracker/LSTCore/standalone
source setup.sh; cmsenv; source setup.sh   # Run ONCE at start of session
lst_make_tracklooper -m            # Clean build (CPU + CUDA by default)
lst_make_tracklooper -C            # CPU only
lst_make_tracklooper -G            # CUDA only
lst_make_tracklooper -R            # ROCm only
lst_make_tracklooper -A            # All backends
lst_make_tracklooper -mcCd         # Clean CPU build with DNN/cut value branches
```

Build flags:
- `-m`: Clean build (make clean first)
- `-c`: Deprecated, but still used
- `-C`: CPU backend only
- `-G`: CUDA backend only
- `-R`: ROCm backend only
- `-d`: Enable cut value / DNN training branches in ntuple output

### Code Checks
```bash
scram b -j 12 code-checks >& c.log
scram b -j 12 code-format >& f.log
```

### Rebuild only the efficiency binary

```bash
cd efficiency && make
```

## Running LST

### Standalone Execution
```bash
# Required before running lst_cuda (recent addition — CUDA driver workaround):
export LD_LIBRARY_PATH=/mnt/data1/kk829/cuda_driver_libs_580:$LD_LIBRARY_PATH

lst_cuda -i trackingNtuple-100.root --jet -o output.root    # Run CUDA backend
lst_cpu -i trackingNtuple-100.root --jet -o output.root     # Run CPU backend (avoid: ~9× slower)

# Options:
#   -i <file>      Input file; ALWAYS use trackingNtuple-100.root or trackingNtuple-1000.root.
#                  NEVER use -i PU200 or any other named dataset.
#   --jet          ALWAYS include this flag unless explicitly told otherwise. Enables jet-branch
#                  reading (sim_genjet_deltaR etc.) for ΔR efficiency plots. Requires
#                  trackingNtuple-100.root (has jet branches); crashes on other inputs.
#   -n <nevents>   Number of events (-1 for all)
#   -v <level>     Verbosity (0=silent, 1=timing, 2=multiplicity)
#   -s <streams>   Concurrent streams
#   -p <ptCut>     Minimum pT cut in GeV (default: 0.8)
#   --allobj       Write all object branches (MD, LS, T3, T4, T5, pLS, pT3, pT5, etc.)
#                  Use this instead of the now-removed -l flag
```

### DNN Training Ntuple Generation
For generating ntuples with all branches needed for DNN training:
```bash
# Build with DNN branches enabled
lst_make_tracklooper -mcCd

# Generate training ntuple
lst_cpu -i trackingNtuple-100.root --allobj -n 300 -l -p 0.8 -s 32 -v 1 -o training_ntuple.root
```

Key flags for training data:
- `--allobj`: Writes all object branches (t5_*, pLS_*, t3_*, etc.)
- `-l`: Include lower-level objects (T3, T5, pT3, pT5 details)
- `-s 32`: Use 32 streams for parallel processing
- `-p 0.8`: pT cut at 0.8 GeV

### Performance Analysis
```bash
createPerfNumDenHists -i output.root -o histograms.root
lst_plot_performance.py histograms.root -t "tag"

# Comparison:
lst_plot_performance.py ref.root new.root -L Baseline,New -t "compare" --compare
```

### Timing Benchmarks
```bash
lst_timing PU200                      # Run timing with CUDA backend (default)
lst_timing -b cpu PU200               # Run timing with CPU backend
lst_timing PU200 explicit 200         # Run with specific config, 200 events

# Output: runs multiple stream configurations (1,2,4,6,8 for GPU; 1,4,16,32,64 for CPU)
# and reports average timing per stage
```

### All-in-One Script
```bash
lst_run -f -m -s trackingNtuple-100.root -n -1 -t myTag
# -f: compile, -m: clean make, -s: sample, -n: events, -t: tag
```

Each call to `lst_run` produces:
- `<tag>_<sample>_NEVT<n>__LSTNtuple.root` — per-object ntuple
- `<tag>_<sample>_NEVT<n>__LSTNumDen.root` — pre-computed num/den histograms
- `performance/<date>-<run>-<sample>/efficiency.root` — efficiency plots

### CMSSW Workflows
```bash
runTheMatrix.py -w upgrade -n -e -l 24834.703  # CPU workflow
runTheMatrix.py -w upgrade -n -e -l 24834.704  # GPU workflow
makeTrackValidationPlots.py --extended step4_24834.704.root  # MTV plots
```

## Code formatting / linting

```bash
# From LST/ directory
make format         # clang-format in-place
make check          # clang-tidy (read-only)
make check-fix      # clang-tidy + auto-fix
```

The `.clang-format` and `.clang-tidy` configs are at `RecoTracker/LSTCore/src/alpaka/`.

## Architecture

### Tracking object hierarchy (bottom-up)

```
Hit → MiniDoublet (MD) → LineSegment (LS) / PixelLineSegment (pLS)
    → Triplet (T3) → Quadruplet (T4) → Quintuplet (T5)
    → PixelTriplet (pT3) → PixelQuintuplet (pT5)
    → TrackCandidate (TC)
```

Each object type has a corresponding SoA layout in `interface/` (`*SoA.h`, `*HostCollection.h`, `*DeviceCollection.h` for the Alpaka device mirror).

### Directory Structure

| Path | Role |
|---|---|
| `interface/` | Public SoA data layouts (host) and geometry types |
| `interface/alpaka/` | Device SoA collections and the `LST` public API |
| `src/` | Host-side geometry: `EndcapGeometry.cc`, `ModuleConnectionMap.cc`, `LSTESData.cc` |
| `src/alpaka/` | Alpaka GPU kernel headers (`MiniDoublet.h`, `Segment.h`, …) and `LST.cc` / `LSTEvent.dev.cc` |
| `standalone/` | Independent build system and analysis tools |
| `standalone/LST/` | Compiled shared libraries (`liblst_{cpu,cuda,rocm}.so`) |
| `standalone/bin/` | `lst_run`, `lst_make_tracklooper` scripts; compiled `lst_{cpu,cuda,rocm}` executables |
| `standalone/code/core/` | Standalone analysis: `trkCore.cc` (event loop orchestration), `AccessHelper.cc` (hit/object traversal), `write_lst_ntuple.cc` (ROOT branch filling), `AnalysisConfig.h` (global `ana` config struct) |
| `standalone/code/rooutil/` | ROOT utility helpers (cxxopts, cutflow, event-index map) |
| `standalone/efficiency/src/` | `performance.cc` (main for `createPerfNumDenHists`), `LSTEff.cc` (ntuple reader), `helper.cc` |
| `standalone/efficiency/python/` | `lst_plot_performance.py` plotting script |

### Key Patterns

**Alpaka Namespacing**: All device code uses `ALPAKA_ACCELERATOR_NAMESPACE::lst` for backend abstraction. GPU kernels under `src/alpaka/` are compiled three times, once per backend (`_cpu.o`, `_cuda.o`, `_rocm.o`); the `ALPAKA_ACCELERATOR_NAMESPACE` macro selects the active backend at compile time.

**Structure of Arrays (SoA)**: Data structures are defined as SoA for GPU efficiency. Each object type has:
- `*SoA.h` - Layout definition
- `*HostCollection.h` - CPU-side collection
- `*DeviceCollection.h` - GPU-side collection (in `interface/alpaka/`)

**LST Object Types** (defined in `interface/Common.h`):
```cpp
enum LSTObjType : int8_t { T5 = 4, pT3 = 5, pT5 = 7, pLS = 8, T4 = 9 };
```

### Key Files
- `interface/alpaka/LST.h` - Main algorithm interface
- `interface/LSTESData.h` - EventSetup data (geometry, module maps)
- `interface/Common.h` - Constants, types, and parameter structs
- `src/alpaka/LST.cc` - Algorithm orchestration; the public type `ALPAKA_ACCELERATOR_NAMESPACE::lst::LSTEvent` is used throughout the analysis code via a `using` alias
- `src/alpaka/LSTEvent.dev.cc` - Event processing kernels; the `LSTEvent` class (`src/alpaka/LSTEvent.h`) owns all device and host SoA collections for a single event and drives the reconstruction sequence

### Data Flow

```
LSTInputDeviceCollection → LSTEvent processing → TrackCandidatesDeviceCollection
```

1. `bin/lst.cc` reads a tracking ntuple (ROOT TTree `trackingNtuple/tree`) using `Trktree` (a generated class in `code/core/Trktree.h`), and calls into the `trkCore` helpers.
2. Input hits are loaded into `LSTInputHostCollection` via `LSTPrepareInput.h`.
3. `LSTEvent` is constructed; reconstruction kernels run step by step (MDs → Segments → Triplets → … → TrackCandidates).
4. `write_lst_ntuple.cc` copies results back to host and fills the output ROOT ntuple branches.
5. `efficiency/` reads that ntuple to compute efficiency, fake rate, and duplicate rate histograms.

## Related Package

`RecoTracker/LST` contains CMSSW integration:
- `plugins/alpaka/` - EDProducers for framework integration
- `python/` - Configuration files
- `src/` - EventSetup data producers

