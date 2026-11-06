#ifndef RecoTracker_MkFitAlpaka_src_alpaka_engine_EngineSelectBridgeKernels_h
#define RecoTracker_MkFitAlpaka_src_alpaka_engine_EngineSelectBridgeKernels_h

// Kernels of the engine <-> select (K2) bridge (EngineSelectBridge.h). Launched only from src/alpaka/Cands.dev.cc.

#include <alpaka/alpaka.hpp>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/prefixScan.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include "RecoTracker/MkFitAlpaka/interface/cands/CandEngineTypes.h"
#include "RecoTracker/MkFitAlpaka/interface/cands/CandsSoA.h"
#include "RecoTracker/MkFitAlpaka/src/alpaka/engine/EngineKernels.h"
#include "RecoTracker/MkFitAlpaka/src/alpaka/select/SelectSoA.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev {

  // K1 -> K2 list in one kernel of one block: per chunk of kChunk seeds, an exclusive prefix scan of seeds.nActive
  // with a carry, then each thread writes the listed candidates of its seeds at their offset (seed_cand_idx order:
  // seed-major, ic ascending). list.n() = total.
  class KernelEngineBuildList {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  ::mkfitdev::SeedCandsSoA::ConstView seeds,
                                  ::mkfitdev::SelListSoA::View list,
                                  int nSeeds,
                                  const int32_t* nDev) const {
      nSeeds = engineRows(nSeeds, nDev);
      constexpr int kChunk = 1024;
      auto& buf = alpaka::declareSharedVar<int32_t[kChunk], __COUNTER__>(acc);
      auto& ws = alpaka::declareSharedVar<int32_t[32], __COUNTER__>(acc);
      auto& carry = alpaka::declareSharedVar<int32_t, __COUNTER__>(acc);
      const int tid = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u];
      const int bdim = alpaka::getWorkDiv<alpaka::Block, alpaka::Threads>(acc)[0u];
      if (tid == 0)
        carry = 0;
      alpaka::syncBlockThreads(acc);
      for (int off = 0; off < nSeeds; off += kChunk) {
        const int len = (nSeeds - off) < kChunk ? (nSeeds - off) : kChunk;
        for (int i = tid; i < len; i += bdim)
          buf[i] = seeds.nActive(off + i);
        alpaka::syncBlockThreads(acc);
        cms::alpakatools::blockPrefixScan(acc, buf, len, ws);  // inclusive
        alpaka::syncBlockThreads(acc);
        const int carryIn = carry;
        for (int i = tid; i < len; i += bdim) {
          const int s = off + i;
          const uint8_t mask = seeds.activeMask(s);
          if (mask == 0)
            continue;
          int k = carryIn + buf[i] - seeds.nActive(s);
          for (int ic = 0; ic < ::mkfitdev::kMaxCandsPerSeed; ++ic) {
            if (!(mask & (1u << ic)))
              continue;
            list[k].row() = ::mkfitdev::candSlotRow(s, seeds.curBuf(s), ic);
            list[k].layer() = seeds.layer(s);
            list[k].region() = seeds.region(s);
            ++k;
          }
        }
        alpaka::syncBlockThreads(acc);
        if (tid == 0)
          carry = carryIn + buf[len - 1];
        alpaka::syncBlockThreads(acc);
      }
      if (tid == 0)
        list.n() = carry;
    }
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev

#endif
