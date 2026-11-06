#ifndef RecoTracker_MkFitAlpaka_src_alpaka_hits_EventOfHitsKernels_h
#define RecoTracker_MkFitAlpaka_src_alpaka_hits_EventOfHitsKernels_h

// Device EventOfHits: per-layer phi/q binning of all hits reproducing mkfit::LayerOfHits
// (HitStructures.cc registerHit/endRegistrationOfHits, binnor.h finalize_registration, suckInDeads).
//
// MkFitCore order inside a layer: hits registered in increasing original index, then sorted by the masked bin key
// (q N-bin << 24 | 16-bit fine phi bin) with an LSD radix sort, which is stable => order = (key, original index).
// Here: counting (atomic histogram per N-bin) + block prefix scans give each bin's first/count; hits are then
// scattered by a deterministic in-bin rank = #{members with (key, row) < own (key, row)}. The order does
// not depend on atomics. (MkFitCore uses std::sort, which is not stable, for layers with < 256 hits: equal keys there
// may come out in a different order.)

#include <cstdint>
#include <stdexcept>
#include <vector>

#include <alpaka/alpaka.hpp>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/prefixScan.h"
#include "HeterogeneousCore/AlpakaInterface/interface/warpsize.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include "RecoTracker/MkFitAlpaka/interface/hits/EventOfHitsSoA.h"
#include "RecoTracker/MkFitAlpaka/interface/hits/HitSoA.h"
#include "RecoTracker/MkFitAlpaka/interface/hits/alpaka/EventOfHitsDeviceCollections.h"
#include "RecoTracker/MkFitAlpaka/interface/hits/HitsMath.h"
#include "RecoTracker/MkFitAlpaka/interface/math/vdtMath.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev::hits {

  using ::mkfitdev::BinnedHitSoA;
  using ::mkfitdev::BinSoA;
  using ::mkfitdev::DeadRegionDev;
  using ::mkfitdev::HitSoA;
  using ::mkfitdev::LayerSoA;
  using ::mkfitdev::hitsmath::toBin;
  using ::mkfitdev::vdt::fast_atan2f;

  constexpr uint32_t kInvalid = 0xffffffffu;
  constexpr uint32_t kGBinMask = (1u << 24) - 1;  // global bin index bits of the per-hit (fine phi, bin) word
  constexpr uint32_t kBlock = 256;                // threads per block on GPU backends (= phi bins per q row)

  // LayerOfHits::HitInfo
  struct HitInfoDev {
    float phi, q, q_half_length, qbar;
  };

  // Per hit (LayerOfHits::registerHit): the hit info and the masked binnor key, i.e. B_pair(phiM, qM).mask_A2_M_bins()
  // = q N-bin << 24 | phi M-bin (16 bits: phi N-bin << 8 | 8 fine bits). One copy of the arithmetic for the GPU per-hit
  // kernel and the CPU per-layer build. Branch-free: on CPU backends a loop over it vectorizes.
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE uint32_t binHit(float x,
                                                      float y,
                                                      float z,
                                                      float hlArg,  // barrel ? e22 : e00 + e11 (loaded by the caller)
                                                      bool barrel,
                                                      bool pixel,
                                                      float qmin,
                                                      float qMLbhp,
                                                      float qMFac,
                                                      uint32_t qLastM,
                                                      float phiRMin,
                                                      float phiMFac,
                                                      HitInfoDev& info) {
    const float phi = fast_atan2f(y, x);   // Hit::phi()
    const float r = sqrtf(x * x + y * y);  // Hit::r()
    const float q = barrel ? z : r;        // Hit::z() / Hit::r()
    // hl_fac = is_pixel() ? 3 : sqrt(3); half_length, qbar as registerHit
    // (one sqrt of the selected argument: the same operations as MkFitCore on either branch)
    const float hlFac = pixel ? 3.0f : sqrtf(3.0f);
    info = HitInfoDev{phi, q, hlFac * sqrtf(hlArg), barrel ? r : z};
    // register_entry_safe: axis_pow2_u1::from_R_to_M_bin_safe, axis::from_R_to_M_bin_safe
    const uint32_t phiM = toBin((phi - phiRMin) * phiMFac) & ::mkfitdev::kPhiMaskM;
    const uint32_t qMIn = uint32_t(toBin((q - qmin) * qMFac));
    const uint32_t qM = q <= qmin ? 0u : (q >= qMLbhp ? qLastM : qMIn);
    // B_pair(phiM, qM).mask_A2_M_bins(): drop the 8 fine bits of q
    return ((qM << ::mkfitdev::kPhiBitsM) | phiM) & ~(((1u << 8) - 1) << ::mkfitdev::kPhiBitsM);
  }

  // 1. Per hit (LayerOfHits::registerHit): phi, q, masked binnor key, global N-bin, hit info; histogram per bin.
  struct KernelBinHits {
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  HitSoA::ConstView hits,
                                  LayerSoA::ConstView layers,
                                  uint32_t nHits,
                                  uint32_t* __restrict__ gbinKey,
                                  HitInfoDev* __restrict__ info,
                                  uint32_t* __restrict__ binCnt) const {
      // raw column pointers in the per-hit loop (the SoA element accessors range-check every access)
      const auto hm = hits.metadata();
      const float* __restrict__ hx = hm.addressOf_x();
      const float* __restrict__ hy = hm.addressOf_y();
      const float* __restrict__ hz = hm.addressOf_z();
      const float* __restrict__ he00 = hm.addressOf_e00();
      const float* __restrict__ he11 = hm.addressOf_e11();
      const float* __restrict__ he22 = hm.addressOf_e22();
      const int8_t* __restrict__ hlay = hm.addressOf_layer();
      const auto lm = layers.metadata();
      const uint8_t* __restrict__ lBarrel = lm.addressOf_isBarrel();
      const uint8_t* __restrict__ lPixel = lm.addressOf_isPixel();
      const float* __restrict__ lQMin = lm.addressOf_qRMin();
      const float* __restrict__ lQMLbhp = lm.addressOf_qMLbhp();
      const float* __restrict__ lQMFac = lm.addressOf_qMFac();
      const uint16_t* __restrict__ lQLastM = lm.addressOf_qLastMBin();
      const uint32_t* __restrict__ lBinBegin = lm.addressOf_binBegin();
      const float phiRMin = layers.phiRMin(), phiMFac = layers.phiMFac();
      // Chunked like uniform_elements, with plain inner loops: on CPU backends the first loop (no atomics, branch-free)
      // can be vectorized, the histogram is a separate scalar loop; on GPU backends each chunk is one element.
      const uint32_t elems = alpaka::getWorkDiv<alpaka::Thread, alpaka::Elems>(acc)[0u];
      const uint32_t thread = alpaka::getIdx<alpaka::Grid, alpaka::Threads>(acc)[0u];
      const uint32_t threads = alpaka::getWorkDiv<alpaka::Grid, alpaka::Threads>(acc)[0u];
      for (uint32_t first = thread * elems; first < nHits; first += threads * elems) {
        const uint32_t last = first + elems < nHits ? first + elems : nHits;
        for (uint32_t i = first; i < last; ++i) {
          const int32_t lraw = hlay[i];
          const int32_t l = lraw < 0 ? 0 : lraw;  // unregistered hits: computed on layer 0, marked invalid below
          const uint32_t masked = binHit(hx[i],
                                         hy[i],
                                         hz[i],
                                         lBarrel[l] ? he22[i] : he00[i] + he11[i],
                                         lBarrel[l],
                                         lPixel[l],
                                         lQMin[l],
                                         lQMLbhp[l],
                                         lQMFac[l],
                                         lQLastM[l],
                                         phiRMin,
                                         phiMFac,
                                         info[i]);
          const uint32_t qN = masked >> 24;
          const uint32_t phiN = (masked >> 8) & 0xffu;
          // global N-bin (< 2^24, eventOfHitsUnbuildable) with the key's 8 fine phi bits on top: all the later
          // steps need of the key (inside one bin the masked key differs only in those bits)
          gbinKey[i] =
              lraw < 0 ? kInvalid : (((masked & 0xffu) << 24) | (lBinBegin[l] + qN * ::mkfitdev::kNPhiBins + phiN));
        }
        for (uint32_t i = first; i < last; ++i) {
          const uint32_t g = gbinKey[i];
          if (g != kInvalid)
            alpaka::atomicAdd(acc, &binCnt[g & kGBinMask], 1u, alpaka::hierarchy::Blocks{});
        }
      }
    }
  };

  // 2. Per q row (256 phi bins): inclusive scan of the bin counts, row total.
  struct KernelScanRows {
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  uint32_t nRows,
                                  uint32_t const* __restrict__ binCnt,
                                  uint32_t* __restrict__ binIncl,
                                  uint32_t* __restrict__ rowTot) const {
      auto& ws = alpaka::declareSharedVar<uint32_t[cms::alpakatools::warpSize], __COUNTER__>(acc);
      for (uint32_t r : cms::alpakatools::independent_groups(acc, nRows)) {
        cms::alpakatools::blockPrefixScan(acc,
                                          binCnt + r * ::mkfitdev::kNPhiBins,
                                          binIncl + r * ::mkfitdev::kNPhiBins,
                                          int32_t(::mkfitdev::kNPhiBins),
                                          ws);
        alpaka::syncBlockThreads(acc);
        if (cms::alpakatools::once_per_block(acc))
          rowTot[r] = binIncl[r * ::mkfitdev::kNPhiBins + ::mkfitdev::kNPhiBins - 1];
        alpaka::syncBlockThreads(acc);
      }
    }
  };

  // 3. Per layer: inclusive scan of its row totals (<= 256 rows), layer total, row -> layer map.
  struct KernelScanLayerRows {
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  LayerSoA::ConstView layers,
                                  uint32_t nLayers,
                                  uint32_t const* __restrict__ rowTot,
                                  uint32_t* __restrict__ rowIncl,
                                  uint32_t* __restrict__ rowLayer,
                                  uint32_t* __restrict__ layerTot) const {
      auto& ws = alpaka::declareSharedVar<uint32_t[cms::alpakatools::warpSize], __COUNTER__>(acc);
      for (uint32_t l : cms::alpakatools::independent_groups(acc, nLayers)) {
        const uint32_t r0 = layers[l].binBegin() / ::mkfitdev::kNPhiBins;
        const uint32_t nr = layers[l].nQBins();
        cms::alpakatools::blockPrefixScan(acc, rowTot + r0, rowIncl + r0, int32_t(nr), ws);
        alpaka::syncBlockThreads(acc);
        for (uint32_t k : cms::alpakatools::independent_group_elements(acc, nr)) {
          rowLayer[r0 + k] = l;
          if (k == nr - 1)
            layerTot[l] = rowIncl[r0 + k];
        }
        alpaka::syncBlockThreads(acc);
      }
    }
  };

  // 4. One block: scan of layer totals -> hitBegin, nHits.
  struct KernelScanLayers {
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  LayerSoA::View layers,
                                  uint32_t nLayers,
                                  uint32_t const* __restrict__ layerTot,
                                  uint32_t* __restrict__ layerIncl) const {
      auto& ws = alpaka::declareSharedVar<uint32_t[cms::alpakatools::warpSize], __COUNTER__>(acc);
      for ([[maybe_unused]] uint32_t g : cms::alpakatools::independent_groups(acc, 1u)) {
        cms::alpakatools::blockPrefixScan(acc, layerTot, layerIncl, int32_t(nLayers), ws);
        alpaka::syncBlockThreads(acc);
        for (uint32_t l : cms::alpakatools::independent_group_elements(acc, nLayers)) {
          layers[l].nHits() = layerTot[l];
          layers[l].hitBegin() = layerIncl[l] - layerTot[l];
        }
      }
    }
  };

  // 5. Per bin: C_pair (first = internal index in layer, 0 when empty), global start of the bin's hits.
  struct KernelFinalizeBins {
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  LayerSoA::View layers,
                                  BinSoA::View bins,
                                  uint32_t nBins,
                                  uint32_t const* __restrict__ binCnt,
                                  uint32_t const* __restrict__ binIncl,
                                  uint32_t const* __restrict__ rowTot,
                                  uint32_t const* __restrict__ rowIncl,
                                  uint32_t const* __restrict__ rowLayer,
                                  uint32_t* __restrict__ binStart) const {
      const uint32_t* __restrict__ lHitBegin = layers.metadata().addressOf_hitBegin();
      uint32_t* __restrict__ bContent = bins.metadata().addressOf_content();
      uint8_t* __restrict__ bDead = bins.metadata().addressOf_dead();
      for (uint32_t b : cms::alpakatools::uniform_elements(acc, nBins)) {
        const uint32_t r = b / ::mkfitdev::kNPhiBins;
        const uint32_t l = rowLayer[r];
        const uint32_t c = binCnt[b];
        const uint32_t first = (rowIncl[r] - rowTot[r]) + (binIncl[b] - c);  // internal index in the layer
        binStart[b] = lHitBegin[l] + first;
        bDead[b] = 0;  // replaces a memset of the bin table; KernelDeadBins runs later
        if (c == 0) {
          bContent[b] = 0;
          continue;
        }
        if (first > ::mkfitdev::kBinFirstMask)
          alpaka::atomicAdd(acc, &layers.nOverflowFirst(), 1u, alpaka::hierarchy::Blocks{});
        if (c > ::mkfitdev::kBinCountMask)
          alpaka::atomicAdd(acc, &layers.nOverflowCount(), 1u, alpaka::hierarchy::Blocks{});
        bContent[b] =
            (first & ::mkfitdev::kBinFirstMask) | ((c & ::mkfitdev::kBinCountMask) << ::mkfitdev::kBinFirstBits);
      }
    }
  };

  // 6. Per hit: unordered bin membership; only the (fine phi, row) word is staged at the hit's slot (4 B scattered
  //    per hit); the ranking below reads the hit's original index and info by row (order fixed in step 7).
  struct KernelFillMembers {
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  uint32_t nHits,
                                  uint32_t const* __restrict__ gbinKey,
                                  uint32_t const* __restrict__ binStart,
                                  uint32_t* __restrict__ binFill,
                                  uint32_t* __restrict__ stKeyRow) const {
      for (uint32_t i : cms::alpakatools::uniform_elements(acc, nHits)) {
        const uint32_t g = gbinKey[i];
        if (g == kInvalid)
          continue;
        const uint32_t b = g & kGBinMask;
        const uint32_t p = binStart[b] + alpaka::atomicAdd(acc, &binFill[b], 1u, alpaka::hierarchy::Blocks{});
        // inside one bin the key differs only in its 8 lowest bits (fine phi): (fine phi, row) is unique and its
        // order is the MkFitCore order; rows < 2^24 (eventOfHitsUnbuildable: nHits < 2^23)
        stKeyRow[p] = (g & ~kGBinMask) | i;
      }
    }
  };
  constexpr uint32_t kStRowMask = (1u << 24) - 1;

  // 7. Per bin: deterministic in-bin rank of each member by (key, row) (= the stable LSD radix order of MkFitCore);
  // write
  //    the MkFitCore-ordered rows (LayerOfHits::endRegistrationOfHits: m_hit_infos[i] = hinfos[ranks[i]], ranks ->
  //    original index).
  struct KernelRankAndStore {
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  BinnedHitSoA::View out,
                                  uint32_t nBins,
                                  uint32_t const* __restrict__ binCnt,
                                  uint32_t const* __restrict__ binStart,
                                  uint32_t const* __restrict__ stKeyRow,
                                  HitSoA::ConstView hits,
                                  HitInfoDev const* __restrict__ info) const {
      const uint32_t nPixel = hits.nPixel();
      auto om = out.metadata();
      uint32_t* __restrict__ oRank = om.addressOf_rank();
      float* __restrict__ oPhi = om.addressOf_phi();
      float* __restrict__ oQ = om.addressOf_q();
      float* __restrict__ oHL = om.addressOf_qHalfLength();
      float* __restrict__ oQbar = om.addressOf_qbar();
      for (uint32_t b : cms::alpakatools::uniform_elements(acc, nBins)) {
        const uint32_t c = binCnt[b];
        if (c == 0)
          continue;
        const uint32_t s = binStart[b];
        if (c == 1) {  // most bins: rank 0
          const uint32_t row = stKeyRow[s] & kStRowMask;
          const HitInfoDev hi = info[row];
          oRank[s] = ::mkfitdev::originalIndex(row, nPixel);
          oPhi[s] = hi.phi;
          oQ[s] = hi.q;
          oHL[s] = hi.q_half_length;
          oQbar[s] = hi.qbar;
          continue;
        }
        for (uint32_t m = s; m < s + c; ++m) {
          const uint32_t km = stKeyRow[m];
          uint32_t rank = 0;
          for (uint32_t n = s; n < s + c; ++n)
            rank += stKeyRow[n] < km ? 1u : 0u;
          const uint32_t pos = s + rank;
          const uint32_t row = km & kStRowMask;
          const HitInfoDev hi = info[row];
          oRank[pos] = ::mkfitdev::originalIndex(row, nPixel);
          oPhi[pos] = hi.phi;
          oQ[pos] = hi.q;
          oHL[pos] = hi.q_half_length;
          oQbar[pos] = hi.qbar;
        }
      }
    }
  };

  // 8. Per dead region: LayerOfHits::suckInDeads.
  struct KernelDeadBins {
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  LayerSoA::ConstView layers,
                                  BinSoA::View bins,
                                  uint32_t nDeads,
                                  DeadRegionDev const* __restrict__ deads) const {
      for (uint32_t i : cms::alpakatools::uniform_elements(acc, nDeads)) {
        const DeadRegionDev d = deads[i];
        const int32_t l = d.layer;
        const float qmin = layers[l].qRMin();
        auto qBinChecked = [&](float q) -> uint16_t {
          return q <= qmin ? uint16_t(0)
                           : (q >= layers[l].qNLbhp() ? layers[l].qLastNBin() : toBin((q - qmin) * layers[l].qNFac()));
        };
        auto phiBin = [&](float phi) -> uint16_t { return toBin((phi - layers.phiRMin()) * layers.phiNFac()); };
        const uint16_t q_bin_1 = qBinChecked(d.q1);
        const uint16_t q_bin_2 = qBinChecked(d.q2) + 1;
        const uint16_t phi_bin_1 = phiBin(d.phi1);
        const uint16_t phi_bin_2 = (phiBin(d.phi2) + 1) & ::mkfitdev::kPhiMaskN;
        const uint32_t layerBins = layers[l].nQBins() * ::mkfitdev::kNPhiBins;
        for (uint16_t q_bin = q_bin_1; q_bin != q_bin_2; q_bin++) {
          const uint32_t qoff = q_bin * ::mkfitdev::kNPhiBins;
          for (uint16_t pb = phi_bin_1; pb != phi_bin_2; pb = (pb + 1) & ::mkfitdev::kPhiMaskN) {
            if (qoff + pb < layerBins)  // MkFitCore writes past the layer here (UB); guard
              bins[layers[l].binBegin() + qoff + pb].dead() = 1;
          }
        }
      }
    }
  };

#if defined(ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED) || defined(ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED)
  // (CPU backends only: a GPU build would carry an unused single-thread kernel with a stack array)
  // CPU backends: kernels 1-7 in one sequential kernel (one block, one thread), organised like MkFitCore: the hits are
  // grouped by layer while their infos are computed (one pass), then each layer is finished in cache: bin histogram,
  // bin table written once, and a stable two-digit counting sort (fine phi, then bin) = MkFitCore's LSD radix order
  // (key, original index). The GPU kernels make two extra passes over the hits and three over all bins.
  struct KernelBuildCpu {
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  HitSoA::ConstView hits,
                                  LayerSoA::View layers,
                                  BinSoA::View bins,
                                  BinnedHitSoA::View out,
                                  uint32_t nHits,
                                  uint32_t nLayers,
                                  uint32_t* __restrict__ lNext,  // [nLayers]
                                  uint32_t* __restrict__ gKey,   // [nHits] layer-grouped: fine phi << 24 | bin in layer
                                  uint32_t* __restrict__ gOrig,  // [nHits] layer-grouped original index
                                  HitInfoDev* __restrict__ gInfo,      // [nHits] layer-grouped hit info
                                  uint32_t* __restrict__ order,        // [nHits] per-layer (fine phi, index) order
                                  uint32_t* __restrict__ cnt) const {  // [256 * 256] per-layer bin counts / cursors
      if (!cms::alpakatools::once_per_grid(acc))
        return;
      const auto hm = hits.metadata();
      const float* __restrict__ hx = hm.addressOf_x();
      const float* __restrict__ hy = hm.addressOf_y();
      const float* __restrict__ hz = hm.addressOf_z();
      const float* __restrict__ he00 = hm.addressOf_e00();
      const float* __restrict__ he11 = hm.addressOf_e11();
      const float* __restrict__ he22 = hm.addressOf_e22();
      const int8_t* __restrict__ hlay = hm.addressOf_layer();
      const uint32_t nPixel = hits.nPixel();
      const auto lm = layers.metadata();
      const uint8_t* __restrict__ lBarrel = lm.addressOf_isBarrel();
      const uint8_t* __restrict__ lPixel = lm.addressOf_isPixel();
      const float* __restrict__ lQMin = lm.addressOf_qRMin();
      const float* __restrict__ lQMLbhp = lm.addressOf_qMLbhp();
      const float* __restrict__ lQMFac = lm.addressOf_qMFac();
      const uint16_t* __restrict__ lQLastM = lm.addressOf_qLastMBin();
      const uint32_t* __restrict__ lNQBins = lm.addressOf_nQBins();
      const uint32_t* __restrict__ lBinBegin = lm.addressOf_binBegin();
      uint32_t* __restrict__ lHitBegin = layers.metadata().addressOf_hitBegin();
      uint32_t* __restrict__ lNHits = layers.metadata().addressOf_nHits();
      const float phiRMin = layers.phiRMin(), phiMFac = layers.phiMFac();

      // hits per layer -> hitBegin, nHits (layers concatenated in layer order); lNext = write cursor per layer
      for (uint32_t l = 0; l < nLayers; ++l)
        lNext[l] = 0;
      for (uint32_t i = 0; i < nHits; ++i)
        if (hlay[i] >= 0)
          ++lNext[hlay[i]];
      for (uint32_t l = 0, run = 0; l < nLayers; ++l) {
        lHitBegin[l] = run;
        lNHits[l] = lNext[l];
        run += lNext[l];
        lNext[l] = lHitBegin[l];
      }

      // registerHit, chunked: a branch-free (vectorizable) loop computes, a scalar loop appends to the hit's layer in
      // increasing original index (= MkFitCore registration order)
      constexpr uint32_t kChunk = 16;
      for (uint32_t first = 0; first < nHits; first += kChunk) {
        const uint32_t m = nHits - first < kChunk ? nHits - first : kChunk;
        uint32_t ck[kChunk];
        HitInfoDev ci[kChunk];
        for (uint32_t j = 0; j < m; ++j) {
          const uint32_t i = first + j;
          const int32_t l = hlay[i] < 0 ? 0 : hlay[i];  // unregistered hits: computed on layer 0, dropped below
          const uint32_t masked = binHit(hx[i],
                                         hy[i],
                                         hz[i],
                                         lBarrel[l] ? he22[i] : he00[i] + he11[i],
                                         lBarrel[l],
                                         lPixel[l],
                                         lQMin[l],
                                         lQMLbhp[l],
                                         lQMFac[l],
                                         lQLastM[l],
                                         phiRMin,
                                         phiMFac,
                                         ci[j]);
          ck[j] = ((masked & 0xffu) << 24) | ((masked >> 24) * ::mkfitdev::kNPhiBins + ((masked >> 8) & 0xffu));
        }
        for (uint32_t j = 0; j < m; ++j) {
          const int32_t l = hlay[first + j];
          if (l < 0)
            continue;
          const uint32_t p = lNext[l]++;
          gKey[p] = ck[j];
          gOrig[p] = ::mkfitdev::originalIndex(first + j, nPixel);
          gInfo[p] = ci[j];
        }
      }

      // per layer (LayerOfHits::endRegistrationOfHits + binnor::finalize_registration), everything in cache
      uint32_t* __restrict__ bContent = bins.metadata().addressOf_content();
      uint8_t* __restrict__ bDead = bins.metadata().addressOf_dead();
      auto om = out.metadata();
      uint32_t* __restrict__ oRank = om.addressOf_rank();
      float* __restrict__ oPhi = om.addressOf_phi();
      float* __restrict__ oQ = om.addressOf_q();
      float* __restrict__ oHL = om.addressOf_qHalfLength();
      float* __restrict__ oQbar = om.addressOf_qbar();
      uint32_t nOverFirst = 0, nOverCount = 0;
      for (uint32_t l = 0; l < nLayers; ++l) {
        const uint32_t base = lHitBegin[l], n = lNHits[l];
        const uint32_t nb = lNQBins[l] * ::mkfitdev::kNPhiBins;  // <= 2^16: the q axis has 8 N-bits
        const uint32_t* __restrict__ key = gKey + base;
        uint32_t fine[::mkfitdev::kNPhiBins];
        for (uint32_t b = 0; b < nb; ++b)
          cnt[b] = 0;
        for (uint32_t f = 0; f < ::mkfitdev::kNPhiBins; ++f)
          fine[f] = 0;
        for (uint32_t k = 0; k < n; ++k) {
          ++cnt[key[k] & kGBinMask];
          ++fine[key[k] >> 24];
        }
        // bin table (C_pair: first = internal index in the layer, 0 when empty; m_dead_bins cleared, KernelDeadBins
        // runs later); the counts become the bins' write cursors
        uint32_t* __restrict__ content = bContent + lBinBegin[l];
        uint8_t* __restrict__ dead = bDead + lBinBegin[l];
        for (uint32_t b = 0, run = 0; b < nb; ++b) {
          const uint32_t c = cnt[b];
          nOverFirst += (c != 0 && run > ::mkfitdev::kBinFirstMask) ? 1u : 0u;
          nOverCount += c > ::mkfitdev::kBinCountMask ? 1u : 0u;
          content[b] = c == 0 ? 0u
                              : ((run & ::mkfitdev::kBinFirstMask) |
                                 ((c & ::mkfitdev::kBinCountMask) << ::mkfitdev::kBinFirstBits));
          dead[b] = 0;
          cnt[b] = run;
          run += c;
        }
        // digit 1: stable counting sort by fine phi; digit 2: stable scatter by bin -> (bin, fine phi, index) order
        for (uint32_t f = 0, run = 0; f < ::mkfitdev::kNPhiBins; ++f) {
          const uint32_t c = fine[f];
          fine[f] = run;
          run += c;
        }
        for (uint32_t k = 0; k < n; ++k)
          order[fine[key[k] >> 24]++] = k;
        for (uint32_t t = 0; t < n; ++t) {
          const uint32_t k = order[t];
          const uint32_t o = base + cnt[key[k] & kGBinMask]++;
          const HitInfoDev hi = gInfo[base + k];
          oRank[o] = gOrig[base + k];
          oPhi[o] = hi.phi;
          oQ[o] = hi.q;
          oHL[o] = hi.q_half_length;
          oQbar[o] = hi.qbar;
        }
      }
      layers.nOverflowFirst() += nOverFirst;
      layers.nOverflowCount() += nOverCount;
    }
  };

#endif

#if defined(ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED)
  // per-thread scratch of KernelBuildCpu on the serial backend (grows to the largest event, never shrinks)
  struct CpuScratch {
    std::vector<uint32_t> lNext, gKey, gOrig, order, cnt;
    std::vector<HitInfoDev> gInfo;
    void reserve(uint32_t nHits, uint32_t nLayers) {
      const size_t n = nHits == 0 ? 1u : nHits;
      if (lNext.size() < nLayers)
        lNext.resize(nLayers);
      if (gKey.size() < n) {
        gKey.resize(n);
        gOrig.resize(n);
        order.resize(n);
        gInfo.resize(n);
      }
      cnt.resize(::mkfitdev::kNPhiBins * ::mkfitdev::kNPhiBins);
    }
  };
  inline CpuScratch& cpuScratch() {
    thread_local CpuScratch scratch;
    return scratch;
  }
#endif

  inline uint32_t blocksFor(uint32_t n) { return n == 0 ? 1u : cms::alpakatools::divide_up_by(n, kBlock); }

  // Allocate a device EventOfHits for nHits input hits and the given static layer table size.
  inline EventOfHitsDevice makeEventOfHitsDevice(Queue& queue, uint32_t nHits, uint32_t nLayers, uint32_t nBins) {
    return EventOfHitsDevice{HitsDeviceCollection(queue, nHits),
                             LayersDeviceCollection(queue, nLayers),
                             BinnedHitsDeviceCollection(queue, nHits),
                             BinsDeviceCollection(queue, nBins)};
  }

  // Build the binning of d.hits into d.layers (hitBegin, nHits, overflow counters), d.binnedHits and d.bins.
  // Inputs on the device: d.hits (HitSoA, layer = -1 for unregistered hits) and the static part of d.layers
  // (EventOfHits LayerAxes.h fillLayers). deads: dead regions (empty = none; MkFitCore calls suckInDeads only when a
  // quality DB is used, and the HLT LST step has only the pixel one). Everything is enqueued on `queue`.
  // Size limits of the device build (checked by the caller, never thrown at event time):
  //   nHits < 2^23: mkfit::HitOnTrack::index is a signed 24-bit field (MkFitCore would wrap silently beyond it);
  //   nLayers <= 1024 and whole 256-bin rows: single-block prefix scans (blockPrefixScan handles <= 1024 elements).
  // Returns nullptr if the event can be built, otherwise the reason (the caller skips mkFit for the event).
  inline const char* eventOfHitsUnbuildable(uint32_t nHits, uint32_t nLayers, uint32_t nBins) {
    if (nHits >= (1u << 23))
      return "more than 2^23 hits in one event (24-bit HitOnTrack index)";
    if (nLayers > 1024 || (nBins / ::mkfitdev::kNPhiBins) * ::mkfitdev::kNPhiBins != nBins || nBins >= kGBinMask)
      return "unexpected layer table (> 1024 layers, partial rows or >= 2^24 - 1 bins)";
    return nullptr;
  }

  // 8. dead regions (copied per event; empty = nothing enqueued)
  inline void enqueueDeadBins(Queue& queue, EventOfHitsViews const& d, std::vector<DeadRegionDev> const& deads) {
    if (deads.empty())
      return;
    const uint32_t nD = deads.size();
    auto dh = cms::alpakatools::make_host_buffer<DeadRegionDev[]>(queue, nD);
    for (uint32_t i = 0; i < nD; ++i)
      dh[i] = deads[i];
    auto dd = cms::alpakatools::make_device_buffer<DeadRegionDev[]>(queue, nD);
    alpaka::memcpy(queue, dd, dh);
    alpaka::exec<Acc1D>(queue,
                        cms::alpakatools::make_workdiv<Acc1D>(blocksFor(nD), kBlock),
                        KernelDeadBins{},
                        d.layers,
                        d.bins,
                        nD,
                        dd.data());
  }

  // Returns false (and enqueues nothing) if eventOfHitsUnbuildable() refuses the sizes.
  // Works on views, so it builds both the standalone EventOfHitsDevice and the blocks of the EventOfHits product.
  inline bool buildEventOfHits(Queue& queue, EventOfHitsViews d, std::vector<DeadRegionDev> const& deads) {
    const uint32_t nHits = d.hits.metadata().size();
    const uint32_t nLayers = d.layers.metadata().size();
    const uint32_t nBins = d.bins.metadata().size();
    const uint32_t nRows = nBins / ::mkfitdev::kNPhiBins;
    if (eventOfHitsUnbuildable(nHits, nLayers, nBins) != nullptr)
      return false;

    using cms::alpakatools::make_device_buffer;
    auto mk = [&](uint32_t n) { return make_device_buffer<uint32_t[]>(queue, n == 0 ? 1u : n); };
    using cms::alpakatools::make_workdiv;
#if defined(ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED) || defined(ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED)
    {
#if defined(ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED)
      // Serial backend: the kernel runs synchronously on this thread, so a per-thread scratch is reused across events.
      // Per-event buffers would be fresh memory every event: CMSSW does not cache buffers of blocking CPU queues
      // (CachedBufAlloc<DevCpu, QueueCpuBlocking> = allocBuf) and jemalloc purges allocations > 8 MiB eagerly.
      CpuScratch& s = cpuScratch();
      s.reserve(nHits, nLayers);
      uint32_t *lNext = s.lNext.data(), *gKey = s.gKey.data(), *gOrig = s.gOrig.data(), *order = s.order.data(),
               *cnt = s.cnt.data();
      HitInfoDev* gInfo = s.gInfo.data();
#else
      auto bLNext = mk(nLayers), bGKey = mk(nHits), bGOrig = mk(nHits), bOrder = mk(nHits);
      auto bCnt = mk(::mkfitdev::kNPhiBins * ::mkfitdev::kNPhiBins);
      auto bGInfo = make_device_buffer<HitInfoDev[]>(queue, nHits == 0 ? 1u : nHits);
      uint32_t *lNext = bLNext.data(), *gKey = bGKey.data(), *gOrig = bGOrig.data(), *order = bOrder.data(),
               *cnt = bCnt.data();
      HitInfoDev* gInfo = bGInfo.data();
#endif
      alpaka::exec<Acc1D>(queue,
                          make_workdiv<Acc1D>(1u, 1u),
                          KernelBuildCpu{},
                          d.hits,
                          d.layers,
                          d.bins,
                          d.binnedHits,
                          nHits,
                          nLayers,
                          lNext,
                          gKey,
                          gOrig,
                          gInfo,
                          order,
                          cnt);
      enqueueDeadBins(queue, d, deads);
      return true;
    }
#endif
    auto gbinKey = mk(nHits), stKeyRow = mk(nHits);
    auto info = make_device_buffer<HitInfoDev[]>(queue, nHits == 0 ? 1u : nHits);
    auto binCnt = mk(nBins), binIncl = mk(nBins), binStart = mk(nBins), binFill = mk(nBins);
    auto rowTot = mk(nRows), rowIncl = mk(nRows), rowLayer = mk(nRows), layerTot = mk(nLayers), layerIncl = mk(nLayers);
    alpaka::memset(queue, binCnt, 0);
    alpaka::memset(queue, binFill, 0);
    // (no memset of the bin table: KernelFinalizeBins writes content and clears dead for every bin)

    auto lv = d.layers;
    alpaka::exec<Acc1D>(queue,
                        make_workdiv<Acc1D>(blocksFor(nHits), kBlock),
                        KernelBinHits{},
                        d.hits,
                        d.layers,
                        nHits,
                        gbinKey.data(),
                        info.data(),
                        binCnt.data());
    alpaka::exec<Acc1D>(queue,
                        make_workdiv<Acc1D>(nRows, kBlock),
                        KernelScanRows{},
                        nRows,
                        binCnt.data(),
                        binIncl.data(),
                        rowTot.data());
    alpaka::exec<Acc1D>(queue,
                        make_workdiv<Acc1D>(nLayers, kBlock),
                        KernelScanLayerRows{},
                        d.layers,
                        nLayers,
                        rowTot.data(),
                        rowIncl.data(),
                        rowLayer.data(),
                        layerTot.data());
    alpaka::exec<Acc1D>(
        queue, make_workdiv<Acc1D>(1u, kBlock), KernelScanLayers{}, lv, nLayers, layerTot.data(), layerIncl.data());
    alpaka::exec<Acc1D>(queue,
                        make_workdiv<Acc1D>(blocksFor(nBins), kBlock),
                        KernelFinalizeBins{},
                        lv,
                        d.bins,
                        nBins,
                        binCnt.data(),
                        binIncl.data(),
                        rowTot.data(),
                        rowIncl.data(),
                        rowLayer.data(),
                        binStart.data());
    alpaka::exec<Acc1D>(queue,
                        make_workdiv<Acc1D>(blocksFor(nHits), kBlock),
                        KernelFillMembers{},
                        nHits,
                        gbinKey.data(),
                        binStart.data(),
                        binFill.data(),
                        stKeyRow.data());
    alpaka::exec<Acc1D>(queue,
                        make_workdiv<Acc1D>(blocksFor(nBins), kBlock),
                        KernelRankAndStore{},
                        d.binnedHits,
                        nBins,
                        binCnt.data(),
                        binStart.data(),
                        stKeyRow.data(),
                        d.hits,
                        info.data());
    enqueueDeadBins(queue, d, deads);
    // scratch buffers are released at scope exit; the caching allocator keeps them stream-ordered
    return true;
  }

  inline bool buildEventOfHits(Queue& queue, EventOfHitsDevice& d, std::vector<DeadRegionDev> const& deads) {
    return buildEventOfHits(
        queue, EventOfHitsViews{d.hits.view(), d.layers.view(), d.binnedHits.view(), d.bins.view()}, deads);
  }

  // Host convenience: copy host hits + layer table to the device and build.
  inline EventOfHitsDevice buildEventOfHits(Queue& queue,
                                            ::mkfitdev::HitsHostCollection const& hitsHost,
                                            ::mkfitdev::LayersHostCollection const& layersHost,
                                            std::vector<DeadRegionDev> const& deads,
                                            bool* built = nullptr) {
    auto d = makeEventOfHitsDevice(
        queue, hitsHost.view().metadata().size(), layersHost.view().metadata().size(), layersHost.view().nBinsTotal());
    alpaka::memcpy(queue, d.hits.buffer(), hitsHost.buffer());
    alpaka::memcpy(queue, d.layers.buffer(), layersHost.buffer());
    const bool ok = buildEventOfHits(queue, d, deads);
    if (built)
      *built = ok;
    return d;
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev::hits

#endif
