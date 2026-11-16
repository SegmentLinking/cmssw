#ifndef RecoTracker_MkFitAlpaka_src_alpaka_v2p2_V2p2Kernels_h
#define RecoTracker_MkFitAlpaka_src_alpaka_v2p2_V2p2Kernels_h

// Kernels of the v2p2 search: MkBuilder::findTracksStandardv2p2 (MkBuilder.cc) with the per-entry body of
// MkFinderV2p2 (begin_layer, process_layer, end_layer, MkFinderV2p2.cc), every seed of the event at one plan entry
// (regions in lock step). Instantiated only in src/alpaka/V2p2.dev.cc; callers use interface/v2p2/alpaka/V2p2Search.h.
//   V0 KernelV2p2PullIn      thread per seed: awaken_candidates, begin_layer, begin_next_Ccrep_in_layer
//   V2 KernelV2p2SelectHits  block per LayerBatch: process_layer_batch phases 1 to 4 (prop_to_layer_edges,
//                            determine_search_windows, select_hits / preselect_hit_batch, order_hits_along_path)
//   V3 KernelV2p2InLayer     the in-layer search of the entry: expand_in_layer, rk_expand_node, the Kalman batch
//                            (rk_flush) and harvest_nodes, to max_hits_per_layer
//   V4 KernelV2p2Select      lanes per seed: select_and_materialise (the lanes share the competitors and the
//                            survivors) and end_layer, then V0 of the next entry (end_layer's awaken_candidates
//                            and the next begin_layer) for the same seed
//   V5 KernelV2p2FinalPick   thread per seed: the final sorting of findTracksStandardv2p2
// The completion machinery of MkFitCore (queues, running Kalman batches, early selection, the node free list) only
// fills SIMD lanes: V3 runs the whole in-layer search of the entry instead, and V4 selects after it.

#include <cstdint>

#include <alpaka/alpaka.hpp>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/warpsize.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include "RecoTracker/MkFitAlpaka/interface/cands/CandEngineTypes.h"
#include "RecoTracker/MkFitAlpaka/interface/cands/CandSeedOps.h"
#include "RecoTracker/MkFitAlpaka/interface/cands/CandsSoA.h"
#include "RecoTracker/MkFitAlpaka/interface/es/ESConfig.h"
#include "RecoTracker/MkFitAlpaka/interface/hits/HitSoA.h"
#include "RecoTracker/MkFitAlpaka/interface/hits/LayerOfHitsAccess.h"
#include "RecoTracker/MkFitAlpaka/interface/matriplex/MatriplexBackend.h"
#include "RecoTracker/MkFitAlpaka/interface/v2p2/V2p2Config.h"
#include "RecoTracker/MkFitAlpaka/interface/v2p2/V2p2SoA.h"
#include "RecoTracker/MkFitAlpaka/interface/v2p2/alpaka/V2p2Search.h"
#include "RecoTracker/MkFitAlpaka/src/alpaka/Packers.h"
#include "RecoTracker/MkFitAlpaka/src/alpaka/cands/CandSeedOpsKernels.h"
#include "RecoTracker/MkFitAlpaka/src/alpaka/prop/PropagationMPlex.h"
#include "RecoTracker/MkFitAlpaka/src/alpaka/prop/PropagationFlags.h"
#include "RecoTracker/MkFitAlpaka/src/alpaka/v2p2/MkFinderV2p2.h"
#include "RecoTracker/MkFitAlpaka/src/alpaka/v2p2/MkFinderV2p2Structures.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev {

  using ::mkfitdev::kV2p2MaxPreselHits;
  using ::mkfitdev::V2p2CandSoA;
  using ::mkfitdev::V2p2Counters;
  using ::mkfitdev::V2p2LayerHitSoA;
  using ::mkfitdev::V2p2NodeSoA;
  using ::mkfitdev::V2p2Params;
  using ::mkfitdev::V2p2SeedSoA;
  using ::mkfitdev::prop::PropagationFlags;

  // GPU backends run a block of threads per work unit, CPU backends one thread with kNN-wide batches
  constexpr bool kV2p2BlockThreads = !cms::alpakatools::requires_single_thread_per_block_v<Acc1D>;
  static_assert(kV2p2BlockThreads == (kNN == 1), "the work divisions below pair one-wide batches with thread blocks");
  // Work units of one single-thread block on CPU backends (CandIncomings of V2 and V3, seeds of the others)
  constexpr uint32_t kV2p2CpuBlockUnits = 128;
  // Lanes per CandIncoming in V2 (at most one warp: the pqueue offers use a warp ballot): a group walks the bins on GPU
  // backends, one lane with kNN-hit batches on CPU ones.
  constexpr uint32_t kV2p2SelectLanes = kV2p2BlockThreads ? 32 : 1;
  // Hits of one walk chunk of V2 (consecutive walk positions, offered to the pqueue in this order)
  constexpr uint32_t kV2p2SelectChunk = kV2p2SelectLanes * kNN;
  // CandIncoming rows per V2 block: on CPU backends its active rows form the kNN-wide LayerBatches
  constexpr uint32_t kV2p2SelectRows = kV2p2BlockThreads ? 1 : kV2p2CpuBlockUnits;
  // V3 threads per block (a Kalman batch each) and CandIncoming rows per block: small GPU blocks spread the
  // latency-bound batches over more SMs, a CPU block takes enough rows to fill its batches
  constexpr uint32_t kV2p2InLayerThreads = kV2p2BlockThreads ? 64 : 1;
  constexpr uint32_t kV2p2InLayerRows = kV2p2BlockThreads ? kV2p2InLayerThreads : kV2p2CpuBlockUnits;
  // V3 rows from V0's list of the entry's CandIncomings on CPU backends; all rows on GPU ones, for more blocks
  constexpr bool kV2p2InLayerFromList = !kV2p2BlockThreads;
  // V3 items of one worklist round: the root items of the block's rows on GPU backends; room for twice that on CPU
  // backends, so that the queue below keeps nearly every node of a round and the Kalman batches stay full
  constexpr uint32_t kV2p2InLayerItems =
      (kV2p2BlockThreads ? 1 : 2) * kV2p2InLayerRows * ::mkfitdev::kV2p2MaxPreselHits;
  // V3 nodes queued for the next round: a node expands into fewer than kV2p2MaxPreselHits items
  constexpr uint32_t kV2p2InLayerQueue = kV2p2InLayerItems / ::mkfitdev::kV2p2MaxPreselHits;
  // Thread-per-seed kernels: small blocks spread the latency-bound seed threads over more SMs (GPU backends only).
  constexpr uint32_t kV2p2SeedBlock = kV2p2BlockThreads ? 32 : kV2p2CpuBlockUnits;
  // V4: lanes per seed (a group shares the seed's selection on GPU backends: a long node list is the tail) and seeds
  // per block
  constexpr uint32_t kV2p2SelectSeedLanes = kV2p2BlockThreads ? 8 : 1;
  constexpr uint32_t kV2p2SelectSeedsPerBlock = kV2p2BlockThreads ? 4 : 1;
  // V4: consecutive tiles of kV2p2SelectSeedsPerBlock seeds per block (CPU backends: as many seeds as kV2p2SeedBlock)
  constexpr uint32_t kV2p2SelectTilesPerBlock = kV2p2BlockThreads ? 1 : kV2p2SeedBlock;
  static_assert((kV2p2SelectSeedLanes & (kV2p2SelectSeedLanes - 1)) == 0, "V4 merges the lanes' tops pairwise");

  // Per region: the final-pick scorer (SteeringRegion::final_pick_track_scorer forward, track_scorer backward) and
  // whether the region has a plan in this direction (findTracksStandardv2p2 skips a region without a backward plan)
  struct V2p2RegionScorers {
    ::mkfitdev::TrackScorer scorer[::mkfitdev::es::kMaxRegions];
    int8_t hasPlan[::mkfitdev::es::kMaxRegions];
  };

  // ---------------------------------------------------------------------------------------------------------
  // V0: awaken_candidates() (setup or the previous entry's end_layer()), begin_layer(), begin_next_Ccrep_in_layer()

  class KernelV2p2PullIn {
  public:
    // the entry's node pool starts empty and the next step's list is cleared (one thread of the grid; this step's
    // V2 and V3 have read the list of that parity)
    ALPAKA_FN_ACC static void resetEntryCounters(Acc1D const& acc, V2p2SeedSoA::View v2seeds, int step) {
      if (cms::alpakatools::once_per_grid(acc)) {
        V2p2Counters& cnt = v2seeds.counters();
        cnt.nPoolRows = 0;
        cnt.nActiveCands[(step + 1) & 1] = 0;
      }
    }

    // one seed
    ALPAKA_FN_ACC static void pullIn(Acc1D const& acc,
                                     SeedCandsSoA::View seeds,
                                     CandSlotsSoA::View slots,
                                     CandHotsSoA::View hots,
                                     V2p2SeedSoA::View v2seeds,
                                     V2p2CandSoA::View v2cands,
                                     const int16_t* stepLayer,
                                     const int16_t* stepPrevLayer,
                                     ::mkfitdev::LayerInfoSoA::ConstView geo,
                                     const V2p2LayerCtx& ctxBase,
                                     int step,
                                     int s,
                                     HotPool const& pool) {
      v2seeds.inLayer(s) = 0;
      v2seeds.nNodes(s) = 0;
      v2seeds.nodeHead(s) = -1;
      v2seeds.maxCands(s) = 0;
      for (int ic = 0; ic < kMaxCandsPerSeed; ++ic)
        v2cands.active(::mkfitdev::v2p2CandRow(s, ic)) = 0;
      const uint32_t ovf0 = seeds.overflowBits(s);
      if (ovf0 & kOverflowHotsBit)
        return;  // failed seed (HoT pool overflow), already Finished
      const int reg = seeds.region(s);
      const int layer = stepLayer[reg];
      const int prevLayer = stepPrevLayer[reg];
      SeedCandsRef ref = makeSeedRef(seeds, slots, hots, s);
      if (prevLayer >= 0)
        awaken_candidates(ref, prevLayer);
      if (layer < 0 || *ref.state != kFinding)
        return;
      reserveHots(acc, pool, s, *ref.nCands);  // a stop node per TrackCand at most
      attachHots(pool, s, ref);
      seeds.layer(s) = layer;
      V2p2LayerCtx ctx = ctxBase;
      begin_layer(ctx, geo, layer);
      int8_t active[kMaxCandsPerSeed];
      for (int ic = 0; ic < kMaxCandsPerSeed; ++ic)
        active[ic] = 0;
      int maxCands = 0;
      begin_next_Ccrep_in_layer(ref, ctx, active, maxCands);
      failSeedOnHotOverflow(acc, seeds, s, ovf0);
      if (seeds.overflowBits(s) & kOverflowHotsBit)
        return;  // the stop HoT did not fit: the seed is out of the search
      int nActive = 0;
      for (int ic = 0; ic < kMaxCandsPerSeed; ++ic) {
        v2cands.active(::mkfitdev::v2p2CandRow(s, ic)) = active[ic];
        nActive += active[ic];
      }
      // the CandIncomings of the entry, for V2
      int k = alpaka::atomicAdd(acc, &v2seeds.counters().nActiveCands[step & 1], nActive, alpaka::hierarchy::Blocks{});
      for (int ic = 0; ic < kMaxCandsPerSeed; ++ic)
        if (active[ic])
          v2cands.activeRow(k++) = ::mkfitdev::v2p2CandRow(s, ic);
      v2seeds.maxCands(s) = int8_t(maxCands);
      v2seeds.inLayer(s) = 1;
    }

    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  SeedCandsSoA::View seeds,
                                  CandSlotsSoA::View slots,
                                  CandHotsSoA::View hots,
                                  V2p2SeedSoA::View v2seeds,
                                  V2p2CandSoA::View v2cands,
                                  const int16_t* stepLayer,      // per region: the entry searched (-1: none)
                                  const int16_t* stepPrevLayer,  // per region: the entry whose seeds awaken
                                  ::mkfitdev::LayerInfoSoA::ConstView geo,
                                  V2p2LayerCtx ctxBase,
                                  int step,
                                  int nSeeds,
                                  const int32_t* nDev) const {
      nSeeds = engineRows(nSeeds, nDev);
      resetEntryCounters(acc, v2seeds, step);
      const HotPool pool = hotPoolOf(seeds, hots);
      for (int32_t s : cms::alpakatools::uniform_elements(acc, nSeeds))
        pullIn(acc, seeds, slots, hots, v2seeds, v2cands, stepLayer, stepPrevLayer, geo, ctxBase, step, s, pool);
    }
  };

  // ---------------------------------------------------------------------------------------------------------
  // V2: process_layer_batch() phases 1 to 4 for LayerBatches of V0's list; the lanes walk the bins in chunks of
  // consecutive walk positions and lane 0 offers the passed hits in walk order, so the pqueues keep MkFitCore's hits.

  class KernelV2p2SelectHits {
    // preselect_hit_batch() of the lane's HitBatch; its passed hits go to the lane's stage entries
    ALPAKA_FN_ACC static void preselectBatch(const V2p2CandWindow<kNN>& w,
                                             const ::mkfitdev::LayerOfHitsAccess& L,
                                             const EngineHitInputs& hits,
                                             int (&candIdcs)[kNN],
                                             uint32_t (&hitIdcs)[kNN],
                                             uint32_t (&hitOrigIdcs)[kNN],
                                             int& nBatch,
                                             int8_t* laneStageCand,
                                             V2p2PassedHitMatch* laneStage,
                                             bool& staged) {
      bool pass[kNN];
      V2p2PassedHitMatch phm[kNN];
      for (int k = nBatch; k < int(kNN); ++k) {
        candIdcs[k] = candIdcs[0];  // valid inputs in the unused Matriplex slots
        hitIdcs[k] = hitIdcs[0];
        hitOrigIdcs[k] = hitOrigIdcs[0];
      }
      preselect_hit_batch<kNN>(w, L, hits, candIdcs, hitIdcs, hitOrigIdcs, nBatch, pass, phm);
      for (int k = 0; k < nBatch; ++k) {
        if (pass[k]) {
          laneStageCand[k] = int8_t(candIdcs[k]);
          laneStage[k] = phm[k];
          staged = true;
        }
      }
      nBatch = 0;
    }

    // lane 0 offers the staged hits of the lanes that staged one, in lane order, which is walk order
    ALPAKA_FN_ACC static void offerStaged(Acc1D const& acc,
                                          uint32_t tid,
                                          const int8_t* stageCand,
                                          const V2p2PassedHitMatch* stage,
                                          V2p2PassedHitMatch (&pq)[kNN][kV2p2MaxPreselHits],
                                          int (&pqSize)[kNN],
                                          int preselStride,
                                          bool& staged) {
      static_assert(cms::alpakatools::warpSize == 1 || kV2p2SelectLanes <= uint32_t(cms::alpakatools::warpSize),
                    "the lanes of V2 must be one warp for the ballot");
      if (alpaka::syncBlockThreadsPredicate<alpaka::BlockOr>(acc, staged ? 1 : 0)) {
        auto lanes = alpaka::warp::ballot(acc, staged ? 1 : 0);
        if (tid == 0) {
          for (uint32_t t = 0; lanes != 0; ++t, lanes >>= 1) {
            if (!(lanes & 1))
              continue;
            for (uint32_t k = 0; k < uint32_t(kNN); ++k) {
              const int i = stageCand[t * kNN + k];
              if (i >= 0)
                pqSize[i] = pq_offer(pq[i], pqSize[i], preselStride, stage[t * kNN + k]);
            }
          }
        }
        alpaka::syncBlockThreads(acc);
      }
      staged = false;
    }

  public:
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  SeedCandsSoA::ConstView seeds,
                                  CandSlotsSoA::ConstView slots,
                                  V2p2SeedSoA::ConstView v2seeds,
                                  V2p2CandSoA::View v2cands,
                                  V2p2LayerHitSoA::View layerHits,
                                  const int16_t* stepLayer,
                                  V2p2EventInputs in,
                                  V2p2LayerCtx ctxBase,
                                  int preselStride,
                                  int step) const {
      const int nItems = v2seeds.counters().nActiveCands[step & 1];
      auto& w = alpaka::declareSharedVar<V2p2CandWindow<kNN>, __COUNTER__>(acc);
      auto& pq = alpaka::declareSharedVar<V2p2PassedHitMatch[kNN][kV2p2MaxPreselHits], __COUNTER__>(acc);
      auto& stage = alpaka::declareSharedVar<V2p2PassedHitMatch[kV2p2SelectChunk], __COUNTER__>(acc);
      auto& stageCand = alpaka::declareSharedVar<int8_t[kV2p2SelectChunk], __COUNTER__>(acc);  // -1: none
      auto& nScanLane = alpaka::declareSharedVar<int32_t[kV2p2SelectLanes][kNN], __COUNTER__>(acc);
      auto& walk = alpaka::declareSharedVar<int8_t[kNN], __COUNTER__>(acc);
      // the hit segments of up to kV2p2SelectLanes q rows and their offsets in the walk
      auto& rowSeg = alpaka::declareSharedVar<V2p2HitSegment[2 * kV2p2SelectLanes], __COUNTER__>(acc);
      auto& segOff = alpaka::declareSharedVar<int32_t[2 * kV2p2SelectLanes + 1], __COUNTER__>(acc);
      const uint32_t tid = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u];
      const int nGroups = cms::alpakatools::divide_up_by(nItems, int(kV2p2SelectRows));

      for (int32_t group : cms::alpakatools::independent_groups(acc, nGroups)) {
        const int itemEnd = std::min(nItems, (group + 1) * int(kV2p2SelectRows));
        int item = group * int(kV2p2SelectRows);
        while (true) {
          // The next LayerBatch: CandIncomings of one layer in list order; uniform over the block
          int rows[kNN];
          int n = 0;
          int layer = -1;
          for (; item < itemEnd && n < int(kNN); ++item) {
            const int row = v2cands.activeRow(item);
            const int rowLayer = stepLayer[seeds.region(::mkfitdev::v2p2CandSeed(row))];
            if (n > 0 && rowLayer != layer)
              break;
            layer = rowLayer;
            rows[n++] = row;
          }
          if (n == 0)
            break;
          const ::mkfitdev::LayerOfHitsAccess L{in.eohLayers, in.binned, in.bins, layer};

          // Phases 1 and 2: prop_to_layer_edges(), determine_search_windows()
          if (tid == 0) {
            V2p2LayerCtx ctx = ctxBase;
            begin_layer(ctx, in.geo, layer);
            // phases 1 and 2 read the TrackCand states only: no HoT chain
            TrackCandView tc[kNN];
            for (int i = 0; i < int(kNN) && i < n; ++i) {
              const int s = ::mkfitdev::v2p2CandSeed(rows[i]);
              const int slot = candSlotRow(s, seeds.curBuf(s), ::mkfitdev::v2p2CandIc(rows[i]));
              tc[i] = TrackCandView{&slots.book(slot), &slots.state(slot), nullptr};
            }
            prop_to_layer_edges(w, tc, n, ctx);
            determine_search_windows(w, tc, L, ctx);
            for (int i = 0; i < int(kNN) && i < n; ++i) {
              v2cands.wsr(rows[i]) = int8_t(w.wsr[i].wsr);
              v2cands.inGap(rows[i]) = int8_t(w.wsr[i].in_gap);
              // The track does not reach this layer: no walk; LayerBatch's rho_dphi / rho_dq stay zero.
              if (Config::V2p2::Policy::use_wsr && w.wsr[i].wsr == ::mkfitdev::WSR_Outside) {
                w.rho_dphi[i] = 0.f;
                w.rho_dq[i] = 0.f;
                walk[i] = 0;
              } else {
                select_hits_setup(w, i, L);
                walk[i] = 1;
              }
            }
          }
          alpaka::syncBlockThreads(acc);

          // Phase 3: select_hits() bin walk and preselect_hit_batch(); a lane's HitBatch fills in walk order
          int nScanned[kNN];
          int pqSize[kNN];  // lane 0 only
          for (int i = 0; i < int(kNN); ++i)
            nScanned[i] = pqSize[i] = 0;
          int candIdcs[kNN];
          uint32_t hitIdcs[kNN];
          uint32_t hitOrigIdcs[kNN];
          int nBatch = 0;
          bool staged = false;  // this lane staged a passed hit in the current chunk
          for (int i = 0; i < int(kNN) && i < n; ++i) {
            if (!walk[i])
              continue;
            const int q1 = w.BL_p.q1[i], q2 = w.BL_p.q2[i];
            // q rows in groups of kV2p2SelectLanes, a lane per row; the group's segments are walked as one sequence
            for (int qg = q1; qg < q2; qg += int(kV2p2SelectLanes)) {
              const int nq = std::min(int(kV2p2SelectLanes), q2 - qg);
              if (int(tid) < nq) {
                V2p2HitSegment seg[2];
                const int nSeg = select_hits_row_segments(w, i, L, uint16_t(qg + int(tid)), seg);
                for (int g = 0; g < 2; ++g)
                  rowSeg[2 * tid + g] = g < nSeg ? seg[g] : V2p2HitSegment{0, 0};
              }
              alpaka::syncBlockThreads(acc);
              if (tid == 0) {
                segOff[0] = 0;
                for (int g = 0; g < 2 * nq; ++g)
                  segOff[g + 1] = segOff[g] + int(rowSeg[g].end - rowSeg[g].begin);
              }
              alpaka::syncBlockThreads(acc);
              const int nWalk = segOff[2 * nq];
              int g = 0;  // the segment of this lane's next walk position
              for (int chunk = 0; chunk < nWalk; chunk += int(kV2p2SelectChunk)) {
                for (int k = 0; k < int(kNN); ++k)
                  stageCand[tid * kNN + k] = -1;
                // the batch fills at most once per chunk: a chunk gives a lane kNN walk positions
                for (int j = 0; j < int(kNN); ++j) {
                  const int p = chunk + int(tid * kNN) + j;
                  if (p >= nWalk)
                    break;
                  while (p >= segOff[g + 1])
                    ++g;
                  const uint32_t hi = rowSeg[g].begin + uint32_t(p - segOff[g]);
                  const uint32_t hiOrig = L.getOriginalHitIndex(hi);
                  if (select_hits_scan_hit(w, i, L, hi, nScanned[i])) {
                    candIdcs[nBatch] = i;
                    hitIdcs[nBatch] = hi;
                    hitOrigIdcs[nBatch++] = hiOrig;
                    if (nBatch == int(kNN))
                      preselectBatch(w,
                                     L,
                                     in.hits,
                                     candIdcs,
                                     hitIdcs,
                                     hitOrigIdcs,
                                     nBatch,
                                     &stageCand[tid * kNN],
                                     &stage[tid * kNN],
                                     staged);
                  }
                }
                offerStaged(acc, tid, stageCand, stage, pq, pqSize, preselStride, staged);
              }
              // rowSeg and segOff are rewritten by the next group
              alpaka::syncBlockThreads(acc);
            }
          }
          if constexpr (kNN > 1) {
            // the partial last batch (one lane on CPU backends)
            for (int k = 0; k < int(kNN); ++k)
              stageCand[tid * kNN + k] = -1;
            if (nBatch > 0)
              preselectBatch(w,
                             L,
                             in.hits,
                             candIdcs,
                             hitIdcs,
                             hitOrigIdcs,
                             nBatch,
                             &stageCand[tid * kNN],
                             &stage[tid * kNN],
                             staged);
            offerStaged(acc, tid, stageCand, stage, pq, pqSize, preselStride, staged);
          }
          for (int i = 0; i < int(kNN) && i < n; ++i)
            nScanLane[tid][i] = nScanned[i];
          alpaka::syncBlockThreads(acc);

          // The density estimate and phase 4: order_hits_along_path()
          if (tid == 0) {
            for (int i = 0; i < int(kNN) && i < n; ++i) {
              int nScannedAll = 0;
              for (uint32_t t = 0; t < kV2p2SelectLanes; ++t)
                nScannedAll += nScanLane[t][i];
              const int s = ::mkfitdev::v2p2CandSeed(rows[i]), ic = ::mkfitdev::v2p2CandIc(rows[i]);
              v2cands.logRho(rows[i]) = select_hits_log_rho(w, i, nScannedAll);
              if (walk[i])
                order_hits_along_path(
                    w, i, pq[i], pqSize[i], &layerHits.hit(::mkfitdev::v2p2LayerHitRow(s, ic, 0, preselStride)));
              v2cands.nLayerHits(rows[i]) = int8_t(walk[i] ? pqSize[i] : 0);
            }
          }
          // the shared LayerBatch is rewritten by the next one of this block
          alpaka::syncBlockThreads(acc);
        }
      }
    }
  };

  // ---------------------------------------------------------------------------------------------------------
  // V3: expand_in_layer() and the worklist behind it (rk_pump): a Kalman batch (rk_flush) and harvest_nodes() per
  // thread and round, then rk_expand_node() of the harvested nodes; an item depends on its parent node only.

  class KernelV2p2InLayer {
    // One harvested item of the batch: harvest_nodes() body; row[i] = the item's pool row, -1 if none
    template <idx_t N>
    ALPAKA_FN_ACC static void harvest(Acc1D const& acc,
                                      const KalmanOpArgs<N>& koa,
                                      const int (&module)[N],
                                      V2p2SeedSoA::View v2seeds,
                                      V2p2NodeSoA::View nodes,
                                      const V2p2Params& vp,
                                      int nodeCapacity,
                                      int (&row)[N]) {
      V2p2Counters& cnt = v2seeds.counters();
      for (int i = 0; i < int(N) && i < koa.N_filled; ++i) {
        row[i] = -1;
        if (!harvest_accept(koa.tsChi2Acc.constAt(i, 0, 0), vp))
          continue;
        const int p = koa.parent_idx[i];
        // the depth of the new node = the parent's hit count
        const int depth = p >= 0 ? int(nodes.nHits(p)) : 0;
        float chi2Sum, logDetVSum;
        int nHits;
        harvest_features(p >= 0 ? nodes.chi2Sum(p) : 0.f,
                         p >= 0 ? nodes.logDetVSum(p) : 0.f,
                         depth,
                         koa.tsChi2.constAt(i, 0, 0),
                         koa.tsDetV.constAt(i, 0, 0),
                         chi2Sum,
                         logDetVSum,
                         nHits);
        const int n = alpaka::atomicAdd(acc, &cnt.nPoolRows, 1, alpaka::hierarchy::Blocks{});
        if (n >= nodeCapacity) {
          alpaka::atomicAdd(acc, &cnt.nNodeOverflow, 1u, alpaka::hierarchy::Blocks{});
          continue;
        }
        const int s = koa.cand_seed[i];
        nodes.seed(n) = s;
        nodes.ic(n) = int8_t(koa.cand_ic[i]);
        nodes.hitPos(n) = int8_t(koa.hit_pos[i]);
        nodes.nHits(n) = int8_t(nHits);
        nodes.parentNode(n) = p;
        nodes.pathKey(n) = ::mkfitdev::v2p2PathKey(p >= 0 ? nodes.pathKey(p) : 0u, depth, koa.hit_pos[i]);
        nodes.hitIndex(n) = koa.hot_index[i];
        nodes.hitLayer(n) = int16_t(koa.hot_layer[i]);
        nodes.module(n) = int16_t(module[i]);
        nodes.chi2(n) = koa.tsChi2.constAt(i, 0, 0);
        nodes.chi2Sum(n) = chi2Sum;
        nodes.logDetVSum(n) = logDetVSum;
        koa.item_state(i, nodes.state(n));
        const int k = alpaka::atomicAdd(acc, &v2seeds.nNodes(s), 1, alpaka::hierarchy::Blocks{});
        if (k < ::mkfitdev::kV2p2NodeListCap)
          v2seeds.nodeList(s).node[k] = n;
        else
          nodes.nextInSeed(n) = alpaka::atomicExch(acc, &v2seeds.nodeHead(s), n, alpaka::hierarchy::Blocks{});
        row[i] = n;
      }
    }

    // rk_flush(): compute_pars() at depth 0, do_kalman_stuff(), harvest
    template <idx_t N>
    ALPAKA_FN_ACC static void flush(Acc1D const& acc,
                                    KalmanOpArgs<N>& koa,
                                    const int (&module)[N],
                                    V2p2SeedSoA::View v2seeds,
                                    V2p2NodeSoA::View nodes,
                                    const V2p2Params& vp,
                                    const PropagationFlags& pfInter,
                                    const PropagationFlags& pfIntra,
                                    int nodeCapacity,
                                    int (&row)[N]) {
      if (koa.N_filled == 0)
        return;
      if (!koa.m_solve_plane)
        koa.compute_pars();
      koa.do_kalman_stuff(pfInter, pfIntra);
      harvest<N>(acc, koa, module, v2seeds, nodes, vp, nodeCapacity, row);
      koa.reset();
    }

    // rk_add_item(): item (CandIncoming row, layer hit lh, parent node or -1) into the next free slot of the batch
    template <idx_t N>
    ALPAKA_FN_ACC static void addItem(KalmanOpArgs<N>& koa,
                                      int (&module)[N],
                                      SeedCandsSoA::ConstView seeds,
                                      CandSlotsSoA::ConstView slots,
                                      V2p2LayerHitSoA::ConstView layerHits,
                                      V2p2NodeSoA::View nodes,
                                      const EngineHitInputs& hits,
                                      int preselStride,
                                      int candRow,
                                      int parent,
                                      int lh) {
      const int s = ::mkfitdev::v2p2CandSeed(candRow), ic = ::mkfitdev::v2p2CandIc(candRow);
      const V2p2PassedHitMatch& phm = layerHits.hit(::mkfitdev::v2p2LayerHitRow(s, ic, lh, preselStride));
      koa.item_begin(s, ic, int(phm.hit_orig_index), phm.layer, parent, lh);
      if (parent < 0)
        koa.load_state_err_chg(phm.mixed_state, slots.state(candSlotRow(s, seeds.curBuf(s), ic)));
      else
        koa.load_state_err_chg(nodes.state(parent));
      module[koa.fill_slot()] = koa.load_hit_module(hits, phm.layer, phm.hit_orig_index);
      koa.item_finished();
    }

    // an item of the CandIncoming row: active, inside the layer, lh among its kept hits
    ALPAKA_FN_ACC static bool candHasHit(V2p2CandSoA::ConstView v2cands, int candRow, int lh) {
      return v2cands.active(candRow) &&
             !(Config::V2p2::Policy::use_wsr && v2cands.wsr(candRow) == ::mkfitdev::WSR_Outside) &&
             lh < v2cands.nLayerHits(candRow);
    }

    // rk_expand_node(): lh after node n's hit, not on a module already on the path
    ALPAKA_FN_ACC static bool expandable(V2p2CandSoA::ConstView v2cands,
                                         V2p2LayerHitSoA::ConstView layerHits,
                                         V2p2NodeSoA::View nodes,
                                         const EngineHitInputs& hits,
                                         int preselStride,
                                         int maxHitsPerLayer,
                                         int n,
                                         int lh) {
      const int s = nodes.seed(n), ic = nodes.ic(n);
      if (!(lh > nodes.hitPos(n) && lh < v2cands.nLayerHits(::mkfitdev::v2p2CandRow(s, ic)) &&
            nodes.nHits(n) < maxHitsPerLayer))
        return false;
      const V2p2PassedHitMatch& phm = layerHits.hit(::mkfitdev::v2p2LayerHitRow(s, ic, lh, preselStride));
      const int module = int(::mkfitdev::hitpack::detIDinLayer(
          hits.packed[::mkfitdev::pack::hitRow(hits, phm.layer, int(phm.hit_orig_index))]));
      return !same_module_on_path(nodes, n, phm.layer, module);
    }

    // an item of the next round, in the block's shared list
    ALPAKA_FN_ACC static void addToRound(Acc1D const& acc,
                                         int32_t& nItems,
                                         int32_t* itemCand,
                                         int32_t* itemNode,
                                         int8_t* itemHit,
                                         int candRow,
                                         int parent,
                                         int lh) {
      const int k = alpaka::atomicAdd(acc, &nItems, 1, alpaka::hierarchy::Threads{});
      itemCand[k] = candRow;
      itemNode[k] = parent;
      itemHit[k] = int8_t(lh);
    }

  public:
    // A block runs the worklist of its CandIncoming rows in rounds: the root items (expand_in_layer), then the
    // expansions of the nodes queued in the previous round (rk_pump). A thread whose node finds the queue full
    // expands it itself, depth first.
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  SeedCandsSoA::ConstView seeds,
                                  CandSlotsSoA::ConstView slots,
                                  V2p2SeedSoA::View v2seeds,
                                  V2p2CandSoA::ConstView v2cands,
                                  V2p2LayerHitSoA::ConstView layerHits,
                                  V2p2NodeSoA::View nodes,
                                  EngineHitInputs hits,
                                  V2p2Params vp,
                                  PropagationFlags pfInter,
                                  PropagationFlags pfIntra,
                                  int preselStride,
                                  int nodeCapacity,
                                  int step,
                                  int nSeeds,
                                  const int32_t* nDev) const {
      nSeeds = engineRows(nSeeds, nDev);
      const int maxHitsPerLayer = vp.max_hits_per_layer;
      auto& deepQueue = alpaka::declareSharedVar<int32_t[kV2p2InLayerQueue], __COUNTER__>(acc);
      auto& itemCand = alpaka::declareSharedVar<int32_t[kV2p2InLayerItems], __COUNTER__>(acc);
      auto& itemNode = alpaka::declareSharedVar<int32_t[kV2p2InLayerItems], __COUNTER__>(acc);
      auto& itemHit = alpaka::declareSharedVar<int8_t[kV2p2InLayerItems], __COUNTER__>(acc);
      // per round parity: a counter is reset one round before it is filled
      auto& nQueued = alpaka::declareSharedVar<int32_t[2], __COUNTER__>(acc);
      auto& nItems = alpaka::declareSharedVar<int32_t[2], __COUNTER__>(acc);
      const int tid = int(alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u]);
      const int nThreads = int(kV2p2InLayerThreads);
      const int nRows = kV2p2InLayerFromList ? v2seeds.counters().nActiveCands[step & 1] : nSeeds * kMaxCandsPerSeed;
      for (int32_t g :
           cms::alpakatools::independent_groups(acc, cms::alpakatools::divide_up_by(nRows, int(kV2p2InLayerRows)))) {
        // the previous group's last round reads the counters
        alpaka::syncBlockThreads(acc);
        if (tid == 0) {
          nQueued[0] = 0;
          nItems[0] = 0;
        }
        alpaka::syncBlockThreads(acc);
        const int rowEnd = std::min(nRows, (g + 1) * int(kV2p2InLayerRows));
        for (int r = g * int(kV2p2InLayerRows) + tid; r < rowEnd; r += nThreads) {
          const int candRow = kV2p2InLayerFromList ? v2cands.activeRow(r) : r;
          if (v2seeds.inLayer(::mkfitdev::v2p2CandSeed(candRow))) {
            for (int lh = 0; candHasHit(v2cands, candRow, lh); ++lh)
              addToRound(acc, nItems[0], itemCand, itemNode, itemHit, candRow, -1, lh);
          }
        }
        alpaka::syncBlockThreads(acc);
        for (int round = 0; round < maxHitsPerLayer && nItems[round & 1] > 0; ++round) {
          const int cur = round & 1, next = cur ^ 1;
          const int nRound = nItems[cur];
          if (tid == 0) {
            nQueued[next] = 0;
            nItems[next] = 0;
          }
          for (int first = tid * int(kNN); first < nRound; first += nThreads * int(kNN)) {
            // a batch of the round's items, then the nodes kept by this thread: the batch's nodes at the bottom,
            // above them a chain of children below max_hits_per_layer
            int keptNode[kNN + ::mkfitdev::kV2p2MaxHitsPerLayer - 1];
            int8_t keptNext[kNN + ::mkfitdev::kV2p2MaxHitsPerLayer - 1];
            int nKept = 0;
            int parent = -1, lh = -1;  // the next depth-first item
            for (bool fromRound = true; fromRound || parent >= 0;) {
              KalmanOpArgs<kNN> koa(!fromRound || round > 0, vp.chi2_trk_fac);
              int module[kNN], row[kNN];
              const int nBatch = fromRound ? std::min(int(kNN), nRound - first) : 1;
              for (int i = 0; i < int(kNN) && i < nBatch; ++i) {
                const int candRow =
                    fromRound ? itemCand[first + i] : ::mkfitdev::v2p2CandRow(nodes.seed(parent), nodes.ic(parent));
                addItem<kNN>(koa,
                             module,
                             seeds,
                             slots,
                             layerHits,
                             nodes,
                             hits,
                             preselStride,
                             candRow,
                             fromRound ? itemNode[first + i] : parent,
                             fromRound ? int(itemHit[first + i]) : lh);
              }
              flush<kNN>(acc, koa, module, v2seeds, nodes, vp, pfInter, pfIntra, nodeCapacity, row);
              for (int i = 0; i < int(kNN) && i < nBatch; ++i) {
                if (row[i] < 0 || nodes.nHits(row[i]) >= maxHitsPerLayer)
                  continue;
                const int q = alpaka::atomicAdd(acc, &nQueued[cur], 1, alpaka::hierarchy::Threads{});
                if (q < int(kV2p2InLayerQueue)) {
                  deepQueue[q] = row[i];
                } else {
                  keptNode[nKept] = row[i];
                  keptNext[nKept] = int8_t(nodes.hitPos(row[i]) + 1);
                  ++nKept;
                }
              }
              fromRound = false;
              parent = -1;
              while (nKept > 0 && parent < 0) {
                const int n = keptNode[nKept - 1];
                const int l = keptNext[nKept - 1]++;
                if (l >= preselStride) {
                  --nKept;
                } else if (expandable(v2cands, layerHits, nodes, hits, preselStride, maxHitsPerLayer, n, l)) {
                  parent = n;
                  lh = l;
                }
              }
            }
          }
          alpaka::syncBlockThreads(acc);
          // rk_pump(): the queued nodes expand into the next round's items
          const int nq = std::min(nQueued[cur], int(kV2p2InLayerQueue));
          for (int q = tid; q < nq; q += nThreads) {
            const int n = deepQueue[q];
            for (int l = nodes.hitPos(n) + 1; l < preselStride; ++l) {
              if (expandable(v2cands, layerHits, nodes, hits, preselStride, maxHitsPerLayer, n, l))
                addToRound(acc,
                           nItems[next],
                           itemCand,
                           itemNode,
                           itemHit,
                           ::mkfitdev::v2p2CandRow(nodes.seed(n), nodes.ic(n)),
                           n,
                           l);
            }
          }
          alpaka::syncBlockThreads(acc);
        }
      }
    }
  };

  // ---------------------------------------------------------------------------------------------------------
  // V4: end_layer() per seed: select_and_materialise() into the other candidate buffer, then the finished check;
  // with nextLayer, V0 of the next entry follows for the same seed (end_layer's awaken_candidates, begin_layer).
  // A group of lanes per seed: each lane reduces a share of the competitors to its top, the tops are merged pairwise,
  // each lane builds a share of the survivors.

  class KernelV2p2Select {
    // one seed in the layer, all lanes of its group; every thread of the block calls it (inLayer false: barriers only)
    ALPAKA_FN_ACC static void select(Acc1D const& acc,
                                     bool inLayer,
                                     SeedCandsSoA::View seeds,
                                     CandSlotsSoA::View slots,
                                     CandHotsSoA::View hots,
                                     V2p2SeedSoA::View v2seeds,
                                     V2p2CandSoA::ConstView v2cands,
                                     V2p2NodeSoA::ConstView nodes,
                                     const int16_t* stepLayer,
                                     ::mkfitdev::LayerInfoSoA::ConstView geo,
                                     const V2p2LayerCtx& ctxBase,
                                     int s,
                                     HotPool const& pool,
                                     const V2p2SelGroup& group) {
      uint32_t ovf0 = 0;
      V2p2LayerCtx ctx;  // read by select_and_materialise only for a seed in the layer
      SeedCandsRef ref{};
      int nextBuf = 0, nList = 0, chain = -1, maxCands = 0;
      CandBook* outBooks = nullptr;
      CandState* outStates = nullptr;
      const int32_t* list = nullptr;
      if (inLayer) {
        ovf0 = seeds.overflowBits(s);
        const int layer = stepLayer[seeds.region(s)];
        ctx = ctxBase;
        begin_layer(ctx, geo, layer);
        ref = makeSeedRef(seeds, slots, hots, s);
        nextBuf = seeds.curBuf(s) ^ 1;
        outBooks = &slots.book(candSlotRow(s, nextBuf, 0));
        outStates = &slots.state(candSlotRow(s, nextBuf, 0));
        const int nNodes = v2seeds.nNodes(s);
        nList = std::min(nNodes, int(::mkfitdev::kV2p2NodeListCap));
        list = v2seeds.nodeList(s).node;
        chain = v2seeds.nodeHead(s);
        maxCands = v2seeds.maxCands(s);
      }
      bool nanScore = false;
      const SeedHotPool hotPool{&pool, s};
      const int n = select_and_materialise(acc,
                                           group,
                                           inLayer,
                                           ref,
                                           hotPool,
                                           outBooks,
                                           outStates,
                                           v2cands,
                                           s,
                                           nodes,
                                           list,
                                           nList,
                                           chain,
                                           maxCands,
                                           ctx,
                                           nanScore);
      if (!inLayer || group.lane != 0)
        return;
      seeds.nCands(s) = n;
      seeds.curBuf(s) = int8_t(nextBuf);
      if (nanScore)
        alpaka::atomicAdd(acc, &v2seeds.counters().nNanScore, 1u, alpaka::hierarchy::Blocks{});
      // A CombCandidate is finished once every TrackCand under it has stopped.
      const SeedCandsRef selected = makeSeedRef(seeds, slots, hots, s);
      if (end_layer_finished(selected))
        seeds.state(s) = kFinished;
      failSeedOnHotOverflow(acc, seeds, s, ovf0);
    }

  public:
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  SeedCandsSoA::View seeds,
                                  CandSlotsSoA::View slots,
                                  CandHotsSoA::View hots,
                                  V2p2SeedSoA::View v2seeds,
                                  V2p2CandSoA::View v2cands,
                                  V2p2NodeSoA::ConstView nodes,
                                  const int16_t* stepLayer,
                                  const int16_t* nextLayer,      // V0 of the next entry; nullptr: none
                                  const int16_t* nextPrevLayer,  // V0 of the next entry
                                  ::mkfitdev::LayerInfoSoA::ConstView geo,
                                  V2p2LayerCtx ctxBase,
                                  int step,
                                  int nSeeds,
                                  const int32_t* nDev) const {
      nSeeds = engineRows(nSeeds, nDev);
      if (nextLayer != nullptr)
        KernelV2p2PullIn::resetEntryCounters(acc, v2seeds, step + 1);
      const HotPool pool = hotPoolOf(seeds, hots);
      auto& tops =
          alpaka::declareSharedVar<V2p2SelTop[kV2p2SelectSeedsPerBlock][kV2p2SelectSeedLanes], __COUNTER__>(acc);
      const uint32_t tid = alpaka::getIdx<alpaka::Block, alpaka::Threads>(acc)[0u];
      const uint32_t group = tid / kV2p2SelectSeedLanes;
      const uint32_t lane = tid % kV2p2SelectSeedLanes;
      const int nTiles = cms::alpakatools::divide_up_by(nSeeds, int(kV2p2SelectSeedsPerBlock));
      const int nTileRuns = cms::alpakatools::divide_up_by(nTiles, int(kV2p2SelectTilesPerBlock));
      for (int32_t run : cms::alpakatools::independent_groups(acc, nTileRuns)) {
        const int tileEnd = std::min(nTiles, (run + 1) * int(kV2p2SelectTilesPerBlock));
        for (int tile = run * int(kV2p2SelectTilesPerBlock); tile < tileEnd; ++tile) {
          const int s = tile * int(kV2p2SelectSeedsPerBlock) + int(group);
          const bool valid = s < nSeeds;
          select(acc,
                 valid && v2seeds.inLayer(s),
                 seeds,
                 slots,
                 hots,
                 v2seeds,
                 v2cands,
                 nodes,
                 stepLayer,
                 geo,
                 ctxBase,
                 s,
                 pool,
                 V2p2SelGroup{int(lane), int(kV2p2SelectSeedLanes), tops[group]});
          // every lane has read the seed's rows before lane 0 rewrites them; tops are reused by the next seed
          alpaka::syncBlockThreads(acc);
          if (valid && nextLayer != nullptr && lane == 0)
            KernelV2p2PullIn::pullIn(
                acc, seeds, slots, hots, v2seeds, v2cands, nextLayer, nextPrevLayer, geo, ctxBase, step + 1, s, pool);
        }
      }
    }
  };

  // ---------------------------------------------------------------------------------------------------------
  // V5: the final sorting of findTracksStandardv2p2: addPreBkwSearchHoles (backward search), then
  // mergeCandsAndBestShortOne(m_job->params(), scorer of the region, update_score, sort)

  class KernelV2p2FinalPick {
  public:
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  SeedCandsSoA::View seeds,
                                  CandSlotsSoA::View slots,
                                  CandHotsSoA::View hots,
                                  V2p2RegionScorers scorers,
                                  int maxCandsPerSeed,
                                  bool fwdSearch,
                                  int nSeeds,
                                  const int32_t* nDev) const {
      nSeeds = engineRows(nSeeds, nDev);
      for (int32_t s : cms::alpakatools::uniform_elements(acc, nSeeds)) {
        if (!scorers.hasPlan[seeds.region(s)])
          continue;
        SeedCandsRef ref = makeSeedRef(seeds, slots, hots, s);
        if (!fwdSearch)
          addPreBkwSearchHoles(ref);
        float pt[kMaxCandsPerSeed];
        for (int ic = 0; ic < *ref.nCands; ++ic)
          pt[ic] = alpaka::math::abs(acc, 1.f / ref.states[ic].par[3]);  // TrackState::pT()
        float ptBest = 0.f;  // read by the merge only with a valid best-short candidate
        if (*ref.bestShortValid)
          ptBest = alpaka::math::abs(acc, 1.f / ref.bestShortState->par[3]);
        mergeCandsAndBestShortOne(ref, maxCandsPerSeed, scorers.scorer[seeds.region(s)], true, true, pt, ptBest);
      }
    }
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev

#endif
