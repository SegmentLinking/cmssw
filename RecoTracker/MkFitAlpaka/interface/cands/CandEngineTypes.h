#ifndef RecoTracker_MkFitAlpaka_interface_cands_CandEngineTypes_h
#define RecoTracker_MkFitAlpaka_interface_cands_CandEngineTypes_h

// Clone-engine scratch and parameter types.
//
// Per plan step, for every listed candidate (row s * kMaxCandsPerSeed + ic):
//   CandSelHitsSoA   written by K2:
//                      prop = layer-propagated state (m_Par/m_Err/m_Chg[iP]),
//                      sel  = hit indices (<= 6, MkFitCore order of m_XHitArr), count, RAW WSR result (BEFORE MkFitCore
//                             find_tracks_handle_missed_layers turns Failed into Outside), in_gap.
//                    The engine reproduces find_tracks_handle_missed_layers itself from the raw WSR (K3b, K4).
//   CandHitChi2SoA   written by K3a, thread per (cand, hit), row (s * kMaxCandsPerSeed + ic) * kMaxHitsPerCand + ih:
//                      chi2 of kalmanPropagateAndComputeChi2Plane and x, y, z of the plane-propagated parameters
//                      (the only propPar elements isStripQCompatible reads).

#include <cstdint>
#include <vector>

#include "DataFormats/SoATemplate/interface/SoALayout.h"
#include "RecoTracker/MkFitAlpaka/interface/cands/CandTypes.h"

namespace mkfitdev {

  // Layer-propagated state of a listed candidate (MkFinder m_Par[iP], m_Err[iP], m_Chg).
  struct CandPropState {
    float par[6];
    float err[21];  // lower triangle, MatriplexSym / SMatrixSym66 order
    int32_t charge;
  };

  // Selected hits of a listed candidate (m_XHitArr / m_XHitSize / m_XWsrResult).
  struct CandSelHits {
    int32_t hit[kMaxHitsPerCand];  // index within the layer's hit wrapper (= HoT index)
    int8_t n;                      // m_XHitSize (<= 0: no hits)
    int8_t wsr;                    // WsrResult (CandOptions.h), raw value from the hit selection
    int8_t inGap;
    int8_t pad;
  };

  struct CandHitChi2 {
    float chi2;  // outChi2 (MkFitCore takes std::abs)
    float px, py, pz;
  };

  GENERATE_SOA_LAYOUT(CandSelHitsSoALayout, SOA_COLUMN(CandPropState, prop), SOA_COLUMN(CandSelHits, sel))
  GENERATE_SOA_LAYOUT(CandHitChi2SoALayout, SOA_COLUMN(CandHitChi2, c2))

  using CandSelHitsSoA = CandSelHitsSoALayout<>;
  using CandHitChi2SoA = CandHitChi2SoALayout<>;

  ALPAKA_FN_HOST_ACC inline constexpr int selRow(int seed, int ic) { return seed * kMaxCandsPerSeed + ic; }
  ALPAKA_FN_HOST_ACC inline constexpr int chi2Row(int seed, int ic, int ih) {
    return (seed * kMaxCandsPerSeed + ic) * kMaxHitsPerCand + ih;
  }

  // Iteration parameters of the clone engine.
  struct EngineIterParams {
    int maxCandsPerSeed;
    int maxHolesPerCand;
    int maxConsecHoles;
    int maxClusterSize;
    float chi2CutMin;
    float pTCutOverlap;
    float minPtCut;
    bool recheckOverlap;
  };

  // Per-layer parameters the engine kernels read (row = mkFit layer id). c2 = the four chi2 window parameters
  // (IterationLayerConfig::get_window_params(in_fwd, true)[c2_sf, c2_0, c2_1, c2_2]); hasC2 = vector not empty.
  struct EngineLayerParams {
    int32_t moduleBegin;  // row of module detIDinLayer 0 of this layer in the module table
    int8_t isPixel;
    int8_t isBarrel;
    int8_t hasC2;
    int8_t pad;
    float c2[4];
  };

  // Module plane table row (ModuleInfo zdir / xdir / pos as packModuleNormDirPnt packs them).
  struct EngineModule {
    float nrm[3];
    float dir[3];
    float pnt[3];
  };

  // Read-only hit inputs: the HitSoA columns the engine needs, as plain pointers (rows = hitBase + index, with
  // hitBase = 0 for pixel layers and nPixel for strip layers), plus the module table.
  struct EngineHitInputs {
    const float* x;
    const float* y;
    const float* z;
    const float* e00;
    const float* e10;
    const float* e11;
    const float* e20;
    const float* e21;
    const float* e22;
    const uint32_t* packed;  // hitpack: detIDinLayer, spanRows
    uint32_t nPixel;
    const EngineModule* modules;
    const EngineLayerParams* layers;
  };

  // TrackerInfo::EtaRegion: Reg_Barrel = 2
  constexpr int kRegBarrel = 2;

  // backward-search gate (EventOfCombCandidates::beginBkwSearch / gateBkwSearch,
  // TrackStructures.cc): minPixelLayers <= 0 = off. A candidate whose backward search added found hits on fewer
  // distinct pixel layers (layer < 64) than required (1 if |d0| to the beam spot < promptMaxD0, else minPixelLayers)
  // gets its pre-search candidate back. bsX/bsY = the EventOfHits beam spot of the event.
  struct BkwSearchGate {
    int minPixelLayers = 0;
    float promptMaxD0 = 0.f;
    uint64_t pixelLayers[4] = {0, 0, 0, 0};
    float bsX = 0.f;
    float bsY = 0.f;
  };

  // Event-wide engine counters (device memory, atomics only here).
  struct EngineCounters {
    uint32_t nOverflowHotsSeeds;  // seeds failed by a HoT pool overflow
    uint32_t nOverflowHits;       // K2 reported more hits than kMaxHitsPerCand (never in MkFitCore)
  };

}  // namespace mkfitdev

// ---- host-side plan tables (moved here from interface/cands/alpaka/CandsEngine.h so host code can use them) ----
namespace mkfitdev {

  // Layer plans of the regions (SteeringParams: m_layer_plan layers, m_fwd_search_pickup, m_bkw_search_pickup).
  struct EnginePlan {
    std::vector<std::vector<int>> layers;  // [region][plan index]
    std::vector<int> fwdPickup, bkwPickup;
  };

  // Lock-step tables: for step t = 0..nSteps-1 and region r, layer[t * nRegions + r] = plan layer of region r at
  // iteration t + 1 after its pickup index (forward: pickup + 1 + t, backward: pickup - 1 - t), -1 past the plan
  // end; prevLayer = the layer before it in the plan walk (prev_layer).
  struct EngineStepTablesHost {
    int nSteps = 0;
    int nRegions = 0;
    std::vector<int16_t> layer, prevLayer;
  };
  inline EngineStepTablesHost makeEngineStepTables(const EnginePlan& plan, bool fwd) {
    EngineStepTablesHost t;
    t.nRegions = int(plan.layers.size());
    for (int r = 0; r < t.nRegions; ++r) {
      const int n = int(plan.layers[r].size());
      const int steps = fwd ? n - 1 - plan.fwdPickup[r] : plan.bkwPickup[r];
      t.nSteps = steps > t.nSteps ? steps : t.nSteps;
    }
    t.layer.assign(t.nSteps * t.nRegions, -1);
    t.prevLayer.assign(t.nSteps * t.nRegions, -1);
    for (int r = 0; r < t.nRegions; ++r) {
      const auto& pl = plan.layers[r];
      const int n = int(pl.size());
      int prev = fwd ? plan.fwdPickup[r] : plan.bkwPickup[r];
      for (int s = 0; s < t.nSteps; ++s) {
        const int idx = fwd ? prev + 1 : prev - 1;
        if (idx < 0 || idx >= n)
          break;
        t.layer[s * t.nRegions + r] = int16_t(pl[idx]);
        t.prevLayer[s * t.nRegions + r] = int16_t(pl[prev]);
        prev = idx;
      }
    }
    return t;
  }

}  // namespace mkfitdev

#endif
