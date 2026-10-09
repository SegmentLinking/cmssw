#ifndef RecoTracker_MkFitAlpaka_interface_cands_EngineFromES_h
#define RecoTracker_MkFitAlpaka_interface_cands_EngineFromES_h

// Host helpers that build the clone engine's tables from the ES product: the region layer plans
// (EnginePlan, for makeEngineStepTables), the iteration parameters, the per-layer
// table (EngineLayerParams) and the module-plane table (EngineModule, same row convention as ESView:
// layers[l].module_begin + short id = the hit's detIDinLayer). Untested in the engine test (it builds its tables from
// the MkFitCore dump); to be exercised by the producer.

#include <vector>

#include "RecoTracker/MkFitAlpaka/interface/cands/CandEngineTypes.h"
#include "RecoTracker/MkFitAlpaka/interface/es/ESConfig.h"
#include "RecoTracker/MkFitAlpaka/interface/es/ESLayouts.h"

namespace mkfitdev {

  inline EnginePlan makeEnginePlan(const ESConfig& c) {
    EnginePlan p;
    p.layers.resize(c.n_regions);
    p.fwdPickup.resize(c.n_regions);
    p.bkwPickup.resize(c.n_regions);
    for (int r = 0; r < c.n_regions; ++r) {
      const SteeringRegion& sr = c.steering_params[r];  // indexed by region id (m_steering_params[region])
      p.layers[r].assign(sr.layer, sr.layer + sr.n_plan);
      p.fwdPickup[r] = sr.fwd_search_pickup;
      p.bkwPickup[r] = sr.bkw_search_pickup;
    }
    return p;
  }

  inline EngineIterParams makeEngineIterParams(const IterParams& ip) {
    EngineIterParams e;
    e.maxCandsPerSeed = ip.maxCandsPerSeed;
    e.maxHolesPerCand = ip.maxHolesPerCand;
    e.maxConsecHoles = ip.maxConsecHoles;
    e.maxClusterSize = int(ip.maxClusterSize);
    e.chi2CutMin = ip.chi2Cut_min;
    e.pTCutOverlap = ip.pTCutOverlap;
    e.minPtCut = ip.minPtCut;
    e.recheckOverlap = ip.recheckOverlap;
    return e;
  }

  // The ES fill throws on non-empty hit-window parameters (empty for the LST step), so hasC2 = 0: the dynamic chi2 cut
  // is chi2Cut_min, as MkFitCore with an empty window vector.
  inline std::vector<EngineLayerParams> makeEngineLayerParams(LayerInfoSoA::ConstView layers, int nLayers) {
    std::vector<EngineLayerParams> out(nLayers);
    for (int l = 0; l < nLayers; ++l) {
      out[l] = EngineLayerParams{layers[l].module_begin(),
                                 int8_t(layers[l].is_pixel() ? 1 : 0),
                                 int8_t(layers[l].layer_type() == static_cast<int>(LayerType::Barrel) ? 1 : 0),
                                 0,
                                 0,
                                 {0.f, 0.f, 0.f, 0.f}};
    }
    return out;
  }

  // packModuleNormDirPnt: norm = zdir, dir = xdir, pnt = pos
  inline std::vector<EngineModule> makeEngineModules(ModuleInfoSoA::ConstView modules, int nModules) {
    std::vector<EngineModule> out(nModules);
    for (int m = 0; m < nModules; ++m) {
      out[m] = EngineModule{{modules[m].zdir_x(), modules[m].zdir_y(), modules[m].zdir_z()},
                            {modules[m].xdir_x(), modules[m].xdir_y(), modules[m].xdir_z()},
                            {modules[m].pos_x(), modules[m].pos_y(), modules[m].pos_z()}};
    }
    return out;
  }

}  // namespace mkfitdev

#endif
