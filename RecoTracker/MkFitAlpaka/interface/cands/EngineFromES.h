#ifndef RecoTracker_MkFitAlpaka_interface_cands_EngineFromES_h
#define RecoTracker_MkFitAlpaka_interface_cands_EngineFromES_h

// Host helpers that build the clone engine's tables from the ES product: the region layer plans
// (EnginePlan, for makeEngineStepTables), the iteration parameters, the per-layer
// table (EngineLayerParams) and the module-plane table (EngineModule, same row convention as ESView:
// layers[l].module_begin + short id = the hit's detIDinLayer).

#include <vector>

#include "RecoTracker/MkFitAlpaka/interface/cands/CandEngineTypes.h"
#include "RecoTracker/MkFitAlpaka/interface/es/ESConfig.h"
#include "RecoTracker/MkFitAlpaka/interface/es/ESLayouts.h"

namespace mkfitdev {

  inline EnginePlan makeEnginePlan(const ESConfig& cfg) {
    EnginePlan plan;
    plan.layers.resize(cfg.n_regions);
    plan.fwdPickup.resize(cfg.n_regions);
    plan.bkwPickup.resize(cfg.n_regions);
    for (int region = 0; region < cfg.n_regions; ++region) {
      const SteeringRegion& sr = cfg.steering_params[region];  // indexed by region id (m_steering_params[region])
      plan.layers[region].assign(sr.layer, sr.layer + sr.n_plan);
      plan.fwdPickup[region] = sr.fwd_search_pickup;
      plan.bkwPickup[region] = sr.bkw_search_pickup;
    }
    return plan;
  }

  inline EngineIterParams makeEngineIterParams(const IterParams& ip) {
    EngineIterParams params;
    params.maxCandsPerSeed = ip.maxCandsPerSeed;
    params.maxHolesPerCand = ip.maxHolesPerCand;
    params.maxConsecHoles = ip.maxConsecHoles;
    params.maxClusterSize = int(ip.maxClusterSize);
    params.chi2CutMin = ip.chi2Cut_min;
    params.pTCutOverlap = ip.pTCutOverlap;
    params.minPtCut = ip.minPtCut;
    return params;
  }

  inline std::vector<EngineLayerParams> makeEngineLayerParams(LayerInfoSoA::ConstView layers, int nLayers) {
    std::vector<EngineLayerParams> out(nLayers);
    for (int layer = 0; layer < nLayers; ++layer)
      out[layer] = EngineLayerParams{layers[layer].module_begin(),
                                     int8_t(layers[layer].is_pixel() ? 1 : 0),
                                     int8_t(layers[layer].layer_type() == static_cast<int>(LayerType::Barrel) ? 1 : 0)};
    return out;
  }

  // packModuleNormDirPnt: norm = zdir, dir = xdir, pnt = pos
  inline std::vector<EngineModule> makeEngineModules(ModuleInfoSoA::ConstView modules, int nModules) {
    std::vector<EngineModule> out(nModules);
    for (int row = 0; row < nModules; ++row) {
      out[row] = EngineModule{{modules[row].zdir_x(), modules[row].zdir_y(), modules[row].zdir_z()},
                              {modules[row].xdir_x(), modules[row].xdir_y(), modules[row].xdir_z()},
                              {modules[row].pos_x(), modules[row].pos_y(), modules[row].pos_z()}};
    }
    return out;
  }

}  // namespace mkfitdev

#endif
