#ifndef RecoTracker_MkFitAlpaka_test_propTestIO_h
#define RecoTracker_MkFitAlpaka_test_propTestIO_h

// Record format shared by the MkFitCore reference writer (testPropMkFitCoreRef.cpp) and the port comparator
// (alpaka/testProp.dev.cc). Symmetric 6x6 errors are stored as 21 floats in packed lower-triangle
// row-major order (i >= j: index i*(i+1)/2 + j), filled through (i, j) accessors on both sides.

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace mkfitdev::proptest {

  enum Op : int {
    kPropR = 0,      // propagateHelixToRMPlex          (finding flags)
    kPropZ = 1,      // propagateHelixToZMPlex          (finding flags)
    kPropPlane = 2,  // propagateHelixToPlaneMPlex      (backward_fit flags)
    kUpdPlane = 3,   // kalmanPropagateAndUpdatePlane   (finding flags, propToHit)
    // final_fit::propagate_update (FinalFit.cc), backward pass: final_fit_pflags, every FinalFitFlags switch on,
    // 2 sub-steps, module material, ms_ref_p
    kFitUpdBkw = 4,
    kChi2Plane = 5,       // kalmanPropagateAndComputeChi2Plane (finding flags, propToHit)
    kUpdPlaneNoProp = 6,  // kalmanPropagateAndUpdatePlane     (finding flags, propToHit = false)
    kPropPlaneNoMat = 7,  // propagateHelixToPlaneMPlex        (backward_fit flags, no material on every track)
    kFitUpdFwd = 8,       // final_fit::propagate_update, forward pass (no sub-steps, no ms_ref_p)
    // mini propagators (MiniPropagators.h) of selectHitIndicesV2: outPar = (x, y, z, px, py, pz), chi2 = dalpha
    kMiniPlexR = 9,   // InitialStatePlex::propagate_to_r (the bin limits, barrel)
    kMiniPlexZ = 10,  // InitialStatePlex::propagate_to_z (the bin limits, endcap)
    kMiniR = 11,      // InitialState::propagate_to_r      (the per-hit preselection, barrel)
    kMiniPlane = 12,  // InitialState::propagate_to_plane  (the per-hit preselection, tilted barrel modules)
    // v2p2 (MkFinderV2p2): the layer crossings, the Hermite cubic and the in-layer Kalman step chains
    // prop_to_layer_edges: copyIn_partial_track_state, init_momentum_vec_and_k with a per-lane field, the two bounding
    // surfaces (propagate_to_r or _to_z, m_isp = m_sp1 between them); outPar = far crossing (x, y, z, px, py, pz),
    // chi2 = its dalpha, outErr[0, 7) = near crossing (x .. pz, dalpha)
    kV2p2Edges = 13,
    // Hermite3D::calculate_coeffs through the two crossings (uniform field), copied slot-wise, evaluate(t = hermT):
    // outPar = (x, y, z, dx, dy, dz), outErr[0, 12) = coefficients, [12] = derfac, [13, 16) = evaluate(t, x, y, z)
    kHermite = 14,
    // Hermite3DOnPlane::init_coeffs + solve (two Newton steps) onto the module plane: outPar = Hermite3D::evaluate at T,
    // chi2 = T, outErr[0, 4) = distance cubic, [4] = T before the steps, [5] = d(T), [6] = dd/dt(T), [7] = d(T),
    // [8] = T after the first step alone
    kHermitePlane = 15,
    // KalmanOpArgs::do_kalman_stuff (MkFinderV2p2Structures.cc), depth 0: from the search's crossing on the module
    // plane (xyz, compute_pars + path_from_perp), inter-layer flags
    kKalmanD0 = 16,
    // KalmanOpArgs::do_kalman_stuff, depth >= 1: path_solve + drift, intra-layer flags
    kKalmanD1 = 17,
    kNOps = 18
  };

  inline const char* opName(int op) {
    static const char* names[kNOps] = {"propagateHelixToRMPlex",
                                       "propagateHelixToZMPlex",
                                       "propagateHelixToPlaneMPlex",
                                       "kalmanPropagateAndUpdatePlane",
                                       "final_fit::propagate_update (bkw)",
                                       "kalmanPropagateAndComputeChi2Plane",
                                       "kalmanPropagateAndUpdatePlane (on plane)",
                                       "propagateHelixToPlaneMPlex (no material)",
                                       "final_fit::propagate_update (fwd)",
                                       "mini InitialStatePlex::propagate_to_r",
                                       "mini InitialStatePlex::propagate_to_z",
                                       "mini InitialState::propagate_to_r",
                                       "mini InitialState::propagate_to_plane",
                                       "v2p2 prop_to_layer_edges",
                                       "v2p2 Hermite3D",
                                       "v2p2 Hermite3DOnPlane",
                                       "v2p2 do_kalman_stuff (depth 0)",
                                       "v2p2 do_kalman_stuff (depth >= 1)"};
    return names[op];
  }

  // the operations whose outPar is a mini-propagator state (x, y, z, px, py, pz)
  inline constexpr bool isMini(int op) {
    return op == kMiniPlexR || op == kMiniPlexZ || op == kMiniR || op == kMiniPlane || op == kV2p2Edges ||
           op == kHermite || op == kHermitePlane;
  }

  // the v2p2 Kalman operations, which also return detV and chi2Acc
  inline constexpr bool isV2p2Kalman(int op) { return op == kKalmanD0 || op == kKalmanD1; }

  // V2p2Params::chi2_trk_fac (MkFitCore default) of the v2p2 Kalman operations
  inline constexpr float kV2p2Chi2TrkFac = 3.0f;

  // FinalFitFlags::bkw_sub_steps of the backward final-fit operation
  inline constexpr int kBkwSubSteps = 2;

  // the operations that compute a chi2
  inline constexpr bool hasChi2(int op) {
    return op == kFitUpdBkw || op == kChi2Plane || op == kFitUpdFwd || isV2p2Kalman(op);
  }

  struct Rec {
    // inputs
    float inPar[6];
    float inErr[21];
    int chg;
    float msRad;
    float msZ;
    float plPnt[3];
    float plNrm[3];
    float plDir[3];
    float msPar[3];
    float msErr[6];  // 3x3 sym packed
    float modRadl;   // the module's own material (final fit)
    float modBbxi;
    float msRefP;  // final_fit::Pass::ms_ref_p (backward final fit): |p| of the start state
    // v2p2: the layer's bounding surfaces in search order (radii in the barrel, z in the endcap), the Hermite parameter
    // of kHermite, and the search's crossing on the module plane (InitialState: x, y, z, px, py, pz, dalpha, inv_pt,
    // inv_k, theta), written by the reference for kKalmanD0
    float edge[2];
    int barrel;
    float hermT;
    float xyz[10];
    // outputs
    float outPar[6];
    float outErr[21];
    float chi2;
    int fail;
    int chgOut;
    float detV;     // v2p2 Kalman: det of the residual covariance
    float chi2Acc;  // v2p2 Kalman: the acceptance chi2 (track covariance scaled by kV2p2Chi2TrkFac^2)
  };

  struct OpBlock {
    int op;
    int pflags;  // PF_* bits
    int propToHit;
    std::vector<Rec> recs;
  };

  struct MaterialBlock {
    int nBinsZ, nBinsR;
    float rngZ, rngR;
    std::vector<float> bbxi, radl;  // binZ * nBinsR + binR
  };

  // file layout: FileHeader, the material bbxi then radl arrays, the number of ops, then per op a BlockHeader followed
  // by its records
  constexpr uint32_t kMagic = 0x50524f50;

  struct FileHeader {
    int magic;
    int nBinsZ, nBinsR;
    float rngZ, rngR;
  };

  struct BlockHeader {
    int op, pflags, propToHit, nRecs;
  };

  inline bool writeFile(const std::string& path, const MaterialBlock& mb, const std::vector<OpBlock>& ops) {
    FILE* file = std::fopen(path.c_str(), "wb");
    if (!file)
      return false;
    const FileHeader head{int(kMagic), mb.nBinsZ, mb.nBinsR, mb.rngZ, mb.rngR};
    const int nops = ops.size();
    bool ok = std::fwrite(&head, sizeof(head), 1, file) == 1 &&
              std::fwrite(mb.bbxi.data(), sizeof(float), mb.bbxi.size(), file) == mb.bbxi.size() &&
              std::fwrite(mb.radl.data(), sizeof(float), mb.radl.size(), file) == mb.radl.size() &&
              std::fwrite(&nops, sizeof(int), 1, file) == 1;
    for (auto const& block : ops) {
      const BlockHeader blockHead{block.op, block.pflags, block.propToHit, int(block.recs.size())};
      ok = ok && std::fwrite(&blockHead, sizeof(blockHead), 1, file) == 1 &&
           std::fwrite(block.recs.data(), sizeof(Rec), block.recs.size(), file) == block.recs.size();
    }
    return std::fclose(file) == 0 && ok;
  }

  inline bool readFile(const std::string& path, MaterialBlock& mb, std::vector<OpBlock>& ops) {
    FILE* file = std::fopen(path.c_str(), "rb");
    if (!file)
      return false;
    FileHeader head{};
    int nops = 0;
    bool ok = std::fread(&head, sizeof(head), 1, file) == 1 && uint32_t(head.magic) == kMagic;
    if (ok) {
      mb = MaterialBlock{head.nBinsZ, head.nBinsR, head.rngZ, head.rngR, {}, {}};
      mb.bbxi.resize(mb.nBinsZ * mb.nBinsR);
      mb.radl.resize(mb.nBinsZ * mb.nBinsR);
      ok = std::fread(mb.bbxi.data(), sizeof(float), mb.bbxi.size(), file) == mb.bbxi.size() &&
           std::fread(mb.radl.data(), sizeof(float), mb.radl.size(), file) == mb.radl.size() &&
           std::fread(&nops, sizeof(int), 1, file) == 1;
    }
    for (int i = 0; ok && i < nops; ++i) {
      BlockHeader blockHead{};
      ok = std::fread(&blockHead, sizeof(blockHead), 1, file) == 1;
      if (ok) {
        OpBlock block{blockHead.op, blockHead.pflags, blockHead.propToHit, std::vector<Rec>(blockHead.nRecs)};
        ok = std::fread(block.recs.data(), sizeof(Rec), block.recs.size(), file) == block.recs.size();
        ops.push_back(std::move(block));
      }
    }
    std::fclose(file);
    return ok;
  }

  inline constexpr int symIdx(int i, int j) { return i >= j ? i * (i + 1) / 2 + j : j * (j + 1) / 2 + i; }

}  // namespace mkfitdev::proptest

#endif
