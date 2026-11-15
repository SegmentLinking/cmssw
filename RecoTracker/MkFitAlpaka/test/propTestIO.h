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
    kNOps = 13
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
                                       "mini InitialState::propagate_to_plane"};
    return names[op];
  }

  inline constexpr bool isMini(int op) {
    return op == kMiniPlexR || op == kMiniPlexZ || op == kMiniR || op == kMiniPlane;
  }

  // FinalFitFlags::bkw_sub_steps of the backward final-fit operation
  inline constexpr int kBkwSubSteps = 2;

  // the operations that compute a chi2
  inline constexpr bool hasChi2(int op) { return op == kFitUpdBkw || op == kChi2Plane || op == kFitUpdFwd; }

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
    // outputs
    float outPar[6];
    float outErr[21];
    float chi2;
    int fail;
    int chgOut;
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
