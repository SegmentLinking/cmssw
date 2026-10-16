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
    kPropR = 0,         // propagateHelixToRMPlex                  (finding_inter_layer flags)
    kPropZ = 1,         // propagateHelixToZMPlex                  (finding_inter_layer flags)
    kPropPlane = 2,     // propagateHelixToPlaneMPlex              (backward_fit flags)
    kChi2Plane = 3,     // kalmanPropagateAndComputeChi2Plane      (finding_intra_layer flags, propToHit)
    kUpdPlane = 4,      // kalmanPropagateAndUpdatePlane           (finding_inter_layer flags, propToHit)
    kUpdChi2Plane = 5,  // kalmanPropagateAndUpdateAndChi2Plane    (forward_fit flags, propToHit)
    kOpPlaneLocal = 6,  // kalmanOperationPlaneLocal(Chi2|Update|LocalCov) on the plane-propagated state
    kNOps = 7
  };

  inline const char* opName(int op) {
    static const char* names[kNOps] = {"propagateHelixToRMPlex",
                                       "propagateHelixToZMPlex",
                                       "propagateHelixToPlaneMPlex",
                                       "kalmanPropagateAndComputeChi2Plane",
                                       "kalmanPropagateAndUpdatePlane",
                                       "kalmanPropagateAndUpdateAndChi2Plane",
                                       "kalmanOperationPlaneLocal"};
    return names[op];
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
    int noMatEff;    // per-slot noMatEffPtr value (kUpdChi2Plane), 0 otherwise
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

  inline bool writeFile(const std::string& fn, const MaterialBlock& mb, const std::vector<OpBlock>& ops) {
    FILE* f = std::fopen(fn.c_str(), "wb");
    if (!f)
      return false;
    const uint32_t magic = 0x50524f50;  // "PROP"
    std::fwrite(&magic, 4, 1, f);
    std::fwrite(&mb.nBinsZ, sizeof(int), 1, f);
    std::fwrite(&mb.nBinsR, sizeof(int), 1, f);
    std::fwrite(&mb.rngZ, sizeof(float), 1, f);
    std::fwrite(&mb.rngR, sizeof(float), 1, f);
    std::fwrite(mb.bbxi.data(), sizeof(float), mb.bbxi.size(), f);
    std::fwrite(mb.radl.data(), sizeof(float), mb.radl.size(), f);
    const int nops = ops.size();
    std::fwrite(&nops, sizeof(int), 1, f);
    for (auto const& b : ops) {
      const int n = b.recs.size();
      std::fwrite(&b.op, sizeof(int), 1, f);
      std::fwrite(&b.pflags, sizeof(int), 1, f);
      std::fwrite(&b.propToHit, sizeof(int), 1, f);
      std::fwrite(&n, sizeof(int), 1, f);
      std::fwrite(b.recs.data(), sizeof(Rec), n, f);
    }
    std::fclose(f);
    return true;
  }

  inline bool readFile(const std::string& fn, MaterialBlock& mb, std::vector<OpBlock>& ops) {
    FILE* f = std::fopen(fn.c_str(), "rb");
    if (!f)
      return false;
    uint32_t magic = 0;
    bool ok = std::fread(&magic, 4, 1, f) == 1 && magic == 0x50524f50;
    ok = ok && std::fread(&mb.nBinsZ, sizeof(int), 1, f) == 1 && std::fread(&mb.nBinsR, sizeof(int), 1, f) == 1;
    ok = ok && std::fread(&mb.rngZ, sizeof(float), 1, f) == 1 && std::fread(&mb.rngR, sizeof(float), 1, f) == 1;
    if (ok) {
      mb.bbxi.resize(mb.nBinsZ * mb.nBinsR);
      mb.radl.resize(mb.nBinsZ * mb.nBinsR);
      ok = std::fread(mb.bbxi.data(), sizeof(float), mb.bbxi.size(), f) == mb.bbxi.size() &&
           std::fread(mb.radl.data(), sizeof(float), mb.radl.size(), f) == mb.radl.size();
    }
    int nops = 0;
    ok = ok && std::fread(&nops, sizeof(int), 1, f) == 1;
    for (int i = 0; ok && i < nops; ++i) {
      OpBlock b;
      int n = 0;
      ok = std::fread(&b.op, sizeof(int), 1, f) == 1 && std::fread(&b.pflags, sizeof(int), 1, f) == 1 &&
           std::fread(&b.propToHit, sizeof(int), 1, f) == 1 && std::fread(&n, sizeof(int), 1, f) == 1;
      if (ok) {
        b.recs.resize(n);
        ok = std::fread(b.recs.data(), sizeof(Rec), n, f) == static_cast<size_t>(n);
        ops.push_back(std::move(b));
      }
    }
    std::fclose(f);
    return ok;
  }

  inline constexpr int symIdx(int i, int j) { return i >= j ? i * (i + 1) / 2 + j : j * (j + 1) / 2 + i; }

}  // namespace mkfitdev::proptest

#endif
