#ifndef RecoTracker_MkFitAlpaka_test_v2p2TestIO_h
#define RecoTracker_MkFitAlpaka_test_v2p2TestIO_h

// Record format shared by the MkFitCore reference writer of the v2p2 finder pieces (testV2p2MkFitCoreRef.cpp) and the
// port comparator (alpaka/testV2p2.dev.cc). Plain PODs written whole; symmetric 6x6 errors are 21 floats in packed
// lower-triangle row-major order (i >= j: i * (i + 1) / 2 + j), 3x3 ones 6 floats in the same order.
// Blocks:
//   PqRec      CandIncoming's bounded pqueue (pq_push / pq_pop with the offer rule of preselect_hit_batch) on streams
//              of passed hits with many equal ddphi, then order_hits_along_path: the heap array and the path order
//   ScoreRec   the layer-step score (V2p2Score.h) and the final-pick scorer v2p2:llh
//   KalRec     KalmanOpArgs::do_kalman_stuff, depth 0 (the search's plane crossing) and depth >= 1 (solve plane)
//   WinRec     MkRZLimits + prop_to_layer_edges + MkBins (crossings, transported covariance, windows) +
//              determine_wsr + the Hermite cubic through the two crossings
//   SelSeed    select_and_materialise on one synthetic seed: TrackCands, CandIncomings, in-layer nodes; outputs the
//              new TrackCands in rank order

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace mkfitdev::v2p2test {

  constexpr uint32_t kMagic = 0x56325032;

  inline constexpr int symIdx(int i, int j) { return i >= j ? i * (i + 1) / 2 + j : j * (j + 1) / 2 + i; }

  // ---- pqueue + path order
  constexpr int kPqMaxStream = 40;
  constexpr int kPqCap = 12;  // V2p2Params::max_presel_hits_limit (both sides static_assert it)

  struct PqRec {
    int nHits;
    int maxPresel;
    float dir;  // order_hits_along_path: +1 if m_sp2.dalpha >= m_sp1.dalpha, else -1
    float ddphi[kPqMaxStream];
    float dalpha[kPqMaxStream];
    uint32_t hitOrig[kPqMaxStream];
    // outputs
    int nKept;
    uint32_t heapHit[kPqCap];  // the pqueue's array after the last offer
    uint32_t pathHit[kPqCap];  // m_layer_hits after order_hits_along_path
  };

  // ---- scores
  struct ScoreParams {
    float hit_bonus, overlap_bonus, chi2_weight, miss_penalty, edge_penalty, gap_penalty, stop_penalty, hit_eff;
    float hit_eff_grp[5];
  };

  struct ScoreRec {
    // LayerStepFeatures
    float chi2_sum, log_det_v_sum, log_rho;
    int n_hits, n_overlap, hole_kind, layer_to;
    int is_outward, is_pixel, is_barrel;
    ScoreParams outward, inward;
    int fake_hit;  // input of v2p2_hole_kind
    // TrackScoreInput of the final-pick scorer
    int n_found, n_tail, n_overlap_hits, n_inside, n_seed_hits;
    float chi2, pt, score_in;
    // outputs
    int group;
    float loglh;
    float step_score;
    int hole_kind_out;
    float llh;
  };

  // ---- Kalman batch
  struct KalRec {
    int solvePlane;
    float chi2TrkFac;
    // depth 0: the InitialState on the plane (x, y, z, px, py, pz, dalpha, inv_pt, inv_k, theta)
    float isXyz[10];
    int isFail;
    float inPar[6];
    float inErr[21];
    int chg;
    float msPar[3];
    float msErr[6];
    float plNrm[3], plDir[3], plPnt[3];
    // outputs
    float outPar[6];
    float outErr[21];
    int chgOut;
    float chi2, detV, chi2Acc;
  };

  // ---- window of one candidate on one layer
  struct WinRec {
    // the layer (LayerInfo) and direction
    int layerId, layerType, isPixel, hasHole, outward;
    float rin, rout, zmin, zmax, holeRMin, holeRMax;
    // the TrackCand state
    float par[6];
    float err[21];
    int chg;
    // outputs: rz_quadrant_check, the crossings (x, y, z, px, py, pz, dalpha) and fail flags
    int quadrantOk;
    float sp1[7], sp2[7];
    int fail1, fail2;
    float cov[6];  // m_cov_0_0, 0_1, 1_1, 2_2, 0_2, 1_2
    float phiMin, phiMax, phiCenter, phiDelta, qMin, qMax, dphiTrack, dqTrack;
    int wsr, inGap;
    float hx[4], hy[4], hz[4], hderfac;
  };

  // ---- end-of-layer selection
  constexpr int kSelMaxCands = 5;   // kMaxCandsPerSeed
  constexpr int kSelMaxNodes = 24;  // in-layer nodes per synthetic seed
  constexpr int kSelMaxHits = 24;   // HoT chain length of a TrackCand
  constexpr int kSelMaxDepth = 8;   // V2p2Params::max_hits_per_layer_limit

  struct SelCand {
    float score;
    float par[6];
    float err[21];
    int charge;
    int nHits;  // the HoT chain, oldest first
    int hitIdx[kSelMaxHits];
    int hitLyr[kSelMaxHits];
    float hitChi2[kSelMaxHits];
    // the CandIncoming of this layer
    int active;
    int wsr, inGap;
    float logRho;
  };

  struct SelNode {
    int ic;
    int parent;  // node index, -1 = root
    int hitPos;
    int hitIndex, hitLayer;
    float chi2;
    float chi2Sum, logDetVSum;  // accumulated parent -> child as harvest_nodes does
    int nHits;
    float par[6];
    float err[21];
    int charge;
  };

  struct SelOut {
    int srcIc;
    int kind;  // 0 path, 1 hole, 2 unchanged
    float score;
    int nFound, nMissing, nOverlap, nInside, nTail;
    float chi2;
    int nHits;  // HoT chain after the layer, oldest first
    int hitIdx[kSelMaxHits + kSelMaxDepth];
    int hitLyr[kSelMaxHits + kSelMaxDepth];
    float par[6];
  };

  struct SelSeed {
    // the layer and the iteration
    int layerId, isPixel, isBarrel, outward;
    int maxCands;  // CCandRep::m_max_cands
    int maxHolesPerCand, maxConsecHoles;
    ScoreParams outwardP, inwardP;
    int nCands;
    SelCand cands[kSelMaxCands];
    int nNodes;
    SelNode nodes[kSelMaxNodes];
    // outputs
    int nOut;
    int nSel;
    int boundaryTie;  // the last kept score equals the best dropped score
    SelOut out[kSelMaxCands];
  };

  // ---- file: FileHeader, the material map, then the five blocks, each a count followed by its records
  struct FileHeader {
    uint32_t magic;
    int matBinsZ, matBinsR;
    float matRangeZ, matRangeR;
    float hitChi2Cut;  // V2p2Params::hit_chi2_cut of MkFitCore, the acceptance compared in the Kalman block
  };

  struct RefData {
    FileHeader head;
    std::vector<float> bbxi, radl;  // binZ * matBinsR + binR
    std::vector<PqRec> pq;
    std::vector<ScoreRec> score;
    std::vector<KalRec> kal;
    std::vector<WinRec> win;
    std::vector<SelSeed> sel;
  };

  template <typename T>
  inline bool writeVec(FILE* f, const std::vector<T>& v) {
    const int n = v.size();
    return std::fwrite(&n, sizeof(int), 1, f) == 1 && std::fwrite(v.data(), sizeof(T), n, f) == size_t(n);
  }

  template <typename T>
  inline bool readVec(FILE* f, std::vector<T>& v) {
    int n = 0;
    if (std::fread(&n, sizeof(int), 1, f) != 1 || n < 0)
      return false;
    v.resize(n);
    return std::fread(v.data(), sizeof(T), n, f) == size_t(n);
  }

  inline bool writeFile(const std::string& path, const RefData& d) {
    FILE* f = std::fopen(path.c_str(), "wb");
    if (!f)
      return false;
    bool ok = std::fwrite(&d.head, sizeof(d.head), 1, f) == 1 && writeVec(f, d.bbxi) && writeVec(f, d.radl) &&
              writeVec(f, d.pq) && writeVec(f, d.score) && writeVec(f, d.kal) && writeVec(f, d.win) &&
              writeVec(f, d.sel);
    return std::fclose(f) == 0 && ok;
  }

  inline bool readFile(const std::string& path, RefData& d) {
    FILE* f = std::fopen(path.c_str(), "rb");
    if (!f)
      return false;
    bool ok = std::fread(&d.head, sizeof(d.head), 1, f) == 1 && d.head.magic == kMagic && readVec(f, d.bbxi) &&
              readVec(f, d.radl) && readVec(f, d.pq) && readVec(f, d.score) && readVec(f, d.kal) && readVec(f, d.win) &&
              readVec(f, d.sel);
    std::fclose(f);
    return ok;
  }

}  // namespace mkfitdev::v2p2test

#endif
