#include "TruthPixelSeeds.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <random>

#include "RecoTracker/LSTCore/interface/Common.h"
#include "Trktree.h"
#include "trkCore.h"


std::map<int, std::vector<int>> buildSimTrkToPixHitMap(std::vector<std::vector<int>> const& pix_simHitIdx,
                                                        std::vector<int> const& simhit_simTrkIdx) {
  std::map<int, std::vector<int>> simTrkToPixHit;
  for (size_t ipix = 0; ipix < pix_simHitIdx.size(); ++ipix) {
    for (int simHitIdx : pix_simHitIdx[ipix]) {
      if (simHitIdx < 0 || static_cast<size_t>(simHitIdx) >= simhit_simTrkIdx.size())
        continue;
      int simTrkIdx = simhit_simTrkIdx[simHitIdx];
      if (simTrkIdx < 0)
        continue;
      simTrkToPixHit[simTrkIdx].push_back(static_cast<int>(ipix));
    }
  }
  return simTrkToPixHit;
}

TruthPixelSeedVectors buildTruthPixelSeeds(std::set<int> const& excludeSimTrkIdx) {
  TruthPixelSeedVectors out;

  auto const& sim_pt = trk.getVF("sim_pt");
  auto const& sim_eta = trk.getVF("sim_eta");
  auto const& sim_phi = trk.getVF("sim_phi");
  auto const& sim_q = trk.getVI("sim_q");
  auto const& sim_pca_dxy = trk.getVF("sim_pca_dxy");
  auto const& sim_pca_dz = trk.getVF("sim_pca_dz");

  auto const& pix_x = trk.getVF("pix_x");
  auto const& pix_y = trk.getVF("pix_y");
  auto const& pix_z = trk.getVF("pix_z");
  auto const& pix_subdet = trk.getVUS("pix_subdet");
  auto const& pix_layer = trk.getVUS("pix_layer");
  auto const& pix_side = trk.getVUS("pix_side");
  auto const& pix_simHitIdx = trk.getVVI("pix_simHitIdx");
  auto const& simhit_simTrkIdx = trk.getVI("simhit_simTrkIdx");

  // Random draw of see_ptErr/see_etaErr from the event's real-seed pool.
  // prepareInput()'s pixtype classification and the pLS DNN embedding both take ptErr/etaErr
  // as features and expect realistic non-zero values.
  auto const& ptErrs = trk.getVF("see_ptErr");
  auto const& etaErrs = trk.getVF("see_etaErr");

  std::mt19937 rng(42);
  std::uniform_int_distribution<size_t> ptErrDist(0, ptErrs.empty() ? 0 : ptErrs.size() - 1);
  std::uniform_int_distribution<size_t> etaErrDist(0, etaErrs.empty() ? 0 : etaErrs.size() - 1);

  auto simTrkToPixHit = buildSimTrkToPixHitMap(pix_simHitIdx, simhit_simTrkIdx);

  for (auto const& [simTrkIdx, pixHits] : simTrkToPixHit) {
    if (simTrkIdx < 0 || static_cast<size_t>(simTrkIdx) >= sim_q.size())
      continue;
    if (sim_q[simTrkIdx] == 0)
      continue;
    if (excludeSimTrkIdx.count(simTrkIdx))
      continue;

    // Pick up to 4 of this track's true pixel hits with distinct (subdet, layer, side),
    // sorted by radius ascending -- innermost first, matching how a real seed's hits run
    // from the vertex outward.
    std::vector<int> candidates = pixHits;
    std::sort(candidates.begin(), candidates.end(), [&](int a, int b) {
      float ra = pix_x[a] * pix_x[a] + pix_y[a] * pix_y[a];
      float rb = pix_x[b] * pix_x[b] + pix_y[b] * pix_y[b];
      return ra < rb;
    });

    std::vector<int> chosen;
    std::vector<std::array<unsigned short, 3>> usedLayers;
    for (int ipix : candidates) {
      std::array<unsigned short, 3> key{pix_subdet[ipix], pix_layer[ipix], pix_side[ipix]};
      if (std::find(usedLayers.begin(), usedLayers.end(), key) != usedLayers.end())
        continue;
      usedLayers.push_back(key);
      chosen.push_back(ipix);
      if (chosen.size() == 4)
        break;
    }
    if (chosen.size() < 3)
      continue;

    // True momentum at PCA.
    float pt = sim_pt[simTrkIdx];
    float eta = sim_eta[simTrkIdx];
    float phi = sim_phi[simTrkIdx];
    float px = pt * std::cos(phi);
    float py = pt * std::sin(phi);
    float pz = pt * std::sinh(eta);

    out.see_px.push_back(px);
    out.see_py.push_back(py);
    out.see_pz.push_back(pz);

    out.see_dxy.push_back(sim_pca_dxy[simTrkIdx]);
    out.see_dz.push_back(sim_pca_dz[simTrkIdx]);
    out.see_ptErr.push_back(ptErrs.empty() ? 0.01f : ptErrs[ptErrDist(rng)]);
    out.see_etaErr.push_back(etaErrs.empty() ? 0.001f : etaErrs[etaErrDist(rng)]);
    out.see_q.push_back(sim_q[simTrkIdx]);

    // Outer reference point ("last hit" trajectory state) = the track's own outermost
    // selected true pixel hit, so it genuinely lies on the helix.
    int outerHit = chosen.back();
    out.see_stateTrajGlbX.push_back(pix_x[outerHit]);
    out.see_stateTrajGlbY.push_back(pix_y[outerHit]);
    out.see_stateTrajGlbZ.push_back(pix_z[outerHit]);

    // Propagate momentum from PCA to the outer pixel hit. On a helix in CMS's 3.8 T
    // field, the momentum phi rotates by -q * 2*arcsin(rt / (2R)) from the vertex to
    // transverse radius rt, where 
    // 
    // R [m] = p_T [GeV/c] / (c × 10⁻⁹ × B [T]) = pt * kR1GeVf 
    // 
    // (kR1GeVf ≈ 87.8 cm/GeV for pt in GeV and positions in cm). 
    // Using the PCA direction for see_stateTrajGlbPx/Py introduces a
    // systematic betaIn bias of O(0.1 rad) for low-pT tracks that exceeds the dBeta cut
    // and collapses pT5 efficiency.
    constexpr float kR1GeVf = 1.f / (2.99792458e-3f * 3.8f);
    float R = pt * kR1GeVf;
    float rt_outer = std::hypot(pix_x[outerHit], pix_y[outerHit]);
    float delta_phi = -float(sim_q[simTrkIdx]) * 2.f * std::asin(std::min(rt_outer / (2.f * R), 1.f));
    out.see_stateTrajGlbPx.push_back(pt * std::cos(phi + delta_phi));
    out.see_stateTrajGlbPy.push_back(pt * std::sin(phi + delta_phi));
    out.see_stateTrajGlbPz.push_back(pz);  // pz unchanged on a helix

    std::vector<int> hitIdx(chosen.begin(), chosen.end());
    std::vector<int> hitType(chosen.size(), static_cast<int>(lst::HitType::Pixel));
    out.see_hitIdx.push_back(std::move(hitIdx));
    out.see_hitType.push_back(std::move(hitType));
  }

  return out;
}

std::set<int> findSimTrkIdxsWithRealSeed(std::vector<std::vector<int>> const& see_hitIdx,
                                          std::vector<std::vector<int>> const& see_hitType,
                                          std::vector<int> const& simhit_simTrkIdx,
                                          std::vector<std::vector<int>> const& ph2_simHitIdx,
                                          std::vector<std::vector<int>> const& pix_simHitIdx) {
  std::set<int> matched;
  for (size_t iSeed = 0; iSeed < see_hitIdx.size(); ++iSeed) {
    std::vector<unsigned int> hitidxs(see_hitIdx[iSeed].begin(), see_hitIdx[iSeed].end());
    std::vector<lst::HitType> hittypes;
    hittypes.reserve(see_hitType[iSeed].size());
    for (int t : see_hitType[iSeed])
      hittypes.push_back(static_cast<lst::HitType>(t));

    for (int simTrkIdx : matchedSimTrkIdxs(hitidxs, hittypes, simhit_simTrkIdx, ph2_simHitIdx, pix_simHitIdx))
      matched.insert(simTrkIdx);
  }
  return matched;
}
