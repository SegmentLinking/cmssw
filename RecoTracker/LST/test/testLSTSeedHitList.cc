// Unit test of the seed hit lists of LSTOutputConverter's seeds from the stored states (LSTSeedHitList.h), on
// synthetic hits:
//  - the collected, sorted and T5/T4-subset lists against a copy of the code of the converter's makeSeed mode: its
//    std::sort on up to 16 hits is the libstdc++ insertion sort, which is stable, so the lists must be identical;
//    above 16 hits the reference breaks the ties of the same comparator by the collection order
//  - the sharesInput skips, the T5 and the T4 subsets
//  - the state's hit moved last, an OT state hit skipped as a duplicate of a pixel-track hit, an OT state hit that
//    the candidate carries only through its pixel track (whose copy of it is a distinct rechit on the same cluster),
//    a state hit that the subset drops (no seed) and one that is not among the hits
//  - the pLS hits: the inner-tracker-only prefix and its fallbacks to all the hits

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <map>
#include <numeric>
#include <random>
#include <span>
#include <vector>

#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>

#include "Geometry/CommonTopologies/interface/GeomDetEnumerators.h"
#include "RecoTracker/LSTCore/interface/Common.h"
#include "RecoTracker/LSTCore/interface/SeedStateSoA.h"

#include "RecoTracker/LST/plugins/LSTSeedHitList.h"

namespace {

  using lst::LSTObjType;

  enum class ModuleType { kPixel, kPSP, kPSS, k2S };

  // A synthetic rechit: what the makeSeed mode reads of a rechit.
  struct FakeHit {
    uint32_t cluster;  // hits with the same cluster share input
    GeomDetEnumerators::SubDetector subDetector;
    float rMin;  // rSpan().first of the surface
    float zMin;  // zSpan().first of the surface
    bool pixelSubdetector;
    ModuleType moduleType;
    unsigned int layer;
  };
  using Hit = FakeHit const*;

  struct FakeAccess {
    lst::SeedHitOrderKey orderKey(Hit hit) const { return {hit->subDetector, hit->rMin, std::abs(hit->zMin)}; }
    bool sharesInput(Hit taken, Hit other) const { return taken->cluster == other->cluster; }
    bool inPixelSubdetector(Hit hit) const { return hit->pixelSubdetector; }
    bool isPSPixelSensor(Hit hit) const { return hit->moduleType == ModuleType::kPSP; }
    bool isPSStripSensor(Hit hit) const { return hit->moduleType == ModuleType::kPSS; }
    unsigned int layer(Hit hit) const { return hit->layer; }
  };

  using TCHits = lst::Params_TC::ArrayUxHits;

  TCHits emptyTCHits() {
    TCHits hits;
    for (auto& layer : hits)
      for (auto& hit : layer)
        hit = lst::kTCEmptyHitIdx;
    return hits;
  }

  void setLayer(TCHits& hits, int layerSlot, uint32_t first, uint32_t second) {
    hits[layerSlot][0] = first;
    hits[layerSlot][1] = second;
  }

  bool isT5orT4(LSTObjType type) { return type == LSTObjType::T5 || type == LSTObjType::T4; }

  // The comparator of the makeSeed mode on the synthetic hits.
  bool makeSeedOrder(Hit a, Hit b) {
    const auto asub = a->subDetector;
    const auto bsub = b->subDetector;
    if (GeomDetEnumerators::isInnerTracker(asub) && GeomDetEnumerators::isOuterTracker(bsub)) {
      return true;
    } else if (GeomDetEnumerators::isOuterTracker(asub) && GeomDetEnumerators::isInnerTracker(bsub)) {
      return false;
    } else if (asub != bsub) {
      return asub < bsub;
    } else {
      if (GeomDetEnumerators::isBarrel(asub)) {
        return a->rMin < b->rMin;
      } else {
        return std::abs(a->zMin) < std::abs(b->zMin);
      }
    }
  }

  struct ReferenceList {
    std::vector<Hit> sorted;              // collected and sorted
    std::vector<Hit> forSeed;             // the T5/T4 subset of sorted; sorted for the other types
    std::map<uint32_t, Hit> duplicateOf;  // an OT slot hit skipped as a duplicate: the hit taken first it shares with
    bool tie = false;                     // two collected hits compare equal
  };

  // The hit lists of the makeSeed mode (LSTOutputConverter.cc, produce), copied with the rechits replaced by the
  // synthetic hits; breakTies orders hits that compare equal by their collection order.
  ReferenceList makeSeedModeHits(LSTObjType iType,
                                 std::vector<Hit> const& pixelSeedHits,
                                 TCHits const& hitIndices,
                                 std::vector<FakeHit> const& otHits,
                                 bool breakTies) {
    ReferenceList list;
    std::vector<Hit> recHits;
    if (!isT5orT4(iType))
      for (auto const& hit : pixelSeedHits)
        recHits.push_back(hit);
    for (unsigned int layerSlot = lst::Params_TC::kPixelLayerSlots; layerSlot < lst::Params_TC::kLayers; ++layerSlot) {
      for (unsigned int hitSlot = 0; hitSlot < lst::Params_TC::kHitsPerLayer; ++hitSlot) {
        unsigned int hitIdx = hitIndices[layerSlot][hitSlot];
        if (hitIdx == lst::kTCEmptyHitIdx)
          continue;
        bool hitOK = true;
        for (auto const& hit : recHits)
          if (hit->cluster == otHits[hitIdx].cluster) {
            hitOK = false;
            list.duplicateOf.emplace(hitIdx, hit);
            break;
          }
        if (hitOK)
          recHits.push_back(&otHits[hitIdx]);
      }
    }
    for (std::size_t i = 0; i < recHits.size(); ++i)
      for (std::size_t j = i + 1; j < recHits.size(); ++j)
        if (!makeSeedOrder(recHits[i], recHits[j]) && !makeSeedOrder(recHits[j], recHits[i]))
          list.tie = true;

    if (breakTies) {
      std::vector<std::size_t> order(recHits.size());
      std::iota(order.begin(), order.end(), 0);
      std::sort(order.begin(), order.end(), [&recHits](std::size_t a, std::size_t b) {
        if (makeSeedOrder(recHits[a], recHits[b]))
          return true;
        if (makeSeedOrder(recHits[b], recHits[a]))
          return false;
        return a < b;
      });
      for (auto i : order)
        list.sorted.push_back(recHits[i]);
    } else {
      std::sort(recHits.begin(), recHits.end(), makeSeedOrder);
      list.sorted = recHits;
    }

    int n = 0;
    unsigned int firstLayer = 0;
    for (auto const& hit : list.sorted) {
      if (iType == LSTObjType::T5) {
        auto hType = hit->moduleType;
        if (hType != ModuleType::kPSP && n < 2)
          continue;  // the first two should be P
      }
      if (iType == LSTObjType::T4) {
        unsigned int hitLayer = hit->layer;
        auto hType = hit->moduleType;
        if (n == 0)
          firstLayer = hitLayer;
        else {
          if (hType == ModuleType::kPSS && hitLayer == firstLayer)
            continue;
        }
      }
      list.forSeed.push_back(hit);
      n++;
    }
    return list;
  }

  std::vector<lst::OTSlotHit<Hit>> otSlotHits(TCHits const& hitIndices, std::vector<FakeHit> const& otHits) {
    std::vector<lst::OTSlotHit<Hit>> slotHits;
    for (int layerSlot = lst::Params_TC::kPixelLayerSlots; layerSlot < lst::Params_TC::kLayers; ++layerSlot)
      for (int hitSlot = 0; hitSlot < lst::Params_TC::kHitsPerLayer; ++hitSlot) {
        const uint32_t hitIdx = hitIndices[layerSlot][hitSlot];
        if (hitIdx != lst::kTCEmptyHitIdx)
          slotHits.push_back({&otHits[hitIdx], hitIdx});
      }
    return slotHits;
  }

  // The builder's list before the state's hit is moved: collected, sorted and the T5/T4 subset.
  std::vector<lst::SeedHitEntry<Hit>> builderEntries(LSTObjType type,
                                                     std::vector<Hit> const& pixelHits,
                                                     std::vector<lst::OTSlotHit<Hit>> const& slotHits,
                                                     bool subset) {
    const FakeAccess access;
    std::vector<lst::SeedHitEntry<Hit>> entries;
    lst::collectTCSeedHits<Hit>(type, pixelHits, slotHits, lst::kInvalidU32Idx, nullptr, access, entries);
    lst::sortSeedHits(entries);
    if (subset)
      lst::selectT5T4SeedHits(type, access, entries);
    return entries;
  }

  std::vector<Hit> hitsOf(std::vector<lst::SeedHitEntry<Hit>> const& entries) {
    std::vector<Hit> hits;
    hits.reserve(entries.size());
    for (auto const& entry : entries)
      hits.push_back(entry.hit);
    return hits;
  }

  struct TCListResult {
    lst::SeedHitListResult result;
    std::vector<Hit> hits;
  };

  // the list of a state on stateHit; an OT stateHit is the hit of otHits at its index, as in the converter
  TCListResult tcList(LSTObjType type,
                      std::vector<Hit> const& pixelHits,
                      std::vector<lst::OTSlotHit<Hit>> const& slotHits,
                      std::vector<FakeHit> const& otHits,
                      uint32_t stateHit) {
    TCListResult out;
    std::vector<lst::SeedHitEntry<Hit>> entries;
    const Hit stateOTHit = lst::stateHitIsPixel(stateHit) ? nullptr : &otHits.at(lst::stateHitIndex(stateHit));
    out.result =
        lst::tcSeedHitList<Hit>(type, pixelHits, slotHits, stateHit, stateOTHit, FakeAccess(), entries, out.hits);
    return out;
  }

  // the list with hit moved last
  std::vector<Hit> movedLast(std::vector<Hit> list, Hit hit) {
    auto const it = std::find(list.begin(), list.end(), hit);
    REQUIRE(it != list.end());
    std::rotate(it, it + 1, list.end());
    return list;
  }

  FakeHit pixelHit(uint32_t cluster, float r, unsigned int layer = 1) {
    return {cluster, GeomDetEnumerators::P2PXB, r, -10.f, true, ModuleType::kPixel, layer};
  }
  FakeHit barrelOTHit(uint32_t cluster, float r, ModuleType type, unsigned int layer) {
    return {cluster, GeomDetEnumerators::P2OTB, r, -50.f, false, type, layer};
  }

  constexpr std::array kTypes{LSTObjType::pT3, LSTObjType::pT5, LSTObjType::T5, LSTObjType::T4};

}  // namespace

TEST_CASE("Seed hit lists against the makeSeed mode on random hits", "[LSTSeedHitList]") {
  std::mt19937 rng(20261021);
  auto uniform = [&rng](int lo, int hi) { return std::uniform_int_distribution<int>(lo, hi)(rng); };
  // few distinct values, for ties; the subdetectors include both groups and more than one per group
  const std::array subDetectorsIT{
      GeomDetEnumerators::P2PXB, GeomDetEnumerators::P2PXEC, GeomDetEnumerators::PixelBarrel};
  const std::array subDetectorsOT{GeomDetEnumerators::P2OTB, GeomDetEnumerators::P2OTEC, GeomDetEnumerators::TOB};
  const std::array spans{5.f, 10.f, 20.f, 40.f, 60.f, 90.f};
  const std::array moduleTypes{ModuleType::kPSP, ModuleType::kPSS, ModuleType::k2S};
  auto randomHit = [&](bool inner, uint32_t nClusters) {
    const auto sub = inner ? subDetectorsIT[uniform(0, 2)] : subDetectorsOT[uniform(0, 2)];
    return FakeHit{static_cast<uint32_t>(uniform(0, nClusters - 1)),
                   sub,
                   spans[uniform(0, spans.size() - 1)],
                   (uniform(0, 1) ? 1.f : -1.f) * spans[uniform(0, spans.size() - 1)],
                   inner,
                   inner ? ModuleType::kPixel : moduleTypes[uniform(0, 2)],
                   static_cast<unsigned int>(uniform(1, 4))};
  };

  constexpr int kTrials = 40000;
  constexpr int kMaxTrackOTHits = 4;
  int nShort = 0, nLong = 0, nShortTie = 0, nLongTie = 0, nSkipped = 0, nSubsetDropped = 0, nMoved = 0;
  int nDuplicateStates = 0, nDroppedStates = 0, nPixelCopyStates = 0;
  for (int trial = 0; trial < kTrials; ++trial) {
    const LSTObjType type = kTypes[trial % kTypes.size()];
    const auto nClusters = static_cast<uint32_t>(uniform(8, 60));
    std::vector<FakeHit> otHits;
    const int nOT = uniform(1, 40);
    otHits.reserve(nOT + kMaxTrackOTHits);
    for (int i = 0; i < nOT; ++i)
      otHits.push_back(randomHit(false, nClusters));
    // a pixel track: inner-tracker hits, then possibly OT hits
    std::vector<FakeHit> pixelTrack;
    const int nIT = uniform(2, 10);
    const int nTrackOT = uniform(0, kMaxTrackOTHits);
    pixelTrack.reserve(nIT + nTrackOT);
    for (int i = 0; i < nIT + nTrackOT; ++i)
      pixelTrack.push_back(randomHit(i < nIT, nClusters));
    std::vector<Hit> pixelHits;
    pixelHits.reserve(pixelTrack.size());
    for (auto const& hit : pixelTrack)
      pixelHits.push_back(&hit);
    // the OT hits of the pixel track as OT hits of their own, distinct rechits on the same clusters that no OT slot
    // holds: the pixel track's copy is the only one in the list of a pT3 or a pT5
    for (int i = nIT; i < nIT + nTrackOT; ++i)
      otHits.push_back(pixelTrack[i]);
    TCHits hitIndices = emptyTCHits();
    const int nLayers = uniform(0, lst::Params_TC::kLayers - lst::Params_TC::kPixelLayerSlots);
    for (int layerSlot = lst::Params_TC::kPixelLayerSlots; layerSlot < lst::Params_TC::kPixelLayerSlots + nLayers;
         ++layerSlot)
      for (int hitSlot = 0; hitSlot < lst::Params_TC::kHitsPerLayer; ++hitSlot)
        if (hitSlot == 0 || uniform(0, 1))
          hitIndices[layerSlot][hitSlot] = uniform(0, nOT - 1);
    auto const slotHits = otSlotHits(hitIndices, otHits);

    auto const sorted = builderEntries(type, pixelHits, slotHits, false);
    auto const entries = builderEntries(type, pixelHits, slotHits, true);
    // every entry's origin addresses its hit
    for (auto const& entry : sorted) {
      const auto decoded = lst::decodeStateHit(entry.origin, otHits.size(), pixelHits.size());
      if (decoded.kind == lst::StateHitKind::kPixel)
        REQUIRE(pixelHits[decoded.index] == entry.hit);
      else
        REQUIRE((decoded.kind == lst::StateHitKind::kOT && &otHits[decoded.index] == entry.hit));
    }

    const bool isShort = sorted.size() <= 16;
    auto const reference = makeSeedModeHits(type, pixelHits, hitIndices, otHits, !isShort);
    REQUIRE(hitsOf(sorted) == reference.sorted);
    REQUIRE(hitsOf(entries) == reference.forSeed);
    // below the threshold, the comparator with the collection order as tie-break gives the same lists
    if (isShort)
      REQUIRE(makeSeedModeHits(type, pixelHits, hitIndices, otHits, true).forSeed == reference.forSeed);
    (isShort ? nShort : nLong) += 1;
    if (reference.tie)
      (isShort ? nShortTie : nLongTie) += 1;
    nSkipped += !reference.duplicateOf.empty();
    nSubsetDropped += reference.forSeed.size() < reference.sorted.size();

    if (entries.empty())
      continue;
    // a hit of the list as the state's hit: moved last, the others in their order
    auto const& state = entries[uniform(0, entries.size() - 1)];
    auto const moved = tcList(type, pixelHits, slotHits, otHits, state.origin);
    REQUIRE(moved.result.status == lst::SeedHitListStatus::kOk);
    REQUIRE(moved.result.stateHitMoved == (state.hit != entries.back().hit));
    REQUIRE(moved.hits == movedLast(reference.forSeed, state.hit));
    nMoved += moved.result.stateHitMoved;
    // an OT slot hit skipped as a duplicate: the hit it duplicates is the state's hit
    for (auto const& [otIndex, duplicated] : reference.duplicateOf) {
      auto const duplicate = tcList(type, pixelHits, slotHits, otHits, lst::encodeStateHitOT(otIndex));
      const bool kept =
          std::find(reference.forSeed.begin(), reference.forSeed.end(), duplicated) != reference.forSeed.end();
      if (kept) {
        REQUIRE(duplicate.result.status == lst::SeedHitListStatus::kOk);
        REQUIRE(duplicate.hits == movedLast(reference.forSeed, duplicated));
        nDuplicateStates += std::find(pixelHits.begin(), pixelHits.end(), duplicated) != pixelHits.end();
      } else {
        REQUIRE(duplicate.result.status == lst::SeedHitListStatus::kStateHitDroppedBySubset);
      }
    }
    // a hit that the subset drops: no seed
    for (auto const& entry : sorted)
      if (std::find(reference.forSeed.begin(), reference.forSeed.end(), entry.hit) == reference.forSeed.end()) {
        REQUIRE(tcList(type, pixelHits, slotHits, otHits, entry.origin).result.status ==
                lst::SeedHitListStatus::kStateHitDroppedBySubset);
        ++nDroppedStates;
      }
    // an OT hit of the pixel track as the state's hit: the first of the candidate's hits (the pixel track's, then the
    // OT slots') on its cluster
    for (uint32_t otIndex = nOT; otIndex < otHits.size(); ++otIndex) {
      std::vector<Hit> candidateHits;
      candidateHits.reserve(pixelHits.size() + slotHits.size());
      if (!isT5orT4(type))
        candidateHits.insert(candidateHits.end(), pixelHits.begin(), pixelHits.end());
      for (auto const& slotHit : slotHits)
        candidateHits.push_back(slotHit.hit);
      auto const first = std::find_if(
          candidateHits.begin(), candidateHits.end(), [&](Hit hit) { return hit->cluster == otHits[otIndex].cluster; });
      auto const copy = tcList(type, pixelHits, slotHits, otHits, lst::encodeStateHitOT(otIndex));
      if (first == candidateHits.end()) {
        REQUIRE(copy.result.status == lst::SeedHitListStatus::kStateHitNotInList);
      } else if (std::find(reference.forSeed.begin(), reference.forSeed.end(), *first) == reference.forSeed.end()) {
        REQUIRE(copy.result.status == lst::SeedHitListStatus::kStateHitDroppedBySubset);
      } else {
        REQUIRE(copy.result.status == lst::SeedHitListStatus::kOk);
        REQUIRE(copy.hits == movedLast(reference.forSeed, *first));
      }
      nPixelCopyStates += !isT5orT4(type);
    }
  }
  std::cout << kTrials << " random lists: " << nShort << " of at most 16 hits (" << nShortTie << " with ties), "
            << nLong << " longer (" << nLongTie << " with ties); " << nSkipped << " with sharesInput skips, "
            << nSubsetDropped << " with subset drops, " << nMoved << " state hits moved, " << nDuplicateStates
            << " states on a skipped duplicate, " << nDroppedStates << " states dropped by the subset, "
            << nPixelCopyStates << " states on an OT hit of the pixel track that no OT slot holds" << std::endl;
  REQUIRE(nShortTie > 1000);
  REQUIRE(nLongTie > 1000);
  REQUIRE(nSkipped > 1000);
  REQUIRE(nSubsetDropped > 1000);
  REQUIRE(nDuplicateStates > 1000);
  REQUIRE(nDroppedStates > 1000);
  REQUIRE(nPixelCopyStates > 1000);
}

TEST_CASE("sharesInput skips", "[LSTSeedHitList]") {
  // the pixel track: two inner-tracker hits and an OT hit on cluster 5
  const std::vector<FakeHit> track{pixelHit(1, 3.f), pixelHit(2, 6.f), barrelOTHit(5, 25.f, ModuleType::kPSP, 1)};
  const std::vector<Hit> pixelHits{&track[0], &track[1], &track[2]};
  // OT slot hits: a duplicate of the track's OT hit, a new hit, a duplicate of that new hit, another new hit
  const std::vector<FakeHit> otHits{barrelOTHit(5, 25.f, ModuleType::kPSP, 1),
                                    barrelOTHit(6, 35.f, ModuleType::kPSP, 2),
                                    barrelOTHit(6, 35.f, ModuleType::kPSS, 2),
                                    barrelOTHit(7, 50.f, ModuleType::k2S, 3)};
  TCHits hitIndices = emptyTCHits();
  setLayer(hitIndices, 2, 0, 1);
  setLayer(hitIndices, 3, 2, 3);
  auto const slotHits = otSlotHits(hitIndices, otHits);

  SECTION("pT5: the duplicates of the track's hits and of earlier OT hits are skipped") {
    const std::vector<Hit> expected{&track[0], &track[1], &track[2], &otHits[1], &otHits[3]};
    REQUIRE(hitsOf(builderEntries(LSTObjType::pT5, pixelHits, slotHits, true)) == expected);
  }
  SECTION("T4: no pixel track, so only the duplicate of the earlier OT hit is skipped") {
    const std::vector<Hit> expected{&otHits[0], &otHits[1], &otHits[3]};
    REQUIRE(hitsOf(builderEntries(LSTObjType::T4, pixelHits, slotHits, true)) == expected);
  }
}

TEST_CASE("T5 and T4 subsets", "[LSTSeedHitList]") {
  SECTION("T5: leading hits skipped until two kept hits are on PS pixel sensors") {
    const std::vector<FakeHit> otHits{barrelOTHit(1, 20.f, ModuleType::k2S, 1),
                                      barrelOTHit(2, 25.f, ModuleType::kPSS, 1),
                                      barrelOTHit(3, 30.f, ModuleType::kPSP, 2),
                                      barrelOTHit(4, 35.f, ModuleType::k2S, 3),
                                      barrelOTHit(5, 40.f, ModuleType::kPSP, 4),
                                      barrelOTHit(6, 45.f, ModuleType::k2S, 5)};
    TCHits hitIndices = emptyTCHits();
    for (uint32_t i = 0; i < otHits.size(); ++i)
      hitIndices[2 + i][0] = i;
    auto const slotHits = otSlotHits(hitIndices, otHits);
    const std::vector<Hit> expected{&otHits[2], &otHits[4], &otHits[5]};
    REQUIRE(hitsOf(builderEntries(LSTObjType::T5, {}, slotHits, true)) == expected);
    // as a T4: the first hit kept, and the PS strip-sensor hit on its layer dropped
    REQUIRE(hitsOf(builderEntries(LSTObjType::T4, {}, slotHits, true)) ==
            std::vector<Hit>{&otHits[0], &otHits[2], &otHits[3], &otHits[4], &otHits[5]});
    // a pT5 has no subset
    REQUIRE(builderEntries(LSTObjType::pT5, {}, slotHits, true).size() == otHits.size());
  }
  SECTION("T4: the PS strip-sensor hits on the layer of the first hit are dropped") {
    const std::vector<FakeHit> otHits{barrelOTHit(1, 20.f, ModuleType::kPSS, 1),
                                      barrelOTHit(2, 21.f, ModuleType::kPSP, 1),
                                      barrelOTHit(3, 22.f, ModuleType::kPSS, 1),
                                      barrelOTHit(4, 23.f, ModuleType::k2S, 1),
                                      barrelOTHit(5, 30.f, ModuleType::kPSS, 2)};
    TCHits hitIndices = emptyTCHits();
    for (uint32_t i = 0; i < otHits.size(); ++i)
      hitIndices[2 + i][0] = i;
    auto const slotHits = otSlotHits(hitIndices, otHits);
    const std::vector<Hit> expected{&otHits[0], &otHits[1], &otHits[3], &otHits[4]};
    REQUIRE(hitsOf(builderEntries(LSTObjType::T4, {}, slotHits, true)) == expected);
  }
}

TEST_CASE("The state's hit", "[LSTSeedHitList]") {
  const std::vector<FakeHit> track{pixelHit(1, 3.f), pixelHit(2, 6.f), barrelOTHit(5, 25.f, ModuleType::kPSP, 1)};
  const std::vector<Hit> pixelHits{&track[0], &track[1], &track[2]};
  // OT hits 0-3 in the OT slots; OT hit 4 on a cluster of none of the candidate's hits
  const std::vector<FakeHit> otHits{barrelOTHit(5, 25.f, ModuleType::kPSP, 1),
                                    barrelOTHit(6, 35.f, ModuleType::kPSP, 2),
                                    barrelOTHit(7, 45.f, ModuleType::k2S, 3),
                                    barrelOTHit(8, 55.f, ModuleType::k2S, 4),
                                    barrelOTHit(9, 65.f, ModuleType::k2S, 5)};
  TCHits hitIndices = emptyTCHits();
  setLayer(hitIndices, 2, 0, 1);
  setLayer(hitIndices, 3, 2, 3);
  auto const slotHits = otSlotHits(hitIndices, otHits);

  SECTION("already last: not moved") {
    auto const list = tcList(LSTObjType::pT5, pixelHits, slotHits, otHits, lst::encodeStateHitOT(3));
    REQUIRE(list.result.status == lst::SeedHitListStatus::kOk);
    REQUIRE_FALSE(list.result.stateHitMoved);
    REQUIRE(list.hits == std::vector<Hit>{&track[0], &track[1], &track[2], &otHits[1], &otHits[2], &otHits[3]});
  }
  SECTION("an OT hit before the last: moved last") {
    auto const list = tcList(LSTObjType::pT5, pixelHits, slotHits, otHits, lst::encodeStateHitOT(1));
    REQUIRE(list.result.status == lst::SeedHitListStatus::kOk);
    REQUIRE(list.result.stateHitMoved);
    REQUIRE(list.hits == std::vector<Hit>{&track[0], &track[1], &track[2], &otHits[2], &otHits[3], &otHits[1]});
  }
  SECTION("a pixel-track hit: moved last") {
    auto const list = tcList(LSTObjType::pT3, pixelHits, slotHits, otHits, lst::encodeStateHitPixel(1));
    REQUIRE(list.result.status == lst::SeedHitListStatus::kOk);
    REQUIRE(list.result.stateHitMoved);
    REQUIRE(list.hits == std::vector<Hit>{&track[0], &track[2], &otHits[1], &otHits[2], &otHits[3], &track[1]});
  }
  SECTION("an OT hit skipped as a duplicate of a pixel-track hit: that pixel-track copy") {
    auto const list = tcList(LSTObjType::pT5, pixelHits, slotHits, otHits, lst::encodeStateHitOT(0));
    REQUIRE(list.result.status == lst::SeedHitListStatus::kOk);
    REQUIRE(list.result.stateHitMoved);
    REQUIRE(list.hits == std::vector<Hit>{&track[0], &track[1], &otHits[1], &otHits[2], &otHits[3], &track[2]});
    // a T5 has no pixel track: the same OT hit is in its list
    auto const t5 = tcList(LSTObjType::T5, pixelHits, slotHits, otHits, lst::encodeStateHitOT(0));
    REQUIRE(t5.result.status == lst::SeedHitListStatus::kOk);
    REQUIRE(t5.hits == std::vector<Hit>{&otHits[1], &otHits[2], &otHits[3], &otHits[0]});
  }
  SECTION("an OT hit that the candidate carries only through its pixel track: the pixel track's copy") {
    // OT hit 0 is in no OT slot; the pixel track's copy of it (track[2], cluster 5) is a distinct rechit
    TCHits withoutHit0 = emptyTCHits();
    setLayer(withoutHit0, 2, 1, lst::kTCEmptyHitIdx);
    setLayer(withoutHit0, 3, 2, 3);
    auto const slots = otSlotHits(withoutHit0, otHits);
    auto const pT5 = tcList(LSTObjType::pT5, pixelHits, slots, otHits, lst::encodeStateHitOT(0));
    REQUIRE(pT5.result.status == lst::SeedHitListStatus::kOk);
    REQUIRE(pT5.result.stateHitMoved);
    REQUIRE(pT5.hits == std::vector<Hit>{&track[0], &track[1], &otHits[1], &otHits[2], &otHits[3], &track[2]});
    TCHits pT3Indices = emptyTCHits();
    setLayer(pT3Indices, 2, 1, lst::kTCEmptyHitIdx);
    setLayer(pT3Indices, 3, 2, lst::kTCEmptyHitIdx);
    auto const pT3 =
        tcList(LSTObjType::pT3, pixelHits, otSlotHits(pT3Indices, otHits), otHits, lst::encodeStateHitOT(0));
    REQUIRE(pT3.result.status == lst::SeedHitListStatus::kOk);
    REQUIRE(pT3.result.stateHitMoved);
    REQUIRE(pT3.hits == std::vector<Hit>{&track[0], &track[1], &otHits[1], &otHits[2], &track[2]});
    // the pixel track's OT hit last in the list: not moved
    const std::vector<FakeHit> outerTrack{pixelHit(1, 3.f), pixelHit(2, 6.f), barrelOTHit(5, 70.f, ModuleType::k2S, 6)};
    const std::vector<FakeHit> outerOTHits{barrelOTHit(5, 70.f, ModuleType::k2S, 6)};
    const std::vector<Hit> outerPixelHits{&outerTrack[0], &outerTrack[1], &outerTrack[2]};
    auto const last = tcList(LSTObjType::pT3, outerPixelHits, slots, outerOTHits, lst::encodeStateHitOT(0));
    REQUIRE(last.result.status == lst::SeedHitListStatus::kOk);
    REQUIRE_FALSE(last.result.stateHitMoved);
    REQUIRE(last.hits ==
            std::vector<Hit>{&outerTrack[0], &outerTrack[1], &otHits[1], &otHits[2], &otHits[3], &outerTrack[2]});
    // a T5 has no pixel track: the hit is not among its hits
    REQUIRE(tcList(LSTObjType::T5, pixelHits, slots, otHits, lst::encodeStateHitOT(0)).result.status ==
            lst::SeedHitListStatus::kStateHitNotInList);
  }
  SECTION("a T5 whose subset drops the state's hit: no seed") {
    const std::vector<FakeHit> t5Hits{barrelOTHit(1, 20.f, ModuleType::k2S, 1),
                                      barrelOTHit(2, 30.f, ModuleType::kPSP, 2),
                                      barrelOTHit(3, 40.f, ModuleType::kPSP, 3),
                                      barrelOTHit(4, 50.f, ModuleType::k2S, 4)};
    TCHits t5Indices = emptyTCHits();
    for (uint32_t i = 0; i < t5Hits.size(); ++i)
      t5Indices[2 + i][0] = i;
    auto const t5Slots = otSlotHits(t5Indices, t5Hits);
    auto const dropped = tcList(LSTObjType::T5, {}, t5Slots, t5Hits, lst::encodeStateHitOT(0));
    REQUIRE(dropped.result.status == lst::SeedHitListStatus::kStateHitDroppedBySubset);
    REQUIRE(dropped.hits.empty());
    auto const kept = tcList(LSTObjType::T5, {}, t5Slots, t5Hits, lst::encodeStateHitOT(2));
    REQUIRE(kept.result.status == lst::SeedHitListStatus::kOk);
    REQUIRE(kept.hits == std::vector<Hit>{&t5Hits[1], &t5Hits[3], &t5Hits[2]});
  }
  SECTION("a hit that is not among the candidate's hits") {
    // an OT hit that shares input with none of the candidate's hits
    REQUIRE(tcList(LSTObjType::pT5, pixelHits, slotHits, otHits, lst::encodeStateHitOT(4)).result.status ==
            lst::SeedHitListStatus::kStateHitNotInList);
    REQUIRE(tcList(LSTObjType::pT3, pixelHits, slotHits, otHits, lst::encodeStateHitPixel(3)).result.status ==
            lst::SeedHitListStatus::kStateHitNotInList);
    // a T5 has no pixel-track hits
    REQUIRE(tcList(LSTObjType::T5, pixelHits, slotHits, otHits, lst::encodeStateHitPixel(0)).result.status ==
            lst::SeedHitListStatus::kStateHitNotInList);
    REQUIRE(tcList(LSTObjType::T4, pixelHits, slotHits, otHits, lst::kInvalidU32Idx).result.status ==
            lst::SeedHitListStatus::kStateHitNotInList);
  }
}

TEST_CASE("pLS hits: the inner-tracker-only prefix and its fallbacks", "[LSTSeedHitList]") {
  const FakeAccess access;
  auto pls = [&access](std::vector<FakeHit> const& track,
                       bool dropOTHits,
                       int maxITHits,
                       bool itStateOk,
                       uint32_t itStateHit,
                       uint32_t nITHits) {
    std::vector<Hit> hits;
    hits.reserve(track.size());
    for (auto const& hit : track)
      hits.push_back(&hit);
    return lst::plsSeedHits<Hit>(hits, dropOTHits, maxITHits, itStateOk, itStateHit, nITHits, access);
  };
  auto check = [](lst::PLSSeedHits const& result, bool rebuild, bool itOnly, uint32_t nHits) {
    REQUIRE(result.rebuildWithoutOT == rebuild);
    REQUIRE(result.itOnly == itOnly);
    REQUIRE(result.nHits == nHits);
  };
  auto const it = [](uint32_t c) { return pixelHit(c, 3.f + c); };
  auto const ot = [](uint32_t c) { return barrelOTHit(c, 25.f + c, ModuleType::kPSP, 1); };
  const std::vector<FakeHit> track{it(0), it(1), it(2), ot(3), ot(4)};
  const uint32_t lastIT = lst::encodeStateHitPixel(2);

  SECTION("the condition holds and the inner-tracker-only state is consistent: its state and the first hits") {
    check(pls(track, true, 3, true, lastIT, 3), true, true, 3);
  }
  SECTION("the condition does not hold: the full state and all the hits") {
    check(pls(track, false, 3, true, lastIT, 3), false, false, 5);
    check(pls(track, true, 2, true, lastIT, 3), false, false, 5);
    check(pls({it(0), it(1), it(2), ot(3), it(4)}, true, 3, true, lastIT, 3), false, false, 5);
    check(pls({it(0), it(1), it(2), it(3), ot(4)}, true, 3, true, lastIT, 4), false, false, 5);
    check(pls({}, true, 3, true, lastIT, 0), false, false, 0);
  }
  SECTION("the condition holds but the inner-tracker-only state is not usable: the full state and all the hits") {
    // status not ok
    check(pls(track, true, 3, false, lastIT, 3), true, false, 5);
    // not on the last inner-tracker hit
    check(pls(track, true, 3, true, lst::encodeStateHitPixel(1), 3), true, false, 5);
    check(pls(track, true, 3, true, lst::encodeStateHitPixel(3), 3), true, false, 5);
    check(pls(track, true, 3, true, lst::encodeStateHitOT(2), 3), true, false, 5);
    check(pls(track, true, 3, true, lst::kInvalidU32Idx, 3), true, false, 5);
    // nITHits not the number of inner-tracker hits of the track
    check(pls(track, true, 3, true, lst::encodeStateHitPixel(1), 2), true, false, 5);
    check(pls(track, true, 3, true, lastIT, 0), true, false, 5);
    check(pls(track, true, 3, true, lst::encodeStateHitPixel(5), 6), true, false, 5);
    // the inner-tracker hits are not the first hits of the track
    const std::vector<FakeHit> mixed{it(0), it(1), ot(2), it(3), ot(4)};
    check(pls(mixed, true, 3, true, lst::encodeStateHitPixel(3), 3), true, false, 5);
    check(pls(mixed, true, 3, true, lst::encodeStateHitPixel(2), 3), true, false, 5);
  }
  SECTION("two inner-tracker hits: as the inner-tracker-only state says") {
    const std::vector<FakeHit> short2{it(0), it(1), ot(2), ot(3)};
    check(pls(short2, true, 3, true, lst::encodeStateHitPixel(1), 2), true, true, 2);
    check(pls(short2, true, 3, false, lst::encodeStateHitPixel(1), 2), true, false, 4);
  }
}
