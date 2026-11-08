#ifndef RecoTracker_LST_plugins_LSTSeedHitList_h
#define RecoTracker_LST_plugins_LSTSeedHitList_h

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <span>
#include <type_traits>
#include <vector>

#include "Geometry/CommonTopologies/interface/GeomDetEnumerators.h"
#include "RecoTracker/LSTCore/interface/Common.h"
#include "RecoTracker/LSTCore/interface/SeedStateSoA.h"

// The hit lists of the seeds that LSTOutputConverter builds from the stored Broken Line Fit states: the lists its
// makeSeed mode builds, with its hit order applied by a stable sort (hits that compare equal keep their order), then
// the hit the state sits on put last, where a seed's state must be.
//
// A hit is a handle of type Hit (TrackingRecHit const* in the converter), read through an accessor with
//   lst::SeedHitOrderKey orderKey(Hit) const;          the subdetector and the spans of the hit's surface
//   bool sharesInput(Hit taken, Hit other) const;      taken->sharesInput(other, TrackingRecHit::all)
//   bool inPixelSubdetector(Hit) const;                geographicalId().subdetId() <= PixelSubdetector::PixelEndcap
//   bool isPSPixelSensor(Hit) const;                   the module type is TrackerGeometry::ModuleType::Ph2PSP
//   bool isPSStripSensor(Hit) const;                   the module type is TrackerGeometry::ModuleType::Ph2PSS
//   unsigned int layer(Hit) const;                     TrackerTopology::layer
namespace lst {

  struct SeedHitOrderKey {
    GeomDetEnumerators::SubDetector subDetector;  // det()->subDetector()
    float rMin;                                   // surface()->rSpan().first
    float absZMin;                                // std::abs(surface()->zSpan().first)
  };

  // The order of the seed hits: the inner tracker first, then by subdetector, then by the surface's rSpan().first in
  // the barrel or |zSpan().first| in the endcaps.
  inline bool seedHitOrder(SeedHitOrderKey const& a, SeedHitOrderKey const& b) {
    if (GeomDetEnumerators::isInnerTracker(a.subDetector) && GeomDetEnumerators::isOuterTracker(b.subDetector))
      return true;
    if (GeomDetEnumerators::isOuterTracker(a.subDetector) && GeomDetEnumerators::isInnerTracker(b.subDetector))
      return false;
    if (a.subDetector != b.subDetector)
      return a.subDetector < b.subDetector;
    if (GeomDetEnumerators::isBarrel(a.subDetector))
      return a.rMin < b.rMin;
    return a.absZMin < b.absZMin;
  }

  // An OT slot hit of a track candidate and its LSTOTHits index.
  template <typename Hit>
  struct OTSlotHit {
    Hit hit;
    uint32_t otIndex;
  };

  // A hit of a seed's list; origin, in the stateHit encoding, is lst::encodeStateHitPixel(k) for hit k of the pixel
  // track and lst::encodeStateHitOT(i) for LSTOTHits hit i.
  template <typename Hit>
  struct SeedHitEntry {
    Hit hit;
    uint32_t origin;
    SeedHitOrderKey key;
  };

  // The hits of a pT3, pT5, T5 or T4 before sorting: for a pT3 or a pT5 the pixel track's hits, then the OT slot hits,
  // in slot order, that share no input with a hit already taken. Returns the origin of the state's hit stateHit: a
  // pixel-track hit is its own origin; an OT hit, whose rechit is stateOTHit, is the first collected hit that shares
  // input with stateOTHit, which is the OT slot hit itself, the hit taken first for an OT slot hit skipped as its
  // duplicate, or the pixel track's own copy of an OT hit that the pixel track carries (a distinct rechit on the same
  // cluster); kInvalidU32Idx if no collected hit shares input with it.
  template <typename Hit, typename Access>
  uint32_t collectTCSeedHits(LSTObjType type,
                             std::type_identity_t<std::span<Hit const>> pixelTrackHits,
                             std::type_identity_t<std::span<OTSlotHit<Hit> const>> otSlotHits,
                             uint32_t stateHit,
                             std::type_identity_t<Hit> stateOTHit,
                             Access const& access,
                             std::vector<SeedHitEntry<Hit>>& entries) {
    entries.clear();
    if (type != LSTObjType::T5 && type != LSTObjType::T4)
      for (uint32_t k = 0; k < pixelTrackHits.size(); ++k)
        entries.push_back({pixelTrackHits[k], encodeStateHitPixel(k), access.orderKey(pixelTrackHits[k])});
    for (auto const& slotHit : otSlotHits) {
      const bool shared = std::any_of(entries.begin(), entries.end(), [&](SeedHitEntry<Hit> const& entry) {
        return access.sharesInput(entry.hit, slotHit.hit);
      });
      if (!shared)
        entries.push_back({slotHit.hit, encodeStateHitOT(slotHit.otIndex), access.orderKey(slotHit.hit)});
    }
    if (stateHitIsPixel(stateHit))
      return stateHit;
    auto const state = std::find_if(entries.begin(), entries.end(), [&](SeedHitEntry<Hit> const& entry) {
      return access.sharesInput(entry.hit, stateOTHit);
    });
    return state == entries.end() ? kInvalidU32Idx : state->origin;
  }

  template <typename Hit>
  void sortSeedHits(std::vector<SeedHitEntry<Hit>>& entries) {
    std::stable_sort(entries.begin(), entries.end(), [](SeedHitEntry<Hit> const& a, SeedHitEntry<Hit> const& b) {
      return seedHitOrder(a.key, b.key);
    });
  }

  // The subset of a sorted T5 or T4 list that the makeSeed mode fits: a T5 skips its leading hits until two kept hits
  // are on PS pixel sensors; a T4 drops the PS strip-sensor hits on the layer of its first hit. Other types: all hits.
  template <typename Hit, typename Access>
  void selectT5T4SeedHits(LSTObjType type, Access const& access, std::vector<SeedHitEntry<Hit>>& entries) {
    if (type != LSTObjType::T5 && type != LSTObjType::T4)
      return;
    std::size_t nKept = 0;
    unsigned int firstLayer = 0;
    for (std::size_t i = 0; i < entries.size(); ++i) {
      Hit const hit = entries[i].hit;
      if (type == LSTObjType::T5) {
        if (!access.isPSPixelSensor(hit) && nKept < 2)
          continue;
      } else {
        const unsigned int layer = access.layer(hit);
        if (nKept == 0)
          firstLayer = layer;
        else if (access.isPSStripSensor(hit) && layer == firstLayer)
          continue;
      }
      entries[nKept++] = entries[i];
    }
    entries.resize(nKept);
  }

  enum class SeedHitListStatus : uint8_t {
    kOk,
    // the state's hit is not among the candidate's hits
    kStateHitNotInList,
    // the T5/T4 subset drops the state's hit: no seed
    kStateHitDroppedBySubset,
  };

  struct SeedHitListResult {
    SeedHitListStatus status;
    bool stateHitMoved;  // the state's hit was not last in the list and was moved there
  };

  // The seed hits of a pT3, pT5, T5 or T4 whose state sits on stateHit (for an OT hit, on the rechit stateOTHit; not
  // read for a pixel-track hit): collected, sorted, the T5/T4 subset, then the state's hit moved last, the others
  // keeping their order. hits is filled for kOk only; entries is a work buffer.
  template <typename Hit, typename Access>
  SeedHitListResult tcSeedHitList(LSTObjType type,
                                  std::type_identity_t<std::span<Hit const>> pixelTrackHits,
                                  std::type_identity_t<std::span<OTSlotHit<Hit> const>> otSlotHits,
                                  uint32_t stateHit,
                                  std::type_identity_t<Hit> stateOTHit,
                                  Access const& access,
                                  std::vector<SeedHitEntry<Hit>>& entries,
                                  std::vector<Hit>& hits) {
    hits.clear();
    const uint32_t stateOrigin =
        collectTCSeedHits<Hit>(type, pixelTrackHits, otSlotHits, stateHit, stateOTHit, access, entries);
    auto const isState = [stateOrigin](SeedHitEntry<Hit> const& entry) { return entry.origin == stateOrigin; };
    if (std::none_of(entries.begin(), entries.end(), isState))
      return {SeedHitListStatus::kStateHitNotInList, false};
    sortSeedHits(entries);
    selectT5T4SeedHits(type, access, entries);
    auto const state = std::find_if(entries.begin(), entries.end(), isState);
    if (state == entries.end())
      return {SeedHitListStatus::kStateHitDroppedBySubset, false};
    const bool moved = state + 1 != entries.end();
    std::rotate(state, state + 1, entries.end());
    hits.reserve(entries.size());
    for (auto const& entry : entries)
      hits.push_back(entry.hit);
    return {SeedHitListStatus::kOk, moved};
  }

  // The condition under which the makeSeed mode rebuilds a pLS seed without its OT hits: the track's last hit is not
  // in the pixel subdetectors and at most maxITHits of its hits are.
  template <typename Hit, typename Access>
  bool dropOTHitsPurePLS(std::type_identity_t<std::span<Hit const>> trackHits, int maxITHits, Access const& access) {
    if (trackHits.empty() || access.inPixelSubdetector(trackHits.back()))
      return false;
    auto const inPixel = [&access](Hit hit) { return access.inPixelSubdetector(hit); };
    return std::count_if(trackHits.begin(), trackHits.end(), inPixel) <= maxITHits;
  }

  struct PLSSeedHits {
    bool rebuildWithoutOT;  // the makeSeed mode would rebuild this seed without its OT hits
    bool itOnly;            // the seed takes the inner-tracker-only state
    uint32_t nHits;         // the seed's hits are the first nHits hits of the track
  };

  // The state and the hits of a pLS seed. When the makeSeed mode would rebuild it without its OT hits (dropOTHits and
  // dropOTHitsPurePLS) and its inner-tracker-only state has status ok (itStateOk), is consistent with the hits (the
  // track's nITHits pixel-subdetector hits are its first nITHits hits) and sits on the last of them (itStateHit), the
  // seed takes that state and those hits; otherwise the full state and all the track's hits.
  template <typename Hit, typename Access>
  PLSSeedHits plsSeedHits(std::type_identity_t<std::span<Hit const>> trackHits,
                          bool dropOTHits,
                          int maxITHits,
                          bool itStateOk,
                          uint32_t itStateHit,
                          uint32_t nITHits,
                          Access const& access) {
    const auto nHits = static_cast<uint32_t>(trackHits.size());
    const bool rebuild = dropOTHits && dropOTHitsPurePLS<Hit>(trackHits, maxITHits, access);
    if (!rebuild || !itStateOk || nITHits == 0 || nITHits > nHits)
      return {rebuild, false, nHits};
    auto const inPixel = [&access](Hit hit) { return access.inPixelSubdetector(hit); };
    const bool prefix = std::all_of(trackHits.begin(), trackHits.begin() + nITHits, inPixel) &&
                        std::none_of(trackHits.begin() + nITHits, trackHits.end(), inPixel);
    const DecodedStateHit decoded = decodeStateHit(itStateHit, 0, nHits);
    if (!prefix || decoded.kind != StateHitKind::kPixel || decoded.index + 1 != nITHits)
      return {true, false, nHits};
    return {true, true, nITHits};
  }

}  // namespace lst

#endif  // RecoTracker_LST_plugins_LSTSeedHitList_h
