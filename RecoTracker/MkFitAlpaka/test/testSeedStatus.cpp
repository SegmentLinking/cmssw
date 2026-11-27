// The mkfit::Track::Status bit positions the seed kernels read and write (interface/seeds/SeedSoA.h): every field set
// alone on a Status and copied to 32 bits as packSeeds does must sit at the device shift and mask. Seeds longer than
// Track::Status::kMaxSeedHits keep that count, through packSeeds and the import's status update as through MkFitCore.

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "RecoTracker/MkFitAlpaka/interface/seeds/SeedSoA.h"
#include "RecoTracker/MkFitAlpaka/interface/seeds/SeedsHostPack.h"
#include "RecoTracker/MkFitCore/interface/Track.h"

namespace {

  uint32_t statusBits(mkfit::Track::Status const& status) {
    static_assert(sizeof(status) == sizeof(uint32_t), "packSeeds copies the Status into 32 bits");
    uint32_t bits;
    std::memcpy(&bits, &status, sizeof(bits));
    return bits;
  }

  int nFailed = 0;

  void check(bool good, const char* what) {
    nFailed += !good;
    std::printf("%s %s\n", good ? "ok  " : "FAIL", what);
  }

}  // namespace

int main() {
  using namespace mkfitdev;
  check(statusBits(mkfit::Track::Status()) == 0u, "default Status is all zero");
  {
    mkfit::Track::Status status;
    status.duplicate = true;
    check(statusBits(status) == kStatusDuplicateMask << kStatusDuplicateShift, "duplicate");
  }
  {
    mkfit::Track::Status status;
    status.algorithm = kStatusAlgorithmMask;
    check(statusBits(status) == kStatusAlgorithmMask << kStatusAlgorithmShift, "algorithm");
  }
  {
    mkfit::Track::Status status;
    status.n_seed_hits = kStatusNSeedHitsMask;
    check(statusBits(status) == kStatusNSeedHitsMask << kStatusNSeedHitsShift, "n_seed_hits");
  }
  {
    mkfit::Track::Status status;
    status.eta_region = kStatusEtaRegionMask;
    check(statusBits(status) == kStatusEtaRegionMask << kStatusEtaRegionShift, "eta_region");
  }
  // statusWithSeedHitsAndRegion = Track::setNSeedHits + setEtaRegion, other fields kept
  bool same = true;
  for (uint32_t nSeedHits = 0; nSeedHits <= kStatusNSeedHitsMask; ++nSeedHits) {
    for (uint32_t region = 0; region <= kStatusEtaRegionMask; ++region) {
      mkfit::Track::Status status;
      status.algorithm = 9;
      status.duplicate = true;
      const uint32_t before = statusBits(status);
      status.n_seed_hits = nSeedHits;
      status.eta_region = region;
      same = same && statusWithSeedHitsAndRegion(before, nSeedHits, region) == statusBits(status);
    }
  }
  check(same, "statusWithSeedHitsAndRegion");

  // counts past the field: Track::setNSeedHits caps them at Status::kMaxSeedHits
  same = true;
  for (int nSeedHits = 0; nSeedHits <= 2 * kStatusMaxSeedHits; ++nSeedHits) {
    mkfit::Track track;
    track.setNSeedHits(nSeedHits);
    same = same && nSeedHitsOfStatus(statusWithSeedHitsAndRegion(0u, nSeedHits, 0)) == track.getNSeedHits();
  }
  check(same, "seed-hit count capped as Track::setNSeedHits");

  // packSeeds + the import's status update (SeedKernels: setNSeedHits(nTotalHits()), setEtaRegion) vs
  // MkBuilder::import_seeds, for a seed longer than the old field and one longer than the row capacity
  {
    const int nHitsLong[] = {17, kMaxSeedHits + 3};
    mkfit::TrackVec in;
    for (int nHits : nHitsLong) {
      mkfit::Track track;
      for (int h = 0; h < nHits; ++h)
        track.addHitIdx(h, h / 2, 0.f);
      in.push_back(track);
    }
    const int nRows = in.size();
    const auto bytes = SeedSoA::computeDataSize(nRows);
    void* mem = std::aligned_alloc(SeedSoA::alignment, bytes);
    SeedSoA layout(static_cast<std::byte*>(mem), nRows);
    SeedSoAView seeds(layout);
    packSeeds(in, [](int, int, float* xyz) { xyz[0] = xyz[1] = xyz[2] = 0.f; }, seeds);
    const int region = 3;
    for (int i = 0; i < nRows; ++i) {
      mkfit::Track ref = in[i];
      ref.setNSeedHits(ref.nTotalHits());
      ref.setEtaRegion(region);
      const int nh = seeds[i].nHits();
      const int got = nSeedHitsOfStatus(statusWithSeedHitsAndRegion(seeds[i].status(), nh, region));
      const bool truncated = in[i].nTotalHits() > kMaxSeedHits;
      // a truncated row keeps kMaxSeedHits hits and that count; it is counted in nOverflowHits
      check(got == (truncated ? kMaxSeedHits : ref.getNSeedHits()) && nh == std::min(in[i].nTotalHits(), kMaxSeedHits),
            truncated ? "packSeeds + import, seed longer than the row: truncated" : "packSeeds + import, 17-hit seed");
      if (!truncated)
        check(statusWithSeedHitsAndRegion(seeds[i].status(), nh, region) == statusBits(ref.getStatus()),
              "packSeeds + import, 17-hit seed: all status bits as MkFitCore");
    }
    check(seeds.nOverflowHits() == 1, "packSeeds counts the truncated seed");
    std::free(mem);
  }
  std::printf("%s: %d check(s) failed\n", nFailed ? "FAIL" : "PASS", nFailed);
  return nFailed ? 1 : 0;
}
