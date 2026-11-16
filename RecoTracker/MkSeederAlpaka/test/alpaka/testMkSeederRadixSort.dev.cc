// Unit test of the seeder's global radix sort (plugins/alpaka/MkSeederRadixSort.h) on the backend it is built for:
// the result must be the stable sort by the requested digits (bytes of keyLow, then keyHigh) computed on the host.

#include <algorithm>
#include <cstdint>
#include <iostream>
#include <numeric>
#include <random>
#include <vector>

#include <alpaka/alpaka.hpp>

#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>

#include "FWCore/Utilities/interface/stringize.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

// the implementation is a translation unit of the plugin library, not part of an installed library: compiled in here
#include "RecoTracker/MkSeederAlpaka/plugins/alpaka/MkSeederRadixSort.dev.cc"

using namespace ALPAKA_ACCELERATOR_NAMESPACE;

namespace {

  using GlobalRadixSort = ALPAKA_ACCELERATOR_NAMESPACE::mkseeder::GlobalRadixSort;

  struct KernelLoadKeys {
    ALPAKA_FN_ACC void operator()(
        Acc1D const& acc, GlobalRadixSort::KeysView keys, const uint64_t* hi, const uint64_t* lo, uint32_t n) const {
      for (uint32_t i : cms::alpakatools::uniform_elements(acc, n)) {
        keys.keyHigh()[i] = hi[i];
        keys.keyLow()[i] = lo[i];
        keys.index()[i] = i;
      }
    }
  };

  struct KernelStoreKeys {
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  ::mkseeder::MkSeederSortKeysSoA::ConstView sorted,
                                  uint64_t* lo,
                                  uint32_t* index,
                                  uint32_t n) const {
      for (uint32_t i : cms::alpakatools::uniform_elements(acc, n)) {
        lo[i] = sorted.keyLow()[i];
        index[i] = sorted.index()[i];
      }
    }
  };

  // sorts the keys on the device with the digits, returns the sorted indices; checks the keyLow's too
  std::vector<uint32_t> sortOnDevice(Queue& queue,
                                     std::vector<uint64_t> const& high,
                                     std::vector<uint64_t> const& low,
                                     uint32_t capacity,
                                     std::initializer_list<int> digits) {
    const uint32_t n = high.size();
    const size_t nbuf = std::max<size_t>(n, 1);
    auto hiH = cms::alpakatools::make_host_buffer<uint64_t[]>(nbuf);
    auto loH = cms::alpakatools::make_host_buffer<uint64_t[]>(nbuf);
    auto outLoH = cms::alpakatools::make_host_buffer<uint64_t[]>(nbuf);
    auto outIdxH = cms::alpakatools::make_host_buffer<uint32_t[]>(nbuf);
    auto countH = cms::alpakatools::make_host_buffer<uint32_t>();
    for (uint32_t i = 0; i < n; ++i)
      hiH[i] = high[i], loH[i] = low[i];
    *countH = n;
    auto hiD = cms::alpakatools::make_device_buffer<uint64_t[]>(queue, nbuf);
    auto loD = cms::alpakatools::make_device_buffer<uint64_t[]>(queue, nbuf);
    auto outLoD = cms::alpakatools::make_device_buffer<uint64_t[]>(queue, nbuf);
    auto outIdxD = cms::alpakatools::make_device_buffer<uint32_t[]>(queue, nbuf);
    auto countD = cms::alpakatools::make_device_buffer<uint32_t>(queue);
    alpaka::memcpy(queue, hiD, hiH);
    alpaka::memcpy(queue, loD, loH);
    alpaka::memcpy(queue, countD, countH);

    GlobalRadixSort sorter(queue, capacity);
    const auto wd = cms::alpakatools::make_workdiv<Acc1D>(cms::alpakatools::divide_up_by(nbuf, 64u), 64u);
    alpaka::exec<Acc1D>(queue, wd, KernelLoadKeys{}, sorter.keys(), hiD.data(), loD.data(), n);
    auto sorted = sorter.sort(queue, countD.data(), digits);
    alpaka::exec<Acc1D>(queue, wd, KernelStoreKeys{}, sorted, outLoD.data(), outIdxD.data(), n);
    alpaka::memcpy(queue, outLoH, outLoD);
    alpaka::memcpy(queue, outIdxH, outIdxD);
    alpaka::wait(queue);
    std::vector<uint32_t> index(n);
    for (uint32_t i = 0; i < n; ++i) {
      index[i] = outIdxH[i];
      REQUIRE(outLoH[i] == low[index[i]]);  // the sorted copy of keyLow belongs to the index
    }
    return index;
  }

  // the host's reference: the stable sort by the digits, least significant first
  std::vector<uint32_t> sortOnHost(std::vector<uint64_t> const& high,
                                   std::vector<uint64_t> const& low,
                                   std::initializer_list<int> digits) {
    std::vector<uint32_t> index(high.size());
    std::iota(index.begin(), index.end(), 0u);
    for (int b : digits) {
      auto digit = [&](uint32_t i) { return b < 8 ? (low[i] >> (8 * b)) & 0xff : (high[i] >> (8 * (b - 8))) & 0xff; };
      std::stable_sort(index.begin(), index.end(), [&](uint32_t a, uint32_t c) { return digit(a) < digit(c); });
    }
    return index;
  }

  void check(uint32_t n, uint32_t capacity, std::initializer_list<int> digits, uint64_t mask, unsigned seed) {
    std::mt19937_64 rng(seed);
    std::vector<uint64_t> high(n), low(n);
    for (uint32_t i = 0; i < n; ++i)
      high[i] = rng() & mask, low[i] = rng() & mask;  // a small mask gives many equal digits and equal keys
    const auto expected = sortOnHost(high, low, digits);
    for (auto const& device : cms::alpakatools::devices<Platform>()) {
      std::cout << "radix sort of " << n << " keys (capacity " << capacity << ") on " << alpaka::getName(device)
                << "\n";
      Queue queue(device);
      REQUIRE(sortOnDevice(queue, high, low, capacity, digits) == expected);
    }
  }

}  // namespace

TEST_CASE("Seeder global radix sort for the " EDM_STRINGIZE(ALPAKA_ACCELERATOR_NAMESPACE) " backend",
          "[" EDM_STRINGIZE(ALPAKA_ACCELERATOR_NAMESPACE) "]") {
  if (cms::alpakatools::devices<Platform>().empty())
    FAIL("No devices available for the " EDM_STRINGIZE(ALPAKA_ACCELERATOR_NAMESPACE) " backend");

  const uint64_t all = ~uint64_t(0);
  SECTION("no keys, one key") {
    check(0, 16, {0, 1, 2, 3, 4, 5, 6, 7}, all, 1);
    check(1, 16, {0, 1, 2, 3, 4, 5, 6, 7}, all, 2);
  }
  SECTION("full 128-bit order") { check(3000, 4096, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}, all, 3); }
  SECTION("many equal digits and keys: stability") { check(5000, 5000, {0, 1, 2, 3, 4, 5, 6, 7, 8, 9}, 0x0303, 4); }
  SECTION("a few digits of keyLow, the key high word untouched") { check(2500, 3000, {0, 1}, all, 5); }
  SECTION("the digits of keyHigh only") { check(2500, 2500, {8, 9, 10}, all, 6); }
  SECTION("less keys than the capacity, several chunks") { check(700, 100000, {0, 1, 2, 3}, 0xfffff, 7); }
}
