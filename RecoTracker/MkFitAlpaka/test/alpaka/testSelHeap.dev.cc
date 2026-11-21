// The hit-selection priority queue SelHeap (select/SelectHitIndicesV2.h) against libstdc++ std::priority_queue with
// the comparator of MkFinder::selectHitIndicesV2, on every backend: streams of hit scores with many exact ties, kept
// and drained as selectHitIndicesV2 does. The selected hits and their order must be identical.

#include <cstdio>
#include <queue>
#include <random>
#include <vector>

#include <alpaka/alpaka.hpp>

#include "FWCore/Utilities/interface/stringize.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/devices.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include "RecoTracker/MkFitAlpaka/src/alpaka/select/SelectHitIndicesV2.h"

using namespace ALPAKA_ACCELERATOR_NAMESPACE;
namespace sel = ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev::select;

namespace {

  constexpr int kNumStreams = 20000;
  constexpr int kMaxStreamHits = 24;

  struct HitStream {
    int nHits;
    float score[kMaxStreamHits];
  };

  struct Selected {
    int nHits;
    unsigned int hitIndex[sel::NEW_MAX_HIT];
  };

  // the queue use of the device selectHitIndicesV2: keep the NEW_MAX_HIT smallest scores, then drain best first
  struct KernelSelect {
    template <typename TAcc>
    ALPAKA_FN_ACC void operator()(TAcc const& acc, const HitStream* streams, Selected* selected, int nStreams) const {
      for (int32_t stream : cms::alpakatools::uniform_elements(acc, nStreams)) {
        HitStream const& in = streams[stream];
        sel::SelHeap pq;
        for (int hit = 0; hit < in.nHits; ++hit) {
          if (pq.n < sel::NEW_MAX_HIT) {
            pq.push({in.score[hit], static_cast<unsigned int>(hit)});
          } else if (in.score[hit] < pq.top().score) {
            pq.pop();
            pq.push({in.score[hit], static_cast<unsigned int>(hit)});
          }
        }
        selected[stream].nHits = pq.n;
        int sz = pq.n;
        while (sz) {
          --sz;
          selected[stream].hitIndex[sz] = pq.top().hit_index;
          pq.pop();
        }
      }
    }
  };

  // MkFinder::selectHitIndicesV2 (MkFinder.cc), its queue and drain as written there
  Selected reference(HitStream const& in) {
    struct PQE {
      float score;
      unsigned int hit_index;
    };
    const auto pqe_cmp = [](const PQE& a, const PQE& b) { return a.score < b.score; };
    std::priority_queue<PQE, std::vector<PQE>, decltype(pqe_cmp)> pqueue(pqe_cmp);
    int pqueue_size = 0;
    for (int hit = 0; hit < in.nHits; ++hit) {
      const float new_ddphi = in.score[hit];
      const unsigned int hi_orig = hit;
      if (pqueue_size < sel::NEW_MAX_HIT) {
        pqueue.push({new_ddphi, hi_orig});
        ++pqueue_size;
      } else if (new_ddphi < pqueue.top().score) {
        pqueue.pop();
        pqueue.push({new_ddphi, hi_orig});
      }
    }
    Selected out{};
    out.nHits = pqueue_size;
    while (pqueue_size) {
      --pqueue_size;
      out.hitIndex[pqueue_size] = pqueue.top().hit_index;
      pqueue.pop();
    }
    return out;
  }

}  // namespace

int main() {
  auto const& devices = cms::alpakatools::devices<Platform>();
  if (devices.empty()) {
    std::printf("No devices available for the %s backend, skipping.\n", EDM_STRINGIZE(ALPAKA_ACCELERATOR_NAMESPACE));
    return 0;
  }
  Queue queue(devices[0]);

  // half the streams draw from a few score values (exact ties everywhere), half from a fine grid (occasional ties)
  std::mt19937 rng(31415);
  std::uniform_int_distribution<int> nHits(0, kMaxStreamHits), coarse(0, 3), fine(0, 999);
  auto streams = cms::alpakatools::make_host_buffer<HitStream[]>(queue, kNumStreams);
  for (int stream = 0; stream < kNumStreams; ++stream) {
    streams[stream].nHits = nHits(rng);
    for (int hit = 0; hit < streams[stream].nHits; ++hit)
      streams[stream].score[hit] = stream % 2 ? 0.25f * coarse(rng) : 1e-3f * fine(rng);
  }
  auto streamsDevice = cms::alpakatools::make_device_buffer<HitStream[]>(queue, kNumStreams);
  auto selectedDevice = cms::alpakatools::make_device_buffer<Selected[]>(queue, kNumStreams);
  auto selected = cms::alpakatools::make_host_buffer<Selected[]>(queue, kNumStreams);
  alpaka::memcpy(queue, streamsDevice, streams);
  const int threads = cms::alpakatools::requires_single_thread_per_block_v<Acc1D> ? 1 : 128;
  alpaka::exec<Acc1D>(
      queue,
      cms::alpakatools::make_workdiv<Acc1D>(cms::alpakatools::divide_up_by(kNumStreams, threads), threads),
      KernelSelect{},
      streamsDevice.data(),
      selectedDevice.data(),
      kNumStreams);
  alpaka::memcpy(queue, selected, selectedDevice);
  alpaka::wait(queue);

  int nDiffer = 0, nTied = 0;
  for (int stream = 0; stream < kNumStreams; ++stream) {
    const Selected ref = reference(streams[stream]);
    bool same = ref.nHits == selected[stream].nHits;
    for (int k = 0; same && k < ref.nHits; ++k)
      same = ref.hitIndex[k] == selected[stream].hitIndex[k];
    nDiffer += !same;
    // streams whose selection had to break a tie between equal scores
    for (int k = 1; k < ref.nHits; ++k)
      if (streams[stream].score[ref.hitIndex[k]] == streams[stream].score[ref.hitIndex[k - 1]]) {
        ++nTied;
        break;
      }
  }
  std::printf("%s %s: %d of %d streams (%d with tied scores in the selection) differ from std::priority_queue\n",
              nDiffer ? "FAIL" : "PASS",
              EDM_STRINGIZE(ALPAKA_ACCELERATOR_NAMESPACE),
              nDiffer,
              kNumStreams,
              nTied);
  return nDiffer ? 3 : 0;
}
