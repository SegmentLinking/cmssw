// Device precomputation for the host output conversion (see MkFitAlpakaOutConvKernels.h).
#include <alpaka/alpaka.hpp>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

#include "MkFitAlpakaOutConvKernels.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev::outconv {
  using namespace cms::alpakatools;
  namespace pca = ::mkfitdev::pca;

  namespace {
    // TSCBL of every track into a scratch PcaOut; the copy to the output SoA is a second kernel, so this one stays
    // spill-free on sm_89
    struct KernelOutConvPca {
      ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                    ::mkfitdev::TrackSoAConstView trk,
                                    int capacity,
                                    pca::BeamIn bl,
                                    pca::PcaOut* scratch) const {
        const int n = trk.nTracks();
        for (auto t : uniform_elements(acc, capacity)) {
          if (int(t) >= n)
            continue;
          pca::TrackIn in;
          pca::ccsToCurvilinear(trk[t].params().v, trk[t].errors().v, trk[t].charge(), in);
          pca::pcaFromFirstHitState(in, bl, scratch[t]);  // sets status on every path, the state when kPcaOk
        }
      }
    };

    struct KernelOutConvStore {
      ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                    ::mkfitdev::TrackSoAConstView trk,
                                    int capacity,
                                    pca::PcaOut const* scratch,
                                    ::mkfitdev::OutConvSoAView out) const {
        const int n = trk.nTracks();
        for (auto t : uniform_elements(acc, capacity)) {
          if (int(t) >= n)
            continue;
          pca::PcaOut const& o = scratch[t];
          auto row = out[t];
          row.pcaStatus() = int8_t(o.status);
          if (o.status != pca::kPcaOk)
            continue;
          auto& s = row.pcaState();
          s.v[0] = o.x.x;
          s.v[1] = o.x.y;
          s.v[2] = o.x.z;
          s.v[3] = o.p.x;
          s.v[4] = o.p.y;
          s.v[5] = o.p.z;
          auto& c = row.pcaCov();
          for (int i = 0, k = 0; i < 5; ++i)
            for (int j = 0; j <= i; ++j)
              c.v[k++] = float(o.C[i][j]);
        }
      }
    };
  }  // namespace

  void launchOutConvStates(Queue& queue,
                           ::mkfitdev::TrackSoAConstView trk,
                           int capacity,
                           pca::BeamIn const& beamLine,
                           ::mkfitdev::OutConvSoAView out) {
    if (capacity <= 0)
      return;
    // heavy double-precision kernel (TTMD + Jacobian): 64-thread blocks; the scratch is queue-ordered
    auto scratch = make_device_buffer<pca::PcaOut[]>(queue, capacity);
    const auto wd = make_workdiv<Acc1D>(divide_up_by(capacity, 64), 64);
    alpaka::exec<Acc1D>(queue, wd, KernelOutConvPca{}, trk, capacity, beamLine, scratch.data());
    alpaka::exec<Acc1D>(queue, wd, KernelOutConvStore{}, trk, capacity, scratch.data(), out);
  }
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev::outconv
