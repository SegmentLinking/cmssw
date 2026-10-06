#include <algorithm>

#include <alpaka/alpaka.hpp>

#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include "HeterogeneousCore/AlpakaMath/interface/deltaPhi.h"

#include "RecoTracker/LSTCore/interface/alpaka/T3Features.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::lst {

  namespace t3featurekernels {
    using namespace t3features;

    // Same definitions as the standalone LST ntuple (write_lst_ntuple.cc, lst_math::Hit)
    ALPAKA_FN_ACC ALPAKA_FN_INLINE float t3Pt(TripletsConst triplets, unsigned int t3) {
      return __H2F(triplets.radius()[t3]) * k2Rinv1GeVf * 2.f;
    }

    ALPAKA_FN_ACC ALPAKA_FN_INLINE float hitEta(Acc1D const& acc, HitsBaseConst hits, unsigned int hit) {
      const float x = hits.xs()[hit], y = hits.ys()[hit], z = hits.zs()[hit];
      const float rt = alpaka::math::sqrt(acc, x * x + y * y);
      const float r3 = alpaka::math::sqrt(acc, x * x + y * y + z * z);
      return static_cast<float>((z > 0) - (z < 0)) * alpaka::math::acosh(acc, r3 / rt);
    }

    ALPAKA_FN_ACC ALPAKA_FN_INLINE void fillT3(Acc1D const& acc,
                                               float* f,
                                               HitsBaseConst hits,
                                               MiniDoubletsConst mds,
                                               SegmentsConst segments,
                                               TripletsConst triplets,
                                               unsigned int t3) {
      const unsigned int ls[2] = {triplets.segmentIndices()[t3][0], triplets.segmentIndices()[t3][1]};
      const unsigned int md[4] = {
          segments.mdIndices()[ls[0]][0], segments.mdIndices()[ls[0]][1], segments.mdIndices()[ls[1]][0], segments.mdIndices()[ls[1]][1]};

      // T3: eta from the anchor hit of the last MD, phi from the anchor hit of the first MD
      const unsigned int firstHit = mds.anchorHitIndices()[md[0]];
      f[kT3 + 0] = t3Pt(triplets, t3);
      f[kT3 + 1] = hitEta(acc, hits, mds.anchorHitIndices()[md[3]]);
      f[kT3 + 2] = cms::alpakatools::phi(acc, hits.xs()[firstHit], hits.ys()[firstHit]);

      // LS: only dPhiChange is stored outside CUT_VALUE_DEBUG; the rest is recomputed exactly as in
      // runSegmentDefaultAlgo{Barrel,Endcap} (Segment.h)
      for (unsigned int k = 0; k < 2; ++k) {
        const unsigned int inner = segments.mdIndices()[ls[k]][0];
        const unsigned int outer = segments.mdIndices()[ls[k]][1];
        const float dPhiChange = __H2F(segments.dPhiChanges()[ls[k]]);
        const float innerAlpha = mds.dphichanges()[inner];
        const float outerAlpha = mds.dphichanges()[outer];
        float* fls = f + kLS + k * kNLS;
        fls[0] = cms::alpakatools::reducePhiRange(acc, mds.anchorPhi()[outer] - mds.anchorPhi()[inner]);
        fls[1] = dPhiChange;
        fls[2] = innerAlpha - dPhiChange;
        fls[3] = outerAlpha - dPhiChange;
        fls[4] = innerAlpha - outerAlpha;
      }

      // MD: hit coordinates come from the input hits, like md_anchor_* / md_other_* in the ntuple
      for (unsigned int m = 0; m < 4; ++m) {
        const unsigned int anchor = mds.anchorHitIndices()[md[m]];
        const unsigned int other = mds.outerHitIndices()[md[m]];
        float* fmd = f + kMD + m * kNMD;
        fmd[0] = hits.xs()[anchor];
        fmd[1] = hits.ys()[anchor];
        fmd[2] = hits.zs()[anchor];
        fmd[3] = hits.xs()[other];
        fmd[4] = hits.ys()[other];
        fmd[5] = hits.zs()[other];
        fmd[6] = mds.dphis()[md[m]];
        fmd[7] = mds.dphichanges()[md[m]];
        fmd[8] = mds.dzs()[md[m]];
      }
    }

    // Number of selected T3s per inner lower module
    struct CountSelectedT3s {
      ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                    ObjectRangesConst ranges,
                                    TripletsConst triplets,
                                    TripletsOccupancyConst occupancy,
                                    unsigned int* counts,
                                    float maxT3Pt) const {
        for (unsigned int mod : cms::alpakatools::uniform_elements(acc, occupancy.metadata().size())) {
          unsigned int n = 0;
          for (unsigned int i = 0; i < occupancy.nTriplets()[mod]; ++i)
            n += t3Pt(triplets, ranges.tripletModuleIndices()[mod] + i) < maxT3Pt;
          counts[mod] = n;
        }
      }
    };

    // Exclusive prefix sum of the per-module counts (single thread; ~1e4 modules), plus the total
    struct ScanSelectedT3s {
      ALPAKA_FN_ACC void operator()(
          Acc1D const& acc, unsigned int const* counts, unsigned int* offsets, unsigned int nModules, T3Features out) const {
        if (cms::alpakatools::once_per_grid(acc)) {
          unsigned int sum = 0;
          for (unsigned int mod = 0; mod < nModules; ++mod) {
            offsets[mod] = sum;
            sum += counts[mod];
          }
          out.nT3s() = sum;
        }
      }
    };

    struct FillT3Features {
      ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                    HitsBaseConst hits,
                                    ObjectRangesConst ranges,
                                    MiniDoubletsConst mds,
                                    SegmentsConst segments,
                                    TripletsConst triplets,
                                    TripletsOccupancyConst occupancy,
                                    unsigned int const* offsets,
                                    T3Features out,
                                    float maxT3Pt) const {
        for (unsigned int mod : cms::alpakatools::uniform_elements(acc, occupancy.metadata().size())) {
          unsigned int row = offsets[mod];
          for (unsigned int i = 0; i < occupancy.nTriplets()[mod]; ++i) {
            const unsigned int t3 = ranges.tripletModuleIndices()[mod] + i;
            if (!(t3Pt(triplets, t3) < maxT3Pt))
              continue;
            out.tripletIndex()[row] = t3;
            fillT3(acc, out.features()[row].data(), hits, mds, segments, triplets, t3);
            ++row;
          }
        }
      }
    };
  }  // namespace t3featurekernels

  T3FeaturesDeviceCollection makeT3Features(Queue& queue,
                                            LSTInputDeviceCollection const& lstInput,
                                            ObjectRangesDeviceCollection const& ranges,
                                            MiniDoubletsDeviceCollection const& miniDoublets,
                                            SegmentsDeviceCollection const& segments,
                                            TripletsDeviceCollection const& triplets,
                                            float maxT3Pt) {
    using namespace t3featurekernels;
    auto const tripletsView = triplets.const_view().triplets();
    auto const occupancy = triplets.const_view().tripletsOccupancy();
    const unsigned int nModules = occupancy.metadata().size();

    // Capacity = all T3 slots, so no device-to-host sync is needed; nT3s() holds the filled size
    T3FeaturesDeviceCollection out(queue, std::max<int>(tripletsView.metadata().size(), 1));

    auto counts = cms::alpakatools::make_device_buffer<unsigned int[]>(queue, nModules);
    auto offsets = cms::alpakatools::make_device_buffer<unsigned int[]>(queue, nModules);

    auto const moduleWorkDiv = cms::alpakatools::make_workdiv<Acc1D>(cms::alpakatools::divide_up_by(nModules, 256), 256);
    alpaka::exec<Acc1D>(queue,
                        moduleWorkDiv,
                        CountSelectedT3s{},
                        ranges.const_view(),
                        tripletsView,
                        occupancy,
                        counts.data(),
                        maxT3Pt);
    alpaka::exec<Acc1D>(queue,
                        cms::alpakatools::make_workdiv<Acc1D>(1, 1),
                        ScanSelectedT3s{},
                        counts.data(),
                        offsets.data(),
                        nModules,
                        out.view());
    alpaka::exec<Acc1D>(queue,
                        moduleWorkDiv,
                        FillT3Features{},
                        lstInput.const_view().hits(),
                        ranges.const_view(),
                        miniDoublets.const_view().miniDoublets(),
                        segments.const_view().segments(),
                        tripletsView,
                        occupancy,
                        offsets.data(),
                        out.view(),
                        maxT3Pt);
    return out;
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::lst
