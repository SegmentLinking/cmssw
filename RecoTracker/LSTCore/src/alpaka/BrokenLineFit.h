#ifndef RecoTracker_LSTCore_src_alpaka_BrokenLineFit_h
#define RecoTracker_LSTCore_src_alpaka_BrokenLineFit_h

#ifndef LST_STANDALONE

#include <cstdint>

#include <Eigen/Core>

#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include "RecoTracker/PixelTrackFitting/interface/FitResult.h"
#include "RecoTracker/PixelTrackFitting/interface/alpaka/BrokenLine.h"

#include "LSTEvent.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::lst {

  // Initialise all fit-result pt entries to -1 before the fit kernels run.
  struct Kernel_InitBLFFit {
    ALPAKA_FN_ACC void operator()(Acc1D const& acc, TrackCandidatesBLFFit fitResults, unsigned int nTC) const {
      for (unsigned int tcIdx : cms::alpakatools::uniform_elements(acc, nTC)) {
        fitResults.pt()[tcIdx] = -1.f;
      }
    }
  };

  // BLF kernel for TCs with exactly N valid OT hits (both sensors per doublet layer).
  // TCs with a different nValid are skipped (handled by another N instantiation).
  template <int N>
  struct Kernel_LSTBLFit {
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  const float bField,
                                  TrackCandidatesBaseConst candsBase,
                                  HitsBaseConst hitsBase,
                                  TrackCandidatesBLFFit fitResults) const {
      const double bFieldD = static_cast<double>(bField);
      const unsigned int nTC = candsBase.nTrackCandidates();
      for (unsigned int tcIdx : cms::alpakatools::uniform_elements(acc, nTC)) {
        auto const& hitSlots = candsBase.hitIndices()[tcIdx];

        // Collect both sensor hits per OT doublet layer slot (inner sensor first,
        // then outer sensor), skipping pixel layer slots and empty slots.
        unsigned int validHitIdxs[Params_TC::kHitsPerLayer * (Params_TC::kLayers - Params_TC::kPixelLayerSlots)];
        int nValid = 0;
        for (int slot = Params_TC::kPixelLayerSlots; slot < Params_TC::kLayers; ++slot) {
          unsigned int h0 = hitSlots[slot][0];
          if (h0 == kTCEmptyHitIdx)
            continue;
          unsigned int h1 = hitSlots[slot][1];
          float x0 = hitsBase.xs()[h0], y0 = hitsBase.ys()[h0], z0 = hitsBase.zs()[h0];
          float x1 = hitsBase.xs()[h1], y1 = hitsBase.ys()[h1], z1 = hitsBase.zs()[h1];
          float d0sq = x0 * x0 + y0 * y0 + z0 * z0;
          float d1sq = x1 * x1 + y1 * y1 + z1 * z1;
          bool swap = d0sq > d1sq;
          validHitIdxs[nValid++] = swap ? h1 : h0;
          validHitIdxs[nValid++] = swap ? h0 : h1;
        }

        if (nValid != N)
          continue;

        Eigen::Matrix<double, 3, N> hits;
        Eigen::Matrix<float, 6, N> hits_ge;
        for (int i = 0; i < N; ++i) {
          const unsigned int hIdx = validHitIdxs[i];
          hits(0, i) = static_cast<double>(hitsBase.xs()[hIdx]);
          hits(1, i) = static_cast<double>(hitsBase.ys()[hIdx]);
          hits(2, i) = static_cast<double>(hitsBase.zs()[hIdx]);
          auto const& ge = hitsBase.ge()[hIdx];
          hits_ge(0, i) = ge[0];
          hits_ge(1, i) = ge[1];
          hits_ge(2, i) = ge[2];
          hits_ge(3, i) = ge[3];
          hits_ge(4, i) = ge[4];
          hits_ge(5, i) = ge[5];
        }

        ::riemannFit::Vector4d fast_fit;
        brokenline::fastFit(acc, hits, fast_fit);

        brokenline::PreparedBrokenLineData<N> data;
        brokenline::prepareBrokenLineData(acc, hits, fast_fit, bFieldD, data);

        brokenline::karimaki_circle_fit circle;
        ::riemannFit::LineFit line;
        brokenline::lineFit(acc, hits_ge, fast_fit, bFieldD, data, line);
        brokenline::circleFit(acc, hits, hits_ge, fast_fit, bFieldD, data, circle);

        fitResults.phi()[tcIdx] = static_cast<float>(circle.par(0));
        fitResults.tip()[tcIdx] = static_cast<float>(circle.par(1));
        fitResults.pt()[tcIdx] = static_cast<float>(bFieldD / alpaka::math::abs(acc, circle.par(2)));
        fitResults.eta()[tcIdx] = static_cast<float>(alpaka::math::asinh(acc, line.par(0)));
        fitResults.zip()[tcIdx] = static_cast<float>(line.par(1));
        fitResults.charge()[tcIdx] = static_cast<int8_t>(circle.qCharge);
        fitResults.chi2()[tcIdx] = static_cast<float>((circle.chi2 + line.chi2) / (2 * N - 5));

        // Circle covariance upper triangle: (phi-phi, phi-tip, tip-tip, phi-k, tip-k, k-k)
        auto& cCircle = fitResults.covCircle()[tcIdx];
        cCircle[0] = static_cast<float>(circle.cov(0, 0));
        cCircle[1] = static_cast<float>(circle.cov(0, 1));
        cCircle[2] = static_cast<float>(circle.cov(1, 1));
        cCircle[3] = static_cast<float>(circle.cov(0, 2));
        cCircle[4] = static_cast<float>(circle.cov(1, 2));
        cCircle[5] = static_cast<float>(circle.cov(2, 2));

        // Line covariance upper triangle: (slope-slope, slope-zip, zip-zip)
        auto& cLine = fitResults.covLine()[tcIdx];
        cLine[0] = static_cast<float>(line.cov(0, 0));
        cLine[1] = static_cast<float>(line.cov(0, 1));
        cLine[2] = static_cast<float>(line.cov(1, 1));
      }
    }
  };

  void launchLSTBrokenLineKernels(Queue& queue,
                                  float bField,
                                  TrackCandidatesBaseConst candsBase,
                                  HitsBaseConst hitsBase,
                                  TrackCandidatesBLFFit fitResults,
                                  unsigned int nTrackCandidates);

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::lst

#endif  // LST_STANDALONE

#endif  // RecoTracker_LSTCore_src_alpaka_BrokenLineFit_h
