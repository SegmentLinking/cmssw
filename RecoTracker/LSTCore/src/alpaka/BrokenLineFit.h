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

  // BLF kernel for TCs with exactly N valid OT hits.
  // TCs with a different nValid are skipped (handled by another N instantiation).
  template <int N>
  struct Kernel_LSTBLFit {
    ALPAKA_FN_ACC void operator()(Acc1D const& acc,
                                  const float bField,
                                  TrackCandidatesBaseConst candsBase,
                                  HitsBaseConst hitsBase,
                                  TrackCandidatesBLFFit fitResults,
                                  LSTObjType targetType) const {
      const double bFieldD = static_cast<double>(bField);
      const unsigned int nTC = candsBase.nTrackCandidates();
      for (unsigned int tcIdx : cms::alpakatools::uniform_elements(acc, nTC)) {
        if (candsBase.trackCandidateType()[tcIdx] != targetType)
          continue;

        auto const& hitSlots = candsBase.hitIndices()[tcIdx];

        // Collect valid OT anchor hit indices; skip pixel layer slots.
        unsigned int validHitIdxs[Params_TC::kLayers - Params_TC::kPixelLayerSlots];
        int nValid = 0;
        for (int slot = Params_TC::kPixelLayerSlots; slot < Params_TC::kLayers; ++slot) {
          unsigned int hIdx = hitSlots[slot][0];
          if (hIdx == kTCEmptyHitIdx)
            continue;
          validHitIdxs[nValid++] = hIdx;
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

  // Launch BLF kernels for all OT-based TC types.  pLS carries no OT hits and is not fitted.
  // T5 and pT5 can have 5, 6, or 7 OT hits (5 base + up to 2 extensions); separate kernel
  // instantiations are used for each count so all hits are used without subsampling.
  inline void launchLSTBrokenLineKernels(Queue& queue,
                                         const float bField,
                                         TrackCandidatesBaseConst candsBase,
                                         HitsBaseConst hitsBase,
                                         TrackCandidatesBLFFit fitResults,
                                         unsigned int nTrackCandidates) {
    if (nTrackCandidates == 0)
      return;

    constexpr uint32_t kBlockSize = 64;
    auto const workDiv =
        cms::alpakatools::make_workdiv<Acc1D>(cms::alpakatools::divide_up_by(nTrackCandidates, kBlockSize), kBlockSize);

    // Initialise all entries to pt=-1; fit kernels only write successful results.
    alpaka::exec<Acc1D>(queue, workDiv, Kernel_InitBLFFit{}, fitResults, nTrackCandidates);

    // pT3: exactly 3 OT layers (DoF = 2*3-5 = 1)
    alpaka::exec<Acc1D>(queue, workDiv, Kernel_LSTBLFit<3>{}, bField, candsBase, hitsBase, fitResults, LSTObjType::pT3);
    // T4: exactly 4 OT layers (DoF = 3)
    alpaka::exec<Acc1D>(queue, workDiv, Kernel_LSTBLFit<4>{}, bField, candsBase, hitsBase, fitResults, LSTObjType::T4);
    // T5: 5, 6, or 7 OT layers (DoF = 5, 7, or 9)
    alpaka::exec<Acc1D>(queue, workDiv, Kernel_LSTBLFit<5>{}, bField, candsBase, hitsBase, fitResults, LSTObjType::T5);
    alpaka::exec<Acc1D>(queue, workDiv, Kernel_LSTBLFit<6>{}, bField, candsBase, hitsBase, fitResults, LSTObjType::T5);
    alpaka::exec<Acc1D>(queue, workDiv, Kernel_LSTBLFit<7>{}, bField, candsBase, hitsBase, fitResults, LSTObjType::T5);
    // pT5: same OT layer structure as T5 (DoF = 5, 7, or 9)
    alpaka::exec<Acc1D>(queue, workDiv, Kernel_LSTBLFit<5>{}, bField, candsBase, hitsBase, fitResults, LSTObjType::pT5);
    alpaka::exec<Acc1D>(queue, workDiv, Kernel_LSTBLFit<6>{}, bField, candsBase, hitsBase, fitResults, LSTObjType::pT5);
    alpaka::exec<Acc1D>(queue, workDiv, Kernel_LSTBLFit<7>{}, bField, candsBase, hitsBase, fitResults, LSTObjType::pT5);
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::lst

#endif  // LST_STANDALONE

#endif  // RecoTracker_LSTCore_src_alpaka_BrokenLineFit_h
