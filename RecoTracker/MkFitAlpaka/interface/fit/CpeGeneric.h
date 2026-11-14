#ifndef RecoTracker_MkFitAlpaka_interface_fit_CpeGeneric_h
#define RecoTracker_MkFitAlpaka_interface_fit_CpeGeneric_h

// Portable (host + device) PixelCPEGeneric with TRACK angles, for the mkFit final fit:
// position and errors.
// Source: RecoLocalTracker/SiPixelRecHits/src/PixelCPEGeneric.cc localPosition (generic algorithm, no irradiation
// correction), CondFormats/SiPixelTransient/src/SiPixelUtils.cc generic_position_formula, PixelCPEBase
// computeAnglesFromTrajectory (cot alpha = ltp dxdz, cot beta = ltp dydz). Module constants as stored in
// pixelCPEforDevice::DetParams (shift = Lorentz shift / 2, chargeWidth = Lorentz shift * width fraction).

#include <cmath>
#include <cstdint>

#include <alpaka/alpaka.hpp>

namespace mkfitdev::cpe {

  // Menu PixelCPEGeneric settings (HLT_75e33 PixelCPEGenericESProducer): eff_charge_cut_low/high X/Y = 0 / 1,
  // size_cut X/Y = 3. Phase-2 inner tracker: no big pixels (pixel fraction 1).
  constexpr float kEffChargeCutLow = 0.f, kEffChargeCutHigh = 1.f, kSizeCut = 3.f;

  struct ClusterEdges {
    int minRow, maxRow, minCol, maxCol;  // SiPixelCluster min/maxPixelRow/Col
    int qfX, qlX, qfY, qlY;              // PixelCPEGenericBase::collect_edge_charges (no truncation)
  };

  struct ModuleCpe {
    float pitchX, pitchY, thickness;
    float shiftX, shiftY;              // 0.5 * Lorentz shift
    float chargeWidthX, chargeWidthY;  // Lorentz shift * width fraction
    float nRows, nCols;                // topology size: local x = (mp - nRows / 2) * pitch
    // errors
    float bx, bz;                // local B field [T] at the module centre (PixelCPEBase::fillDetParams)
    int templ;                   // index into CpeTables::templ (-1: no GenError for this module)
    int bigPerRocX, bigPerRocY;  // RectangularPixelPhase2Topology m_BIG_PIX_PER_ROC_X/Y (0 on Phase-2 IT)
  };

  // One SiPixelGenErrorStore, flattened (CondFormats/SiPixelTransient SiPixelGenError.h), only what qbin needs for
  // the errors: offsets into CpeTables::pool.
  //   cotbetaY[NTy], cotbetaX[NTyx], cotalphaX[NTxx]
  //   enty  NTy x {qavg, syone, sytwo, yrmsgen[4]}                (kEntY floats)
  //   entx  NTyx x NTxx x xrmsgen[4]
  //   sx    NTxx x {entx[0][ix].sxone, entx[0][ix].sxtwo}
  struct GenErrTemplate {
    int id, NTy, NTyx, NTxx, Dtype;
    float qscale, fbin0, fbin1, fbin2, cotalpha0;  // cotalpha0 = enty[0].cotalpha
    int offCotbY, offCotbX, offCotaX, offEnty, offEntx, offSx;
  };
  constexpr int kEntY = 7;

  struct CpeTables {
    const ModuleCpe* modules;
    int nModules;
    const GenErrTemplate* templ;
    int nTempl;
    const float* pool;
  };

  // Per pixel hit: the cluster quantities PixelCPEGeneric reads (SiPixelCluster), computed once per cluster.
  struct ClusterCpe {
    ClusterEdges e;
    float charge;  // SiPixelCluster::charge()
    int module;    // index into CpeTables::modules (-1: unknown -> no CPE, the projected hit is kept)
  };

  constexpr float kMicronsToCm = 1.0e-4f;                                  // PixelCPEBase::micronsToCm
  constexpr float kEdgeClusterErrorX = 50.0f, kEdgeClusterErrorY = 85.0f;  // menu EdgeClusterErrorX/Y [um]

  // std::lower_bound on a sorted float array (first element not less than v)
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE int lowerBound(const float* a, int n, float v) {
    int lo = 0, len = n;
    while (len > 0) {
      const int half = len >> 1;
      if (a[lo + half] < v) {
        lo += half + 1;
        len -= half + 1;
      } else {
        len = half;
      }
    }
    return lo;
  }

  // the interpolation index block of qbin: j = lower_bound, clamped to [1, n-1], ratio as PixelCPEGeneric
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE int interpIndex(const float* a, int n, float v, float& ratio) {
    int j = lowerBound(a, n, v);
    if (j == n) {
      --j;
      ratio = 1.f;
    } else if (j == 0) {
      ++j;
      ratio = 0.f;
    } else {
      ratio = (v - a[j - 1]) / (a[j] - a[j - 1]);
    }
    return j;
  }

  // SiPixelGenError::qbin (SiPixelGenError.cc:538-872) without the irradiation corrections (menu
  // IrradiationBiasCorrection = False) and without the Lorentz width/bias side outputs: the generic errors in
  // microns (sigmax/sigmay = x/yrmsgen of the charge bin, sx1/sx2/sy1/sy2 single-pixel errors). Returns binq.
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE int qbinErrors(GenErrTemplate const& T,
                                                     const float* pool,
                                                     float cotalpha,
                                                     float cotbeta,
                                                     float locBz,
                                                     float locBx,
                                                     float qclus,
                                                     float& sigmay,
                                                     float& sigmax,
                                                     float& sy1,
                                                     float& sy2,
                                                     float& sx1,
                                                     float& sx2) {
    const float acotb = cotbeta < 0.f ? -cotbeta : cotbeta;
    const float cotalpha0 = T.cotalpha0;
    const float qcorrect =
        std::sqrt((1.f + cotbeta * cotbeta + cotalpha * cotalpha) / (1.f + cotbeta * cotbeta + cotalpha0 * cotalpha0));
    float cota = cotalpha;
    float cotb = acotb;
    switch (T.Dtype) {
      case 0:
        break;
      case 1:
        cotb = locBz < 0.f ? cotbeta : -cotbeta;
        break;
      default:  // 2..5 (anything else throws in PixelCPEGeneric; the table builder rejects it)
        if (locBx * locBz < 0.f)
          cota = -cotalpha;
        cotb = locBx > 0.f ? cotbeta : -cotbeta;
        break;
    }
    float yratio;
    const int ihighY = interpIndex(pool + T.offCotbY, T.NTy, cotb, yratio);
    const int ilowY = ihighY - 1;
    const float* eyl = pool + T.offEnty + kEntY * ilowY;
    const float* eyh = pool + T.offEnty + kEntY * ihighY;
    float qavg = (1.f - yratio) * eyl[0] + yratio * eyh[0];
    qavg *= qcorrect;
    const float qtotal = T.qscale * qclus;
    const float fq = qtotal / qavg;
    const int binq = fq > T.fbin0 ? 0 : (fq > T.fbin1 ? 1 : (fq > T.fbin2 ? 2 : 3));
    const float yrmsgen = (1.f - yratio) * eyl[3 + binq] + yratio * eyh[3 + binq];
    sy1 = (1.f - yratio) * eyl[1] + yratio * eyh[1];
    sy2 = (1.f - yratio) * eyl[2] + yratio * eyh[2];

    float yxratio, xxratio;
    const int iyhigh = interpIndex(pool + T.offCotbX, T.NTyx, acotb, yxratio);
    const int iylow = iyhigh - 1;
    const int ihigh = interpIndex(pool + T.offCotaX, T.NTxx, cota, xxratio);
    const int ilow = ihigh - 1;
    const float* sx = pool + T.offSx;
    sx1 = (1.f - xxratio) * sx[2 * ilow] + xxratio * sx[2 * ihigh];
    sx2 = (1.f - xxratio) * sx[2 * ilow + 1] + xxratio * sx[2 * ihigh + 1];
    const float* ex = pool + T.offEntx;
    auto xr = [&](int iy, int ix) { return ex[4 * (iy * T.NTxx + ix) + binq]; };
    const float xrmsgen = (1.f - yxratio) * ((1.f - xxratio) * xr(iylow, ilow) + xxratio * xr(iylow, ihigh)) +
                          yxratio * ((1.f - xxratio) * xr(iyhigh, ilow) + xxratio * xr(iyhigh, ihigh));
    sigmay = yrmsgen;
    sigmax = xrmsgen;
    return binq;
  }

  // siPixelUtils::generic_position_formula
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE float genericPositionFormula(int size,
                                                                   int qf,
                                                                   int ql,
                                                                   float upperEdgeFirstPix,
                                                                   float lowerEdgeLastPix,
                                                                   float lorentzShift,
                                                                   float thickness,
                                                                   float cotAngle,
                                                                   float pitch,
                                                                   float pitchFractionFirst,
                                                                   float pitchFractionLast) {
    const float geomCenter = 0.5f * (upperEdgeFirstPix + lowerEdgeLastPix);
    if (size == 1)
      return geomCenter;
    const float wInner = lowerEdgeLastPix - upperEdgeFirstPix;
    const float wPred = thickness * cotAngle - lorentzShift;
    const float sumOfEdge = pitchFractionFirst + pitchFractionLast;
    const float wPredAbs = wPred < 0.f ? -wPred : wPred;
    float wEff = wPredAbs - wInner;
    if ((size >= kSizeCut) || ((wEff / pitch < kEffChargeCutLow) | (wEff / pitch > kEffChargeCutHigh)))
      wEff = pitch * 0.5f * sumOfEdge;
    const float qDiff = ql - qf;
    float qSum = ql + qf;
    if (qSum == 0)
      qSum = 1.0f;
    return geomCenter + 0.5f * (qDiff / qSum) * wEff;
  }

  // PixelCPEGeneric::localPosition with track angles (cotAlpha = dxdz, cotBeta = dydz of the local track state).
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE void localPositionTrackAngles(
      ModuleCpe const& m, ClusterEdges const& c, float cotAlpha, float cotBeta, float& x, float& y) {
    const float upperX = (float(c.minRow) + 1.f - 0.5f * m.nRows) * m.pitchX;
    const float lowerX = (float(c.maxRow) - 0.5f * m.nRows) * m.pitchX;
    const float upperY = (float(c.minCol) + 1.f - 0.5f * m.nCols) * m.pitchY;
    const float lowerY = (float(c.maxCol) - 0.5f * m.nCols) * m.pitchY;
    x = genericPositionFormula(c.maxRow - c.minRow + 1,
                               c.qfX,
                               c.qlX,
                               upperX,
                               lowerX,
                               m.chargeWidthX,
                               m.thickness,
                               cotAlpha,
                               m.pitchX,
                               1.f,
                               1.f) +
        m.shiftX;
    y = genericPositionFormula(c.maxCol - c.minCol + 1,
                               c.qfY,
                               c.qlY,
                               upperY,
                               lowerY,
                               m.chargeWidthY,
                               m.thickness,
                               cotBeta,
                               m.pitchY,
                               1.f,
                               1.f) +
        m.shiftY;
  }

  // RectangularPixelPhase2Topology::containsBigPixel
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE bool containsBigPixel(int iMin, int iMax, int nPxTot, int nPxBigPerROC) {
    const int firstBig = nPxTot / 2 - nPxBigPerROC;
    const int lastBig = nPxTot / 2 - 1 + nPxBigPerROC;
    return !((nPxBigPerROC == 0) || (iMin > lastBig) || (iMax < firstBig));
  }

  // PixelCPEGeneric::getParameters(cluster, det, ltp) of the menu configuration for the mkFit final fit:
  // localPosition with the track angles + localError with the GenError errors (UseErrorsFromTemplates = True,
  // with_track_angle = true). out = (x, y, exx, exy, eyy) as the cpe_func of MkFitFitProducer. Returns false when the
  // hit has no CPE module or GenError template (the caller keeps the projected hit; never seen in the menu).
  ALPAKA_FN_HOST_ACC ALPAKA_FN_INLINE bool cpeTrackAngles(
      CpeTables const& t, ClusterCpe const& c, float cotAlpha, float cotBeta, float (&out)[5]) {
    if (c.module < 0 || c.module >= t.nModules)
      return false;
    ModuleCpe const& m = t.modules[c.module];
    if (m.templ < 0 || m.templ >= t.nTempl)
      return false;
    float sigmay, sigmax, sy1, sy2, sx1, sx2;
    qbinErrors(t.templ[m.templ], t.pool, cotAlpha, cotBeta, m.bz, m.bx, c.charge, sigmay, sigmax, sy1, sy2, sx1, sx2);
    localPositionTrackAngles(m, c.e, cotAlpha, cotBeta, out[0], out[1]);
    // PixelCPEGenericBase::initializeLocalErrorVariables + setXYErrors (useTemplateErrors = true)
    const int nRows = int(m.nRows), nCols = int(m.nCols);
    const bool edgex = (c.e.minRow == 0) | (c.e.minRow == nRows - 1) | (c.e.maxRow == 0) | (c.e.maxRow == nRows - 1);
    const bool edgey = (c.e.minCol == 0) | (c.e.minCol == nCols - 1) | (c.e.maxCol == 0) | (c.e.maxCol == nCols - 1);
    const bool bigInX = containsBigPixel(c.e.minRow, c.e.maxRow, nRows, m.bigPerRocX);
    const bool bigInY = containsBigPixel(c.e.minCol, c.e.maxCol, nCols, m.bigPerRocY);
    const int sizex = c.e.maxRow - c.e.minRow + 1, sizey = c.e.maxCol - c.e.minCol + 1;
    float xerr = kEdgeClusterErrorX * kMicronsToCm;
    float yerr = kEdgeClusterErrorY * kMicronsToCm;
    if (!edgex)
      xerr = (sizex == 1 ? (bigInX ? sx2 : sx1) : sigmax) * kMicronsToCm;
    if (!edgey)
      yerr = (sizey == 1 ? (bigInY ? sy2 : sy1) : sigmay) * kMicronsToCm;
    out[2] = xerr * xerr;
    out[3] = 0.f;
    out[4] = yerr * yerr;
    return true;
  }

}  // namespace mkfitdev::cpe

#endif
