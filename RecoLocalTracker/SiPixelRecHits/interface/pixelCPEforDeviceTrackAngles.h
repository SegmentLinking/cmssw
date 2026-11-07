#ifndef RecoLocalTracker_SiPixelRecHits_interface_pixelCPEforDeviceTrackAngles_h
#define RecoLocalTracker_SiPixelRecHits_interface_pixelCPEforDeviceTrackAngles_h

// PixelCPEGeneric::getParameters(cluster, det, ltp) with the track angles, for any device, as configured with
// TruncatePixelCharge and IrradiationBiasCorrection off; modules without big pixels.

#include <alpaka/alpaka.hpp>

#include "CondFormats/SiPixelTransient/interface/SiPixelUtils.h"
#include "HeterogeneousCore/AlpakaInterface/interface/alpakastdAlgorithm.h"
#include "RecoLocalTracker/SiPixelRecHits/interface/PixelGenErrorTablesSoA.h"
#include "RecoLocalTracker/SiPixelRecHits/interface/pixelCPEforDevice.h"

namespace pixelCPEforDevice {

  // The cluster quantities PixelCPEGeneric reads (pixel indices in the module)
  struct GenericClusterParams {
    int minRow, maxRow, minCol, maxCol;
    int qFirstX, qLastX;  // charge in the first and last row
    int qFirstY, qLastY;  // charge in the first and last column
    float charge;
  };

  struct HitParameters {
    float x, y;        // local position (cm)
    float xx, xy, yy;  // local error (cm^2)
  };

  // GenError errors (sigmas) of the cluster charge bin, in microns
  struct GenErrorSigmas {
    float sigmaX, sigmaY;              // multi-pixel projections
    float singlePixelX, singlePixelY;  // single-pixel projections
  };

  // The interpolation interval [index - 1, index] of value in the sorted values and the ratio, as in SiPixelGenError::qbin
  ALPAKA_FN_HOST_ACC inline int interpolationIndex(const float* values, int size, float value, float& ratio) {
    const float* upper = alpaka_std::lower_bound(values, values + size, value);
    if (upper == values + size) {
      --upper;
      ratio = 1.f;
    } else if (upper == values) {
      ++upper;
      ratio = 0.f;
    } else {
      ratio = (value - (*(upper - 1))) / ((*upper) - (*(upper - 1)));
    }
    return upper - values;
  }

  // SiPixelGenError::qbin without the irradiation corrections: the errors for these track angles
  ALPAKA_FN_HOST_ACC inline GenErrorSigmas genErrorSigmas(PixelGenErrorTablesSoAConstView tables,
                                                          int store,
                                                          float cotAlpha,
                                                          float cotBeta,
                                                          float localBz,
                                                          float localBx,
                                                          float clusterCharge) {
    using siPixelUtils::interpolate1d;
    using siPixelUtils::interpolate2d;
    auto const header = tables.stores()[store];
    const float* pool = tables.pool().value().data();

    const float absCotBeta = std::abs(cotBeta);
    const float cotAlpha0 = header.cotAlpha0();
    const float qCorrect =
        std::sqrt((1.f + cotBeta * cotBeta + cotAlpha * cotAlpha) / (1.f + cotBeta * cotBeta + cotAlpha0 * cotAlpha0));
    float cota = cotAlpha;
    float cotb = absCotBeta;
    switch (header.detectorType()) {
      case 0:
        break;
      case 1:
        cotb = localBz < 0.f ? cotBeta : -cotBeta;
        break;
      default:  // 2 to 5, others are rejected by the ES producer
        if (localBx * localBz < 0.f)
          cota = -cotAlpha;
        cotb = localBx > 0.f ? cotBeta : -cotBeta;
        break;
    }

    float yRatio;
    const int yHigh = interpolationIndex(pool + header.cotBetaYOffset(), header.nCotBetaY(), cotb, yRatio);
    const float* yEntryLow = pool + header.yEntryOffset() + pixelGenErrorTables::kYEntrySize * (yHigh - 1);
    const float* yEntryHigh = yEntryLow + pixelGenErrorTables::kYEntrySize;
    const float qAverage = interpolate1d(yRatio, yEntryLow[0], yEntryHigh[0]) * qCorrect;
    const float chargeFraction = header.qscale() * clusterCharge / qAverage;
    const int qBin = chargeFraction > header.fbin0()   ? 0
                     : chargeFraction > header.fbin1() ? 1
                     : chargeFraction > header.fbin2() ? 2
                                                       : 3;
    GenErrorSigmas sigmas;
    sigmas.sigmaY = interpolate1d(yRatio, yEntryLow[2 + qBin], yEntryHigh[2 + qBin]);
    sigmas.singlePixelY = interpolate1d(yRatio, yEntryLow[1], yEntryHigh[1]);

    float yxRatio, xxRatio;
    const int yxHigh = interpolationIndex(pool + header.cotBetaXOffset(), header.nCotBetaX(), absCotBeta, yxRatio);
    const int xHigh = interpolationIndex(pool + header.cotAlphaXOffset(), header.nCotAlphaX(), cota, xxRatio);
    const float* singleX = pool + header.singleXOffset();
    sigmas.singlePixelX = interpolate1d(xxRatio, singleX[xHigh - 1], singleX[xHigh]);
    const float* xEntries = pool + header.xEntryOffset() + qBin;
    const int nCotAlphaX = header.nCotAlphaX();
    auto xRms = [&](int iy, int ix) { return xEntries[pixelGenErrorTables::kXEntrySize * (iy * nCotAlphaX + ix)]; };
    sigmas.sigmaX = interpolate2d(yxRatio,
                                  xxRatio,
                                  xRms(yxHigh - 1, xHigh - 1),
                                  xRms(yxHigh - 1, xHigh),
                                  xRms(yxHigh, xHigh - 1),
                                  xRms(yxHigh, xHigh));
    return sigmas;
  }

  // PixelCPEGeneric::localPosition with the track angles
  ALPAKA_FN_HOST_ACC inline void localPositionTrackAngles(CommonParams const& commonParams,
                                                          DetParams const& detParams,
                                                          PixelGenErrorTablesSoAConstView tables,
                                                          GenericClusterParams const& cluster,
                                                          float cotAlpha,
                                                          float cotBeta,
                                                          float& x,
                                                          float& y) {
    auto const modules = tables.modules();
    const float thickness = detParams.isBarrel ? commonParams.theThicknessB : commonParams.theThicknessE;
    // inner edges of the first and last pixels, as the topology's localPosition without big pixels
    const float halfRows = 0.5f * detParams.nRows, halfCols = 0.5f * detParams.nCols;
    const float upperEdgeFirstX = (float(cluster.minRow) + 1.f - halfRows) * detParams.thePitchX;
    const float lowerEdgeLastX = (float(cluster.maxRow) - halfRows) * detParams.thePitchX;
    const float upperEdgeFirstY = (float(cluster.minCol) + 1.f - halfCols) * detParams.thePitchY;
    const float lowerEdgeLastY = (float(cluster.maxCol) - halfCols) * detParams.thePitchY;
    x = siPixelUtils::genericPositionFormula(cluster.maxRow - cluster.minRow + 1,
                                             cluster.qFirstX,
                                             cluster.qLastX,
                                             upperEdgeFirstX,
                                             lowerEdgeLastX,
                                             detParams.chargeWidthX,
                                             thickness,
                                             cotAlpha,
                                             detParams.thePitchX,
                                             1.f,
                                             1.f,
                                             modules.effChargeCutLowX(),
                                             modules.effChargeCutHighX(),
                                             modules.sizeCutX()) +
        detParams.shiftX;
    y = siPixelUtils::genericPositionFormula(cluster.maxCol - cluster.minCol + 1,
                                             cluster.qFirstY,
                                             cluster.qLastY,
                                             upperEdgeFirstY,
                                             lowerEdgeLastY,
                                             detParams.chargeWidthY,
                                             thickness,
                                             cotBeta,
                                             detParams.thePitchY,
                                             1.f,
                                             1.f,
                                             modules.effChargeCutLowY(),
                                             modules.effChargeCutHighY(),
                                             modules.sizeCutY()) +
        detParams.shiftY;
  }

  // PixelCPEGeneric::getParameters with the track angles; false if the module has no GenError
  ALPAKA_FN_HOST_ACC inline bool hitParametersTrackAngles(CommonParams const& commonParams,
                                                          DetParams const& detParams,
                                                          PixelGenErrorTablesSoAConstView tables,
                                                          int module,
                                                          GenericClusterParams const& cluster,
                                                          float cotAlpha,
                                                          float cotBeta,
                                                          HitParameters& hit) {
    auto const modules = tables.modules();
    const int store = modules[module].genErrorStore();
    if (store < 0)
      return false;
    const GenErrorSigmas sigmas = genErrorSigmas(
        tables, store, cotAlpha, cotBeta, modules[module].localBz(), modules[module].localBx(), cluster.charge);
    localPositionTrackAngles(commonParams, detParams, tables, cluster, cotAlpha, cotBeta, hit.x, hit.y);

    // PixelCPEGenericBase::initializeLocalErrorVariables and setXYErrors
    const int lastRow = detParams.nRows - 1, lastCol = detParams.nCols - 1;
    const bool edgeX = cluster.minRow == 0 || cluster.maxRow == lastRow;
    const bool edgeY = cluster.minCol == 0 || cluster.maxCol == lastCol;
    float errorX = modules.edgeClusterErrorX() * micronsToCm;
    float errorY = modules.edgeClusterErrorY() * micronsToCm;
    if (!edgeX)
      errorX = (cluster.maxRow == cluster.minRow ? sigmas.singlePixelX : sigmas.sigmaX) * micronsToCm;
    if (!edgeY)
      errorY = (cluster.maxCol == cluster.minCol ? sigmas.singlePixelY : sigmas.sigmaY) * micronsToCm;
    hit.xx = errorX * errorX;
    hit.xy = 0.f;
    hit.yy = errorY * errorY;
    return true;
  }

}  // namespace pixelCPEforDevice

#endif  // RecoLocalTracker_SiPixelRecHits_interface_pixelCPEforDeviceTrackAngles_h
