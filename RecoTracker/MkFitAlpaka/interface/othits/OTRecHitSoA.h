#ifndef RecoTracker_MkFitAlpaka_interface_othits_OTRecHitSoA_h
#define RecoTracker_MkFitAlpaka_interface_othits_OTRecHitSoA_h

// Phase-2 OT rechit SoA of the device EventOfHits, made on the device by a transliteration of Phase2StripCPE
// (RecoLocalTracker/Phase2TrackerRecHits/src/Phase2StripCPE.cc) from the OT clusters. Row = OT cluster key (flat index
// in the Phase2TrackerCluster1D DetSetVector), which is also the flat index of the legacy Phase2TrackerRecHits output
// (one rechit per cluster, same order).
#include <cstdint>

#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "DataFormats/SoATemplate/interface/SoALayout.h"

namespace mkfitdev {

  GENERATE_SOA_LAYOUT(OTRecHitSoALayout,
                      SOA_SCALAR(uint32_t, nHits),      // rows in use (= OT clusters of the event)
                      SOA_COLUMN(int32_t, module),      // GeomDet index (TrackerGeometry::detUnits() index)
                      SOA_COLUMN(uint16_t, clustSize),  // Phase2TrackerCluster1D::size()
                      SOA_COLUMN(uint32_t, strip),      // firstStrip() | column() << 16 (kernel input)
                      SOA_COLUMN(float, lx),            // localPosition (Phase2StripCPE::localParameters)
                      SOA_COLUMN(float, ly),
                      SOA_COLUMN(float, exx),  // localPositionError xx, yy (xy = 0)
                      SOA_COLUMN(float, eyy))
  using OTRecHitSoA = OTRecHitSoALayout<>;
  using OTRecHitHostCollection = PortableHostCollection<OTRecHitSoA>;

  // strip column word: first strip in the low bits, column above
  constexpr uint32_t kOTFirstStripMask = 0xffffu, kOTColumnShift = 16;

  // Per OT GeomDet (index - firstIndex): RectangularPixelPhase2Topology::localX/localY constants, the Phase2StripCPE
  // Lorentz shift (0.5 * coveredStrips) and errors. Built once per IOV on the host from the CMSSW objects
  // (coveredStrips through Phase2StripCPE::driftDirection, the errors through Phase2StripCPE::localParameters), so the
  // per-IOV numbers are bitwise those of Phase2StripCPE by construction.
  struct OTCpeModule {
    float halfCovered;  // 0.5f * coveredStrips
    float pitchX, pitchY, bigPitchX, bigPitchY, xOffset, yOffset;
    float xHalfTerm, yHalfTerm;  // second-half terms of localX/localY (ispix_secondhalf = 1), Phase2StripCPE order
    float exx, eyy;
    int32_t xA, xB, xShift;  // nrows/2 - 2, nrows/2 - 2 + 2*nrows/ROWS_PER_ROC, 2*nrows/ROWS_PER_ROC
    int32_t yA, yB, yShift;  // ncols/2 - 1, ncols/2 - 1 + ncols/COLS_PER_ROC, ncols/COLS_PER_ROC
    uint32_t detId;
    int32_t valid;  // 1: OT GeomDetUnit with a RectangularPixelPhase2Topology
  };

}  // namespace mkfitdev

#endif
