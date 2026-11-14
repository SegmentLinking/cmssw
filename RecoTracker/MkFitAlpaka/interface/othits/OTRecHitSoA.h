#ifndef RecoTracker_MkFitAlpaka_interface_othits_OTRecHitSoA_h
#define RecoTracker_MkFitAlpaka_interface_othits_OTRecHitSoA_h

// ONE Phase-2 OT rechit SoA made on the device by a transliteration of
// Phase2StripCPE (RecoLocalTracker/Phase2TrackerRecHits/src/Phase2StripCPE.cc) from the OT clusters. Row = OT cluster
// key (flat index in the Phase2TrackerCluster1D DetSetVector), which is also the flat index of the legacy
// Phase2TrackerRecHits output (one rechit per cluster, same order). Read by the LST input, the CA OT layers and the
// device EventOfHits; the legacy Phase2TrackerRecHit1D is made on demand from (module, lx, ly, exx, eyy) + cluster ref.
// The first five fields are filled on the host from the clusters (one pass, no geometry objects); the kernel fills the
// rest. Columns are laid out in declaration order, so the host part is one contiguous prefix copy.
#include <cstdint>

#include "DataFormats/Portable/interface/PortableHostCollection.h"
#include "DataFormats/SoATemplate/interface/SoALayout.h"

namespace mkfitdev {

  GENERATE_SOA_LAYOUT(OTRecHitSoALayout,
                      SOA_SCALAR(uint32_t, nHits),      // rows in use (= OT clusters of the event)
                      SOA_COLUMN(int32_t, module),      // GeomDet index (TrackerGeometry::detUnits() index)
                      SOA_COLUMN(uint32_t, detId),      // raw DetId
                      SOA_COLUMN(uint16_t, clustSize),  // Phase2TrackerCluster1D::size()
                      SOA_COLUMN(uint32_t, strip),      // firstStrip() | column() << 16 (kernel input)
                      SOA_COLUMN(float, lx),            // localPosition (Phase2StripCPE::localParameters)
                      SOA_COLUMN(float, ly),
                      SOA_COLUMN(float, exx),  // localPositionError xx, yy (xy = 0)
                      SOA_COLUMN(float, eyy),
                      SOA_COLUMN(float, gx),  // Surface::toGlobal(localPosition), with the contraction of
                      SOA_COLUMN(float, gy),  // the host build (first product fused)
                      SOA_COLUMN(float, gz))
  using OTRecHitSoA = OTRecHitSoALayout<>;
  using OTRecHitHostCollection = PortableHostCollection<OTRecHitSoA>;

  // Per OT GeomDet (index - firstIndex): RectangularPixelPhase2Topology::localX/localY constants, the Phase2StripCPE
  // Lorentz shift (0.5 * coveredStrips) and errors, the Surface rotation rows and position. Built once per IOV on the
  // host from the CMSSW objects (coveredStrips through Phase2StripCPE::driftDirection, the errors through
  // Phase2StripCPE::localParameters), so the per-IOV numbers are bitwise those of Phase2StripCPE by construction.
  struct OTCpeModule {
    float r[9];         // TkRotation<float> xx xy xz | yx yy yz | zx zy zz
    float p[3];         // Surface position
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
