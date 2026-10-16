#include "RecoLocalTracker/Phase2TrackerRecHits/interface/Phase2StripCPE.h"
#include "Geometry/CommonTopologies/interface/PixelTopology.h"
#include "Geometry/TrackerGeometryBuilder/interface/RectangularPixelPhase2Topology.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/Utilities/interface/Exception.h"

Phase2StripCPE::Phase2StripCPE(edm::ParameterSet& conf,
                               const MagneticField& magf,
                               const TrackerGeometry& geom,
                               const SiPhase2OuterTrackerLorentzAngle& LorentzAngle)
    : magfield_(magf),
      geom_(geom),
      lorentzAngleMap_(LorentzAngle),
      tanLorentzAnglePerTesla_(conf.getParameter<double>("TanLorentzAnglePerTesla")) {
  use_LorentzAngle_DB_ = conf.getParameter<bool>("LorentzAngle_DB");
  fillParam();
}

void Phase2StripCPE::fillPSetDescription(edm::ParameterSetDescription& desc) {
  desc.add<double>("TanLorentzAnglePerTesla", 0.07);
  desc.add<bool>("LorentzAngle_DB", true);
}

Phase2StripCPE::LocalValues Phase2StripCPE::localParameters(const Phase2TrackerCluster1D& cluster,
                                                            const GeomDetUnit& detunit) const {
  auto const& p = m_Params[detunit.index() - m_off];
  auto const position =
      phase2StripCPE::localPosition(p.position, cluster.firstStrip(), cluster.column(), cluster.size());
  return std::make_pair(LocalPoint(position.x, position.y, 0), p.localErr);
}

LocalVector Phase2StripCPE::driftDirection(const Phase2TrackerGeomDetUnit& det) const {
  LocalVector lbfield = (det.surface()).toLocal(magfield_.inTesla(det.surface().position()));

  float langle =
      use_LorentzAngle_DB_ ? lorentzAngleMap_.getLorentzAngle(det.geographicalId().rawId()) : tanLorentzAnglePerTesla_;

  float dir_x = -langle * lbfield.y();
  float dir_y = langle * lbfield.x();
  float dir_z = 1.f;  // E field always in z direction

  return LocalVector(dir_x, dir_y, dir_z);
}

void Phase2StripCPE::fillParam() {
  // in phase 2 they are all pixel topologies...
  auto const& dus = geom_.detUnits();
  m_off = dus.size();
  // skip Barrel and Foward pixels...
  for (unsigned int i = 3; i < 7; ++i) {
    LogDebug("LookingForFirstPhase2OT") << " Subdetector " << i << " GeomDetEnumerator "
                                        << GeomDetEnumerators::tkDetEnum[i] << " offset "
                                        << geom_.offsetDU(GeomDetEnumerators::tkDetEnum[i]) << std::endl;
    if (geom_.offsetDU(GeomDetEnumerators::tkDetEnum[i]) != dus.size()) {
      if (geom_.offsetDU(GeomDetEnumerators::tkDetEnum[i]) < m_off)
        m_off = geom_.offsetDU(GeomDetEnumerators::tkDetEnum[i]);
    }
  }
  LogDebug("LookingForFirstPhase2OT") << " Chosen offset: " << m_off;

  m_Params.resize(dus.size() - m_off);
  // very very minimal, for sure it will need to expand...
  for (auto i = m_off; i != dus.size(); ++i) {
    auto& p = m_Params[i - m_off];

    const Phase2TrackerGeomDetUnit& det = (const Phase2TrackerGeomDetUnit&)(*dus[i]);
    assert(det.index() == int(i));
    auto const* topology = dynamic_cast<const RectangularPixelPhase2Topology*>(&det.specificType().specificTopology());
    if (topology == nullptr)
      throw cms::Exception("Phase2StripCPE")
          << "module " << det.geographicalId().rawId() << " does not have a RectangularPixelPhase2Topology";

    auto pitch_x = topology->pitch().first;
    auto pitch_y = topology->pitch().second;

    // see https://github.com/cms-sw/cmssw/blob/CMSSW_8_1_X/RecoLocalTracker/SiStripRecHitConverter/src/StripCPE.cc
    auto thickness = det.specificSurface().bounds().thickness();
    auto drift = driftDirection(det) * thickness;
    auto lvec = drift + LocalVector(0, 0, -thickness);
    float coveredStrips = lvec.x() / pitch_x;  // simplifies wrt Phase0 tracker because only rectangular modules

    p.position = {pitch_x,
                  pitch_y,
                  topology->pitchbigpixelX(),
                  topology->pitchbigpixelY(),
                  topology->xoffset(),
                  topology->yoffset(),
                  topology->nrows(),
                  topology->ncolumns(),
                  topology->rowsperroc(),
                  topology->colsperroc(),
                  coveredStrips};

    constexpr float o12 = 1. / 12;
    p.localErr = LocalError(o12 * pitch_x * pitch_x, 0, o12 * pitch_y * pitch_y);  // e2_xx, e2_xy, e2_yy
  }
}
