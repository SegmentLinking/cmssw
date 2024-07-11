#include "RecoTracker/LSTCore/interface/EndcapGeometry.h"

SDL::EndcapGeometry::EndcapGeometry(std::string filename) { load(filename); }

void SDL::EndcapGeometry::load(std::string filename) {
  dxdy_slope_.clear();
  centroid_phis_.clear();

  std::ifstream ifile(filename, std::ios::binary);
  if (!ifile.is_open()) {
    throw std::runtime_error("Unable to open file: " + filename);
  }

  while (!ifile.eof()) {
    unsigned int detid;
    float dxdy_slope, centroid_phi;

    // Read the detid, dxdy_slope, and centroid_phi from binary file
    ifile.read(reinterpret_cast<char*>(&detid), sizeof(detid));
    ifile.read(reinterpret_cast<char*>(&dxdy_slope), sizeof(dxdy_slope));
    ifile.read(reinterpret_cast<char*>(&centroid_phi), sizeof(centroid_phi));

    if (ifile) {
      dxdy_slope_[detid] = dxdy_slope;
      centroid_phis_[detid] = centroid_phi;
    } else {
      // End of file or read failed
      if (!ifile.eof()) {
        throw std::runtime_error("Failed to read Endcap Geometry binary data.");
      }
    }
  }

  fillGeoMapArraysExplicit();
}

void SDL::EndcapGeometry::fillGeoMapArraysExplicit() {
  nEndCapMap = centroid_phis_.size();

  geoMapDetId_buf.reserve(nEndCapMap);
  geoMapPhi_buf.reserve(nEndCapMap);

  for (auto it = centroid_phis_.begin(); it != centroid_phis_.end(); ++it) {
    unsigned int detId = it->first;
    float Phi = it->second;
    geoMapPhi_buf.push_back(Phi);
    geoMapDetId_buf.push_back(detId);
  }
}

float SDL::EndcapGeometry::getdxdy_slope(unsigned int detid) const {
  if (dxdy_slope_.find(detid) != dxdy_slope_.end()) {
    return dxdy_slope_.at(detid);
  } else {
    return 0;
  }
}
