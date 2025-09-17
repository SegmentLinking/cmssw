#ifndef RecoTracker_LSTCore_interface_LSTGeometry_IO_h
#define RecoTracker_LSTCore_interface_LSTGeometry_IO_h

#include "RecoTracker/LSTCore/interface/LSTGeometry/ModuleInfo.h"
#include "RecoTracker/LSTCore/interface/LSTGeometry/SensorInfo.h"

#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

namespace lst {

  std::string trim(const std::string& str) {
    size_t first = str.find_first_not_of(" \t\r\n");
    if (first == std::string::npos)
      return "";
    size_t last = str.find_last_not_of(" \t\r\n");
    return str.substr(first, (last - first + 1));
  }

  std::vector<std::string> parseCSVLine(const std::string& line) {
    std::vector<std::string> tokens;
    std::stringstream ss(line);
    std::string token;

    while (std::getline(ss, token, ',')) {
      tokens.push_back(trim(token));
    }

    return tokens;
  }

  std::vector<ModuleInfo> readModuleInfo(const std::string& filename) {
    std::vector<ModuleInfo> modules;
    std::string line;
    std::ifstream file(filename);

    if (!file.is_open()) {
      throw std::runtime_error("Could not open file " + filename);
    }

    // Skip header line
    std::getline(file, line);

    while (std::getline(file, line)) {
      if (line.empty())
        continue;

      std::vector<std::string> tokens = parseCSVLine(line);
      if (tokens.size() != 23)
        continue;

      ModuleInfo m;

      m.detId = std::stoul(tokens[0]);
      m.binaryDetId = std::stoul(tokens[1], nullptr, 2);
      m.section = tokens[2];
      m.layer = std::stoi(tokens[3]);
      m.ring = std::stoi(tokens[4]);
      m.sensorCenterRho_mm = std::stod(tokens[5]);
      m.sensorCenterZ_mm = std::stod(tokens[6]);
      m.tiltAngle_deg = std::stod(tokens[7]);
      m.skewAngle_deg = std::stod(tokens[8]);
      m.yawAngle_deg = std::stod(tokens[9]);
      m.phi_deg = std::stod(tokens[10]);
      m.vtxOneX_mm = std::stod(tokens[11]);
      m.vtxOneY_mm = std::stod(tokens[12]);
      m.vtxTwoX_mm = std::stod(tokens[13]);
      m.vtxTwoY_mm = std::stod(tokens[14]);
      m.vtxThreeX_mm = std::stod(tokens[15]);
      m.vtxThreeY_mm = std::stod(tokens[16]);
      m.vtxFourX_mm = std::stod(tokens[17]);
      m.vtxFourY_mm = std::stod(tokens[18]);
      m.meanWidth_mm = std::stod(tokens[19]);
      m.length_mm = std::stod(tokens[20]);
      m.sensorSpacing_mm = std::stod(tokens[21]);
      m.sensorThickness_mm = std::stod(tokens[22]);

      modules.push_back(m);
    }

    return modules;
  }

  std::vector<SensorInfo> readSensorInfo(const std::string& filename) {
    std::vector<SensorInfo> sensors;
    std::string line;
    std::ifstream file(filename);

    if (!file.is_open()) {
      throw std::runtime_error("Could not open file " + filename);
    }

    // Skip header line
    std::getline(file, line);

    while (std::getline(file, line)) {
      if (line.empty())
        continue;

      std::vector<std::string> tokens = parseCSVLine(line);
      if (tokens.size() != 8)
        continue;

      SensorInfo s;

      s.detId = std::stoul(tokens[0]);
      s.binaryDetId = std::stoul(tokens[1], nullptr, 2);
      s.section = tokens[2];
      s.layer = std::stoi(tokens[3]);
      s.ring = std::stoi(tokens[4]);
      s.sensorCenterRho_mm = std::stod(tokens[5]);
      s.sensorCenterZ_mm = std::stod(tokens[6]);
      s.phi_deg = std::stod(tokens[7]);

      sensors.push_back(s);
    }

    return sensors;
  }

}  // namespace lst

#endif