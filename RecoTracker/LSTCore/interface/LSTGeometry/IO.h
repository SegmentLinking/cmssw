#ifndef RecoTracker_LSTCore_interface_LSTGeometry_IO_h
#define RecoTracker_LSTCore_interface_LSTGeometry_IO_h

#include "Common.h"
#include "RecoTracker/LSTCore/interface/LSTGeometry/Common.h"
#include "RecoTracker/LSTCore/interface/LSTGeometry/ModuleInfo.h"
#include "RecoTracker/LSTCore/interface/LSTGeometry/SensorInfo.h"
#include "RecoTracker/LSTCore/interface/LSTGeometry/PixelMapMethods.h"
#include "RecoTracker/LSTCore/interface/LSTGeometry/OrientationMethods.h"

#include <vector>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <format>
#include <filesystem>

namespace lstgeometry {

  std::string trim(std::string const& str) {
    size_t first = str.find_first_not_of(" \t\r\n");
    if (first == std::string::npos)
      return "";
    size_t last = str.find_last_not_of(" \t\r\n");
    return str.substr(first, (last - first + 1));
  }

  std::vector<std::string> parseCSVLine(std::string const& line) {
    std::vector<std::string> tokens;
    std::stringstream ss(line);
    std::string token;

    while (std::getline(ss, token, ',')) {
      tokens.push_back(trim(token));
    }

    return tokens;
  }

  std::vector<ModuleInfo> readModuleInfo(std::string const& filename) {
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

      ModuleInfo m{static_cast<unsigned int>(std::stoul(tokens[0])),
                   std::stod(tokens[5]) / 10.0,
                   std::stod(tokens[6]) / 10.0,
                   degToRad(std::stod(tokens[7])),
                   degToRad(std::stod(tokens[8])),
                   degToRad(std::stod(tokens[9])),
                   degToRad(std::stod(tokens[10])),
                   std::stod(tokens[11]) / 10.0,
                   std::stod(tokens[12]) / 10.0,
                   std::stod(tokens[13]) / 10.0,
                   std::stod(tokens[14]) / 10.0,
                   std::stod(tokens[15]) / 10.0,
                   std::stod(tokens[16]) / 10.0,
                   std::stod(tokens[17]) / 10.0,
                   std::stod(tokens[18]) / 10.0,
                   std::stod(tokens[19]) / 10.0,
                   std::stod(tokens[20]) / 10.0,
                   std::stod(tokens[21]) / 10.0,
                   MatrixD8x3::Zero()};

      modules.push_back(m);
    }

    return modules;
  }

  std::unordered_map<unsigned int, SensorInfo> readSensorInfo(std::string const& filename) {
    std::unordered_map<unsigned int, SensorInfo> sensors;
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

      SensorInfo s{
          static_cast<unsigned int>(std::stoul(tokens[0])),
          std::stod(tokens[5]) / 10.0,
          std::stod(tokens[6]) / 10.0,
          degToRad(std::stod(tokens[7])),
      };

      sensors[s.detId] = s;
    }

    return sensors;
  }

  std::vector<double> readAverages(std::string const& filename) {
    std::vector<double> averages;
    std::string line;
    std::ifstream file(filename);

    if (!file.is_open()) {
      throw std::runtime_error("Could not open file " + filename);
    }

    while (std::getline(file, line)) {
      if (line.empty())
        continue;

      std::vector<std::string> tokens = parseCSVLine(line);
      if (tokens.size() != 1)
        continue;

      averages.push_back(std::stod(tokens[0]));
    }

    return averages;
  }

  void writeCentroids(std::unordered_map<unsigned int, Centroid> const& centroids,
                      std::string const& base_filename,
                      bool binary = true) {
    std::filesystem::path filepath(base_filename);
    std::filesystem::create_directories(filepath.parent_path());

    std::string filename = base_filename + (binary ? ".bin" : ".txt");
    std::ofstream file(filename, binary ? std::ios::binary : std::ios::out);

    if (binary) {
      for (auto& [detid, centroid] : centroids) {
        float x = centroid.x;
        float y = centroid.y;
        float z = centroid.z;
        unsigned int moduleType = centroid.moduleType;
        file.write(reinterpret_cast<const char*>(&detid), sizeof(detid));
        file.write(reinterpret_cast<const char*>(&x), sizeof(x));
        file.write(reinterpret_cast<const char*>(&y), sizeof(y));
        file.write(reinterpret_cast<const char*>(&z), sizeof(z));
        file.write(reinterpret_cast<const char*>(&moduleType), sizeof(moduleType));
      }
    } else {
      for (auto& [detid, centroid] : centroids) {
        file << detid << "," << centroid.x << "," << centroid.y << "," << centroid.z << "," << centroid.moduleType
             << std::endl;
      }
    }
  }

  void writeSlopes(std::unordered_map<unsigned int, SlopeData> const& slopes,
                   std::unordered_map<unsigned int, SensorInfo> const& sensors,
                   std::string const& base_filename,
                   bool binary = true) {
    std::filesystem::path filepath(base_filename);
    std::filesystem::create_directories(filepath.parent_path());

    std::string filename = base_filename + (binary ? ".bin" : ".txt");
    std::ofstream file(filename, binary ? std::ios::binary : std::ios::out);

    if (binary) {
      for (auto& [detid, slope] : slopes) {
        float drdz_slope = slope.drdz_slope;
        float dxdy_slope = slope.dxdy_slope;
        float phi = sensors.at(detid).phi_rad;
        file.write(reinterpret_cast<const char*>(&detid), sizeof(detid));
        if (drdz_slope != kDefaultSlope) {
          file.write(reinterpret_cast<const char*>(&drdz_slope), sizeof(drdz_slope));
          file.write(reinterpret_cast<const char*>(&dxdy_slope), sizeof(dxdy_slope));
        } else {
          file.write(reinterpret_cast<const char*>(&dxdy_slope), sizeof(dxdy_slope));
          file.write(reinterpret_cast<const char*>(&phi), sizeof(phi));
        }
      }
    } else {
      for (auto& [detid, slope] : slopes) {
        float drdz_slope = slope.drdz_slope;
        float dxdy_slope = slope.dxdy_slope;
        float phi = sensors.at(detid).phi_rad;
        file << detid << ",";
        if (drdz_slope != kDefaultSlope) {
          file << drdz_slope << "," << dxdy_slope << std::endl;
        } else {
          file << dxdy_slope << "," << phi << std::endl;
        }
      }
    }
  }

  void writeModuleConnections(std::unordered_map<unsigned int, std::unordered_set<unsigned int>> const& connections,
                              std::string const& base_filename,
                              bool binary = true) {
    std::filesystem::path filepath(base_filename);
    std::filesystem::create_directories(filepath.parent_path());

    std::string filename = base_filename + (binary ? ".bin" : ".txt");
    std::ofstream file(filename, binary ? std::ios::binary : std::ios::out);

    if (binary) {
      for (auto& [detid, set] : connections) {
        file.write(reinterpret_cast<const char*>(&detid), sizeof(detid));
        unsigned int length = set.size();
        file.write(reinterpret_cast<const char*>(&length), sizeof(length));
        for (unsigned int i : set) {
          file.write(reinterpret_cast<const char*>(&i), sizeof(i));
        }
      }
    } else {
      for (auto& [detid, set] : connections) {
        file << detid << "," << set.size();
        for (unsigned int i : set) {
          file << "," << i;
        }
        file << std::endl;
      }
    }
  }

  void writePixelMaps(PixelMap const& maps, std::string const& base_filename, bool binary = true) {
    std::filesystem::path filepath(base_filename);
    std::filesystem::create_directories(filepath.parent_path());

    if (binary) {
      for (auto& [layersubdetcharge, map] : maps) {
        auto& [layer, subdet, charge] = layersubdetcharge;

        std::string charge_str = charge > 0 ? "_pos" : (charge < 0 ? "_neg" : "");
        std::string filename = std::format("{}{}_layer{}_subdet{}.bin", base_filename, charge_str, layer, subdet);

        std::ofstream file(filename, std::ios::binary);

        for (unsigned int isuperbin = 0; isuperbin < map.size(); isuperbin++) {
          auto const& set = map.at(isuperbin);

          file.write(reinterpret_cast<const char*>(&isuperbin), sizeof(isuperbin));
          unsigned int length = set.size();
          file.write(reinterpret_cast<const char*>(&length), sizeof(length));
          for (unsigned int i : set) {
            file.write(reinterpret_cast<const char*>(&i), sizeof(i));
          }
        }
      }
    } else {
      for (auto& [layersubdetcharge, map] : maps) {
        auto& [layer, subdet, charge] = layersubdetcharge;

        std::string charge_str = charge > 0 ? "_pos" : (charge < 0 ? "_neg" : "");
        std::string filename = std::format("{}{}_layer{}_subdet{}.txt", base_filename, charge_str, layer, subdet);

        std::ofstream file(filename);

        for (unsigned int isuperbin = 0; isuperbin < map.size(); isuperbin++) {
          auto const& set = map.at(isuperbin);

          unsigned int length = set.size();
          file << isuperbin << "," << length;
          for (unsigned int i : set) {
            file << "," << i;
          }
          file << std::endl;
        }
      }
    }
  }

}  // namespace lstgeometry

#endif
