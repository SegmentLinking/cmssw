#ifndef RecoTracker_LSTCore_interface_LSTGeometry_IO_h
#define RecoTracker_LSTCore_interface_LSTGeometry_IO_h

#include "Common.h"
#include "RecoTracker/LSTCore/interface/LSTGeometry/ModuleInfo.h"
#include "RecoTracker/LSTCore/interface/LSTGeometry/SensorInfo.h"
#include "RecoTracker/LSTCore/interface/LSTGeometry/PixelMapMethods.h"

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

      SensorInfo s;

      s.detId = std::stoul(tokens[0]);
      s.binaryDetId = std::stoul(tokens[1], nullptr, 2);
      s.section = tokens[2];
      s.layer = std::stoi(tokens[3]);
      s.ring = std::stoi(tokens[4]);
      s.sensorCenterRho_mm = std::stod(tokens[5]);
      s.sensorCenterZ_mm = std::stod(tokens[6]);
      s.phi_deg = std::stod(tokens[7]);

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
  
  void writeCentroids(std::unordered_map<unsigned int, Centroid> const& centroids, std::string const& base_filename, bool binary = true) {
      std::string filename = base_filename + (binary ? ".bin" : ".txt");
      std::ofstream file(filename, binary ? std::ios::binary : std::ios::out);
      
      if(binary) {
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
            file << detid << "," << centroid.x << "," << centroid.y << "," << centroid.z << "," << centroid.moduleType << std::endl;
        }
      }
  }
  
  void writeSlopes(std::unordered_map<unsigned int, SlopeData> const& slopes, std::unordered_map<unsigned int, SensorInfo> const& sensors, std::string const& base_filename, bool binary = true) {
      std::string filename = base_filename + (binary ? ".bin" : ".txt");
      std::ofstream file(filename, binary ? std::ios::binary : std::ios::out);
      
      if(binary) {
          for (auto& [detid, slope] : slopes) {
              float drdz_slope = slope.drdz_slope;
              float dxdy_slope = slope.dxdy_slope;
              float phi = degToRad(sensors.at(detid).phi_deg);
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
            float phi = degToRad(sensors.at(detid).phi_deg);
            file << detid << ",";
            if (drdz_slope != kDefaultSlope) {
                file << drdz_slope << "," << dxdy_slope << std::endl;
            } else {
                file << dxdy_slope << "," << phi << std::endl;
            }
        }
      }
  }
  
  void writeModuleConnections(std::unordered_map<unsigned int, std::unordered_set<unsigned int>> const& connections, std::string const& base_filename, bool binary = true) {
      std::string filename = base_filename + (binary ? ".bin" : ".txt");
      std::ofstream file(filename, binary ? std::ios::binary : std::ios::out);
      
      if(binary) {
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
      
      if(binary) {
          for (auto& [layersubdet, map] : maps) {
              auto& [layer, subdet] = layersubdet;
              
              std::string filename_all = std::format("{}_layer{}_subdet{}.bin", base_filename, layer, subdet);
              std::string filename_pos = std::format("{}_pos_layer{}_subdet{}.bin", base_filename, layer, subdet);
              std::string filename_neg = std::format("{}_neg_layer{}_subdet{}.bin", base_filename, layer, subdet);
              
              std::ofstream file_all(filename_all, std::ios::binary);
              std::ofstream file_pos(filename_pos, std::ios::binary);
              std::ofstream file_neg(filename_neg, std::ios::binary);
              
              for (auto& [key, set] : map) {
                  auto& [ipt, ieta, iphi, iz, charge] = key;
                  unsigned int isuperbin = (ipt * kNPhi * kNEta * kNZ) + (ieta * kNPhi * kNZ) + (iphi * kNZ) + iz;
                  
                  file_all.write(reinterpret_cast<const char*>(&isuperbin), sizeof(isuperbin));
                  (charge > 0 ? file_pos : file_neg).write(reinterpret_cast<const char*>(&isuperbin), sizeof(isuperbin));
                  unsigned int length = set.size();
                  file_all.write(reinterpret_cast<const char*>(&length), sizeof(length));
                  (charge > 0 ? file_pos : file_neg).write(reinterpret_cast<const char*>(&length), sizeof(length));
                  for (unsigned int i : set) {
                      file_all.write(reinterpret_cast<const char*>(&i), sizeof(i));
                      (charge > 0 ? file_pos : file_neg).write(reinterpret_cast<const char*>(&i), sizeof(i));
                  }
              }
          }
      } else {
        for (auto& [layersubdet, map] : maps) {
            auto& [layer, subdet] = layersubdet;
            
            std::string filename_all = std::format("{}_layer{}_subdet{}.txt", base_filename, layer, subdet);
            std::string filename_pos = std::format("{}_pos_layer{}_subdet{}.txt", base_filename, layer, subdet);
            std::string filename_neg = std::format("{}_neg_layer{}_subdet{}.txt", base_filename, layer, subdet);
            
            std::ofstream file_all(filename_all);
            std::ofstream file_pos(filename_pos);
            std::ofstream file_neg(filename_neg);
            
            for (auto& [key, set] : map) {
                auto& [ipt, ieta, iphi, iz, charge] = key;
                unsigned int isuperbin = (ipt * kNPhi * kNEta * kNZ) + (ieta * kNPhi * kNZ) + (iphi * kNZ) + iz;
                
                unsigned int length = set.size();
                file_all << isuperbin << "," << length;
                (charge > 0 ? file_pos : file_neg) << isuperbin << "," << length;
                for (unsigned int i : set) {
                    file_all << "," << i;
                    (charge > 0 ? file_pos : file_neg) << "," << i;
                }
                file_all << std::endl;
                (charge > 0 ? file_pos : file_neg) << std::endl;
            }
        }
    }
  }

}  // namespace lstgeometry

#endif