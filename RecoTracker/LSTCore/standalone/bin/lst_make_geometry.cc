#include <string>

#include "cxxopts.h"

#include "RecoTracker/LSTCore/interface/LSTGeometry/CornerMethods.h"
#include "RecoTracker/LSTCore/interface/LSTGeometry/CentroidMethods.h"
#include "RecoTracker/LSTCore/interface/LSTGeometry/OrientationMethods.h"
#include "RecoTracker/LSTCore/interface/LSTGeometry/IO.h"
#include "RecoTracker/LSTCore/interface/LSTGeometry/PixelMapMethods.h"
#include "RecoTracker/LSTCore/interface/LSTGeometry/ModuleMapMethods.h"

using namespace lstgeometry;

int main(int argc, char **argv) {

    cxxopts::Options options("\nLST Geometry\n\n");
    options.add_options()(
        "module_info_file",
        "The path to the csv file containing module information.",
        cxxopts::value<std::string>()->default_value("../data/module_info_OT800_IT711.csv"))(
        "sensor_info_file",
        "The path to the csv file containing sensor information.",
        cxxopts::value<std::string>()->default_value("../data/DetId_sensors_list_OT800_IT711.csv"))(
        "average_r_file",
        "The path to the text file containing the average r positions of the Barrel layers.",
        cxxopts::value<std::string>()->default_value("../data/average_r_OT800_IT711.txt"))(
        "average_z_file",
        "The path to the text file containing the average z positions of the Endcap layers.",
        cxxopts::value<std::string>()->default_value("../data/average_z_OT800_IT711.txt"));
        
    auto result = options.parse(argc, argv);
    
    std::string module_info_file = result["module_info_file"].as<std::string>();
    std::string sensor_info_file = result["sensor_info_file"].as<std::string>();
    std::string average_r_file = result["average_r_file"].as<std::string>();
    std::string average_z_file = result["average_z_file"].as<std::string>();
    
    auto modules_info = readModuleInfo(module_info_file);
    auto sensors_info = readSensorInfo(sensor_info_file);
    auto average_r = readAverages(average_r_file);
    auto average_z = readAverages(average_z_file);
    
    for (auto& mod : modules_info)
        transformSensorCorners(mod);
    
    auto assigned_corners = assignCornersToSensors(modules_info, sensors_info);
    
    auto centroids = computeCentroids(sensors_info);
    
    auto [barrel_slopes, endcap_slopes] = processCorners(assigned_corners);
    
    auto det_geom = DetectorGeometry(assigned_corners, average_r, average_z);
    det_geom.buildByLayer();
    
    auto pixel_map = computePixelMap(centroids, det_geom);
    
    auto detids_etaphi_layer_ref = det_geom.getDetIds(
                [](const auto& x){
                    auto mod = Module(x.first);
                return ((mod.subdet() == 5 && mod.isLower() == 1 && mod.layer() != 6) ||
                (mod.subdet() == 4 && mod.isLower() == 1 && mod.layer() != 5 &&
                    !(mod.ring() == 15 && mod.layer() == 1) &&
                    !(mod.ring() == 15 && mod.layer() == 2) &&
                    !(mod.ring() == 12 && mod.layer() == 3) &&
                    !(mod.ring() == 12 && mod.layer() == 4)
                ));
                }
                );
                
    std::unordered_map<unsigned int, std::vector<unsigned int>> straight_line_connections;
    std::unordered_map<unsigned int, std::vector<unsigned int>> curved_line_connections;
    
    for (auto ref_detid : detids_etaphi_layer_ref) {
        straight_line_connections[ref_detid] = getStraightLineConnections(ref_detid, centroids, det_geom);
        curved_line_connections[ref_detid] = getCurvedLineConnections(ref_detid, centroids, det_geom);
    }
    
    std::cout << "Done!" << std::endl;
    
  return 0;
}