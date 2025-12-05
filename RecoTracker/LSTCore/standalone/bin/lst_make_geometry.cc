#include <string>

#include "cxxopts.h"

#include "RecoTracker/LSTCore/interface/LSTGeometry/IO.h"
#include "RecoTracker/LSTCore/interface/LSTGeometry/LSTGeometryMethods.h"

using namespace lstgeometry;

int main(int argc, char** argv) {
  cxxopts::Options options("\nLST Geometry\n\n");
  options.add_options()("module_info_file",
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
      cxxopts::value<std::string>()->default_value("../data/average_z_OT800_IT711.txt"))(
      "output_dir", "The path to the output directory.", cxxopts::value<std::string>()->default_value("../data/"))(
      "output_as_binary",
      "Boolean flag specifying whether to write outputs as binary or text files.",
      cxxopts::value<bool>()->default_value("true"))(
      "pt_cut", "pT cutoff value.", cxxopts::value<double>()->default_value("0.8"));

  auto result = options.parse(argc, argv);

  std::string module_info_file = result["module_info_file"].as<std::string>();
  std::string sensor_info_file = result["sensor_info_file"].as<std::string>();
  std::string average_r_file = result["average_r_file"].as<std::string>();
  std::string average_z_file = result["average_z_file"].as<std::string>();
  std::string output_dir = result["output_dir"].as<std::string>();
  bool output_as_bin = result["output_as_binary"].as<bool>();
  double ptCut = result["pt_cut"].as<double>();

  auto modules_info = readModuleInfo(module_info_file);
  auto sensors_info = readSensorInfo(sensor_info_file);
  auto average_r = readAverages(average_r_file);
  auto average_z = readAverages(average_z_file);

  auto lstGeometry = makeLSTGeometry(modules_info, sensors_info, average_r, average_z, ptCut);

  writeCentroids(lstGeometry->centroids, output_dir + "sensor_centroids", output_as_bin);
  writeSlopes(
      lstGeometry->barrel_slopes, lstGeometry->sensor_info, output_dir + "tilted_barrel_orientation", output_as_bin);
  writeSlopes(lstGeometry->endcap_slopes, lstGeometry->sensor_info, output_dir + "endcap_orientation", output_as_bin);
  writePixelMaps(lstGeometry->pixel_map, output_dir + "pixelmap/pLS_map", output_as_bin);
  writeModuleConnections(
      lstGeometry->merged_line_connections, output_dir + "module_connection_tracing_merged", output_as_bin);

  return 0;
}
