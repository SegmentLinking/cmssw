// Unit test of the seeder's JSON readers (interface/MkSeederConfig.h): the working point's files parse, a removed
// option is accepted only with its off value, malformed or incomplete files throw.

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <string>

#include <nlohmann/json.hpp>

#define CATCH_CONFIG_MAIN
#include <catch2/catch_all.hpp>

#include "FWCore/ParameterSet/interface/FileInPath.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "RecoTracker/MkSeederAlpaka/interface/MkSeederConfig.h"

namespace {

  const char* const kConfigFile = "RecoTracker/MkFit/data/mkfit-phase2-seeder-6l-l6c10-effB.json";
  const char* const kMlpFile = "RecoTracker/MkFit/data/mkfit-phase2-seeder-triplet-mlp.json";

  std::string path(const char* fileInPath) { return edm::FileInPath(fileInPath).fullPath(); }

  nlohmann::json readJson(const std::string& file) {
    std::ifstream in(file);
    return nlohmann::json::parse(in);
  }

  // a scratch file in the test's temporary directory, removed with the object
  class TempFile {
  public:
    explicit TempFile(std::string const& name, std::string const& content)
        : path_(std::filesystem::temp_directory_path() / ("testMkSeederConfig_" + std::to_string(::getpid()) + name)) {
      std::ofstream(path_) << content;
    }
    ~TempFile() { std::filesystem::remove(path_); }
    std::string str() const { return path_.string(); }

  private:
    std::filesystem::path path_;
  };

  // the working point's JSON with the changes of `edit`
  template <typename F>
  mkseeder::MkSeederConfig readEdited(F&& edit) {
    nlohmann::json j = readJson(path(kConfigFile));
    edit(j);
    TempFile file(".json", j.dump());
    return mkseeder::readMkSeederConfig(file.str());
  }

}  // namespace

TEST_CASE("MkSeederConfig reader", "[MkSeederAlpaka]") {
  SECTION("the working point's file parses and gives the values of the file") {
    const auto j = readJson(path(kConfigFile));
    const auto c = mkseeder::readMkSeederConfig(path(kConfigFile));
    REQUIRE(c.ptMin == Catch::Approx(j["pt_min"].get<float>()));
    REQUIRE(c.d0Max == Catch::Approx(j["d0_max"].get<float>()));
    REQUIRE(c.zVertexHalfWidth == Catch::Approx(j["zv"].get<float>()));
    REQUIRE(c.maxHoles == j["max_holes"].get<int>());
    REQUIRE(c.patterns.size() == j["patterns"].size());
    REQUIRE(c.shapeWindows.size() == j["shape_win"].size());
    REQUIRE(c.chainOuterTrackerLayers.size() == j["chain_ot"].size());
    REQUIRE(c.dupMaxSigma == Catch::Approx(j["dup_sig"].get<float>()));
    REQUIRE(c.fitPriorScale == Catch::Approx(j["fit"]["prior_scale"].get<float>()));
    for (auto const& p : c.patterns)
      for (int layer : p.layers)
        REQUIRE(layer >= -1);
  }

  SECTION("unknown keys are ignored, the removed options are accepted with their off values") {
    REQUIRE_NOTHROW(readEdited([](nlohmann::json& j) {
      j["not_an_option"] = 7;
      j["dedup_helix"] = 0;
      j["trip_max_score"] = 0.f;
      j["ext_eta"] = 99.f;
    }));
  }

  SECTION("the pruned combinations") {
    const auto c = readEdited([](nlohmann::json& j) { j["prune"] = {{0, 1, 17}, {0, 1, 3, 4}}; });
    REQUIRE(c.prunedCombinations.size() == 2);
    REQUIRE(c.prunedCombinations[0] == std::vector<int>{0, 1, 17});
    REQUIRE(c.prunedCombinations[1] == std::vector<int>{0, 1, 3, 4});
  }

  SECTION("the L3-miss gap window and the stage-c network cuts") {
    const auto off = readEdited([](nlohmann::json& j) {
      j.erase("miss_gap_dz");
      j.erase("trip_net_c");
    });
    REQUIRE(off.missGapDz == 0.f);
    for (float cut : off.stageCNetCuts)
      REQUIRE(cut == mkseeder::kStageCNetOff);
    const auto c = readEdited([](nlohmann::json& j) {
      j["miss_gap_dz"] = 0.5f;
      j["trip_net_c"] = {-3.5f, -4.5f, -4.f, -3.5f, -3.f};
    });
    REQUIRE(c.missGapDz == Catch::Approx(0.5f));
    REQUIRE(c.stageCNetCuts[1] == Catch::Approx(-4.5f));
    REQUIRE(c.stageCNetCuts[4] == Catch::Approx(-3.f));
  }

  SECTION("a removed option with another value than its off value throws") {
    REQUIRE_THROWS_AS(readEdited([](nlohmann::json& j) { j["dedup_helix"] = 1; }), cms::Exception);
    REQUIRE_THROWS_AS(readEdited([](nlohmann::json& j) { j["d_mode"] = 2; }), cms::Exception);
    REQUIRE_THROWS_AS(readEdited([](nlohmann::json& j) { j["trip_max_score"] = 0.5f; }), cms::Exception);
    REQUIRE_THROWS_AS(readEdited([](nlohmann::json& j) { j["ext_eta"] = 1.5f; }), cms::Exception);
    REQUIRE_THROWS_AS(readEdited([](nlohmann::json& j) { j["fit"]["mode"] = 0; }), cms::Exception);
  }

  SECTION("a missing required value, a bad size or a bad type throws") {
    REQUIRE_THROWS_AS(readEdited([](nlohmann::json& j) { j.erase("pix_fill_dedup"); }), cms::Exception);
    REQUIRE_THROWS_AS(readEdited([](nlohmann::json& j) { j["pix_fill_dedup"] = 0; }), cms::Exception);
    REQUIRE_THROWS_AS(readEdited([](nlohmann::json& j) { j["dup_sig"] = 0.f; }), cms::Exception);
    REQUIRE_THROWS_AS(readEdited([](nlohmann::json& j) { j["ext_wx"] = {0.1f}; }), cms::Exception);
    REQUIRE_THROWS_AS(readEdited([](nlohmann::json& j) { j["pt_min"] = "high"; }), cms::Exception);
    REQUIRE_THROWS_AS(readEdited([](nlohmann::json& j) { j["prune"] = {{0, 1}}; }), cms::Exception);
  }

  SECTION("a missing or malformed file throws") {
    REQUIRE_THROWS_AS(mkseeder::readMkSeederConfig("/nonexistent/mkseeder.json"), cms::Exception);
    TempFile broken(".json", "{ \"pt_min\": ");
    REQUIRE_THROWS_AS(mkseeder::readMkSeederConfig(broken.str()), cms::Exception);
  }
}

TEST_CASE("MkSeederTripletMlpWeights reader", "[MkSeederAlpaka]") {
  using W = mkseeder::MkSeederTripletMlpWeights;
  SECTION("the network file parses") {
    const auto j = readJson(path(kMlpFile));
    const auto w = mkseeder::readMkSeederTripletMlpWeights(path(kMlpFile));
    REQUIRE(w.featureMean[0] == Catch::Approx(j["featureMean"][0].get<float>()));
    REQUIRE(w.hidden1Weights[W::kNFeatures * W::kNHidden1 - 1] ==
            Catch::Approx(j["hidden1Weights"][W::kNFeatures * W::kNHidden1 - 1].get<float>()));
    REQUIRE(w.outputBias == Catch::Approx(j["outputBias"].get<float>()));
    REQUIRE(w.zFold == Catch::Approx(j.value("zFold", 0.f)));
  }
  SECTION("a wrong network shape or a missing array throws") {
    auto j = readJson(path(kMlpFile));
    j["nHidden1"] = 33;
    TempFile wrongShape(".json", j.dump());
    REQUIRE_THROWS_AS(mkseeder::readMkSeederTripletMlpWeights(wrongShape.str()), cms::Exception);
    j = readJson(path(kMlpFile));
    j["featureMean"].erase(0);
    TempFile shortArray(".json", j.dump());
    REQUIRE_THROWS_AS(mkseeder::readMkSeederTripletMlpWeights(shortArray.str()), cms::Exception);
    j = readJson(path(kMlpFile));
    j.erase("outputBias");
    TempFile noBias(".json", j.dump());
    REQUIRE_THROWS_AS(mkseeder::readMkSeederTripletMlpWeights(noBias.str()), cms::Exception);
    REQUIRE_THROWS_AS(mkseeder::readMkSeederTripletMlpWeights("/nonexistent/mlp.json"), cms::Exception);
  }
}
