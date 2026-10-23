// Compares pixelCPEforDevice::hitParametersTrackAngles on the device with PixelCPEGeneric::getParameters on the host,
// for synthetic clusters and track angles on every moduleStride-th module and the first module of every GenError store.

#include <algorithm>
#include <cmath>
#include <sstream>
#include <string>
#include <vector>

#include <alpaka/alpaka.hpp>

#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"
#include "DataFormats/TrajectoryState/interface/LocalTrajectoryParameters.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/stringize.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ESGetToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/global/EDProducer.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "RecoLocalTracker/ClusterParameterEstimator/interface/PixelClusterParameterEstimator.h"
#include "RecoLocalTracker/Records/interface/PixelCPEFastParamsRecord.h"
#include "RecoLocalTracker/Records/interface/TkPixelCPERecord.h"
#include "RecoLocalTracker/SiPixelRecHits/interface/PixelGenErrorTablesHost.h"
#include "RecoLocalTracker/SiPixelRecHits/interface/alpaka/PixelCPEFastParamsCollection.h"
#include "RecoLocalTracker/SiPixelRecHits/interface/alpaka/PixelGenErrorTablesDevice.h"
#include "RecoLocalTracker/SiPixelRecHits/interface/pixelCPEforDeviceClusterParams.h"

#include "PixelCPETrackAnglesTestKernel.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  using namespace pixelCPEforDevice::test;

  class TestPixelCPETrackAngles : public global::EDProducer<> {
  public:
    TestPixelCPETrackAngles(edm::ParameterSet const& config)
        : EDProducer<>(config),
          fastParamsToken_(esConsumes(edm::ESInputTag("", config.getParameter<std::string>("PixelCPEFastParams")))),
          tablesToken_(esConsumes(edm::ESInputTag("", config.getParameter<std::string>("PixelGenErrorTables")))),
          hostFastParamsToken_(esConsumes(edm::ESInputTag("", config.getParameter<std::string>("PixelCPEFastParams")))),
          hostTablesToken_(esConsumes(edm::ESInputTag("", config.getParameter<std::string>("PixelGenErrorTables")))),
          cpeToken_(esConsumes(edm::ESInputTag("", config.getParameter<std::string>("PixelCPE")))),
          geometryToken_(esConsumes()),
          moduleStride_(config.getParameter<int>("moduleStride")),
          positionTolerance_(config.getParameter<double>("positionTolerance")),
          errorTolerance_(config.getParameter<double>("errorTolerance")) {}

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      desc.add<std::string>("PixelCPEFastParams", "PixelCPEFastParamsPhase2");
      desc.add<std::string>("PixelGenErrorTables", "PixelGenErrorTablesPhase2");
      desc.add<std::string>("PixelCPE", "PixelCPEGeneric");
      desc.add<int>("moduleStride", 50);
      desc.add<double>("positionTolerance", 1e-6)->setComment("cm");
      desc.add<double>("errorTolerance", 1e-5)->setComment("relative, on the squared errors");
      descriptions.addWithDefaultLabel(desc);
    }

    void produce(edm::StreamID, device::Event& iEvent, device::EventSetup const& iSetup) const override {
      auto const& hostParams = *iSetup.getData(hostFastParamsToken_).data();
      auto const& hostTables = iSetup.getData(hostTablesToken_);
      auto const& cpe = iSetup.getData(cpeToken_);
      auto const& geometry = iSetup.getData(geometryToken_);

      std::vector<TrackAnglesCase> cases;
      std::vector<PixelClusterParameterEstimator::ReturnType> expected;
      makeCases(hostParams, hostTables, cpe, geometry, cases, expected);
      const int32_t size = cases.size();

      auto& queue = iEvent.queue();
      auto hostCases = cms::alpakatools::make_host_buffer<TrackAnglesCase[]>(queue, size);
      std::copy(cases.begin(), cases.end(), hostCases.data());
      auto deviceCases = cms::alpakatools::make_device_buffer<TrackAnglesCase[]>(queue, size);
      alpaka::memcpy(queue, deviceCases, hostCases);
      auto deviceResults = cms::alpakatools::make_device_buffer<TrackAnglesResult[]>(queue, size);
      pixelCPEforDeviceTest::runTrackAngles(queue,
                                            iSetup.getData(fastParamsToken_).data(),
                                            iSetup.getData(tablesToken_).const_view(),
                                            deviceCases.data(),
                                            deviceResults.data(),
                                            size);
      auto hostResults = cms::alpakatools::make_host_buffer<TrackAnglesResult[]>(queue, size);
      alpaka::memcpy(queue, hostResults, deviceResults);
      alpaka::wait(queue);

      compare(cases, expected, hostResults.data());
    }

  private:
    void makeCases(pixelCPEforDevice::ParamsOnDeviceT<pixelTopology::Phase2> const& params,
                   PixelGenErrorTablesHost const& tables,
                   PixelClusterParameterEstimator const& cpe,
                   TrackerGeometry const& geometry,
                   std::vector<TrackAnglesCase>& cases,
                   std::vector<PixelClusterParameterEstimator::ReturnType>& expected) const {
      struct Pixel {
        int row, col, adc;
      };
      const float cotAlphas[] = {-0.6f, -0.2f, 0.05f, 0.3f, 0.6f};
      const float cotBetas[] = {-6.0f, -3.0f, -1.0f, -0.2f, 0.0f, 0.5f, 1.2f, 4.0f};
      auto const modules = tables.const_view().modules();
      std::vector<bool> storeSeen(tables.const_view().stores().metadata().size(), false);
      for (int module = 0; module < pixelTopology::Phase2::numberOfModules; ++module) {
        const int store = modules[module].genErrorStore();
        const bool firstOfStore = store >= 0 && !storeSeen[store];
        if (module % moduleStride_ != 0 && !firstOfStore)
          continue;
        if (store >= 0)
          storeSeen[store] = true;
        auto const& detParams = params.detParams(module);
        auto const* det = geometry.idToDetUnit(DetId(detParams.rawId));
        if (det == nullptr)
          continue;
        const int nRows = detParams.nRows, nCols = detParams.nCols;
        const int row0 = nRows / 2 + 3, col0 = nCols / 2 + 5;
        const std::vector<std::vector<Pixel>> clusters = {
            {{row0, col0, 21000}},
            {{row0, col0, 12000}, {row0 + 1, col0, 9000}},
            {{row0, col0, 8000}, {row0, col0 + 1, 15000}, {row0, col0 + 2, 7000}},
            {{row0, col0, 6000},
             {row0 + 1, col0, 9000},
             {row0 + 2, col0 + 1, 7000},
             {row0 + 1, col0 + 2, 11000},
             {row0, col0 + 3, 5000},
             {row0 + 2, col0 + 3, 8000}},
            {{row0, col0, 35000}, {row0 + 1, col0 + 1, 4000}},
            {{0, col0, 10000}, {1, col0, 9000}},                       // first row
            {{nRows - 1, col0, 14000}},                                // last row
            {{row0, 0, 9000}, {row0, 1, 12000}},                       // first column
            {{row0, nCols - 2, 7000}, {row0 + 1, nCols - 1, 13000}}};  // last column
        for (auto const& pixels : clusters) {
          SiPixelCluster cluster(SiPixelCluster::PixelPos(pixels[0].row, pixels[0].col), pixels[0].adc);
          double sumRow = pixels[0].row, sumCol = pixels[0].col;
          for (std::size_t k = 1; k < pixels.size(); ++k) {
            cluster.add(SiPixelCluster::PixelPos(pixels[k].row, pixels[k].col), pixels[k].adc);
            sumRow += pixels[k].row;
            sumCol += pixels[k].col;
          }
          const float x = (sumRow / pixels.size() + 0.5 - 0.5 * nRows) * detParams.thePitchX;
          const float y = (sumCol / pixels.size() + 0.5 - 0.5 * nCols) * detParams.thePitchY;
          for (float cotAlpha : cotAlphas)
            for (float cotBeta : cotBetas) {
              cases.push_back({module, pixelCPEforDevice::genericClusterParams(cluster), cotAlpha, cotBeta});
              const LocalTrajectoryParameters track(0.1f, cotAlpha, cotBeta, x, y, 1.f);
              expected.push_back(cpe.getParameters(cluster, *det, track));
            }
        }
      }
    }

    void compare(std::vector<TrackAnglesCase> const& cases,
                 std::vector<PixelClusterParameterEstimator::ReturnType> const& expected,
                 TrackAnglesResult const* results) const {
      auto relative = [](double value, double reference) {
        return reference == 0 ? std::abs(value) : std::abs(value - reference) / std::abs(reference);
      };
      double maxDx = 0, maxDy = 0, maxRelXX = 0, maxRelYY = 0, maxXY = 0;
      long nBad = 0, nInvalid = 0;
      std::string first;
      for (std::size_t i = 0; i < cases.size(); ++i) {
        auto const& position = std::get<0>(expected[i]);
        auto const& error = std::get<1>(expected[i]);
        auto const& hit = results[i].hit;
        bool bad = !results[i].valid;
        if (bad) {
          ++nInvalid;
        } else {
          const double dx = std::abs(double(hit.x) - position.x()), dy = std::abs(double(hit.y) - position.y());
          const double relXX = relative(hit.xx, error.xx()), relYY = relative(hit.yy, error.yy());
          const double xy = std::max(std::abs(hit.xy), std::abs(error.xy()));
          maxDx = std::max(maxDx, dx);
          maxDy = std::max(maxDy, dy);
          maxRelXX = std::max(maxRelXX, relXX);
          maxRelYY = std::max(maxRelYY, relYY);
          maxXY = std::max(maxXY, xy);
          bad = dx > positionTolerance_ || dy > positionTolerance_ || relXX > errorTolerance_ ||
                relYY > errorTolerance_ || xy != 0;
        }
        if (bad) {
          ++nBad;
          if (first.empty()) {
            std::ostringstream message;
            message << "module " << cases[i].module << " rows " << cases[i].cluster.minRow << "-"
                    << cases[i].cluster.maxRow << " columns " << cases[i].cluster.minCol << "-"
                    << cases[i].cluster.maxCol << " cot alpha " << cases[i].cotAlpha << " cot beta " << cases[i].cotBeta
                    << ": host (" << position.x() << ", " << position.y() << ", " << error.xx() << ", " << error.xy()
                    << ", " << error.yy() << ") device (" << hit.x << ", " << hit.y << ", " << hit.xx << ", " << hit.xy
                    << ", " << hit.yy << ")";
            first = message.str();
          }
        }
      }
      std::ostringstream summary;
      summary << cases.size() << " hits, " << nBad << " beyond tolerance (" << nInvalid
              << " without GenError); max |dx| " << 1e4 * maxDx << " um, max |dy| " << 1e4 * maxDy
              << " um, max relative xx " << maxRelXX << ", yy " << maxRelYY << ", max |xy| " << maxXY;
      edm::LogPrint("TestPixelCPETrackAngles") << EDM_STRINGIZE(ALPAKA_ACCELERATOR_NAMESPACE) << ": " << summary.str();
      if (nBad > 0)
        throw cms::Exception("TestPixelCPETrackAngles") << summary.str() << "; first: " << first;
    }

    const device::ESGetToken<PixelCPEFastParamsPhase2, PixelCPEFastParamsRecord> fastParamsToken_;
    const device::ESGetToken<PixelGenErrorTablesDevice, PixelCPEFastParamsRecord> tablesToken_;
    const edm::ESGetToken<PixelCPEFastParamsHost<pixelTopology::Phase2>, PixelCPEFastParamsRecord> hostFastParamsToken_;
    const edm::ESGetToken<PixelGenErrorTablesHost, PixelCPEFastParamsRecord> hostTablesToken_;
    const edm::ESGetToken<PixelClusterParameterEstimator, TkPixelCPERecord> cpeToken_;
    const edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> geometryToken_;
    const int moduleStride_;
    const double positionTolerance_;
    const double errorTolerance_;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
DEFINE_FWK_ALPAKA_MODULE(TestPixelCPETrackAngles);
