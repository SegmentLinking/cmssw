// Phase-2 OT rechits of the device mkFit hit input, made from the OT clusters. The host lists the cluster detsets; per
// cluster key, the GeomDet index, size and first strip | column are written on the host (CPU backends) or by a kernel
// from the stored clusters (GPU backends), then a kernel applies Phase2StripCPE (MkFitAlpakaOTRecHitsKernels).
// Its per-module constants come from a per-IOV table built from the CMSSW objects: coveredStrips through
// Phase2StripCPE::driftDirection (as Phase2StripCPE::fillParam), errors through Phase2StripCPE::localParameters,
// topology constants from the RectangularPixelPhase2Topology accessors.
// Output: mkfitdev::OTRecHitDeviceCollection, one row per OT cluster key.
#include <algorithm>
#include <array>
#include <atomic>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <type_traits>
#include <vector>

#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/Phase2TrackerCluster/interface/Phase2TrackerCluster1D.h"
#include "DataFormats/TrackerRecHit2D/interface/Phase2TrackerRecHit1D.h"
#include "FWCore/Framework/interface/Run.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "FWCore/Utilities/interface/ESInputTag.h"
#include "Geometry/CommonTopologies/interface/GeomDetEnumerators.h"
#include "Geometry/CommonTopologies/interface/PixelGeomDetUnit.h"
#include "Geometry/CommonTopologies/interface/ProxyPixelTopology.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/RectangularPixelPhase2Topology.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDPutToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/Event.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EventSetup.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/global/EDProducer.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "RecoLocalTracker/Phase2TrackerRecHits/interface/Phase2StripCPE.h"
#include "RecoLocalTracker/Records/interface/TkPhase2OTCPERecord.h"
#include "RecoTracker/MkFit/interface/MkFitSeedWrapper.h"
#include "RecoTracker/MkFitAlpaka/interface/alpaka/Backend.h"
#include "RecoTracker/MkFitAlpaka/interface/othits/OTCpe.h"
#include "RecoTracker/MkFitAlpaka/interface/othits/OTRecHitSoA.h"
#include "RecoTracker/MkFitAlpaka/interface/othits/alpaka/OTRecHitDeviceCollection.h"

#include "MkFitAlpakaOTRecHitsKernels.h"
#include "MkFitAlpakaReleaseOrder.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  namespace {
    // the SoA's strip word: first strip | column << 16
    uint32_t stripWord(Phase2TrackerCluster1D const& cluster) {
      return (cluster.firstStrip() & ::mkfitdev::kOTFirstStripMask) | (cluster.column() << ::mkfitdev::kOTColumnShift);
    }

    struct OTCpeTable {
      std::vector<::mkfitdev::OTCpeModule> modules;  // index = GeomDet index - firstIndex
      int32_t firstIndex = 0;
      // raw DetId -> table index of the valid entries, open addressing (linear probing, load <= 1/2): one cache line
      // per detset instead of TrackerGeometry::idToDetUnit's hash-map node plus the GeomDet read
      std::vector<uint64_t> idSlots;  // (detId << 32) | index; 0 = empty (DetId 0 is never an OT module)
      uint32_t idShift = 32;

      static constexpr uint32_t kFibonacciHash = 0x9e3779b1u;  // 2^32 / golden ratio
      static uint32_t slotOf(uint32_t id, uint32_t shift) { return (id * kFibonacciHash) >> shift; }
      void buildIdSlots() {
        uint32_t nValid = 0;
        for (auto const& entry : modules)
          nValid += entry.valid;
        uint32_t bits = 1;
        while ((1u << bits) < 2 * nValid)
          ++bits;
        idShift = 32 - bits;
        idSlots.assign(size_t(1) << bits, 0);
        const uint32_t mask = (1u << bits) - 1;
        for (uint32_t index = 0; index < modules.size(); ++index) {
          if (!modules[index].valid)
            continue;
          const uint32_t id = modules[index].detId;
          uint32_t slot = slotOf(id, idShift);
          while (idSlots[slot] != 0) {
            if (uint32_t(idSlots[slot] >> 32) == id)
              throw cms::Exception("MkFitAlpakaOTRecHits") << "DetId " << id << " twice in the OT CPE table";
            slot = (slot + 1) & mask;
          }
          idSlots[slot] = (uint64_t(id) << 32) | index;
        }
      }
      // table index of a valid entry with this DetId, or -1
      int32_t indexOf(uint32_t id) const {
        const uint32_t mask = idSlots.size() - 1;
        for (uint32_t slot = slotOf(id, idShift);; slot = (slot + 1) & mask) {
          const uint64_t entry = idSlots[slot];
          if (entry == 0)
            return -1;
          if (uint32_t(entry >> 32) == id)
            return int32_t(uint32_t(entry));
        }
      }
    };

    using DetSetSpan = ::mkfitdev::othits::OTDetSetSpan;
    // the device expansion reads the clusters as stored: two uint16 per cluster (checkClusterLayout, first event)
    static_assert(sizeof(Phase2TrackerCluster1D) == 2 * sizeof(uint16_t));
    static_assert(std::is_trivially_copyable_v<Phase2TrackerCluster1D>);

  }  // namespace

  // the run cache holds the per-IOV table built at the run transition (globalBeginRun)
  class MkFitAlpakaOTRecHitsProducer : public global::EDProducer<edm::RunCache<const OTCpeTable>> {
  public:
    explicit MkFitAlpakaOTRecHitsProducer(edm::ParameterSet const& iConfig)
        : EDProducer<edm::RunCache<const OTCpeTable>>(iConfig),
          clustersToken_{consumes(iConfig.getParameter<edm::InputTag>("src"))},
          geomToken_{esConsumes()},
          cpeToken_{esConsumes(iConfig.getParameter<edm::ESInputTag>("Phase2StripCPE"))},
          geomRunToken_{esConsumes<edm::Transition::BeginRun>()},
          cpeRunToken_{esConsumes<edm::Transition::BeginRun>(iConfig.getParameter<edm::ESInputTag>("Phase2StripCPE"))},
          putToken_{produces()},
          queueToken_{produces("queue")} {
      // consumed for scheduling only: the SoA is made once the building's seeds exist; nothing reads the seeds
      if (auto const& seedsTag = iConfig.getParameter<edm::InputTag>("seeds"); !seedsTag.label().empty())
        seedsToken_ = consumes(seedsTag);
    }

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      desc.add<edm::InputTag>("src", edm::InputTag("hltSiPhase2Clusters"))->setComment("Phase-2 OT clusters");
      desc.add<edm::ESInputTag>("Phase2StripCPE", edm::ESInputTag("phase2StripCPEESProducer", "Phase2StripCPE"))
          ->setComment("the Phase2StripCPE (as hltSiPhase2RecHits); its per-module numbers fill the table");
      desc.add("seeds", edm::InputTag{""})
          ->setComment("scheduling only: the SoA is made once the building's seeds exist (it is deleted early)");
      descriptions.addWithDefaultLabel(desc);
    }

    // The per-IOV table is built here, once, before the run's first event: otherwise every stream's first event waits
    // on the table mutex while one of them builds it. produce() still checks the IOV (and rebuilds on a change).
    std::shared_ptr<const OTCpeTable> globalBeginRun(edm::Run const&, edm::EventSetup const& es) const override {
      return table(
          es.getData(cpeRunToken_), es.getData(geomRunToken_), es.get<TkPhase2OTCPERecord>().cacheIdentifier());
    }
    void globalEndRun(edm::Run const&, edm::EventSetup const&) const override {}

    void produce(edm::StreamID, device::Event& iEvent, device::EventSetup const& iSetup) const override {
      const auto& clusters = iEvent.get(clustersToken_);
      const auto& geom = iSetup.getData(geomToken_);
      edm::EventSetup const& es = iSetup;
      const auto cpeTable = table(iSetup.getData(cpeToken_), geom, es.get<TkPhase2OTCPERecord>().cacheIdentifier());
      const auto* devTable = deviceTable(iEvent.queue(), cpeTable);

      const uint32_t nClusters = clusters.dataSize();
      std::vector<DetSetSpan> spans;
      mkfitdev::OTRecHitDeviceCollection product(iEvent.queue(), std::max<int32_t>(int32_t(nClusters), 1));
      if constexpr (mkfitdev::kCpuBackend) {
        fillHost(clusters, *cpeTable, nClusters, product.view(), spans);
      } else {
        // the host lists the detsets (one DetId-index lookup each); the raw clusters and the spans go to the device,
        // where one kernel expands module / size / strip per cluster key (the same integers as fillHost)
        auto& queue = iEvent.queue();
        fillSpans(clusters, *cpeTable, nClusters, spans);
        const uint32_t nSpans = spans.size();
        // one staging buffer and one device buffer: the stored clusters, then the spans
        static_assert(sizeof(Phase2TrackerCluster1D) % alignof(DetSetSpan) == 0);
        const size_t rawBytes = size_t(nClusters) * sizeof(Phase2TrackerCluster1D);
        const auto bytes = static_cast<uint32_t>(rawBytes + spans.size() * sizeof(DetSetSpan) + sizeof(DetSetSpan));
        auto hostBuffer = cms::alpakatools::make_host_buffer<std::byte[]>(queue, bytes);
        auto* hostRaw = reinterpret_cast<uint16_t*>(hostBuffer.data());
        if (nClusters > 0)
          std::memcpy(hostRaw, clusters.data().data(), rawBytes);
        std::copy(spans.begin(), spans.end(), reinterpret_cast<DetSetSpan*>(hostBuffer.data() + rawBytes));
        if (not layoutChecked_.load(std::memory_order_relaxed))
          checkClusterLayout(clusters, hostRaw, nClusters);
        auto deviceBuffer = cms::alpakatools::make_device_buffer<std::byte[]>(queue, bytes);
        alpaka::memcpy(queue, deviceBuffer, hostBuffer);
        mkfitdev::othits::runOTExpand(queue,
                                      product.view(),
                                      reinterpret_cast<DetSetSpan const*>(deviceBuffer.data() + rawBytes),
                                      nSpans,
                                      reinterpret_cast<uint16_t const*>(deviceBuffer.data()),
                                      devTable,
                                      cpeTable->firstIndex,
                                      nClusters);
      }
      mkfitdev::othits::runOTCpe(iEvent.queue(), product.view(), devTable, cpeTable->firstIndex, nClusters);
      iEvent.emplace(putToken_, std::move(product));
      // the SoA is deleted early: its reader orders this queue after its reads
      iEvent.emplace(queueToken_, mkfitdev::queueHandle(iEvent.queue()));
    }

  private:
    // per-IOV table (TkPhase2OTCPERecord depends on the geometry, the field and the Lorentz angle records)
    std::shared_ptr<const OTCpeTable> table(ClusterParameterEstimator<Phase2TrackerCluster1D> const& cpeBase,
                                            TrackerGeometry const& geom,
                                            unsigned long long id) const {
      std::lock_guard<std::mutex> lock(tableMutex_);
      if (table_ && tableId_ == id)
        return table_;
      auto const* cpe = dynamic_cast<Phase2StripCPE const*>(&cpeBase);
      if (cpe == nullptr)
        throw cms::Exception("MkFitAlpakaOTRecHits") << "the configured OT CPE is not a Phase2StripCPE";
      auto newTable = std::make_shared<OTCpeTable>();
      // first OT GeomDetUnit index, as Phase2StripCPE::fillParam
      auto const& dus = geom.detUnits();
      uint32_t firstIndex = dus.size();
      for (unsigned int i = 3; i < 7; ++i) {
        const auto offset = geom.offsetDU(GeomDetEnumerators::tkDetEnum[i]);
        if (offset != dus.size() && offset < firstIndex)
          firstIndex = offset;
      }
      newTable->firstIndex = int32_t(firstIndex);
      newTable->modules.resize(dus.size() - firstIndex);
      for (auto i = firstIndex; i != dus.size(); ++i) {
        auto& cpeModule = newTable->modules[i - firstIndex];
        std::memset(&cpeModule, 0, sizeof(cpeModule));
        auto const* det = dynamic_cast<PixelGeomDetUnit const*>(dus[i]);
        if (det == nullptr || det->index() != int(i))
          continue;
        // PixelGeomDetUnit holds a ProxyPixelTopology (surface deformations); localX/localY forward to the wrapped one
        PixelTopology const* pixelTopology = &det->specificTopology();
        if (auto const* proxy = dynamic_cast<ProxyPixelTopology const*>(pixelTopology))
          pixelTopology = &proxy->specificTopology();
        auto const* topo = dynamic_cast<RectangularPixelPhase2Topology const*>(pixelTopology);
        if (topo == nullptr)
          continue;
        // Phase2StripCPE::fillParam, verbatim
        auto pitch_x = topo->pitch().first;
        auto thickness = det->specificSurface().bounds().thickness();
        auto drift = cpe->driftDirection(*det) * thickness;
        auto lvec = drift + LocalVector(0, 0, -thickness);
        float coveredStrips = lvec.x() / pitch_x;
        cpeModule.halfCovered = 0.5f * coveredStrips;
        // errors exactly as the CPE returns them
        const auto params = cpe->localParameters(Phase2TrackerCluster1D(0, 0, 1), *det);
        cpeModule.exx = params.second.xx();
        cpeModule.eyy = params.second.yy();
        if (params.second.xy() != 0.f)
          throw cms::Exception("MkFitAlpakaOTRecHits") << "Phase2StripCPE local error xy != 0 for det index " << i;
        // RectangularPixelPhase2Topology::localX/localY constants (same integer and float expressions)
        const int nrows = topo->nrows(), ncols = topo->ncolumns(), rpr = topo->rowsperroc(), cpr = topo->colsperroc();
        cpeModule.pitchX = topo->pitch().first;
        cpeModule.pitchY = topo->pitch().second;
        cpeModule.bigPitchX = topo->pitchbigpixelX();
        cpeModule.bigPitchY = topo->pitchbigpixelY();
        cpeModule.xOffset = topo->xoffset();
        cpeModule.yOffset = topo->yoffset();
        const int ispixSecondHalf = 1;  // the int factor of the second-half terms of localX/localY
        cpeModule.xHalfTerm = ispixSecondHalf * 2 * cpeModule.bigPitchX * nrows / rpr;
        cpeModule.yHalfTerm = ispixSecondHalf * cpeModule.bigPitchY * ncols / cpr;
        cpeModule.xA = nrows / 2 - 2;
        cpeModule.xShift = 2 * nrows / rpr;
        cpeModule.xB = nrows / 2 - 2 + 2 * nrows / rpr;
        cpeModule.yA = ncols / 2 - 1;
        cpeModule.yShift = ncols / cpr;
        cpeModule.yB = ncols / 2 - 1 + ncols / cpr;
        cpeModule.detId = det->geographicalId().rawId();
        cpeModule.valid = 1;
      }
      newTable->buildIdSlots();
      table_ = std::move(newTable);
      tableId_ = id;
      return table_;
    }

    // device copy per (device, table): one copy + wait per IOV and device, shared by all streams
    const ::mkfitdev::OTCpeModule* deviceTable(Queue& queue, std::shared_ptr<const OTCpeTable> const& hostTable) const {
      if constexpr (mkfitdev::kCpuBackend) {
        return hostTable->modules.data();
      } else {
        const auto key = std::make_pair(static_cast<long>(alpaka::getNativeHandle(alpaka::getDev(queue))),
                                        reinterpret_cast<uintptr_t>(hostTable.get()));
        std::lock_guard<std::mutex> lock(tableMutex_);
        auto it = devTables_.find(key);
        if (it == devTables_.end()) {
          const uint32_t nRows = std::max<size_t>(hostTable->modules.size(), 1);
          auto entry = std::make_unique<DeviceTable>(DeviceTable{
              hostTable,
              cms::alpakatools::make_device_buffer<::mkfitdev::OTCpeModule[]>(alpaka::getDev(queue), nRows)});
          if (!hostTable->modules.empty())
            alpaka::memcpy(
                queue,
                entry->buf,
                cms::alpakatools::make_host_view(const_cast<::mkfitdev::OTCpeModule*>(hostTable->modules.data()),
                                                 hostTable->modules.size()));
          alpaka::wait(queue);
          it = devTables_.emplace(key, std::move(entry)).first;
        }
        return it->second->buf.data();
      }
    }

    // CPU backends: the spans, then per cluster key the GeomDet index, size and first strip | column
    static void fillHost(Phase2TrackerCluster1DCollectionNew const& clusters,
                         OTCpeTable const& cpeTable,
                         uint32_t nClusters,
                         ::mkfitdev::OTRecHitSoA::View view,
                         std::vector<DetSetSpan>& spans) {
      fillSpans(clusters, cpeTable, nClusters, spans);
      view.nHits() = nClusters;
      int32_t* moduleColumn = view.metadata().addressOf_module();
      uint16_t* clustSizeColumn = view.metadata().addressOf_clustSize();
      uint32_t* stripColumn = view.metadata().addressOf_strip();
      const auto* stored = clusters.data().data();
      for (const auto& span : spans) {
        const int32_t gind = span.tableIndex + cpeTable.firstIndex;
        for (uint32_t k = span.first; k < span.first + span.size; ++k) {
          moduleColumn[k] = gind;
          clustSizeColumn[k] = stored[k].size();
          stripColumn[k] = stripWord(stored[k]);
        }
      }
    }

    // the non-empty detsets (first cluster key, size, table index); every cluster key must be covered
    static void fillSpans(Phase2TrackerCluster1DCollectionNew const& clusters,
                          OTCpeTable const& cpeTable,
                          uint32_t nClusters,
                          std::vector<DetSetSpan>& spans) {
      spans.reserve(clusters.size());
      const auto* stored = clusters.data().data();
      uint32_t covered = 0;
      for (const auto& ds : clusters) {
        if (ds.empty())
          continue;
        const uint32_t id = ds.detId();
        const int32_t tableIndex = cpeTable.indexOf(id);
        if (tableIndex < 0)
          throw cms::Exception("MkFitAlpakaOTRecHits") << "OT cluster on DetId " << id << " without a CPE table entry";
        spans.push_back({uint32_t(&*ds.begin() - stored), uint32_t(ds.size()), tableIndex});
        covered += ds.size();
      }
      if (covered != nClusters)
        throw cms::Exception("MkFitAlpakaOTRecHits")
            << "OT clusters: detsets cover " << covered << " of " << nClusters << " keys";
    }

    // once per job: the device decode of the stored clusters equals the accessors (Phase2TrackerCluster1D layout)
    void checkClusterLayout(Phase2TrackerCluster1DCollectionNew const& clusters,
                            uint16_t const* raw,
                            uint32_t nClusters) const {
      if (nClusters == 0)
        return;
      const auto* stored = clusters.data().data();
      for (uint32_t k = 0; k < nClusters; ++k) {
        uint16_t size;
        uint32_t strip;
        ::mkfitdev::othits::decodeCluster(raw[2 * k], raw[2 * k + 1], size, strip);
        if (size != stored[k].size() || strip != stripWord(stored[k]))
          throw cms::Exception("MkFitAlpakaOTRecHits")
              << "Phase2TrackerCluster1D layout: the device decode differs from the accessors at key " << k;
      }
      layoutChecked_.store(true, std::memory_order_relaxed);
    }

    struct DeviceTable {
      std::shared_ptr<const OTCpeTable> owner;
      cms::alpakatools::device_buffer<Device, ::mkfitdev::OTCpeModule[]> buf;
    };

    const edm::EDGetTokenT<Phase2TrackerCluster1DCollectionNew> clustersToken_;
    const edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> geomToken_;
    const edm::ESGetToken<ClusterParameterEstimator<Phase2TrackerCluster1D>, TkPhase2OTCPERecord> cpeToken_;
    const edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> geomRunToken_;
    const edm::ESGetToken<ClusterParameterEstimator<Phase2TrackerCluster1D>, TkPhase2OTCPERecord> cpeRunToken_;
    const device::EDPutToken<mkfitdev::OTRecHitDeviceCollection> putToken_;
    const edm::EDPutTokenT<unsigned long long> queueToken_;
    edm::EDGetTokenT<MkFitSeedWrapper> seedsToken_;

    mutable std::atomic<bool> layoutChecked_{false};
    mutable std::mutex tableMutex_;
    mutable std::shared_ptr<const OTCpeTable> table_;
    mutable unsigned long long tableId_ = 0;
    mutable std::map<std::pair<long, uintptr_t>, std::unique_ptr<DeviceTable>> devTables_;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

DEFINE_FWK_ALPAKA_MODULE(MkFitAlpakaOTRecHitsProducer);
