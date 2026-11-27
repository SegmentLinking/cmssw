// The Phase-2 OT rechits on the device. Replaces the host
// Phase2TrackerRecHits producer (hltSiPhase2RecHits) for every reader that can take the SoA: one host pass over the
// OT clusters writes (GeomDet index, DetId, size, first strip | column) per cluster key into the SoA's prefix columns,
// one kernel (MkFitAlpakaOTRecHitsKernels.dev.cc) applies Phase2StripCPE (pitch, centre, the per-module Lorentz shift
// and pitch^2/12 errors from a per-IOV table) and Surface::toGlobal. The per-IOV table is built from the CMSSW objects:
// coveredStrips through Phase2StripCPE::driftDirection (SiPhase2OuterTrackerLorentzAngle x the local B field of the
// module, exactly the fillParam), errors through Phase2StripCPE::localParameters, topology constants from the
// RectangularPixelPhase2Topology accessors.
// produceCAHits (replaces Phase2OTRecHitsSoAConverter = hltPhase2OtRecHitsSoA): also puts the CA OT layers'
// hit SoA (P-module hits of the OT barrel, reco::TrackingRecHitsSoACollection) and the host hitModuleStart vector, made
// by a selection kernel from the OT rechit SoA.
// instance "queue" = a host product with the native handle of the queue the OT rechit SoA was
// allocated on (0 on CPU backends). With the SoA deleted early, its device readers on other queues order that queue
// after their reads.
#include <algorithm>
#include <array>
#include <atomic>
#include <cstring>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
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
#include "RecoTracker/MkFitAlpaka/interface/othits/OTCpe.h"
#include "RecoTracker/MkFitAlpaka/interface/othits/OTRecHitSoA.h"
#include "RecoTracker/MkFitAlpaka/interface/othits/alpaka/OTRecHitDeviceCollection.h"

#include "DataFormats/BeamSpot/interface/BeamSpot.h"
#include "DataFormats/SiStripDetId/interface/StripSubdetector.h"
#include "DataFormats/SiPixelDetId/interface/PixelSubdetector.h"
#include "DataFormats/TrackingRecHitSoA/interface/TrackingRecHitsHost.h"
#include "DataFormats/TrackingRecHitSoA/interface/alpaka/TrackingRecHitsSoACollection.h"
#include "MkFitAlpakaOTCAHitsKernels.h"
#include "MkFitAlpakaOTRecHitsKernels.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  namespace {
    struct OTCpeTable {
      std::vector<::mkfitdev::OTCpeModule> modules;  // index = GeomDet index - firstIndex
      int32_t firstIndex = 0;
      // CA OT layers (Phase2OTRecHitsSoAConverter::beginRun): P modules of the OT barrel in detUnits order
      std::vector<int32_t> pOffset;  // index = GeomDet index - firstIndex; -1: not a P module of the OT barrel
      uint32_t nP = 0;
      uint16_t modulesInPixel = 0;
      // raw DetId -> table index of the valid entries, open addressing (linear probing, load <= 1/2): one cache line
      // per detset instead of TrackerGeometry::idToDetUnit's hash-map node plus the GeomDet read
      std::vector<uint64_t> idSlots;  // (detId << 32) | index; 0 = empty (DetId 0 is never an OT module)
      uint32_t idShift = 32;

      static uint32_t slotOf(uint32_t id, uint32_t shift) { return (id * 0x9e3779b1u) >> shift; }
      void buildIdSlots() {
        uint32_t nValid = 0;
        for (auto const& m : modules)
          nValid += m.valid;
        uint32_t bits = 1;
        while ((1u << bits) < 2 * nValid)
          ++bits;
        idShift = 32 - bits;
        idSlots.assign(size_t(1) << bits, 0);
        const uint32_t mask = (1u << bits) - 1;
        for (uint32_t mi = 0; mi < modules.size(); ++mi) {
          if (!modules[mi].valid)
            continue;
          const uint32_t id = modules[mi].detId;
          uint32_t sl = slotOf(id, idShift);
          while (idSlots[sl] != 0) {
            if (uint32_t(idSlots[sl] >> 32) == id)
              throw cms::Exception("MkFitAlpakaOTRecHits") << "DetId " << id << " twice in the OT CPE table";
            sl = (sl + 1) & mask;
          }
          idSlots[sl] = (uint64_t(id) << 32) | mi;
        }
      }
      // table index of a valid entry with this DetId, or -1
      int32_t indexOf(uint32_t id) const {
        const uint32_t mask = idSlots.size() - 1;
        for (uint32_t sl = slotOf(id, idShift);; sl = (sl + 1) & mask) {
          const uint64_t v = idSlots[sl];
          if (v == 0)
            return -1;
          if (uint32_t(v >> 32) == id)
            return int32_t(uint32_t(v));
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
          queuePutToken_{produces("queue")},
          caHits_{iConfig.getParameter<bool>("produceCAHits")} {
      if (caHits_) {
        beamSpotToken_ = consumes(iConfig.getParameter<edm::InputTag>("beamSpot"));
        pixelSoAToken_ = consumes(iConfig.getParameter<edm::InputTag>("pixelRecHitSoASource"));
        caPutToken_ = produces();
        hmsPutToken_ = produces();
        keyStartPutToken_ = produces("caKeyStart");
      }
    }

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      desc.add<edm::InputTag>("src", edm::InputTag("hltSiPhase2Clusters"))->setComment("Phase-2 OT clusters");
      desc.add<edm::ESInputTag>("Phase2StripCPE", edm::ESInputTag("phase2StripCPEESProducer", "Phase2StripCPE"))
          ->setComment("the Phase2StripCPE (as hltSiPhase2RecHits); its per-module numbers fill the table");
      desc.add<bool>("produceCAHits", false)
          ->setComment("also put the CA OT layers' hit SoA + hitModuleStart (= Phase2OTRecHitsSoAConverter products)");
      desc.add<edm::InputTag>("beamSpot", edm::InputTag("hltOnlineBeamSpot"))->setComment("produceCAHits");
      desc.add<edm::InputTag>("pixelRecHitSoASource", edm::InputTag("hltPhase2SiPixelRecHitsSoA"))
          ->setComment("produceCAHits: pixel rechit SoA (host), for the pixel hit count");
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
      const auto t = table(iSetup.getData(cpeToken_), geom, es.get<TkPhase2OTCPERecord>().cacheIdentifier());
      const auto* devTable = deviceTable(iEvent.queue(), t);

      const uint32_t n = clusters.dataSize();
      std::vector<DetSetSpan> spans;
      mkfitdev::OTRecHitDeviceCollection product(iEvent.queue(), std::max<int32_t>(int32_t(n), 1));
      if constexpr (std::is_same_v<Device, alpaka::DevCpu>) {
        fillHost(clusters, *t, n, product.view(), spans);
      } else {
        // the host lists the detsets (one DetId-index lookup each); the raw clusters and the spans go to the device,
        // where one kernel expands module / detId / size / strip per cluster key (the same integers as fillHost)
        auto& queue = iEvent.queue();
        fillSpans(clusters, *t, n, spans);
        const uint32_t nSpans = spans.size();
        // one staging buffer and one device buffer: the stored clusters (4 bytes each), then the spans
        static_assert(sizeof(Phase2TrackerCluster1D) % alignof(DetSetSpan) == 0);
        const size_t rawBytes = size_t(n) * sizeof(Phase2TrackerCluster1D);
        const auto bytes = static_cast<uint32_t>(rawBytes + spans.size() * sizeof(DetSetSpan) + sizeof(DetSetSpan));
        auto host = cms::alpakatools::make_host_buffer<std::byte[]>(queue, bytes);
        auto* hostRaw = reinterpret_cast<uint16_t*>(host.data());
        if (n > 0)
          std::memcpy(hostRaw, clusters.data().data(), rawBytes);
        std::copy(spans.begin(), spans.end(), reinterpret_cast<DetSetSpan*>(host.data() + rawBytes));
        if (not layoutChecked_.load(std::memory_order_relaxed))
          checkClusterLayout(clusters, hostRaw, n);
        auto dev = cms::alpakatools::make_device_buffer<std::byte[]>(queue, bytes);
        alpaka::memcpy(queue, dev, host);
        mkfitdev::othits::runOTExpand(queue,
                                      product.view(),
                                      reinterpret_cast<DetSetSpan const*>(dev.data() + rawBytes),
                                      nSpans,
                                      reinterpret_cast<uint16_t const*>(dev.data()),
                                      devTable,
                                      t->firstIndex,
                                      n);
      }
      mkfitdev::othits::runOTCpe(iEvent.queue(), product.view(), devTable, t->firstIndex, n);
      if (caHits_)
        produceCA(iEvent, spans, t, devTable, n, product);
      iEvent.emplace(putToken_, std::move(product));
      unsigned long long h = 0;  // allocation queue of the SoA (0 on CPU backends: blocking queue)
#if !(defined(ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED) || defined(ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED))
      h = reinterpret_cast<unsigned long long>(alpaka::getNativeHandle(iEvent.queue()));
#endif
      iEvent.emplace(queuePutToken_, h);
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
      auto t = std::make_shared<OTCpeTable>();
      // first OT GeomDetUnit index, as Phase2StripCPE::fillParam
      auto const& dus = geom.detUnits();
      uint32_t off = dus.size();
      for (unsigned int i = 3; i < 7; ++i) {
        const auto o = geom.offsetDU(GeomDetEnumerators::tkDetEnum[i]);
        if (o != dus.size() && o < off)
          off = o;
      }
      t->firstIndex = int32_t(off);
      t->modules.resize(dus.size() - off);
      for (auto i = off; i != dus.size(); ++i) {
        auto& m = t->modules[i - off];
        std::memset(&m, 0, sizeof(m));
        auto const* det = dynamic_cast<PixelGeomDetUnit const*>(dus[i]);
        if (det == nullptr || det->index() != int(i))
          continue;
        // PixelGeomDetUnit holds a ProxyPixelTopology (surface deformations); localX/localY forward to the wrapped one
        PixelTopology const* tp = &det->specificTopology();
        if (auto const* proxy = dynamic_cast<ProxyPixelTopology const*>(tp))
          tp = &proxy->specificTopology();
        auto const* topo = dynamic_cast<RectangularPixelPhase2Topology const*>(tp);
        if (topo == nullptr)
          continue;
        // Phase2StripCPE::fillParam, verbatim
        auto pitch_x = topo->pitch().first;
        auto thickness = det->specificSurface().bounds().thickness();
        auto drift = cpe->driftDirection(*det) * thickness;
        auto lvec = drift + LocalVector(0, 0, -thickness);
        float coveredStrips = lvec.x() / pitch_x;
        m.halfCovered = 0.5f * coveredStrips;
        // errors exactly as the CPE returns them
        const auto lv = cpe->localParameters(Phase2TrackerCluster1D(0, 0, 1), *det);
        m.exx = lv.second.xx();
        m.eyy = lv.second.yy();
        if (lv.second.xy() != 0.f)
          throw cms::Exception("MkFitAlpakaOTRecHits") << "Phase2StripCPE local error xy != 0 for det index " << i;
        // RectangularPixelPhase2Topology::localX/localY constants (same integer and float expressions)
        const int nrows = topo->nrows(), ncols = topo->ncolumns(), rpr = topo->rowsperroc(), cpr = topo->colsperroc();
        m.pitchX = topo->pitch().first;
        m.pitchY = topo->pitch().second;
        m.bigPitchX = topo->pitchbigpixelX();
        m.bigPitchY = topo->pitchbigpixelY();
        m.xOffset = topo->xoffset();
        m.yOffset = topo->yoffset();
        const int one = 1;
        m.xHalfTerm = one * 2 * m.bigPitchX * nrows / rpr;
        m.yHalfTerm = one * m.bigPitchY * ncols / cpr;
        m.xA = nrows / 2 - 2;
        m.xShift = 2 * nrows / rpr;
        m.xB = nrows / 2 - 2 + 2 * nrows / rpr;
        m.yA = ncols / 2 - 1;
        m.yShift = ncols / cpr;
        m.yB = ncols / 2 - 1 + ncols / cpr;
        // Surface rotation rows and position (GloballyPositioned<float>)
        auto const& s = det->surface();
        auto const& r = s.rotation();
        const float rr[9] = {r.xx(), r.xy(), r.xz(), r.yx(), r.yy(), r.yz(), r.zx(), r.zy(), r.zz()};
        for (int j = 0; j < 9; ++j)
          m.r[j] = rr[j];
        m.p[0] = s.position().x();
        m.p[1] = s.position().y();
        m.p[2] = s.position().z();
        m.detId = det->geographicalId().rawId();
        m.valid = 1;
      }
      // CA OT layers: P modules of the OT barrel in detUnits order (Phase2OTRecHitsSoAConverter::beginRun)
      t->pOffset.assign(t->modules.size(), -1);
      uint32_t nPix = 0, nP = 0;
      for (auto const* du : dus) {
        const DetId d = du->geographicalId();
        if (d.subdetId() == PixelSubdetector::PixelBarrel || d.subdetId() == PixelSubdetector::PixelEndcap)
          ++nPix;
        if (geom.getDetectorType(d) == TrackerGeometry::ModuleType::Ph2PSP && d.subdetId() == StripSubdetector::TOB) {
          const int32_t mi = du->index() - t->firstIndex;
          if (mi < 0 || mi >= int32_t(t->modules.size()))
            throw cms::Exception("MkFitAlpakaOTRecHits") << "P module outside the OT index range";
          t->pOffset[mi] = int32_t(nP++);
        }
      }
      t->nP = nP;
      t->modulesInPixel = uint16_t(nPix);
      t->buildIdSlots();
      table_ = std::move(t);
      tableId_ = id;
      return table_;
    }

    // device copy per (device, table): one copy + wait per IOV and device, shared by all streams
    const ::mkfitdev::OTCpeModule* deviceTable(Queue& queue, std::shared_ptr<const OTCpeTable> const& t) const {
      return deviceTables(queue, t).first;
    }
    std::pair<const ::mkfitdev::OTCpeModule*, const int32_t*> deviceTables(
        Queue& queue, std::shared_ptr<const OTCpeTable> const& t) const {
      if constexpr (std::is_same_v<Device, alpaka::DevCpu>) {
        return {t->modules.data(), t->pOffset.data()};
      } else {
        const auto key = std::make_pair(static_cast<long>(alpaka::getNativeHandle(alpaka::getDev(queue))),
                                        reinterpret_cast<uintptr_t>(t.get()));
        std::lock_guard<std::mutex> lock(tableMutex_);
        auto it = devTables_.find(key);
        if (it == devTables_.end()) {
          const uint32_t n = std::max<size_t>(t->modules.size(), 1);
          auto e = std::make_unique<DeviceTable>(
              DeviceTable{t,
                          cms::alpakatools::make_device_buffer<::mkfitdev::OTCpeModule[]>(alpaka::getDev(queue), n),
                          cms::alpakatools::make_device_buffer<int32_t[]>(alpaka::getDev(queue), n)});
          if (!t->modules.empty()) {
            alpaka::memcpy(queue,
                           e->buf,
                           cms::alpakatools::make_host_view(const_cast<::mkfitdev::OTCpeModule*>(t->modules.data()),
                                                            t->modules.size()));
            alpaka::memcpy(
                queue,
                e->pOff,
                cms::alpakatools::make_host_view(const_cast<int32_t*>(t->pOffset.data()), t->pOffset.size()));
          }
          alpaka::wait(queue);
          it = devTables_.emplace(key, std::move(e)).first;
        }
        return {it->second->buf.data(), it->second->pOff.data()};
      }
    }

    // the one host pass: per cluster key the GeomDet index, DetId, size and first strip | column
    static void fillHost(Phase2TrackerCluster1DCollectionNew const& clusters,
                         OTCpeTable const& t,
                         uint32_t n,
                         ::mkfitdev::OTRecHitSoA::View v,
                         std::vector<DetSetSpan>& spans) {
      v.nHits() = n;
      spans.reserve(clusters.size());
      int32_t* mod = v.metadata().addressOf_module();
      uint32_t* did = v.metadata().addressOf_detId();
      uint16_t* sz = v.metadata().addressOf_clustSize();
      uint32_t* st = v.metadata().addressOf_strip();
      const auto* data0 = clusters.data().data();
      uint32_t covered = 0;
      for (const auto& ds : clusters) {
        if (ds.empty())
          continue;
        const uint32_t id = ds.detId();
        const int32_t mi = t.indexOf(id);
        if (mi < 0)
          throw cms::Exception("MkFitAlpakaOTRecHits") << "OT cluster on DetId " << id << " without a CPE table entry";
        const int32_t gind = mi + t.firstIndex;
        const uint32_t k0 = &*ds.begin() - data0;
        spans.push_back({k0, uint32_t(ds.size()), mi});
        uint32_t k = k0;
        for (const auto& c : ds) {
          mod[k] = gind;
          did[k] = id;
          sz[k] = c.size();
          st[k] = (c.firstStrip() & 0xffffu) | (c.column() << 16);
          ++k;
        }
        covered += k - k0;
      }
      if (covered != n)
        throw cms::Exception("MkFitAlpakaOTRecHits")
            << "OT clusters: detsets cover " << covered << " of " << n << " keys";
    }

    // device path: the spans only (fillHost's checks, no per-cluster loop)
    static void fillSpans(Phase2TrackerCluster1DCollectionNew const& clusters,
                          OTCpeTable const& t,
                          uint32_t n,
                          std::vector<DetSetSpan>& spans) {
      spans.reserve(clusters.size());
      const auto* data0 = clusters.data().data();
      uint32_t covered = 0;
      for (const auto& ds : clusters) {
        if (ds.empty())
          continue;
        const uint32_t id = ds.detId();
        const int32_t mi = t.indexOf(id);
        if (mi < 0)
          throw cms::Exception("MkFitAlpakaOTRecHits") << "OT cluster on DetId " << id << " without a CPE table entry";
        spans.push_back({uint32_t(&*ds.begin() - data0), uint32_t(ds.size()), mi});
        covered += ds.size();
      }
      if (covered != n)
        throw cms::Exception("MkFitAlpakaOTRecHits")
            << "OT clusters: detsets cover " << covered << " of " << n << " keys";
    }

    // once per job: the device decode of the stored clusters equals the accessors (Phase2TrackerCluster1D layout)
    void checkClusterLayout(Phase2TrackerCluster1DCollectionNew const& clusters,
                            uint16_t const* raw,
                            uint32_t n) const {
      if (n == 0)
        return;
      const auto* c = clusters.data().data();
      for (uint32_t k = 0; k < n; ++k) {
        uint16_t size;
        uint32_t strip;
        ::mkfitdev::othits::decodeCluster(raw[2 * k], raw[2 * k + 1], size, strip);
        if (size != c[k].size() || strip != ((c[k].firstStrip() & 0xffffu) | (c[k].column() << 16)))
          throw cms::Exception("MkFitAlpakaOTRecHits")
              << "Phase2TrackerCluster1D layout: the device decode differs from the accessors at key " << k;
      }
      layoutChecked_.store(true, std::memory_order_relaxed);
    }

    // CA OT layers: per-event P-module ranges on the host (from the cluster detset sizes), rows on the device
    void produceCA(device::Event& iEvent,
                   std::vector<DetSetSpan> const& spans,
                   std::shared_ptr<const OTCpeTable> const& tsp,
                   const ::mkfitdev::OTCpeModule* devTable,
                   uint32_t n,
                   mkfitdev::OTRecHitDeviceCollection const& ot) const {
      auto& queue = iEvent.queue();
      OTCpeTable const& t = *tsp;
      const auto& bs = iEvent.get(beamSpotToken_);
      const int nPixelHits = iEvent.get(pixelSoAToken_).view().trackingHits().metadata().size();
      const uint32_t nP = t.nP;
      if (nP == 0)
        throw cms::Exception("MkFitAlpakaOTRecHits") << "no P module in the OT barrel";
      // hitStart[0..nP] (exclusive prefix of the P-module hit counts), keyStart[0..nP)
      auto hostSK = cms::alpakatools::make_host_buffer<uint32_t[]>(queue, 2 * nP + 1);
      uint32_t* hitStart = hostSK.data();
      uint32_t* keyStart = hostSK.data() + nP + 1;
      std::fill(hitStart, hitStart + 2 * nP + 1, 0u);
      for (const auto& sp : spans) {
        const int32_t off = t.pOffset[sp.mi];
        if (off < 0)
          continue;
        hitStart[off + 1] = sp.size;
        keyStart[off] = sp.first;
      }
      for (uint32_t i = 0; i < nP; ++i)
        hitStart[i + 1] += hitStart[i];
      const uint32_t nPHits = hitStart[nP];
      std::vector<uint32_t> hms(nP + 1);
      for (uint32_t i = 0; i <= nP; ++i)
        hms[i] = hitStart[i] + uint32_t(nPixelHits);
      reco::TrackingRecHitsSoACollection ca(queue, nPHits, nP);
      // moduleStart (nP + 1 entries) to the hitModules block
      auto mv = ca.view().hitModules();
      alpaka::memcpy(
          queue,
          cms::alpakatools::make_device_view(alpaka::getDev(queue), mv.metadata().addressOf_moduleStart(), nP + 1),
          cms::alpakatools::make_host_view(hms.data(), nP + 1));
      // hitStart / keyStart to the device (CPU backends: used in place)
      const auto tabs = deviceTables(queue, tsp);
      const uint32_t* dSK = hostSK.data();
      std::optional<cms::alpakatools::device_buffer<Device, uint32_t[]>> devSK;
      if constexpr (!std::is_same_v<Device, alpaka::DevCpu>) {
        devSK.emplace(cms::alpakatools::make_device_buffer<uint32_t[]>(queue, 2 * nP + 1));
        alpaka::memcpy(queue, *devSK, hostSK);
        dSK = devSK->data();
      }
      mkfitdev::othits::CAHitsParams p{bs.x0(), bs.y0(), bs.z0(), t.firstIndex, n, t.modulesInPixel};
      mkfitdev::othits::runOTCAHits(
          queue, ot.const_view(), devTable, tabs.second, dSK, dSK + nP + 1, p, ca.view().trackingHits());
      iEvent.emplace(caPutToken_, std::move(ca));
      iEvent.emplace(hmsPutToken_, std::move(hms));
      // first OT cluster key of every P module (CA row hms[i] - nPixelHits + j <-> key keyStart[i] + j): hltInputLSTDevice
      // builds its CA-row -> key map from it
      iEvent.emplace(keyStartPutToken_, keyStart, keyStart + nP);
    }

    struct DeviceTable {
      std::shared_ptr<const OTCpeTable> owner;
      cms::alpakatools::device_buffer<Device, ::mkfitdev::OTCpeModule[]> buf;
      cms::alpakatools::device_buffer<Device, int32_t[]> pOff;
    };

    const edm::EDGetTokenT<Phase2TrackerCluster1DCollectionNew> clustersToken_;
    const edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> geomToken_;
    const edm::ESGetToken<ClusterParameterEstimator<Phase2TrackerCluster1D>, TkPhase2OTCPERecord> cpeToken_;
    const edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> geomRunToken_;
    const edm::ESGetToken<ClusterParameterEstimator<Phase2TrackerCluster1D>, TkPhase2OTCPERecord> cpeRunToken_;
    const device::EDPutToken<mkfitdev::OTRecHitDeviceCollection> putToken_;
    const edm::EDPutTokenT<unsigned long long> queuePutToken_;
    const bool caHits_;
    edm::EDGetTokenT<::reco::BeamSpot> beamSpotToken_;
    edm::EDGetTokenT<::reco::TrackingRecHitHost> pixelSoAToken_;
    device::EDPutToken<reco::TrackingRecHitsSoACollection> caPutToken_;
    edm::EDPutTokenT<std::vector<uint32_t>> hmsPutToken_;
    edm::EDPutTokenT<std::vector<uint32_t>> keyStartPutToken_;

    mutable std::atomic<bool> layoutChecked_{false};
    mutable std::mutex tableMutex_;
    mutable std::shared_ptr<const OTCpeTable> table_;
    mutable unsigned long long tableId_ = 0;
    mutable std::map<std::pair<long, uintptr_t>, std::unique_ptr<DeviceTable>> devTables_;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

DEFINE_FWK_ALPAKA_MODULE(MkFitAlpakaOTRecHitsProducer);
