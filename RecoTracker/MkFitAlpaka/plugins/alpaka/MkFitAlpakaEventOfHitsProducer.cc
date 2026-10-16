// Portable EventOfHits: builds the device EventOfHits (HitSoA + per-layer binning reproducing LayerOfHits) from the
// pixel rechit SoA and the device OT rechit SoA, with the MkFitCore host MkFitGeometry.
// Dead modules: the pixel-quality dead regions are collected on the host exactly as MkFitEventOfHitsProducer does
// (the strip quality DB is off in the HLT LST step and is not supported here), the dead-bin table is filled on device.
//
// the device EventOfHits is put into the Event as the event product
// mkfitdev::EventOfHitsDeviceCollection (blocks hits, layers, binnedHits, bins). Events beyond the device build limits
// are skipped with a warning
// (no exception at event time). Skip contract: a skipped event gets an EMPTY EventOfHits (all four
// blocks of size 0; nLayers == 0 marks it, readable on the host from the metadata without a sync). The build module
// then sets the status counter eventSkipped and puts empty TrackSoAs.
//
// The HitSoA is made on the device from the pixel rechit SoA (hltPhase2SiPixelRecHitsSoA) and the device OT rechit
// SoA, with a per-module table (rotation, position, layer, uniqueIdInLayer; an ES product built per IOV), bitwise equal
// to the RecoTracker/MkFit hit converters, which leave the menu. What stays on the host: one pass over the legacy pixel
// clusters (row permutation moduleStart + originalId and the legacy sizes; the legacy cluster order is not the SoA
// order).
//
// guard - a pixel row found through originalId is accepted only if the SoA local position is
// bitwise the legacy rechit's; otherwise (and for clusters without originalId) the row is found by position. Both
// cases are counted (LogWarning on the first, totals at endJob). on
// CPU backends the raw host rows are per-stream scratch (grow-only capacity, reused every event).
//
// instance "queue" = a host product with the native handle of the queue the EventOfHits
// buffer was allocated on (0 on CPU backends). With the product deleted early (canDeleteEarly, after its last
// consumer, MkFitAlpakaFitDeviceProducer), the caching allocator orders the block's reuse after the work of that queue
// only; the fit checks that it ran on this queue and waits otherwise.
//
// the device OT rechit SoA is deleted early, after its last reader (canDeleteEarly).
// otSoAQueue = the SoA's allocation queue: when this module runs on another queue, that queue is made to wait (on the
// device) for the kernel that reads the SoA, so the caching allocator cannot hand the block out while it is read.
#include <algorithm>
#include <array>
#include <optional>
#include <atomic>
#include <cstddef>
#include <type_traits>
#include <cstring>
#include <vector>

#include "CondFormats/DataRecord/interface/SiPixelQualityRcd.h"
#include "CondFormats/SiPixelObjects/interface/SiPixelQuality.h"
#include "DataFormats/TrackerCommon/interface/TrackerDetSide.h"
#include "DataFormats/TrackerCommon/interface/TrackerTopology.h"
#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "RecoTracker/MkFit/interface/MkFitClusterIndexToHit.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/global/EDProducer.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/Event.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EventSetup.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
#include "RecoTracker/MkFit/interface/MkFitGeometry.h"
#include "RecoTracker/MkFitAlpaka/interface/hits/HitModuleTableESData.h"
#include "RecoTracker/MkFitAlpaka/plugins/alpaka/MkFitAlpakaReleaseOrder.h"
#include "RecoTracker/MkFitCore/interface/HitStructures.h"
#include "RecoTracker/Record/interface/TrackerRecoGeometryRecord.h"

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDPutToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ESGetToken.h"
#include "RecoTracker/MkFitAlpaka/interface/EventOfHitsProduct.h"
#include "RecoTracker/MkFitAlpaka/interface/alpaka/EventOfHitsProduct.h"
#include "RecoTracker/MkFitAlpaka/interface/es/ESData.h"
#include "RecoTracker/MkFitAlpaka/interface/hits/LayerAxes.h"
#include "RecoTracker/MkFitAlpaka/interface/hits/alpaka/EventOfHitsBuild.h"
#include "RecoTracker/MkFitAlpaka/interface/hits/alpaka/DeviceHitsBuild.h"
#include "RecoTracker/MkFitAlpaka/interface/othits/alpaka/OTRecHitDeviceCollection.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDGetToken.h"
#include "DataFormats/SiPixelCluster/interface/SiPixelCluster.h"
#include "DataFormats/Phase2TrackerCluster/interface/Phase2TrackerCluster1D.h"
#include "DataFormats/TrackerRecHit2D/interface/Phase2TrackerRecHit1D.h"
#include "DataFormats/TrackerRecHit2D/interface/SiPixelRecHitCollection.h"
#include "DataFormats/TrackingRecHitSoA/interface/TrackingRecHitsHost.h"
#include "DataFormats/TrackingRecHitSoA/interface/alpaka/TrackingRecHitsSoACollection.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include <memory>
#include <stdexcept>

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  namespace {
    // per-stream raw host rows of the device hit input (CPU backends); rows past the event's count are unused
    struct EohScratch {
      int32_t cap = 0;
      uint64_t nAllocs = 0;
      std::optional<::mkfitdev::DeviceHitRawHostCollection> raw;
      // CPU backends, the EventOfHits product's memory (grow-only, 25% headroom)
      std::optional<cms::alpakatools::host_buffer<std::byte[]>> product;
      std::size_t productCap = 0;
      uint64_t nProductAllocs = 0;
    };
    // per-event counts of the pixel cluster key -> SoA row guard
    struct RawGuard {
      uint32_t keyMismatch = 0;     // originalId row whose local position is not the legacy one (row found by position)
      uint32_t positionSearch = 0;  // clusters without originalId (row found by position)
    };
  }  // namespace

  class MkFitAlpakaEventOfHitsProducer : public global::EDProducer<edm::StreamCache<EohScratch>> {
  public:
    explicit MkFitAlpakaEventOfHitsProducer(edm::ParameterSet const& iConfig)
        : EDProducer<edm::StreamCache<EohScratch>>(iConfig),
          mkFitGeomToken_{esConsumes()},
          productToken_{produces()},
          queueToken_{produces("queue")},
          usePixelQualityDB_{iConfig.getParameter<bool>("usePixelQualityDB")},
          produceSrcRows_{iConfig.getParameter<bool>("producePixelSrcRows")},
          produceIndexToHit_{iConfig.getParameter<bool>("producePixelIndexToHit")} {
      if (produceSrcRows_)
        srcRowsToken_ = produces("pixelSrcRows");
      if (produceIndexToHit_)
        indexToHitToken_ = produces("pixelIndexToHit");
      pixelSoAToken_ = consumes(iConfig.getParameter<edm::InputTag>("pixelSoA"));
      pixelRecHitsToken_ = consumes(iConfig.getParameter<edm::InputTag>("pixelRecHits"));
      // OT rows from the device OT rechit SoA (no legacy OT rechits read)
      otSoAToken_ = consumes(iConfig.getParameter<edm::InputTag>("otSoA"));
      if (auto const q = iConfig.getParameter<edm::InputTag>("otSoAQueue"); !q.label().empty())
        otSoAQueueToken_ = consumes(q);
      pixelClustersToken_ = consumes(iConfig.getParameter<edm::InputTag>("pixelClusters"));
      otClustersToken_ = consumes(iConfig.getParameter<edm::InputTag>("otClusters"));
      moduleTableToken_ = esConsumes(edm::ESInputTag("", iConfig.getParameter<std::string>("moduleTable")));
      esHostToken_ = esConsumes(iConfig.getParameter<edm::ESInputTag>("esData"));
      if (usePixelQualityDB_) {
        pixelQualityToken_ = esConsumes();
        geomToken_ = esConsumes();
      }
    }

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      desc.add("usePixelQualityDB", true)
          ->setComment("Use SiPixelQuality DB information (as MkFitEventOfHitsProducer)");
      desc.add("esData", edm::ESInputTag{"", ""})
          ->setComment("MkFitAlpakaESProducer ComponentName (the static layer table, q axes)");
      desc.add("pixelSoA", edm::InputTag{"hltPhase2SiPixelRecHitsSoA"})->setComment("pixel rechit SoA (host)");
      desc.add("pixelRecHits", edm::InputTag{"hltSiPixelRecHits"})
          ->setComment("legacy pixel rechits (cluster key -> SoA row, legacy cluster sizes)");
      desc.add("otSoA", edm::InputTag{"hltMkFitAlpakaOTRecHits"})
          ->setComment("device OT rechit SoA (MkFitAlpakaOTRecHitsProducer): the OT rows of the hits");
      desc.add("otSoAQueue", edm::InputTag{""})
          ->setComment(
              "early deletion of otSoA: its producer's \"queue\" product; on another queue this module"
              "orders that queue after its read of the SoA (empty = off)");
      desc.add("pixelClusters", edm::InputTag{"hltSiPixelClusters"})
          ->setComment("legacy pixel clusters of the rechits (indexed by key, as convertHits)");
      desc.add("otClusters", edm::InputTag{"hltSiPhase2Clusters"})->setComment("OT clusters of the rechits");
      desc.add<std::string>("moduleTable", "MkFitAlpakaHitModuleTable")
          ->setComment("ComponentName of the MkFitAlpakaEventOfHitsModuleTableESProducer product (per-module table)");
      desc.add("producePixelSrcRows", false)
          ->setComment(
              "also put the pixel cluster key -> pixel rechit SoA row map ('pixelSrcRows', host "
              "std::vector<uint32_t>, kNoHit for keys without a rechit; EMPTY when a row had to be found by "
              "position) for hltInputLSTDevice's pixel-key map");
      desc.add("producePixelIndexToHit", false)
          ->setComment(
              "also put the pixel MkFitClusterIndexToHit ('pixelIndexToHit'), filled in the "
              "same pass over the legacy rechits as convertHits");
      descriptions.addWithDefaultLabel(desc);
    }

    std::unique_ptr<EohScratch> beginStream(edm::StreamID) const override { return std::make_unique<EohScratch>(); }

    void endStream(edm::StreamID sid) const override { productAllocs_ += streamCache(sid)->nProductAllocs; }

    void endJob() override {
      if (!otSoAQueueToken_.isUninitialized())
        edm::LogInfo("MkFitAlpakaEventOfHits")
            << "OT rechit SoA release: same queue " << nOtSameQueue_ << ", ordered " << nOtOrdered_;
      if (productAllocs_ > 0)
        edm::LogInfo("MkFitAlpakaEventOfHits") << "product block allocations " << productAllocs_ << ", largest product "
                                               << 1e-6 * productMaxBytes_ << " MB";
      edm::LogInfo("MkFitAlpakaEventOfHits") << "pixel rows found by position: originalId mismatches " << nKeyMismatch_
                                             << ", clusters without originalId " << nPositionSearch_;
    }

    void produce(edm::StreamID sid, device::Event& iEvent, device::EventSetup const& iSetup) const override {
      const auto& mkFitGeom = iSetup.getData(mkFitGeomToken_);

      // dead regions as MkFitEventOfHitsProducer (pixel quality part)
      std::vector<::mkfitdev::DeadRegionDev> deads;
      if (usePixelQualityDB_) {
        const auto& trackerGeom = iSetup.getData(geomToken_);
        const auto& pixelQuality = iSetup.getData(pixelQualityToken_);
        for (const auto& bp : pixelQuality.getBadComponentList()) {
          const DetId detid(bp.DetID);
          const auto& surf = trackerGeom.idToDet(detid)->surface();
          bool isBarrel = (mkFitGeom.topology()->side(detid) == static_cast<unsigned>(TrackerDetSide::Barrel));
          const auto ilay = mkFitGeom.mkFitLayerNumber(detid);
          const auto q1 = isBarrel ? surf.zSpan().first : surf.rSpan().first;
          const auto q2 = isBarrel ? surf.zSpan().second : surf.rSpan().second;
          if (bp.errorType == 0)
            deads.push_back({surf.phiSpan().first, surf.phiSpan().second, q1, q2, ilay});
        }
      }

      // raw host rows of the hit input (pixel rows, then strip rows)
      uint32_t nPix = 0, nStr = 0;
      std::optional<::mkfitdev::DeviceHitRawHostCollection> raw;
      const ::mkfitdev::DeviceHitRawHostCollection* rawRef = nullptr;  // raw or the per-stream scratch
      {
        const auto& pixRH = iEvent.get(pixelRecHitsToken_);
        nPix = wrapperRows(pixRH);
        // one rechit per OT cluster in cluster-key order (Phase2TrackerRecHits): converter rows = OT clusters
        nStr = iEvent.get(otClustersToken_).dataSize();
        const int32_t nRaw = int32_t(nPix);  // the OT rows come from the device OT rechit SoA
        ::mkfitdev::DeviceHitRawHostCollection* rawP = nullptr;
        if constexpr (kScratchReuse) {  // grow-only per-stream rows (25% headroom), reused every event
          EohScratch& sc = *streamCache(sid);
          if (!sc.raw || nRaw > sc.cap) {
            sc.cap = ((nRaw + nRaw / 4 + 4095) / 4096) * 4096;
            sc.raw.reset();
            sc.raw.emplace(iEvent.queue(), sc.cap);
            ++sc.nAllocs;
          }
          rawP = &*sc.raw;
        } else {
          raw.emplace(iEvent.queue(), nRaw);
          rawP = &*raw;
        }
        RawGuard guard;
        std::optional<MkFitClusterIndexToHit> indexToHit;
        if (produceIndexToHit_)
          indexToHit.emplace();
        fillRaw(iEvent.get(pixelSoAToken_),
                pixRH,
                iEvent.getHandle(pixelClustersToken_),
                nPix,
                rawP->view(),
                guard,
                indexToHit ? &indexToHit->hits() : nullptr);
        checkGuard(iEvent.id().event(), guard);
        if (indexToHit)
          iEvent.emplace(indexToHitToken_, std::move(*indexToHit));
        if (produceSrcRows_) {  // fillRaw's srcRow (moduleStart[det] + originalId, position-checked); empty if not exact
          std::vector<uint32_t> rows;
          if (guard.keyMismatch == 0 && guard.positionSearch == 0) {
            const uint32_t* src = rawP->view().metadata().addressOf_srcRow();
            rows.assign(src, src + nPix);
          }
          iEvent.emplace(srcRowsToken_, std::move(rows));
        }
        rawRef = rawP;
      }
      const auto axes = layerAxisInputsES(iSetup.getData(esHostToken_));
      // the EventOfHits event product (one PortableCollection over the four blocks)
      const uint32_t nHits = nPix + nStr, nLayers = axes.size(), nBins = ::mkfitdev::totalBins(axes);
      if (nLayers > ::mkfitdev::kMaxLayers)
        throw cms::Exception("MkFitAlpakaEventOfHits")
            << nLayers << " mkFit layers do not fit the int8_t layer column of HitSoA";
      const std::array<int32_t, 4> sizes{{int32_t(nHits), int32_t(nLayers), int32_t(nHits), int32_t(nBins)}};
      mkfitdev::EventOfHitsDeviceCollection product = makeProduct(sid, iEvent.queue(), sizes);
      fillFromDeviceInputs(iEvent, iSetup, product, *rawRef, nPix, nStr, axes);
      otReleaseGuard(iEvent);
      if (!mkfitdev::hits::runBuildEventOfHits(iEvent.queue(), product, deads)) {
        skipEvent(iEvent, nHits, nLayers);
        return;
      }
      iEvent.emplace(productToken_, std::move(product));
      putQueue(iEvent);
    }

  private:
    // convertHits row count: max(last hit's cluster key + 1, dataSize()), 0 for an empty collection
    template <typename C>
    static uint32_t wrapperRows(C const& hits) {
      if (hits.empty())
        return 0;
      return std::max<uint32_t>(hits.data().back().firstClusterRef().index() + 1, hits.dataSize());
    }

    // host part of the device hit input: pixel rows = SoA row of each legacy cluster key (moduleStart[det] +
    // originalId; persisted clusters have no originalId, then the SoA row of the module with the bitwise-equal local
    // position, as the legacy rechit position is a copy of the SoA one) + legacy sizes; the OT rows come from the
    // device OT rechit SoA (setOT)
    static void fillRaw(::reco::TrackingRecHitHost const& soa,
                        SiPixelRecHitCollection const& pixRH,
                        edm::Handle<SiPixelClusterCollectionNew> const& pixCluH,
                        uint32_t nPix,
                        ::mkfitdev::DeviceHitRawSoA::View rv,
                        RawGuard& guard,
                        std::vector<TrackingRecHit const*>* indexToHit) {
      const auto& pixClu = *pixCluH;
      uint32_t* src = rv.metadata().addressOf_srcRow();
      for (uint32_t i = 0; i < nPix; ++i)
        src[i] = ::mkfitdev::kNoHit;
      const auto hv = soa.const_view().trackingHits();
      const auto mv = soa.const_view().hitModules();
      const uint32_t nSoa = soa.nHits();
      uint32_t* spans = rv.metadata().addressOf_spans();
      // MkFitClusterIndexToHit as convertHits: max(last hit's key + 1, dataSize()) entries, each hit at its key
      if (indexToHit && !pixRH.empty())
        indexToHit->resize(nPix, nullptr);
      for (const auto& ds : pixRH) {
        if (ds.empty())
          continue;
        const uint32_t gind = ds.begin()->det()->index();
        const uint32_t b = mv[gind].moduleStart(), e = mv[gind + 1].moduleStart();
        for (const auto& h : ds) {
          const auto ref = h.firstClusterRef();
          if (ref.id() != pixCluH.id())  // convertHits reads the cluster by key from the collection of the hits' refs
            throw cms::Exception("MkFitAlpakaEventOfHits") << "pixelClusters is not the collection of the rechits";
          const uint32_t k = ref.index();
          const auto& clu = pixClu.data()[k];
          uint32_t j = ::mkfitdev::kNoHit;
          const float lx = h.localPosition().x(), ly = h.localPosition().y();
          if (clu.originalId() != SiPixelCluster::invalidClusterId) {
            j = b + clu.originalId();
            // the SoA row of originalId must carry the legacy local position (bitwise: the legacy one is a copy)
            if (!(j < e && j < nSoa && hv[j].xLocal() == lx && hv[j].yLocal() == ly)) {
              ++guard.keyMismatch;
              j = ::mkfitdev::kNoHit;
            }
          } else {
            ++guard.positionSearch;
          }
          if (j == ::mkfitdev::kNoHit) {
            for (uint32_t r = b; r < e && r < nSoa; ++r)
              if (hv[r].xLocal() == lx && hv[r].yLocal() == ly) {
                j = r;
                break;
              }
          }
          if (j >= nSoa || hv[j].detectorIndex() != gind)
            throw cms::Exception("MkFitAlpakaEventOfHits")
                << "pixel cluster key " << k << " (det index " << gind << ") has no row in the pixel rechit SoA";
          src[k] = j;
          spans[k] = uint32_t(clu.sizeX()) | (uint32_t(clu.sizeY()) << 16);
          if (indexToHit) {
            if (k >= indexToHit->size())
              indexToHit->resize(k + 1, nullptr);
            (*indexToHit)[k] = &h;
          }
        }
      }
    }

    // counters per job; the first event with a guard hit warns
    void checkGuard(unsigned long long evt, RawGuard const& g) const {
      if (g.keyMismatch == 0 && g.positionSearch == 0)
        return;
      nKeyMismatch_ += g.keyMismatch;
      nPositionSearch_ += g.positionSearch;
      if (!warned_.exchange(true))
        edm::LogWarning("MkFitAlpakaEventOfHits")
            << "event " << evt << ": device hit input found " << g.keyMismatch + g.positionSearch
            << " pixel rows by position (" << g.keyMismatch << " originalId mismatches, " << g.positionSearch
            << " clusters without originalId); totals at endJob";
    }

    static void setRaw(mkfitdev::hits::DeviceHitInputs& in, ::mkfitdev::DeviceHitRawSoA::ConstView v) {
      const auto m = v.metadata();
      in.srcRow = m.addressOf_srcRow();
      in.module = m.addressOf_module();
      in.spans = m.addressOf_spans();
      in.lx = m.addressOf_lx();
      in.ly = m.addressOf_ly();
      in.exx = m.addressOf_exx();
      in.exy = m.addressOf_exy();
      in.eyy = m.addressOf_eyy();
    }

    // OT rows straight from the device OT rechit SoA (device memory; row = OT cluster key)
    void setOT(device::Event& iEvent, mkfitdev::hits::DeviceHitInputs& in) const {
      const auto m = iEvent.get(otSoAToken_).const_view().metadata();
      if (static_cast<uint32_t>(m.size()) < in.nStrip)
        throw cms::Exception("MkFitAlpakaEventOfHits") << "OT rechit SoA has fewer rows than OT clusters";
      in.otModule = m.addressOf_module();
      in.otSize = m.addressOf_clustSize();
      in.otLx = m.addressOf_lx();
      in.otLy = m.addressOf_ly();
      in.otExx = m.addressOf_exx();
      in.otEyy = m.addressOf_eyy();
    }

    template <typename T>
    static auto hostView(const T* p, uint32_t n) {
      return cms::alpakatools::make_host_view(const_cast<T*>(p), n);
    }

    void fillFromDeviceInputs(device::Event& iEvent,
                              device::EventSetup const& iSetup,
                              mkfitdev::EventOfHitsDeviceCollection& product,
                              ::mkfitdev::DeviceHitRawHostCollection const& raw,
                              uint32_t nPix,
                              uint32_t nStr,
                              std::vector<::mkfitdev::LayerAxisInput> const& axes) const {
      auto& queue = iEvent.queue();
      // per-module table: the ES product (one device copy per IOV)
      const ::mkfitdev::HitModuleDev* moduleTable = iSetup.getData(moduleTableToken_).data();
      const auto& soa = iEvent.get(pixelSoAToken_);
      const auto soaView = soa.const_view();
      const auto hitsView = soaView.trackingHits();
      const auto hm = hitsView.metadata();  // refers to hitsView: keep the view alive
      const uint32_t nSoa = soa.nHits();
      mkfitdev::hits::DeviceHitInputs in{};
      in.nPixel = nPix;
      in.nStrip = nStr;
      if constexpr (std::is_same_v<Device, alpaka::DevCpu>) {
        in.xLocal = hm.addressOf_xLocal();
        in.yLocal = hm.addressOf_yLocal();
        in.xerrLocal = hm.addressOf_xerrLocal();
        in.yerrLocal = hm.addressOf_yerrLocal();
        in.detectorIndex = hm.addressOf_detectorIndex();
        in.modules = moduleTable;
        setRaw(in, raw.const_view());
        setOT(iEvent, in);
        ::mkfitdev::fillLayers(axes, nPix, product.view().layers());
        mkfitdev::hits::runFillHitsFromDeviceInputs(queue, in, product.view().hits());
      } else {
        // pixel SoA columns and raw rows to the device
        auto dX = cms::alpakatools::make_device_buffer<float[]>(queue, nSoa);
        auto dY = cms::alpakatools::make_device_buffer<float[]>(queue, nSoa);
        auto dEX = cms::alpakatools::make_device_buffer<float[]>(queue, nSoa);
        auto dEY = cms::alpakatools::make_device_buffer<float[]>(queue, nSoa);
        auto dDet = cms::alpakatools::make_device_buffer<uint16_t[]>(queue, nSoa);
        alpaka::memcpy(queue, dX, hostView(hm.addressOf_xLocal(), nSoa));
        alpaka::memcpy(queue, dY, hostView(hm.addressOf_yLocal(), nSoa));
        alpaka::memcpy(queue, dEX, hostView(hm.addressOf_xerrLocal(), nSoa));
        alpaka::memcpy(queue, dEY, hostView(hm.addressOf_yerrLocal(), nSoa));
        alpaka::memcpy(queue, dDet, hostView(hm.addressOf_detectorIndex(), nSoa));
        mkfitdev::hits::DeviceHitRawDeviceCollection rawD(queue, raw.view().metadata().size());
        alpaka::memcpy(queue, rawD.buffer(), raw.buffer());
        in.xLocal = dX.data();
        in.yLocal = dY.data();
        in.xerrLocal = dEX.data();
        in.yerrLocal = dEY.data();
        in.detectorIndex = dDet.data();
        in.modules = moduleTable;
        setRaw(in, rawD.const_view());
        setOT(iEvent, in);
        copyLayers(queue, product, nPix, axes);
        mkfitdev::hits::runFillHitsFromDeviceInputs(queue, in, product.view().hits());
        // the temporaries above are queue-ordered (caching allocator); pageable sources are staged by the copy call
      }
    }

    // layers block: filled on the host, copied alone (blocks in declaration order: hits, layers, binnedHits, bins)
    static void copyLayers(Queue& queue,
                           mkfitdev::EventOfHitsDeviceCollection& product,
                           uint32_t nPix,
                           std::vector<::mkfitdev::LayerAxisInput> const& axes) {
      auto pView = product.view();
      const std::array<int32_t, 4> sizes{{pView.hits().metadata().size(),
                                          pView.layers().metadata().size(),
                                          pView.binnedHits().metadata().size(),
                                          pView.bins().metadata().size()}};
      ::mkfitdev::EventOfHitsHostCollection staging(queue, sizes);
      auto sView = staging.view();
      auto sLayers = sView.layers();
      ::mkfitdev::fillLayers(axes, nPix, sLayers);
      auto sBinned = sView.binnedHits();
      auto pLayers = pView.layers();
      auto* sl = reinterpret_cast<std::byte*>(sLayers.metadata().addressOf_qRMin());
      auto* sb = reinterpret_cast<std::byte*>(sBinned.metadata().addressOf_rank());
      auto* dl = reinterpret_cast<std::byte*>(pLayers.metadata().addressOf_qRMin());
      const uint32_t nBytes = static_cast<uint32_t>(sb - sl);
      alpaka::memcpy(queue,
                     cms::alpakatools::make_device_view(alpaka::getDev(queue), dl, nBytes),
                     cms::alpakatools::make_host_view(sl, nBytes));
    }

    // Static layer table inputs from the ES product (filled from the same LayerOfHits::Initializator per layer,
    // esFill.cc), no per-event TrackerInfo walk; the axis types are MkFitCore's.
    static_assert(std::is_same_v<::mkfitdev::AxisPhi, mkfit::LayerOfHits::axis_phi_t>);
    static_assert(std::is_same_v<::mkfitdev::AxisQ, mkfit::LayerOfHits::axis_eta_t>);
    static std::vector<::mkfitdev::LayerAxisInput> layerAxisInputsES(::mkfitdev::ESDataHost const& es) {
      const auto lv = es.layers->const_view();
      std::vector<::mkfitdev::LayerAxisInput> v;
      v.reserve(lv.metadata().size());
      for (int l = 0; l < lv.metadata().size(); ++l)
        v.push_back({lv[l].q_min(),
                     lv[l].q_max(),
                     lv[l].n_q(),
                     lv[l].layer_type() == static_cast<int>(::mkfitdev::LayerType::Barrel),
                     lv[l].is_pixel()});
      return v;
    }

    // An event beyond the device build limits is not built (no exception at event time); downstream device
    // modules must treat a missing/empty EventOfHits as "skip mkFit for this event".
    // the product is always put; a skipped event gets the empty one (nLayers == 0).
    void skipEvent(device::Event& iEvent, uint32_t nHits, uint32_t nLayers) const {
      edm::LogWarning("MkFitAlpakaEventOfHits")
          << "event " << iEvent.id().event() << ": device EventOfHits not built (" << nHits << " hits, " << nLayers
          << " layers exceed the device build limits); mkFit is skipped for this event";
      const std::array<int32_t, 4> empty{{0, 0, 0, 0}};
      iEvent.emplace(productToken_, mkfitdev::EventOfHitsDeviceCollection(iEvent.queue(), empty));
      putQueue(iEvent);
    }

    // CPU backends take the product's memory from the per-stream pool; GPU backends allocate per event (caching
    // allocator)
    mkfitdev::EventOfHitsDeviceCollection makeProduct(edm::StreamID sid,
                                                      Queue& queue,
                                                      std::array<int32_t, 4> const& sizes) const {
#if defined(ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED) || defined(ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED)
      {
        EohScratch& sc = *streamCache(sid);
        const std::size_t n = ::mkfitdev::EventOfHitsPooledHostCollection::bytes(sizes);
        for (uint64_t m = productMaxBytes_.load(); n > m && !productMaxBytes_.compare_exchange_weak(m, n);) {
        }
        if (!sc.product || n > sc.productCap) {
          sc.productCap = ((n + n / 4 + 65535) / 65536) * 65536;
          sc.product.reset();
          sc.product.emplace(
              cms::alpakatools::make_host_buffer<std::byte[]>(queue, static_cast<alpaka_common::Idx>(sc.productCap)));
          ++sc.nProductAllocs;
        }
        return mkfitdev::EventOfHitsDeviceCollection(*sc.product, sizes);
      }
#endif
      return mkfitdev::EventOfHitsDeviceCollection(queue, sizes);
    }

    // the OT rechit SoA is deleted early: order its allocation queue after this module's read (the hit fill kernel
    // enqueued just before)
    void otReleaseGuard(device::Event& iEvent) const {
      if (otSoAQueueToken_.isUninitialized())
        return;
      if (mkfitdev::orderReleaseAfterReads(iEvent.queue(), iEvent.get(otSoAQueueToken_), "MkFitAlpakaEventOfHits"))
        ++nOtOrdered_;
      else
        ++nOtSameQueue_;
    }

    // the native handle of the queue of the EventOfHits allocation (0 on CPU backends: blocking queue)
    void putQueue(device::Event& iEvent) const {
      unsigned long long h = 0;
#if !(defined(ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED) || defined(ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED))
      h = reinterpret_cast<unsigned long long>(alpaka::getNativeHandle(iEvent.queue()));
#endif
      iEvent.emplace(queueToken_, h);
    }

    edm::EDGetTokenT<::reco::TrackingRecHitHost> pixelSoAToken_;
    edm::EDGetTokenT<SiPixelRecHitCollection> pixelRecHitsToken_;
    device::EDGetToken<mkfitdev::OTRecHitDeviceCollection> otSoAToken_;
    edm::EDGetTokenT<unsigned long long> otSoAQueueToken_;  //
    mutable std::atomic<uint64_t> nOtSameQueue_{0}, nOtOrdered_{0};
    edm::EDGetTokenT<SiPixelClusterCollectionNew> pixelClustersToken_;
    edm::EDGetTokenT<Phase2TrackerCluster1DCollectionNew> otClustersToken_;
    const edm::ESGetToken<MkFitGeometry, TrackerRecoGeometryRecord> mkFitGeomToken_;
    const device::EDPutToken<mkfitdev::EventOfHitsDeviceCollection> productToken_;
    const edm::EDPutTokenT<unsigned long long> queueToken_;
    edm::ESGetToken<SiPixelQuality, SiPixelQualityRcd> pixelQualityToken_;
    edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> geomToken_;
    const bool usePixelQualityDB_;
    edm::EDPutTokenT<std::vector<uint32_t>> srcRowsToken_;
    edm::EDPutTokenT<MkFitClusterIndexToHit> indexToHitToken_;
    // per-stream scratch (raw rows, product memory) on CPU backends; GPU backends use the caching allocator
    static constexpr bool kScratchReuse = std::is_same_v<Device, alpaka::DevCpu>;
    const bool produceSrcRows_;
    const bool produceIndexToHit_;
    mutable std::atomic<uint64_t> nKeyMismatch_{0}, nPositionSearch_{0};
    mutable std::atomic<bool> warned_{false};
    mutable std::atomic<uint64_t> productAllocs_{0}, productMaxBytes_{0};
    edm::ESGetToken<::mkfitdev::ESDataHost, TrackerRecoGeometryRecord> esHostToken_;
    device::ESGetToken<::mkfitdev::HitModuleTableESData<Device>, TrackerRecoGeometryRecord> moduleTableToken_;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

DEFINE_FWK_ALPAKA_MODULE(MkFitAlpakaEventOfHitsProducer);
