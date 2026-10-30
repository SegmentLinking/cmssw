// Builds the device mkFit EventOfHits: the HitSoA (bitwise the hits of the RecoTracker/MkFit hit converters) and the
// per-layer binning of mkfit::LayerOfHits.
// Inputs: the pixel rechit SoA, the device OT rechit SoA (MkFitAlpakaOTRecHitsProducer), the per-module table
// (MkFitAlpakaEventOfHitsModuleTableESProducer), and the legacy pixel rechits and clusters, read in one host pass
// that maps each legacy cluster key to its SoA row (the legacy cluster order is not the SoA order).
// Dead regions: from the pixel quality DB as in MkFitEventOfHitsProducer (no strip quality DB).
// Output: mkfitdev::EventOfHitsDeviceCollection (blocks hits, layers, binnedHits, bins), optionally the pixel
// MkFitClusterIndexToHit. An event beyond the device build limits gets an empty product (nLayers == 0, readable on
// the host without a sync) and a warning; the building module then skips mkFit for it.
// On CPU backends the raw host rows and the product's memory come from a pool of grow-only scratch sets, one per
// event in flight, returned to the pool when the product is deleted.
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
#include "FWCore/Utilities/interface/ReusableObjectHolder.h"
#include "Geometry/Records/interface/TrackerDigiGeometryRecord.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/global/EDProducer.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/Event.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EventSetup.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
#include "RecoTracker/MkFit/interface/MkFitGeometry.h"
#include "RecoTracker/MkFit/interface/MkFitSeedWrapper.h"
#include "RecoTracker/MkFitAlpaka/interface/hits/HitModuleTableESData.h"
#include "RecoTracker/MkFitCore/interface/HitStructures.h"
#include "RecoTracker/Record/interface/TrackerRecoGeometryRecord.h"

#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDPutToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/ESGetToken.h"
#include "RecoTracker/MkFitAlpaka/interface/EventOfHitsProduct.h"
#include "RecoTracker/MkFitAlpaka/interface/alpaka/Backend.h"
#include "RecoTracker/MkFitAlpaka/interface/alpaka/EventOfHitsProduct.h"
#include "RecoTracker/MkFitAlpaka/interface/es/ESData.h"
#include "RecoTracker/MkFitAlpaka/interface/hits/LayerAxes.h"
#include "RecoTracker/MkFitAlpaka/interface/hits/alpaka/EventOfHitsBuild.h"
#include "RecoTracker/MkFitAlpaka/interface/hits/alpaka/DeviceHitsBuild.h"
#include "RecoTracker/MkFitAlpaka/interface/othits/alpaka/OTRecHitDeviceCollection.h"
#include "MkFitAlpakaReleaseOrder.h"
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
    // scratch set of the CPU backends
    struct EohScratch {
      // raw host rows of the device hit input (rows past the event's count are unused)
      int32_t cap = 0;
      std::optional<::mkfitdev::DeviceHitRawHostCollection> raw;
      // memory of the EventOfHits product
      std::optional<cms::alpakatools::host_buffer<std::byte[]>> product;
      std::size_t productCap = 0;
    };

    // grow-only capacity: the request plus a quarter of headroom, rounded up to whole granules
    // (std::size_t rather than divide_up_by's uint32_t: the product capacity is in bytes)
    constexpr std::size_t capacityWithHeadroom(std::size_t request, std::size_t granule) {
      return ((request + request / 4 + granule - 1) / granule) * granule;
    }
    constexpr std::size_t kRawRowsGranule = 4096;        // rows
    constexpr std::size_t kProductBytesGranule = 65536;  // bytes

    // per-event counts of the pixel cluster key -> SoA row guard
    struct RawGuard {
      uint32_t keyMismatch = 0;     // originalId row whose local position is not the legacy one (row found by position)
      uint32_t positionSearch = 0;  // clusters without originalId (row found by position)
    };
  }  // namespace

  class MkFitAlpakaEventOfHitsProducer : public global::EDProducer<> {
  public:
    explicit MkFitAlpakaEventOfHitsProducer(edm::ParameterSet const& iConfig)
        : EDProducer<>(iConfig),
          mkFitGeomToken_{esConsumes()},
          productToken_{produces()},
          queueToken_{produces("queue")},
          usePixelQualityDB_{iConfig.getParameter<bool>("usePixelQualityDB")},
          produceIndexToHit_{iConfig.getParameter<bool>("producePixelIndexToHit")} {
      if (produceIndexToHit_)
        indexToHitToken_ = produces("pixelIndexToHit");
      pixelSoAToken_ = consumes(iConfig.getParameter<edm::InputTag>("pixelSoA"));
      pixelRecHitsToken_ = consumes(iConfig.getParameter<edm::InputTag>("pixelRecHits"));
      // OT rows from the device OT rechit SoA (no legacy OT rechits read)
      otSoAToken_ = consumes(iConfig.getParameter<edm::InputTag>("otSoA"));
      if (auto const& otSoAQueueTag = iConfig.getParameter<edm::InputTag>("otSoAQueue"); !otSoAQueueTag.label().empty())
        otSoAQueueToken_ = consumes(otSoAQueueTag);
      pixelClustersToken_ = consumes(iConfig.getParameter<edm::InputTag>("pixelClusters"));
      // consumed for scheduling only: the EventOfHits is made once the building's seeds exist; nothing reads the seeds
      if (auto const& seedsTag = iConfig.getParameter<edm::InputTag>("seeds"); !seedsTag.label().empty())
        seedsToken_ = consumes(seedsTag);
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
          ->setComment("the OT SoA's allocation queue (instance 'queue'), set when the SoA is deleted early");
      desc.add("pixelClusters", edm::InputTag{"hltSiPixelClusters"})
          ->setComment("legacy pixel clusters of the rechits (indexed by key, as convertHits)");
      desc.add("otClusters", edm::InputTag{"hltSiPhase2Clusters"})->setComment("OT clusters of the rechits");
      desc.add("seeds", edm::InputTag{""})
          ->setComment(
              "scheduling only: the EventOfHits is built once the building's seeds exist, so it lives only "
              "through the building and the fit (deleted early)");
      desc.add<std::string>("moduleTable", "MkFitAlpakaHitModuleTable")
          ->setComment("ComponentName of the MkFitAlpakaEventOfHitsModuleTableESProducer product (per-module table)");
      desc.add("producePixelIndexToHit", false)
          ->setComment(
              "also put the pixel MkFitClusterIndexToHit ('pixelIndexToHit'), filled in the "
              "same pass over the legacy rechits as convertHits");
      descriptions.addWithDefaultLabel(desc);
    }

    void endJob() override {
      if (nKeyMismatch_ > 0 || nPositionSearch_ > 0)
        edm::LogWarning("MkFitAlpakaEventOfHits")
            << "pixel rows found by position: originalId mismatches " << nKeyMismatch_
            << ", clusters without originalId " << nPositionSearch_;
    }

    void produce(edm::StreamID, device::Event& iEvent, device::EventSetup const& iSetup) const override {
      const auto& mkFitGeom = iSetup.getData(mkFitGeomToken_);
      // CPU backends: a scratch set of the pool, held by the product until it is deleted
      std::shared_ptr<EohScratch> scratchSet;
      if constexpr (mkfitdev::kCpuBackend)
        scratchSet = scratch_.makeOrGet([] { return new EohScratch(); });

      // dead regions as MkFitEventOfHitsProducer (pixel quality part)
      std::vector<::mkfitdev::DeadRegionDev> deads;
      if (usePixelQualityDB_) {
        const auto& trackerGeom = iSetup.getData(geomToken_);
        const auto& pixelQuality = iSetup.getData(pixelQualityToken_);
        for (const auto& badModule : pixelQuality.getBadComponentList()) {
          const DetId detid(badModule.DetID);
          const auto& surf = trackerGeom.idToDet(detid)->surface();
          bool isBarrel = (mkFitGeom.topology()->side(detid) == static_cast<unsigned>(TrackerDetSide::Barrel));
          const auto ilay = mkFitGeom.mkFitLayerNumber(detid);
          const auto q1 = isBarrel ? surf.zSpan().first : surf.rSpan().first;
          const auto q2 = isBarrel ? surf.zSpan().second : surf.rSpan().second;
          if (badModule.errorType == 0)
            deads.push_back({surf.phiSpan().first, surf.phiSpan().second, q1, q2, ilay});
        }
      }

      // raw host rows of the pixel hits (the OT rows come from the device OT rechit SoA)
      uint32_t nPix = 0, nStr = 0;
      std::optional<::mkfitdev::DeviceHitRawHostCollection> raw;
      const ::mkfitdev::DeviceHitRawHostCollection* rawRef = nullptr;  // raw or the pooled scratch
      {
        const auto& pixRH = iEvent.get(pixelRecHitsToken_);
        nPix = wrapperRows(pixRH);
        // one rechit per OT cluster in cluster-key order (Phase2TrackerRecHits): converter rows = OT clusters
        nStr = iEvent.get(otClustersToken_).dataSize();
        const int32_t nRaw = int32_t(nPix);
        ::mkfitdev::DeviceHitRawHostCollection* rawP = nullptr;
        if constexpr (mkfitdev::kCpuBackend) {
          EohScratch& pooled = *scratchSet;
          if (!pooled.raw || nRaw > pooled.cap) {
            pooled.cap = int32_t(capacityWithHeadroom(std::size_t(nRaw), kRawRowsGranule));
            pooled.raw.reset();
            pooled.raw.emplace(iEvent.queue(), pooled.cap);
          }
          rawP = &*pooled.raw;
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
        rawRef = rawP;
      }
      const auto axes = layerAxisInputsES(iSetup.getData(esHostToken_));
      // the EventOfHits event product (one PortableCollection over the four blocks)
      const uint32_t nHits = nPix + nStr, nLayers = axes.size(), nBins = ::mkfitdev::totalBins(axes);
      if (nLayers > ::mkfitdev::kMaxLayers)
        throw cms::Exception("MkFitAlpakaEventOfHits")
            << nLayers << " mkFit layers do not fit the int8_t layer column of HitSoA";
      // block sizes: hits, layers, binnedHits, bins
      const ::mkfitdev::EventOfHitsSizes sizes{{int32_t(nHits), int32_t(nLayers), int32_t(nHits), int32_t(nBins)}};
      mkfitdev::EventOfHitsDeviceCollection product = makeProduct(scratchSet, iEvent.queue(), sizes);
      fillFromDeviceInputs(iEvent, iSetup, product, *rawRef, nPix, nStr, axes);
      if (!otSoAQueueToken_.isUninitialized())
        mkfitdev::orderReleaseAfterReads(iEvent.queue(), iEvent.get(otSoAQueueToken_));
      // the EventOfHits is deleted early: its last reader orders this queue after its reads
      iEvent.emplace(queueToken_, mkfitdev::queueHandle(iEvent.queue()));
      if (!mkfitdev::hits::runBuildEventOfHits(iEvent.queue(), product, deads)) {
        skipEvent(iEvent, nHits, nLayers);
        return;
      }
      iEvent.emplace(productToken_, std::move(product));
    }

  private:
    // convertHits row count: max(last hit's cluster key + 1, dataSize()), 0 for an empty collection
    template <typename THits>
    static uint32_t wrapperRows(THits const& hits) {
      if (hits.empty())
        return 0;
      return std::max<uint32_t>(hits.data().back().firstClusterRef().index() + 1, hits.dataSize());
    }

    // Host part of the device hit input, per legacy pixel cluster key: the SoA row (moduleStart[det] + originalId;
    // clusters without originalId, such as persisted ones, take the row of the module with the bitwise-equal local
    // position, as the legacy rechit position is a copy of the SoA one) and the legacy cluster sizes.
    static void fillRaw(::reco::TrackingRecHitHost const& soa,
                        SiPixelRecHitCollection const& pixRH,
                        edm::Handle<SiPixelClusterCollectionNew> const& pixCluH,
                        uint32_t nPix,
                        ::mkfitdev::DeviceHitRawSoA::View rawView,
                        RawGuard& guard,
                        std::vector<TrackingRecHit const*>* indexToHit) {
      const auto& pixClu = *pixCluH;
      uint32_t* src = rawView.metadata().addressOf_srcRow();
      for (uint32_t i = 0; i < nPix; ++i)
        src[i] = ::mkfitdev::kNoHit;
      const auto soaHits = soa.const_view().trackingHits();
      const auto soaModules = soa.const_view().hitModules();
      const uint32_t nSoa = soa.nHits();
      uint32_t* spans = rawView.metadata().addressOf_spans();
      // MkFitClusterIndexToHit as convertHits: max(last hit's key + 1, dataSize()) entries, each hit at its key
      if (indexToHit && !pixRH.empty())
        indexToHit->resize(nPix, nullptr);
      for (const auto& ds : pixRH) {
        if (ds.empty())
          continue;
        const uint32_t gind = ds.begin()->det()->index();
        const uint32_t moduleBegin = soaModules[gind].moduleStart(), moduleEnd = soaModules[gind + 1].moduleStart();
        for (const auto& hit : ds) {
          const auto ref = hit.firstClusterRef();
          if (ref.id() != pixCluH.id())  // convertHits reads the cluster by key from the collection of the hits' refs
            throw cms::Exception("MkFitAlpakaEventOfHits") << "pixelClusters is not the collection of the rechits";
          const uint32_t key = ref.index();
          const auto& clu = pixClu.data()[key];
          uint32_t row = ::mkfitdev::kNoHit;
          const float lx = hit.localPosition().x(), ly = hit.localPosition().y();
          if (clu.originalId() != SiPixelCluster::invalidClusterId) {
            row = moduleBegin + clu.originalId();
            // the SoA row of originalId must carry the legacy local position (bitwise: the legacy one is a copy)
            if (!(row < moduleEnd && row < nSoa && soaHits[row].xLocal() == lx && soaHits[row].yLocal() == ly)) {
              ++guard.keyMismatch;
              row = ::mkfitdev::kNoHit;
            }
          } else {
            ++guard.positionSearch;
          }
          if (row == ::mkfitdev::kNoHit) {
            for (uint32_t candidate = moduleBegin; candidate < moduleEnd && candidate < nSoa; ++candidate)
              if (soaHits[candidate].xLocal() == lx && soaHits[candidate].yLocal() == ly) {
                row = candidate;
                break;
              }
          }
          if (row >= nSoa || soaHits[row].detectorIndex() != gind)
            throw cms::Exception("MkFitAlpakaEventOfHits")
                << "pixel cluster key " << key << " (det index " << gind << ") has no row in the pixel rechit SoA";
          src[key] = row;
          spans[key] = uint32_t(clu.sizeX()) | (uint32_t(clu.sizeY()) << ::mkfitdev::kSpanYShift);
          if (indexToHit) {
            if (key >= indexToHit->size())
              indexToHit->resize(key + 1, nullptr);
            (*indexToHit)[key] = &hit;
          }
        }
      }
    }

    // counters per job; the first event with a guard hit warns
    void checkGuard(unsigned long long evt, RawGuard const& guard) const {
      if (guard.keyMismatch == 0 && guard.positionSearch == 0)
        return;
      nKeyMismatch_ += guard.keyMismatch;
      nPositionSearch_ += guard.positionSearch;
      if (!warned_.exchange(true))
        edm::LogWarning("MkFitAlpakaEventOfHits")
            << "event " << evt << ": device hit input found " << guard.keyMismatch + guard.positionSearch
            << " pixel rows by position (" << guard.keyMismatch << " originalId mismatches, " << guard.positionSearch
            << " clusters without originalId); totals at endJob";
    }

    static void setRaw(mkfitdev::hits::DeviceHitInputs& in, ::mkfitdev::DeviceHitRawSoA::ConstView rawView) {
      const auto columns = rawView.metadata();
      in.srcRow = columns.addressOf_srcRow();
      in.module = columns.addressOf_module();
      in.spans = columns.addressOf_spans();
      in.lx = columns.addressOf_lx();
      in.ly = columns.addressOf_ly();
      in.exx = columns.addressOf_exx();
      in.exy = columns.addressOf_exy();
      in.eyy = columns.addressOf_eyy();
    }

    // OT rows straight from the device OT rechit SoA (device memory; row = OT cluster key)
    void setOT(device::Event& iEvent, mkfitdev::hits::DeviceHitInputs& in) const {
      const auto otColumns = iEvent.get(otSoAToken_).const_view().metadata();
      if (static_cast<uint32_t>(otColumns.size()) < in.nStrip)
        throw cms::Exception("MkFitAlpakaEventOfHits") << "OT rechit SoA has fewer rows than OT clusters";
      in.otModule = otColumns.addressOf_module();
      in.otSize = otColumns.addressOf_clustSize();
      in.otLx = otColumns.addressOf_lx();
      in.otLy = otColumns.addressOf_ly();
      in.otExx = otColumns.addressOf_exx();
      in.otEyy = otColumns.addressOf_eyy();
    }

    template <typename T>
    static auto hostView(const T* data, uint32_t size) {
      return cms::alpakatools::make_host_view(const_cast<T*>(data), size);
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
      const auto hitColumns = hitsView.metadata();  // refers to hitsView: keep the view alive
      const uint32_t nSoa = soa.nHits();
      mkfitdev::hits::DeviceHitInputs in{};
      in.nPixel = nPix;
      in.nStrip = nStr;
      if constexpr (mkfitdev::kCpuBackend) {
        in.xLocal = hitColumns.addressOf_xLocal();
        in.yLocal = hitColumns.addressOf_yLocal();
        in.xerrLocal = hitColumns.addressOf_xerrLocal();
        in.yerrLocal = hitColumns.addressOf_yerrLocal();
        in.detectorIndex = hitColumns.addressOf_detectorIndex();
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
        alpaka::memcpy(queue, dX, hostView(hitColumns.addressOf_xLocal(), nSoa));
        alpaka::memcpy(queue, dY, hostView(hitColumns.addressOf_yLocal(), nSoa));
        alpaka::memcpy(queue, dEX, hostView(hitColumns.addressOf_xerrLocal(), nSoa));
        alpaka::memcpy(queue, dEY, hostView(hitColumns.addressOf_yerrLocal(), nSoa));
        alpaka::memcpy(queue, dDet, hostView(hitColumns.addressOf_detectorIndex(), nSoa));
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

    // layers block: filled on the host, copied alone
    static void copyLayers(Queue& queue,
                           mkfitdev::EventOfHitsDeviceCollection& product,
                           uint32_t nPix,
                           std::vector<::mkfitdev::LayerAxisInput> const& axes) {
      auto pLayers = product.view().layers();
      // staged in a host LayerSoA alone (the block has the same layout)
      PortableHostCollection<::mkfitdev::LayerSoA> staging(queue, pLayers.metadata().size());
      ::mkfitdev::fillLayers(axes, nPix, staging.view());
      auto* layersBegin = reinterpret_cast<std::byte*>(pLayers.metadata().addressOf_qRMin());
      const auto nBytes = ::mkfitdev::LayerSoA::computeDataSize(pLayers.metadata().size());
      alpaka::memcpy(
          queue, cms::alpakatools::make_device_view(alpaka::getDev(queue), layersBegin, nBytes), staging.buffer());
    }

    // Static layer table inputs from the ES product (filled from the same LayerOfHits::Initializator per layer,
    // esFill.cc), no per-event TrackerInfo walk; the axis types are MkFitCore's.
    static_assert(std::is_same_v<::mkfitdev::AxisPhi, mkfit::LayerOfHits::axis_phi_t>);
    static_assert(std::is_same_v<::mkfitdev::AxisQ, mkfit::LayerOfHits::axis_eta_t>);
    static std::vector<::mkfitdev::LayerAxisInput> layerAxisInputsES(::mkfitdev::ESDataHost const& es) {
      const auto layers = es.layers->const_view();
      std::vector<::mkfitdev::LayerAxisInput> inputs;
      inputs.reserve(layers.metadata().size());
      for (int il = 0; il < layers.metadata().size(); ++il)
        inputs.push_back({layers[il].q_min(),
                          layers[il].q_max(),
                          layers[il].n_q(),
                          layers[il].layer_type() == static_cast<int>(::mkfitdev::LayerType::Barrel),
                          layers[il].is_pixel()});
      return inputs;
    }

    // An event beyond the device build limits is not built (no exception at event time): the product is always put,
    // a skipped event gets the empty one (nLayers == 0), and downstream device modules skip mkFit for it.
    void skipEvent(device::Event& iEvent, uint32_t nHits, uint32_t nLayers) const {
      edm::LogWarning("MkFitAlpakaEventOfHits")
          << "event " << iEvent.id().event() << ": device EventOfHits not built (" << nHits << " hits, " << nLayers
          << " layers exceed the device build limits); mkFit is skipped for this event";
      const ::mkfitdev::EventOfHitsSizes empty{};
      iEvent.emplace(productToken_, mkfitdev::EventOfHitsDeviceCollection(iEvent.queue(), empty));
    }

    // CPU backends take the product's memory from the scratch set; GPU backends allocate per event (caching
    // allocator)
    mkfitdev::EventOfHitsDeviceCollection makeProduct(std::shared_ptr<EohScratch> const& scratchSet,
                                                      Queue& queue,
                                                      ::mkfitdev::EventOfHitsSizes const& sizes) const {
#if defined(ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED) || defined(ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED)
      {
        EohScratch& pooled = *scratchSet;
        const std::size_t bytes = ::mkfitdev::EventOfHitsPooledHostCollection::bytes(sizes);
        if (!pooled.product || bytes > pooled.productCap) {
          pooled.productCap = capacityWithHeadroom(bytes, kProductBytesGranule);
          pooled.product.reset();
          pooled.product.emplace(cms::alpakatools::make_host_buffer<std::byte[]>(
              queue, static_cast<alpaka_common::Idx>(pooled.productCap)));
        }
        return mkfitdev::EventOfHitsDeviceCollection(*pooled.product, sizes, scratchSet);
      }
#endif
      return mkfitdev::EventOfHitsDeviceCollection(queue, sizes);
    }

    // event data
    edm::EDGetTokenT<::reco::TrackingRecHitHost> pixelSoAToken_;
    edm::EDGetTokenT<SiPixelRecHitCollection> pixelRecHitsToken_;
    device::EDGetToken<mkfitdev::OTRecHitDeviceCollection> otSoAToken_;
    edm::EDGetTokenT<unsigned long long> otSoAQueueToken_;
    edm::EDGetTokenT<SiPixelClusterCollectionNew> pixelClustersToken_;
    edm::EDGetTokenT<Phase2TrackerCluster1DCollectionNew> otClustersToken_;
    edm::EDGetTokenT<MkFitSeedWrapper> seedsToken_;
    // event setup
    const edm::ESGetToken<MkFitGeometry, TrackerRecoGeometryRecord> mkFitGeomToken_;
    edm::ESGetToken<::mkfitdev::ESDataHost, TrackerRecoGeometryRecord> esHostToken_;
    device::ESGetToken<::mkfitdev::HitModuleTableESData<Device>, TrackerRecoGeometryRecord> moduleTableToken_;
    edm::ESGetToken<SiPixelQuality, SiPixelQualityRcd> pixelQualityToken_;
    edm::ESGetToken<TrackerGeometry, TrackerDigiGeometryRecord> geomToken_;
    // products
    const device::EDPutToken<mkfitdev::EventOfHitsDeviceCollection> productToken_;
    const edm::EDPutTokenT<unsigned long long> queueToken_;
    edm::EDPutTokenT<MkFitClusterIndexToHit> indexToHitToken_;
    // configuration
    const bool usePixelQualityDB_;
    const bool produceIndexToHit_;
    // scratch sets (raw rows, product memory) on CPU backends; GPU backends use the caching allocator
    mutable edm::ReusableObjectHolder<EohScratch> scratch_;
    // pixel rows found by position (data-quality counters, see fillRaw)
    mutable std::atomic<uint64_t> nKeyMismatch_{0}, nPositionSearch_{0};
    mutable std::atomic<bool> warned_{false};
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

DEFINE_FWK_ALPAKA_MODULE(MkFitAlpakaEventOfHitsProducer);
