// The converter's missing-hit navigation on the device, one set of launches per event
// after the device fit (MkFitAlpakaNavDeviceKernels.dev.cc). Tables: the flat layer tables of the ES product NavFlatTables (built once
// per IOV, MkFitAlpakaNavFlatTablesESProducer) and the compatibleLayers candidate lists of the navigation school
//, copied to each device once per IOV. Output: per (track, direction, layer) the
// front det of compatibleDets or a NavResultCode; MkFitAlpakaOutputTrackConverter (navDevice) reads it on the host.
#include <algorithm>
#include <array>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <type_traits>
#include <utility>
#include <vector>

#include <alpaka/alpaka.hpp>

#include "FWCore/MessageLogger/interface/MessageLogger.h"
#include "FWCore/ParameterSet/interface/ConfigurationDescriptions.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "FWCore/ParameterSet/interface/ParameterSetDescription.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "FWCore/Utilities/interface/InputTag.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDGetToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EDPutToken.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/Event.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/EventSetup.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/MakerMacros.h"
#include "HeterogeneousCore/AlpakaCore/interface/alpaka/global/EDProducer.h"
#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "RecoTracker/MkFitAlpaka/interface/alpaka/NavResultProduct.h"
#include "RecoTracker/MkFitAlpaka/interface/alpaka/tracks/TrackSoADeviceCollection.h"
#include "RecoTracker/Record/interface/NavigationSchoolRecord.h"
#include "RecoTracker/Record/interface/TrackerRecoGeometryRecord.h"
#include "TrackingTools/DetLayers/interface/BarrelDetLayer.h"
#include "TrackingTools/DetLayers/interface/DetLayer.h"
#include "TrackingTools/DetLayers/interface/ForwardDetLayer.h"
#include "TrackingTools/DetLayers/interface/NavigationSchool.h"
#include "TrackingTools/DetLayers/interface/TkLayerLess.h"

#include "RecoTracker/MkFitAlpaka/interface/navdev/NavFlatTables.h"
#include "MkFitAlpakaNavDeviceKernels.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE {

  class MkFitAlpakaNavDeviceProducer : public global::EDProducer<> {
  public:
    explicit MkFitAlpakaNavDeviceProducer(edm::ParameterSet const& iConfig)
        : EDProducer<>(iConfig),
          tracksToken_{consumes(iConfig.getParameter<edm::InputTag>("tracks"))},
          flatToken_{esConsumes()},
          navToken_{esConsumes(iConfig.getParameter<edm::ESInputTag>("NavigationSchool"))},
          putToken_{produces()} {}

    static void fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
      edm::ParameterSetDescription desc;
      desc.add<edm::InputTag>("tracks", edm::InputTag("hltInitialStepTrackCandidatesMkFitFitDevice"))
          ->setComment("device TrackSoA of the device final fit (MkFitAlpakaFitDeviceProducer)");
      desc.add<edm::ESInputTag>("NavigationSchool", edm::ESInputTag{"", "SimpleNavigationSchool"})
          ->setComment("must be the output converter's navigation school");
      descriptions.addWithDefaultLabel(desc);
    }

    void produce(edm::StreamID, device::Event& iEvent, device::EventSetup const& iSetup) const override {
      auto& queue = iEvent.queue();
      // CPU backends: an empty product, the converter searches on the host (faster: no device-side compatibleLayers)
      if constexpr (std::is_same_v<Device, alpaka::DevCpu>) {
        iEvent.emplace(putToken_, mkfitdev::NavResultDeviceCollection(queue, 0));
        return;
      }
      auto const& trk = iEvent.get(tracksToken_);
      auto const& flat = iSetup.getData(flatToken_);
      auto const& school = iSetup.getData(navToken_);
      DeviceTables const& dt = tables(queue, flat, school);
      const int capacity = trk.const_view().metadata().size();  // 0 for a skipped / seedless event
      // rows per (track, direction, layer), then the set header rows per (track, direction) (NavResultSoA.h)
      mkfitdev::NavResultDeviceCollection out(queue, capacity * 2 * (dt.view.nLayers + 1));
      if (capacity > 0) {
        namespace nd = ::mkfitdev::navdev;
        namespace nw = mkfitdev::navdevice;
        auto starts = cms::alpakatools::make_device_buffer<nd::DetTestStart[]>(queue, capacity);
        // the compacted call lists (one per layer kind; at most one call per (track, direction, layer)) and their counts
        auto const& ls = dt.view.listStart;
        auto calls = cms::alpakatools::make_device_buffer<int[]>(
            queue, std::max<size_t>(size_t(2) * capacity * ls[nw::kNavKinds], 1));
        auto nCalls = cms::alpakatools::make_device_buffer<int[]>(queue, nw::kNavKinds);
        alpaka::memset(queue, nCalls, 0);
        const size_t nThreads = nw::navSearchThreads(capacity);  // per-thread scratch of the search kernels
        auto arena = cms::alpakatools::make_device_buffer<nd::NavGroups[]>(queue, nThreads * nd::kNavArena);
        auto memo = cms::alpakatools::make_device_buffer<nd::DetTestResult[]>(queue, nThreads * nd::kNavMemo);
        auto memoDet = cms::alpakatools::make_device_buffer<int[]>(queue, nThreads * nd::kNavMemo);
        auto parts = cms::alpakatools::make_device_buffer<nw::NavPart[]>(
            queue, std::max<size_t>(size_t(2) * capacity * (ls[2] - ls[1]), 1));
        // the converter's Chi2MeasurementEstimator(30., -3.0, 0.5, 2.0, 0.5, 1.e12)
        const mkfitdev::navdevice::NavEstimator est{2.0, 0.25, -3.0, 0.5};
        nw::launchNavDevice(queue,
                            trk.const_view(),
                            capacity,
                            dt.view,
                            est,
                            starts.data(),
                            calls.data(),
                            nCalls.data(),
                            arena.data(),
                            memo.data(),
                            memoDet.data(),
                            parts.data(),
                            out.view());
      }
      iEvent.emplace(putToken_, std::move(out));
    }

  private:
    template <typename T>
    using DBuf = cms::alpakatools::device_buffer<Device, T[]>;
    struct DeviceTables {
      void const* flatKey = nullptr;
      void const* schoolKey = nullptr;
      std::optional<DBuf<::mkfitdev::navdev::NavDet>> dets;
      std::optional<DBuf<int>> idx, layerFlat, mkFitToLayer;
      std::optional<DBuf<::mkfitdev::navdev::NavRing>> rings;
      std::optional<DBuf<::mkfitdev::navdev::NavSubDisk>> subDisks;
      std::optional<DBuf<::mkfitdev::navdev::NavRod>> rods;
      std::optional<DBuf<::mkfitdev::navdev::NavBarrel>> barrels;
      std::optional<DBuf<float>> zs;
      std::optional<cms::alpakatools::device_buffer<Device, ::mkfitdev::nav::Table>> layers;
      mkfitdev::navdevice::NavDeviceTables view{};
    };

    template <typename T>
    static std::optional<DBuf<T>> toDevice(Queue& queue, std::vector<T> const& v) {
      auto d = cms::alpakatools::make_device_buffer<T[]>(alpaka::getDev(queue), std::max<size_t>(v.size(), 1));
      if (!v.empty())
        alpaka::memcpy(queue, d, cms::alpakatools::make_host_view(const_cast<T*>(v.data()), Idx(v.size())));
      return d;
    }

    // the compatibleLayers candidate lists: per layer the lists of
    // SimpleBarrel/ForwardNavigableLayer::nextLayers, rebuilt from the school's static lists + TkLayerLess
    static void buildLayerTable(NavigationSchool const& school,
                                ::mkfitdev::navdev::NavFlatTables const& flat,
                                ::mkfitdev::nav::Table& T) {
      namespace nav = ::mkfitdev::nav;
      using DLV = std::vector<const DetLayer*>;
      auto pick = [](DLV const& l, int what) {  // 0 barrel, 1 forward z < 0, 2 forward z > 0, 3 forward
        DLV r;
        for (auto c : l) {
          const bool b = c->isBarrel();
          const double z = c->position().z();
          if ((what == 0 && b) || (what == 1 && !b && z < 0) || (what == 2 && !b && z > 0) || (what == 3 && !b))
            r.push_back(c);
        }
        return r;
      };
      // a + b in the order of SimpleBarrelNavigableLayer (std::sort with TkLayerLess)
      auto sorted = [](DLV a, DLV const& b, TkLayerLess const& less) {
        a.insert(a.end(), b.begin(), b.end());
        std::sort(a.begin(), a.end(), less);
        return a;
      };
      auto const& all = flat.allLayers;
      T.nLayers = all.size();
      T.nIdx = 0;
      for (size_t i = 0; i < all.size(); ++i) {
        const DetLayer* l = all[i];
        nav::Layer& L = T.layers[i];
        L.barrel = l->isBarrel();
        const auto& bounds = l->surface().bounds();
        L.thickness = bounds.thickness();
        if (L.barrel) {
          L.rz = static_cast<BarrelDetLayer const*>(l)->specificSurface().radius();
          L.halfLength = bounds.length() * 0.5f;
          L.rin = L.rout = 0;
        } else {
          auto const& disk = static_cast<ForwardDetLayer const*>(l)->specificSurface();
          L.rz = disk.position().z();
          L.rin = disk.innerRadius();
          L.rout = disk.outerRadius();
          L.halfLength = 0;
        }
        const DLV sOut = school.nextLayers(*l, insideOut), sIn = school.nextLayers(*l, outsideIn);
        std::array<DLV, nav::kNLists> lists;
        if (L.barrel) {
          const DLV oB = pick(sOut, 0), oL = pick(sOut, 1), oR = pick(sOut, 2);
          const DLV iB = pick(sIn, 0), iL = pick(sIn, 1), iR = pick(sIn, 2);
          lists[nav::kNegOuter] = sorted(oB, oL, TkLayerLess());
          lists[nav::kPosOuter] = sorted(oB, oR, TkLayerLess());
          lists[nav::kNegInner] = sorted(iB, iL, TkLayerLess(outsideIn));
          lists[nav::kPosInner] = sorted(iB, iR, TkLayerLess(outsideIn));
          lists[nav::kIB] = iB, lists[nav::kIL] = iL, lists[nav::kIR] = iR;
          lists[nav::kOB] = oB, lists[nav::kOL] = oL, lists[nav::kOR] = oR;
        } else {
          lists[nav::kFOut] = sOut, lists[nav::kFIn] = sIn;
          lists[nav::kFIF] = pick(sIn, 3), lists[nav::kFOB] = pick(sOut, 0);
          lists[nav::kFIB] = pick(sIn, 0), lists[nav::kFOF] = pick(sOut, 3);
        }
        for (int k = 0; k < nav::kNLists; ++k) {
          L.off[k] = T.nIdx;
          L.len[k] = lists[k].size();
          for (auto c : lists[k]) {
            auto const it = flat.layerIndex.find(c);
            if (T.nIdx >= nav::kMaxIdx || it == flat.layerIndex.end())
              throw cms::Exception("Configuration")
                  << "MkFitAlpakaNavDeviceProducer: candidate list overflow or a layer outside allLayers()";
            T.idx[T.nIdx++] = it->second;
          }
        }
      }
    }

    // the device copy of the tables for this queue's device, rebuilt when the ES products change (once per IOV)
    DeviceTables const& tables(Queue& queue,
                               ::mkfitdev::navdev::NavFlatTables const& flat,
                               NavigationSchool const& school) const {
      const auto dev = alpaka::getDev(queue);
      std::lock_guard<std::mutex> lk(mutex_);
      DeviceTables& dt = cache_[alpaka::getNativeHandle(dev)];
      if (dt.flatKey == &flat && dt.schoolKey == &school)
        return dt;
      if (flat.allLayers.size() > size_t(::mkfitdev::nav::kMaxLayers))
        throw cms::Exception("Configuration") << "MkFitAlpakaNavDeviceProducer: " << flat.allLayers.size()
                                              << " tracker layers > " << ::mkfitdev::nav::kMaxLayers;
      auto hLayers = cms::alpakatools::make_host_buffer<::mkfitdev::nav::Table>(queue);
      buildLayerTable(school, flat, *hLayers.data());
      std::vector<int> layerFlat;
      for (auto const& a : flat.layerFlat)
        layerFlat.insert(layerFlat.end(), a.begin(), a.end());
      dt.dets = toDevice(queue, flat.dets);
      dt.idx = toDevice(queue, flat.idx);
      dt.rings = toDevice(queue, flat.rings);
      dt.subDisks = toDevice(queue, flat.subDisks);
      dt.rods = toDevice(queue, flat.rods);
      dt.barrels = toDevice(queue, flat.barrels);
      dt.zs = toDevice(queue, flat.zs);
      dt.layerFlat = toDevice(queue, layerFlat);
      dt.mkFitToLayer = toDevice(queue, flat.mkFitToLayer);
      dt.layers.emplace(cms::alpakatools::make_device_buffer<::mkfitdev::nav::Table>(dev));
      alpaka::memcpy(queue, *dt.layers, hLayers);
      alpaka::wait(queue);  // once per IOV: every stream's queue may use the tables from now on
      dt.view.flat = ::mkfitdev::navdev::NavTables{dt.dets->data(),
                                                   dt.idx->data(),
                                                   dt.rings->data(),
                                                   dt.subDisks->data(),
                                                   dt.rods->data(),
                                                   dt.barrels->data(),
                                                   dt.zs->data()};
      dt.view.layers = dt.layers->data();
      dt.view.layerFlat = dt.layerFlat->data();
      dt.view.mkFitToLayer = dt.mkFitToLayer->data();
      dt.view.nMkFit = flat.mkFitToLayer.size();
      dt.view.nLayers = flat.allLayers.size();
      // the layers of each device call list, as KernelNavStart assigns them
      int nList[mkfitdev::navdevice::kNavKinds] = {};
      for (auto const& a : flat.layerFlat)
        if (a[1] >= 0 && (a[0] == 0 || a[0] == 2))
          ++nList[a[0] == 0 ? 0 : (flat.barrels[a[1]].stacked ? 1 : 2)];
      dt.view.listStart[0] = 0;
      for (int k = 0; k < mkfitdev::navdevice::kNavKinds; ++k)
        dt.view.listStart[k + 1] = dt.view.listStart[k] + nList[k];
      dt.flatKey = &flat;
      dt.schoolKey = &school;
      return dt;
    }

    const device::EDGetToken<mkfitdev::TrackSoADeviceCollection> tracksToken_;
    const edm::ESGetToken<::mkfitdev::navdev::NavFlatTables, TrackerRecoGeometryRecord> flatToken_;
    const edm::ESGetToken<NavigationSchool, NavigationSchoolRecord> navToken_;
    const device::EDPutToken<mkfitdev::NavResultDeviceCollection> putToken_;
    mutable std::mutex mutex_;
    mutable std::map<decltype(alpaka::getNativeHandle(std::declval<Device>())), DeviceTables> cache_;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE

DEFINE_FWK_ALPAKA_MODULE(MkFitAlpakaNavDeviceProducer);
