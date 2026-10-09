#ifndef RecoLocalTracker_Phase2TrackerRecHits_Phase2TrackerRecHitOnDemand_h
#define RecoLocalTracker_Phase2TrackerRecHits_Phase2TrackerRecHitOnDemand_h

// A Phase2TrackerRecHit1D made on demand from its OT cluster
// key with the operations of Phase2TrackerRecHits::produce (the CPE's localParameters of the cluster on its
// GeomDetUnit, edmNew::makeRefTo of the cluster), so it equals the legacy collection's hit at that key (the legacy
// producer makes one rechit per cluster, in cluster order: rechit flat index = cluster key). Readers that need only
// the hits of their tracks use it instead of the full legacy collection (hltSiPhase2RecHits).

#include <algorithm>
#include <cstdint>
#include <vector>

#include "DataFormats/Common/interface/DetSetVectorNew.h"
#include "DataFormats/Common/interface/Handle.h"
#include "DataFormats/DetId/interface/DetId.h"
#include "DataFormats/Phase2TrackerCluster/interface/Phase2TrackerCluster1D.h"
#include "DataFormats/TrackerRecHit2D/interface/Phase2TrackerRecHit1D.h"
#include "FWCore/Utilities/interface/Exception.h"
#include "Geometry/TrackerGeometryBuilder/interface/TrackerGeometry.h"
#include "RecoLocalTracker/ClusterParameterEstimator/interface/ClusterParameterEstimator.h"

class Phase2TrackerRecHitOnDemand {
public:
  using CPE = ClusterParameterEstimator<Phase2TrackerCluster1D>;

  // one light pass over the cluster detsets (first key and DetId of each); the GeomDetUnit of a detset is looked up
  // once, at its first made hit (a per-detset memo instead of a TrackerGeometry hash lookup per
  // made hit; same pointer, so the same hits). One instance per module and event, used from one thread.
  Phase2TrackerRecHitOnDemand(edm::Handle<Phase2TrackerCluster1DCollectionNew> const& clusters,
                              TrackerGeometry const& geom,
                              CPE const& cpe)
      : clusters_(clusters), geom_(&geom), cpe_(&cpe) {
    auto const* d0 = clusters->data().data();
    first_.reserve(clusters->size());
    ids_.reserve(clusters->size());
    for (auto const& ds : *clusters) {
      if (ds.empty())
        continue;
      const uint32_t k = &*ds.begin() - d0;
      if (!first_.empty() && k < first_.back())
        throw cms::Exception("Phase2TrackerRecHitOnDemand") << "cluster detsets not in data order";
      first_.push_back(k);
      ids_.push_back(ds.detId());
    }
    dets_.assign(ids_.size(), nullptr);
  }

  uint32_t size() const { return clusters_->dataSize(); }

  GeomDetUnit const* detOfKey(uint32_t key) const {
    auto const i = std::upper_bound(first_.begin(), first_.end(), key) - first_.begin() - 1;
    auto& du = dets_[i];
    if (du == nullptr)
      du = geom_->idToDetUnit(DetId(ids_[i]));
    return du;
  }

  // the legacy hit at cluster key `key` (Phase2TrackerRecHits::produce)
  Phase2TrackerRecHit1D make(uint32_t key) const {
    GeomDetUnit const* du = detOfKey(key);
    auto const& cluster = clusters_->data()[key];
    const CPE::LocalValues lv = cpe_->localParameters(cluster, *du);
    return Phase2TrackerRecHit1D(lv.first, lv.second, *du, edmNew::makeRefTo(clusters_, &cluster));
  }

private:
  edm::Handle<Phase2TrackerCluster1DCollectionNew> clusters_;
  TrackerGeometry const* geom_;
  CPE const* cpe_;
  std::vector<uint32_t> first_;
  std::vector<uint32_t> ids_;
  mutable std::vector<GeomDetUnit const*> dets_;  // memo of idToDetUnit(ids_[i]), filled on first use
};

#endif
