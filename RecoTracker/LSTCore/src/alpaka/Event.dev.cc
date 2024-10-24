#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"

#include "Event.h"

#include "MiniDoublet.h"
#include "PixelQuintuplet.h"
#include "PixelTriplet.h"
#include "Quintuplet.h"
#include "Segment.h"
#include "TrackCandidate.h"
#include "Triplet.h"

using Device = ALPAKA_ACCELERATOR_NAMESPACE::Device;
using Queue = ALPAKA_ACCELERATOR_NAMESPACE::Queue;
using Acc1D = ALPAKA_ACCELERATOR_NAMESPACE::Acc1D;
using Acc3D = ALPAKA_ACCELERATOR_NAMESPACE::Acc3D;

using namespace ALPAKA_ACCELERATOR_NAMESPACE::lst;

void Event::initSync(bool verbose) {
  alpaka::wait(queue_);  // other calls can be asynchronous
  addObjects_ = verbose;

  //reset the arrays
  for (int i = 0; i < 6; i++) {
    n_hits_by_layer_barrel_[i] = 0;
    n_minidoublets_by_layer_barrel_[i] = 0;
    n_segments_by_layer_barrel_[i] = 0;
    n_triplets_by_layer_barrel_[i] = 0;
    n_trackCandidates_by_layer_barrel_[i] = 0;
    n_quintuplets_by_layer_barrel_[i] = 0;
    if (i < 5) {
      n_hits_by_layer_endcap_[i] = 0;
      n_minidoublets_by_layer_endcap_[i] = 0;
      n_segments_by_layer_endcap_[i] = 0;
      n_triplets_by_layer_endcap_[i] = 0;
      n_trackCandidates_by_layer_endcap_[i] = 0;
      n_quintuplets_by_layer_endcap_[i] = 0;
    }
  }
}

void Event::resetEventSync() {
  alpaka::wait(queue_);  // synchronize to reset consistently
  //reset the arrays
  for (int i = 0; i < 6; i++) {
    n_hits_by_layer_barrel_[i] = 0;
    n_minidoublets_by_layer_barrel_[i] = 0;
    n_segments_by_layer_barrel_[i] = 0;
    n_triplets_by_layer_barrel_[i] = 0;
    n_trackCandidates_by_layer_barrel_[i] = 0;
    n_quintuplets_by_layer_barrel_[i] = 0;
    if (i < 5) {
      n_hits_by_layer_endcap_[i] = 0;
      n_minidoublets_by_layer_endcap_[i] = 0;
      n_segments_by_layer_endcap_[i] = 0;
      n_triplets_by_layer_endcap_[i] = 0;
      n_trackCandidates_by_layer_endcap_[i] = 0;
      n_quintuplets_by_layer_endcap_[i] = 0;
    }
  }
  hitsInGPU_.reset();
  hitsBuffers_.reset();
  miniDoubletsDC_.reset();
  rangesInGPU_.reset();
  rangesBuffers_.reset();
  segmentsDC_.reset();
  tripletsDC_.reset();
  quintupletsDC_.reset();
  trackCandidatesDC_.reset();
  pixelTripletsDC_.reset();
  pixelQuintupletsDC_.reset();

  hitsInCPU_.reset();
  rangesInCPU_.reset();
  miniDoubletsHC_.reset();
  segmentsHC_.reset();
  tripletsHC_.reset();
  quintupletsHC_.reset();
  pixelTripletsHC_.reset();
  pixelQuintupletsHC_.reset();
  trackCandidatesHC_.reset();
  modulesInCPU_.reset();
}

void Event::addHitToEvent(std::vector<float> const& x,
                          std::vector<float> const& y,
                          std::vector<float> const& z,
                          std::vector<unsigned int> const& detId,
                          std::vector<unsigned int> const& idxInNtuple) {
  // Use the actual number of hits instead of a max.
  unsigned int nHits = x.size();

  // Initialize space on device/host for next event.
  if (!hitsInGPU_) {
    hitsInGPU_.emplace();
    hitsBuffers_.emplace(nModules_, nHits, devAcc_, queue_);
    hitsInGPU_->setData(*hitsBuffers_);
  }

  if (!rangesInGPU_) {
    rangesInGPU_.emplace();
    rangesBuffers_.emplace(nModules_, nLowerModules_, devAcc_, queue_);
    rangesInGPU_->setData(*rangesBuffers_);
  }

  // Need a view here before transferring to the device.
  auto nHits_view = alpaka::createView(cms::alpakatools::host(), &nHits, (Idx)1u);

  // Copy the host arrays to the GPU.
  alpaka::memcpy(queue_, hitsBuffers_->xs_buf, x, nHits);
  alpaka::memcpy(queue_, hitsBuffers_->ys_buf, y, nHits);
  alpaka::memcpy(queue_, hitsBuffers_->zs_buf, z, nHits);
  alpaka::memcpy(queue_, hitsBuffers_->detid_buf, detId, nHits);
  alpaka::memcpy(queue_, hitsBuffers_->idxs_buf, idxInNtuple, nHits);
  alpaka::memcpy(queue_, hitsBuffers_->nHits_buf, nHits_view);
  alpaka::wait(queue_);  // FIXME: remove synch after inputs refactored to be in pinned memory

  Vec3D const threadsPerBlock1{1, 1, 256};
  Vec3D const blocksPerGrid1{1, 1, max_blocks};
  WorkDiv3D const hit_loop_workdiv = createWorkDiv(blocksPerGrid1, threadsPerBlock1, elementsPerThread);

  alpaka::exec<Acc3D>(queue_,
                      hit_loop_workdiv,
                      HitLoopKernel{},
                      Endcap,
                      TwoS,
                      nModules_,
                      nEndCapMap_,
                      endcapGeometryBuffers_.geoMapDetId_buf.data(),
                      endcapGeometryBuffers_.geoMapPhi_buf.data(),
                      *modulesBuffers_.data(),
                      *hitsInGPU_,
                      nHits);

  Vec3D const threadsPerBlock2{1, 1, 256};
  Vec3D const blocksPerGrid2{1, 1, max_blocks};
  WorkDiv3D const module_ranges_workdiv = createWorkDiv(blocksPerGrid2, threadsPerBlock2, elementsPerThread);

  alpaka::exec<Acc3D>(
      queue_, module_ranges_workdiv, ModuleRangesKernel{}, *modulesBuffers_.data(), *hitsInGPU_, nLowerModules_);
}

void Event::addPixelSegmentToEvent(std::vector<unsigned int> const& hitIndices0,
                                   std::vector<unsigned int> const& hitIndices1,
                                   std::vector<unsigned int> const& hitIndices2,
                                   std::vector<unsigned int> const& hitIndices3,
                                   std::vector<float> const& dPhiChange,
                                   std::vector<float> const& ptIn,
                                   std::vector<float> const& ptErr,
                                   std::vector<float> const& px,
                                   std::vector<float> const& py,
                                   std::vector<float> const& pz,
                                   std::vector<float> const& eta,
                                   std::vector<float> const& etaErr,
                                   std::vector<float> const& phi,
                                   std::vector<int> const& charge,
                                   std::vector<unsigned int> const& seedIdx,
                                   std::vector<int> const& superbin,
                                   std::vector<PixelType> const& pixelType,
                                   std::vector<char> const& isQuad) {
  unsigned int size = ptIn.size();

  if (size > n_max_pixel_segments_per_module) {
    printf(
        "*********************************************************\n"
        "* Warning: Pixel line segments will be truncated.       *\n"
        "* You need to increase n_max_pixel_segments_per_module. *\n"
        "*********************************************************\n");
    size = n_max_pixel_segments_per_module;
  }

  unsigned int mdSize = 2 * size;
  uint16_t pixelModuleIndex = pixelMapping_.pixelModuleIndex;

  if (!miniDoubletsDC_) {
    // Create a view for the element nLowerModules_ inside rangesBuffers_->miniDoubletModuleOccupancy
    auto dst_view_miniDoubletModuleOccupancy =
        alpaka::createSubView(rangesBuffers_->miniDoubletModuleOccupancy_buf, (Idx)1u, (Idx)nLowerModules_);

    // Create a host buffer for a value to be passed to the device
    auto pixelMaxMDs_buf_h = cms::alpakatools::make_host_buffer<int[]>(queue_, (Idx)1u);
    *pixelMaxMDs_buf_h.data() = n_max_pixel_md_per_modules;

    alpaka::memcpy(queue_, dst_view_miniDoubletModuleOccupancy, pixelMaxMDs_buf_h);

    WorkDiv1D const createMDArrayRangesGPU_workDiv = createWorkDiv<Vec1D>({1}, {1024}, {1});

    alpaka::exec<Acc1D>(
        queue_, createMDArrayRangesGPU_workDiv, CreateMDArrayRangesGPU{}, *modulesBuffers_.data(), *rangesInGPU_);

    auto nTotalMDs_buf_h = cms::alpakatools::make_host_buffer<unsigned int[]>(queue_, (Idx)1u);
    alpaka::memcpy(queue_, nTotalMDs_buf_h, rangesBuffers_->device_nTotalMDs_buf);
    alpaka::wait(queue_);  // wait to get the data before manipulation

    *nTotalMDs_buf_h.data() += n_max_pixel_md_per_modules;
    unsigned int nTotalMDs = *nTotalMDs_buf_h.data();

    std::array<int, 2> const mds_sizes{{static_cast<int>(nTotalMDs), static_cast<int>(nLowerModules_ + 1)}};
    miniDoubletsDC_.emplace(mds_sizes, queue_);

    auto mdsOccupancy = miniDoubletsDC_->view<MiniDoubletsOccupancySoA>();
    auto nMDs_view = alpaka::createView(devAcc_, mdsOccupancy.nMDs(), mdsOccupancy.metadata().size());
    auto totOccupancyMDs_view =
        alpaka::createView(devAcc_, mdsOccupancy.totOccupancyMDs(), mdsOccupancy.metadata().size());
    alpaka::memset(queue_, nMDs_view, 0u);
    alpaka::memset(queue_, totOccupancyMDs_view, 0u);
  }
  if (!segmentsDC_) {
    // can be optimized here: because we didn't distinguish pixel segments and outer-tracker segments and call them both "segments", so they use the index continuously.
    // If we want to further study the memory footprint in detail, we can separate the two and allocate different memories to them

    WorkDiv1D const createSegmentArrayRanges_workDiv = createWorkDiv<Vec1D>({1}, {1024}, {1});

    alpaka::exec<Acc1D>(queue_,
                        createSegmentArrayRanges_workDiv,
                        CreateSegmentArrayRanges{},
                        *modulesBuffers_.data(),
                        *rangesInGPU_,
                        miniDoubletsDC_->const_view<MiniDoubletsSoA>());

    auto nTotalSegments_view = alpaka::createView(cms::alpakatools::host(), &nTotalSegments_, (Idx)1u);

    alpaka::memcpy(queue_, nTotalSegments_view, rangesBuffers_->device_nTotalSegs_buf);
    alpaka::wait(queue_);  // wait to get the value before manipulation

    nTotalSegments_ += n_max_pixel_segments_per_module;

    std::array<int, 3> const segments_sizes{{static_cast<int>(nTotalSegments_),
                                             static_cast<int>(nLowerModules_ + 1),
                                             static_cast<int>(n_max_pixel_segments_per_module)}};
    segmentsDC_.emplace(segments_sizes, queue_);

    auto nSegments_view =
        alpaka::createView(devAcc_, segmentsDC_->view<SegmentsOccupancySoA>().nSegments(), nLowerModules_ + 1);
    auto totOccupancySegments_view = alpaka::createView(
        devAcc_, segmentsDC_->view<SegmentsOccupancySoA>().totOccupancySegments(), nLowerModules_ + 1);
    alpaka::memset(queue_, nSegments_view, 0u);
    alpaka::memset(queue_, totOccupancySegments_view, 0u);
  }

  auto hitIndices0_dev = allocBufWrapper<unsigned int>(devAcc_, size, queue_);
  auto hitIndices1_dev = allocBufWrapper<unsigned int>(devAcc_, size, queue_);
  auto hitIndices2_dev = allocBufWrapper<unsigned int>(devAcc_, size, queue_);
  auto hitIndices3_dev = allocBufWrapper<unsigned int>(devAcc_, size, queue_);
  auto dPhiChange_dev = allocBufWrapper<float>(devAcc_, size, queue_);

  alpaka::memcpy(queue_, hitIndices0_dev, hitIndices0, size);
  alpaka::memcpy(queue_, hitIndices1_dev, hitIndices1, size);
  alpaka::memcpy(queue_, hitIndices2_dev, hitIndices2, size);
  alpaka::memcpy(queue_, hitIndices3_dev, hitIndices3, size);
  alpaka::memcpy(queue_, dPhiChange_dev, dPhiChange, size);

  SegmentsPixel segmentsPixel = segmentsDC_->view<SegmentsPixelSoA>();
  alpaka::memcpy(queue_, alpaka::createView(devAcc_, segmentsPixel.ptIn(), size), ptIn, size);
  alpaka::memcpy(queue_, alpaka::createView(devAcc_, segmentsPixel.ptErr(), size), ptErr, size);
  alpaka::memcpy(queue_, alpaka::createView(devAcc_, segmentsPixel.px(), size), px, size);
  alpaka::memcpy(queue_, alpaka::createView(devAcc_, segmentsPixel.py(), size), py, size);
  alpaka::memcpy(queue_, alpaka::createView(devAcc_, segmentsPixel.pz(), size), pz, size);
  alpaka::memcpy(queue_, alpaka::createView(devAcc_, segmentsPixel.etaErr(), size), etaErr, size);
  alpaka::memcpy(queue_, alpaka::createView(devAcc_, segmentsPixel.isQuad(), size), isQuad, size);
  alpaka::memcpy(queue_, alpaka::createView(devAcc_, segmentsPixel.eta(), size), eta, size);
  alpaka::memcpy(queue_, alpaka::createView(devAcc_, segmentsPixel.phi(), size), phi, size);
  alpaka::memcpy(queue_, alpaka::createView(devAcc_, segmentsPixel.charge(), size), charge, size);
  alpaka::memcpy(queue_, alpaka::createView(devAcc_, segmentsPixel.seedIdx(), size), seedIdx, size);
  alpaka::memcpy(queue_, alpaka::createView(devAcc_, segmentsPixel.superbin(), size), superbin, size);
  alpaka::memcpy(queue_, alpaka::createView(devAcc_, segmentsPixel.pixelType(), size), pixelType, size);

  // Create source views for size and mdSize
  auto src_view_size = alpaka::createView(cms::alpakatools::host(), &size, (Idx)1u);
  auto src_view_mdSize = alpaka::createView(cms::alpakatools::host(), &mdSize, (Idx)1u);

  SegmentsOccupancy segmentsOccupancy = segmentsDC_->view<SegmentsOccupancySoA>();
  auto nSegments_view = alpaka::createView(devAcc_, segmentsOccupancy.nSegments(), (Idx)nLowerModules_ + 1);
  auto dst_view_segments = alpaka::createSubView(nSegments_view, (Idx)1u, (Idx)pixelModuleIndex);
  alpaka::memcpy(queue_, dst_view_segments, src_view_size);

  auto totOccupancySegments_view =
      alpaka::createView(devAcc_, segmentsOccupancy.totOccupancySegments(), (Idx)nLowerModules_ + 1);
  auto dst_view_totOccupancySegments = alpaka::createSubView(totOccupancySegments_view, (Idx)1u, (Idx)pixelModuleIndex);
  alpaka::memcpy(queue_, dst_view_totOccupancySegments, src_view_size);

  auto mdsOccupancy = miniDoubletsDC_->view<MiniDoubletsOccupancySoA>();
  auto nMDs_view = alpaka::createView(devAcc_, mdsOccupancy.nMDs(), (Idx)mdsOccupancy.metadata().size());
  auto dst_view_nMDs = alpaka::createSubView(nMDs_view, (Idx)1u, (Idx)pixelModuleIndex);
  alpaka::memcpy(queue_, dst_view_nMDs, src_view_mdSize);

  auto totOccupancyMDs_view =
      alpaka::createView(devAcc_, mdsOccupancy.totOccupancyMDs(), (Idx)mdsOccupancy.metadata().size());
  auto dst_view_totOccupancyMDs = alpaka::createSubView(totOccupancyMDs_view, (Idx)1u, (Idx)pixelModuleIndex);
  alpaka::memcpy(queue_, dst_view_totOccupancyMDs, src_view_mdSize);

  alpaka::wait(queue_);  // FIXME: remove synch after inputs refactored to be in pinned memory

  Vec3D const threadsPerBlock{1, 1, 256};
  Vec3D const blocksPerGrid{1, 1, max_blocks};
  WorkDiv3D const addPixelSegmentToEvent_workdiv = createWorkDiv(blocksPerGrid, threadsPerBlock, elementsPerThread);

  alpaka::exec<Acc3D>(queue_,
                      addPixelSegmentToEvent_workdiv,
                      AddPixelSegmentToEventKernel{},
                      *modulesBuffers_.data(),
                      *rangesInGPU_,
                      *hitsInGPU_,
                      miniDoubletsDC_->view<MiniDoubletsSoA>(),
                      segmentsDC_->view<SegmentsSoA>(),
                      segmentsDC_->view<SegmentsPixelSoA>(),
                      hitIndices0_dev.data(),
                      hitIndices1_dev.data(),
                      hitIndices2_dev.data(),
                      hitIndices3_dev.data(),
                      dPhiChange_dev.data(),
                      pixelModuleIndex,
                      size);
}

void Event::createMiniDoublets() {
  // Create a view for the element nLowerModules_ inside rangesBuffers_->miniDoubletModuleOccupancy
  auto dst_view_miniDoubletModuleOccupancy =
      alpaka::createSubView(rangesBuffers_->miniDoubletModuleOccupancy_buf, (Idx)1u, (Idx)nLowerModules_);

  // Create a host buffer for a value to be passed to the device
  auto pixelMaxMDs_buf_h = cms::alpakatools::make_host_buffer<int[]>(queue_, (Idx)1u);
  *pixelMaxMDs_buf_h.data() = n_max_pixel_md_per_modules;

  alpaka::memcpy(queue_, dst_view_miniDoubletModuleOccupancy, pixelMaxMDs_buf_h);

  WorkDiv1D const createMDArrayRangesGPU_workDiv = createWorkDiv<Vec1D>({1}, {1024}, {1});

  alpaka::exec<Acc1D>(
      queue_, createMDArrayRangesGPU_workDiv, CreateMDArrayRangesGPU{}, *modulesBuffers_.data(), *rangesInGPU_);

  auto nTotalMDs_buf_h = cms::alpakatools::make_host_buffer<unsigned int[]>(queue_, (Idx)1u);
  alpaka::memcpy(queue_, nTotalMDs_buf_h, rangesBuffers_->device_nTotalMDs_buf);
  alpaka::wait(queue_);  // wait to get the data before manipulation

  *nTotalMDs_buf_h.data() += n_max_pixel_md_per_modules;
  unsigned int nTotalMDs = *nTotalMDs_buf_h.data();

  if (!miniDoubletsDC_) {
    std::array<int, 2> const mds_sizes{{static_cast<int>(nTotalMDs), static_cast<int>(nLowerModules_ + 1)}};
    miniDoubletsDC_.emplace(mds_sizes, queue_);

    auto mdsOccupancy = miniDoubletsDC_->view<MiniDoubletsOccupancySoA>();
    auto nMDs_view = alpaka::createView(devAcc_, mdsOccupancy.nMDs(), mdsOccupancy.metadata().size());
    auto totOccupancyMDs_view =
        alpaka::createView(devAcc_, mdsOccupancy.totOccupancyMDs(), mdsOccupancy.metadata().size());
    alpaka::memset(queue_, nMDs_view, 0u);
    alpaka::memset(queue_, totOccupancyMDs_view, 0u);
  }

  Vec3D const threadsPerBlockCreateMDInGPU{1, 16, 32};
  Vec3D const blocksPerGridCreateMDInGPU{1, nLowerModules_ / threadsPerBlockCreateMDInGPU[1], 1};
  WorkDiv3D const createMiniDoubletsInGPUv2_workDiv =
      createWorkDiv(blocksPerGridCreateMDInGPU, threadsPerBlockCreateMDInGPU, elementsPerThread);

  alpaka::exec<Acc3D>(queue_,
                      createMiniDoubletsInGPUv2_workDiv,
                      CreateMiniDoubletsInGPUv2{},
                      *modulesBuffers_.data(),
                      *hitsInGPU_,
                      miniDoubletsDC_->view<MiniDoubletsSoA>(),
                      miniDoubletsDC_->view<MiniDoubletsOccupancySoA>(),
                      *rangesInGPU_);

  WorkDiv1D const addMiniDoubletRangesToEventExplicit_workDiv = createWorkDiv<Vec1D>({1}, {1024}, {1});

  alpaka::exec<Acc1D>(queue_,
                      addMiniDoubletRangesToEventExplicit_workDiv,
                      AddMiniDoubletRangesToEventExplicit{},
                      *modulesBuffers_.data(),
                      miniDoubletsDC_->view<MiniDoubletsOccupancySoA>(),
                      *rangesInGPU_,
                      *hitsInGPU_);

  if (addObjects_) {
    addMiniDoubletsToEventExplicit();
  }
}

void Event::createSegmentsWithModuleMap() {
  if (!segmentsDC_) {
    std::array<int, 3> const segments_sizes{{static_cast<int>(nTotalSegments_),
                                             static_cast<int>(nLowerModules_ + 1),
                                             static_cast<int>(n_max_pixel_segments_per_module)}};
    segmentsDC_.emplace(segments_sizes, queue_);

    auto nSegments_view =
        alpaka::createView(devAcc_, segmentsDC_->view<SegmentsOccupancySoA>().nSegments(), nLowerModules_ + 1);
    auto totOccupancySegments_view = alpaka::createView(
        devAcc_, segmentsDC_->view<SegmentsOccupancySoA>().totOccupancySegments(), nLowerModules_ + 1);
    alpaka::memset(queue_, nSegments_view, 0u);
    alpaka::memset(queue_, totOccupancySegments_view, 0u);
  }

  Vec3D const threadsPerBlockCreateSeg{1, 1, 64};
  Vec3D const blocksPerGridCreateSeg{1, 1, nLowerModules_};
  WorkDiv3D const createSegmentsInGPUv2_workDiv =
      createWorkDiv(blocksPerGridCreateSeg, threadsPerBlockCreateSeg, elementsPerThread);

  alpaka::exec<Acc3D>(queue_,
                      createSegmentsInGPUv2_workDiv,
                      CreateSegmentsInGPUv2{},
                      *modulesBuffers_.data(),
                      miniDoubletsDC_->const_view<MiniDoubletsSoA>(),
                      miniDoubletsDC_->const_view<MiniDoubletsOccupancySoA>(),
                      segmentsDC_->view<SegmentsSoA>(),
                      segmentsDC_->view<SegmentsOccupancySoA>(),
                      *rangesInGPU_);

  WorkDiv1D const addSegmentRangesToEventExplicit_workDiv = createWorkDiv<Vec1D>({1}, {1024}, {1});

  alpaka::exec<Acc1D>(queue_,
                      addSegmentRangesToEventExplicit_workDiv,
                      AddSegmentRangesToEventExplicit{},
                      *modulesBuffers_.data(),
                      segmentsDC_->view<SegmentsOccupancySoA>(),
                      *rangesInGPU_);

  if (addObjects_) {
    addSegmentsToEventExplicit();
  }
}

void Event::createTriplets() {
  if (!tripletsDC_) {
    WorkDiv1D const createTripletArrayRanges_workDiv = createWorkDiv<Vec1D>({1}, {1024}, {1});

    alpaka::exec<Acc1D>(queue_,
                        createTripletArrayRanges_workDiv,
                        CreateTripletArrayRanges{},
                        *modulesBuffers_.data(),
                        *rangesInGPU_,
                        segmentsDC_->const_view<SegmentsOccupancySoA>());

    // TODO: Why are we pulling this back down only to put it back on the device in a new struct?
    auto maxTriplets_buf_h = cms::alpakatools::make_host_buffer<unsigned int[]>(queue_, (Idx)1u);

    alpaka::memcpy(queue_, maxTriplets_buf_h, rangesBuffers_->device_nTotalTrips_buf);
    alpaka::wait(queue_);  // wait to get the value before using it

    std::array<int, 2> const triplets_sizes{
        {static_cast<int>(*maxTriplets_buf_h.data()), static_cast<int>(nLowerModules_)}};
    tripletsDC_.emplace(triplets_sizes, queue_);

    auto tripletsOccupancy = tripletsDC_->view<TripletsOccupancySoA>();
    auto nTriplets_view =
        alpaka::createView(devAcc_, tripletsOccupancy.nTriplets(), tripletsOccupancy.metadata().size());
    alpaka::memset(queue_, nTriplets_view, 0u);
    auto totOccupancyTriplets_view =
        alpaka::createView(devAcc_, tripletsOccupancy.totOccupancyTriplets(), tripletsOccupancy.metadata().size());
    alpaka::memset(queue_, totOccupancyTriplets_view, 0u);
    auto triplets = tripletsDC_->view<TripletsSoA>();
    auto partOfPT5_view = alpaka::createView(devAcc_, triplets.partOfPT5(), triplets.metadata().size());
    alpaka::memset(queue_, partOfPT5_view, 0u);
    auto partOfT5_view = alpaka::createView(devAcc_, triplets.partOfT5(), triplets.metadata().size());
    alpaka::memset(queue_, partOfT5_view, 0u);
    auto partOfPT3_view = alpaka::createView(devAcc_, triplets.partOfPT3(), triplets.metadata().size());
    alpaka::memset(queue_, partOfPT3_view, 0u);
  }

  uint16_t nonZeroModules = 0;
  unsigned int max_InnerSeg = 0;

  // Allocate and copy nSegments from device to host (only nLowerModules in OT, not the +1 with pLSs)
  auto nSegments_buf_h = cms::alpakatools::make_host_buffer<unsigned int[]>(queue_, nLowerModules_);
  auto nSegments_buf_d =
      alpaka::createView(devAcc_, segmentsDC_->const_view<SegmentsOccupancySoA>().nSegments(), nLowerModules_);
  alpaka::memcpy(queue_, nSegments_buf_h, nSegments_buf_d, nLowerModules_);

  // ... same for module_nConnectedModules
  // FIXME: replace by ES host data
  auto module_nConnectedModules_buf_h = cms::alpakatools::make_host_buffer<uint16_t[]>(queue_, nLowerModules_);
  alpaka::memcpy(queue_, module_nConnectedModules_buf_h, modulesBuffers_.nConnectedModules_buf, nLowerModules_);

  alpaka::wait(queue_);  // wait for nSegments and module_nConnectedModules before using

  auto const* nSegments = nSegments_buf_h.data();
  auto const* module_nConnectedModules = module_nConnectedModules_buf_h.data();

  // Allocate host index and fill it directly
  auto index_buf_h = cms::alpakatools::make_host_buffer<uint16_t[]>(queue_, nLowerModules_);
  auto* index = index_buf_h.data();

  for (uint16_t innerLowerModuleIndex = 0; innerLowerModuleIndex < nLowerModules_; innerLowerModuleIndex++) {
    uint16_t nConnectedModules = module_nConnectedModules[innerLowerModuleIndex];
    unsigned int nInnerSegments = nSegments[innerLowerModuleIndex];
    if (nConnectedModules != 0 and nInnerSegments != 0) {
      index[nonZeroModules] = innerLowerModuleIndex;
      nonZeroModules++;
    }
    max_InnerSeg = std::max(max_InnerSeg, nInnerSegments);
  }

  // Allocate and copy to device index
  auto index_gpu_buf = allocBufWrapper<uint16_t>(devAcc_, nLowerModules_, queue_);
  alpaka::memcpy(queue_, index_gpu_buf, index_buf_h, nonZeroModules);

  Vec3D const threadsPerBlockCreateTrip{1, 16, 16};
  Vec3D const blocksPerGridCreateTrip{max_blocks, 1, 1};
  WorkDiv3D const createTriplets_workDiv =
      createWorkDiv(blocksPerGridCreateTrip, threadsPerBlockCreateTrip, elementsPerThread);

  alpaka::exec<Acc3D>(queue_,
                      createTriplets_workDiv,
                      CreateTriplets{},
                      *modulesBuffers_.data(),
                      miniDoubletsDC_->const_view<MiniDoubletsSoA>(),
                      segmentsDC_->const_view<SegmentsSoA>(),
                      segmentsDC_->const_view<SegmentsOccupancySoA>(),
                      tripletsDC_->view<TripletsSoA>(),
                      tripletsDC_->view<TripletsOccupancySoA>(),
                      *rangesInGPU_,
                      index_gpu_buf.data(),
                      nonZeroModules);

  WorkDiv1D const addTripletRangesToEventExplicit_workDiv = createWorkDiv<Vec1D>({1}, {1024}, {1});

  alpaka::exec<Acc1D>(queue_,
                      addTripletRangesToEventExplicit_workDiv,
                      AddTripletRangesToEventExplicit{},
                      *modulesBuffers_.data(),
                      tripletsDC_->const_view<TripletsOccupancySoA>(),
                      *rangesInGPU_);

  if (addObjects_) {
    addTripletsToEventExplicit();
  }
}

void Event::createTrackCandidates(bool no_pls_dupclean, bool tc_pls_triplets) {
  if (!trackCandidatesDC_) {
    trackCandidatesDC_.emplace(n_max_nonpixel_track_candidates + n_max_pixel_track_candidates, queue_);
    auto buf = trackCandidatesDC_->buffer();
    alpaka::memset(queue_, buf, 0u);
  }

  Vec3D const threadsPerBlock_crossCleanpT3{1, 16, 64};
  Vec3D const blocksPerGrid_crossCleanpT3{1, 4, 20};
  WorkDiv3D const crossCleanpT3_workDiv =
      createWorkDiv(blocksPerGrid_crossCleanpT3, threadsPerBlock_crossCleanpT3, elementsPerThread);

  alpaka::exec<Acc3D>(queue_,
                      crossCleanpT3_workDiv,
                      CrossCleanpT3{},
                      *modulesBuffers_.data(),
                      *rangesInGPU_,
                      pixelTripletsDC_->view(),
                      segmentsDC_->const_view<SegmentsPixelSoA>(),
                      pixelQuintupletsDC_->const_view());

  WorkDiv1D const addpT3asTrackCandidates_workDiv = createWorkDiv<Vec1D>({1}, {512}, {1});

  alpaka::exec<Acc1D>(queue_,
                      addpT3asTrackCandidates_workDiv,
                      AddpT3asTrackCandidates{},
                      nLowerModules_,
                      pixelTripletsDC_->const_view(),
                      trackCandidatesDC_->view(),
                      segmentsDC_->const_view<SegmentsPixelSoA>(),
                      *rangesInGPU_);

  // Pull nEligibleT5Modules from the device.
  auto nEligibleModules_buf_h = cms::alpakatools::make_host_buffer<uint16_t[]>(queue_, 1u);
  alpaka::memcpy(queue_, nEligibleModules_buf_h, rangesBuffers_->nEligibleT5Modules_buf);
  alpaka::wait(queue_);  // wait to get the value before using
  auto const nEligibleModules = *nEligibleModules_buf_h.data();

  Vec3D const threadsPerBlockRemoveDupQuints{1, 16, 32};
  Vec3D const blocksPerGridRemoveDupQuints{1, std::max(nEligibleModules / 16, 1), std::max(nEligibleModules / 32, 1)};
  WorkDiv3D const removeDupQuintupletsBeforeTC_workDiv =
      createWorkDiv(blocksPerGridRemoveDupQuints, threadsPerBlockRemoveDupQuints, elementsPerThread);

  alpaka::exec<Acc3D>(queue_,
                      removeDupQuintupletsBeforeTC_workDiv,
                      RemoveDupQuintupletsBeforeTC{},
                      quintupletsDC_->view<QuintupletsSoA>(),
                      quintupletsDC_->view<QuintupletsOccupancySoA>(),
                      *rangesInGPU_);

  Vec3D const threadsPerBlock_crossCleanT5{32, 1, 32};
  Vec3D const blocksPerGrid_crossCleanT5{(13296 / 32) + 1, 1, max_blocks};
  WorkDiv3D const crossCleanT5_workDiv =
      createWorkDiv(blocksPerGrid_crossCleanT5, threadsPerBlock_crossCleanT5, elementsPerThread);

  alpaka::exec<Acc3D>(queue_,
                      crossCleanT5_workDiv,
                      CrossCleanT5{},
                      *modulesBuffers_.data(),
                      quintupletsDC_->view<QuintupletsSoA>(),
                      quintupletsDC_->const_view<QuintupletsOccupancySoA>(),
                      pixelQuintupletsDC_->const_view(),
                      pixelTripletsDC_->const_view(),
                      *rangesInGPU_);

  Vec3D const threadsPerBlock_addT5asTrackCandidate{1, 8, 128};
  Vec3D const blocksPerGrid_addT5asTrackCandidate{1, 8, 10};
  WorkDiv3D const addT5asTrackCandidate_workDiv =
      createWorkDiv(blocksPerGrid_addT5asTrackCandidate, threadsPerBlock_addT5asTrackCandidate, elementsPerThread);

  alpaka::exec<Acc3D>(queue_,
                      addT5asTrackCandidate_workDiv,
                      AddT5asTrackCandidate{},
                      nLowerModules_,
                      quintupletsDC_->const_view<QuintupletsSoA>(),
                      quintupletsDC_->const_view<QuintupletsOccupancySoA>(),
                      trackCandidatesDC_->view(),
                      *rangesInGPU_);

  if (!no_pls_dupclean) {
    Vec3D const threadsPerBlockCheckHitspLS{1, 16, 16};
    Vec3D const blocksPerGridCheckHitspLS{1, max_blocks * 4, max_blocks / 4};
    WorkDiv3D const checkHitspLS_workDiv =
        createWorkDiv(blocksPerGridCheckHitspLS, threadsPerBlockCheckHitspLS, elementsPerThread);

    alpaka::exec<Acc3D>(queue_,
                        checkHitspLS_workDiv,
                        CheckHitspLS{},
                        *modulesBuffers_.data(),
                        segmentsDC_->const_view<SegmentsOccupancySoA>(),
                        segmentsDC_->view<SegmentsPixelSoA>(),
                        true);
  }

  Vec3D const threadsPerBlock_crossCleanpLS{1, 16, 32};
  Vec3D const blocksPerGrid_crossCleanpLS{1, 4, 20};
  WorkDiv3D const crossCleanpLS_workDiv =
      createWorkDiv(blocksPerGrid_crossCleanpLS, threadsPerBlock_crossCleanpLS, elementsPerThread);

  alpaka::exec<Acc3D>(queue_,
                      crossCleanpLS_workDiv,
                      CrossCleanpLS{},
                      *modulesBuffers_.data(),
                      *rangesInGPU_,
                      pixelTripletsDC_->const_view(),
                      trackCandidatesDC_->view(),
                      segmentsDC_->const_view<SegmentsSoA>(),
                      segmentsDC_->const_view<SegmentsOccupancySoA>(),
                      segmentsDC_->view<SegmentsPixelSoA>(),
                      miniDoubletsDC_->const_view<MiniDoubletsSoA>(),
                      *hitsInGPU_,
                      quintupletsDC_->const_view<QuintupletsSoA>());

  Vec3D const threadsPerBlock_addpLSasTrackCandidateInGPU{1, 1, 384};
  Vec3D const blocksPerGrid_addpLSasTrackCandidateInGPU{1, 1, max_blocks};
  WorkDiv3D const addpLSasTrackCandidateInGPU_workDiv = createWorkDiv(
      blocksPerGrid_addpLSasTrackCandidateInGPU, threadsPerBlock_addpLSasTrackCandidateInGPU, elementsPerThread);

  alpaka::exec<Acc3D>(queue_,
                      addpLSasTrackCandidateInGPU_workDiv,
                      AddpLSasTrackCandidateInGPU{},
                      nLowerModules_,
                      trackCandidatesDC_->view(),
                      segmentsDC_->const_view<SegmentsOccupancySoA>(),
                      segmentsDC_->const_view<SegmentsPixelSoA>(),
                      tc_pls_triplets);

  // Check if either n_max_pixel_track_candidates or n_max_nonpixel_track_candidates was reached
  auto nTrackCanpT5Host_buf = allocBufWrapper<unsigned int>(cms::alpakatools::host(), 1, queue_);
  auto nTrackCanpT3Host_buf = allocBufWrapper<unsigned int>(cms::alpakatools::host(), 1, queue_);
  auto nTrackCanpLSHost_buf = allocBufWrapper<unsigned int>(cms::alpakatools::host(), 1, queue_);
  auto nTrackCanT5Host_buf = allocBufWrapper<unsigned int>(cms::alpakatools::host(), 1, queue_);
  alpaka::memcpy(
      queue_, nTrackCanpT5Host_buf, alpaka::createView(devAcc_, &(*trackCandidatesDC_)->nTrackCandidatespT5(), 1u));
  alpaka::memcpy(
      queue_, nTrackCanpT3Host_buf, alpaka::createView(devAcc_, &(*trackCandidatesDC_)->nTrackCandidatespT3(), 1u));
  alpaka::memcpy(
      queue_, nTrackCanpLSHost_buf, alpaka::createView(devAcc_, &(*trackCandidatesDC_)->nTrackCandidatespLS(), 1u));
  alpaka::memcpy(
      queue_, nTrackCanT5Host_buf, alpaka::createView(devAcc_, &(*trackCandidatesDC_)->nTrackCandidatesT5(), 1u));
  alpaka::wait(queue_);  // wait to get the values before using them

  auto nTrackCandidatespT5 = *nTrackCanpT5Host_buf.data();
  auto nTrackCandidatespT3 = *nTrackCanpT3Host_buf.data();
  auto nTrackCandidatespLS = *nTrackCanpLSHost_buf.data();
  auto nTrackCandidatesT5 = *nTrackCanT5Host_buf.data();
  if ((nTrackCandidatespT5 + nTrackCandidatespT3 + nTrackCandidatespLS == n_max_pixel_track_candidates) ||
      (nTrackCandidatesT5 == n_max_nonpixel_track_candidates)) {
    printf(
        "****************************************************************************************************\n"
        "* Warning: Track candidates were possibly truncated.                                               *\n"
        "* You may need to increase either n_max_pixel_track_candidates or n_max_nonpixel_track_candidates. *\n"
        "* Run the code with the WARNINGS flag activated for more details.                                  *\n"
        "****************************************************************************************************\n");
  }
}

void Event::createPixelTriplets() {
  if (!pixelTripletsDC_) {
    pixelTripletsDC_.emplace(n_max_pixel_triplets, queue_);
    auto nPixelTriplets_view = alpaka::createView(devAcc_, &(*pixelTripletsDC_)->nPixelTriplets(), 1u);
    alpaka::memset(queue_, nPixelTriplets_view, 0u);
    auto totOccupancyPixelTriplets_view =
        alpaka::createView(devAcc_, &(*pixelTripletsDC_)->totOccupancyPixelTriplets(), 1u);
    alpaka::memset(queue_, totOccupancyPixelTriplets_view, 0u);
  }
  SegmentsOccupancy segmentsOccupancy = segmentsDC_->view<SegmentsOccupancySoA>();
  SegmentsPixelConst segmentsPixel = segmentsDC_->view<SegmentsPixelSoA>();

  auto superbins_buf = allocBufWrapper<int>(cms::alpakatools::host(), n_max_pixel_segments_per_module, queue_);
  auto pixelTypes_buf = allocBufWrapper<PixelType>(cms::alpakatools::host(), n_max_pixel_segments_per_module, queue_);

  alpaka::memcpy(
      queue_, superbins_buf, alpaka::createView(devAcc_, segmentsPixel.superbin(), n_max_pixel_segments_per_module));
  alpaka::memcpy(
      queue_, pixelTypes_buf, alpaka::createView(devAcc_, segmentsPixel.pixelType(), n_max_pixel_segments_per_module));
  auto const* superbins = superbins_buf.data();
  auto const* pixelTypes = pixelTypes_buf.data();

  unsigned int nInnerSegments;
  auto nInnerSegments_src_view = alpaka::createView(cms::alpakatools::host(), &nInnerSegments, (size_t)1u);

  // Create a sub-view for the device buffer
  unsigned int totalModules = nLowerModules_ + 1;
  auto dev_view_nSegments_buf = alpaka::createView(devAcc_, segmentsOccupancy.nSegments(), totalModules);
  auto dev_view_nSegments = alpaka::createSubView(dev_view_nSegments_buf, (Idx)1u, (Idx)nLowerModules_);

  alpaka::memcpy(queue_, nInnerSegments_src_view, dev_view_nSegments);
  alpaka::wait(queue_);  // wait to get nInnerSegments (also superbins and pixelTypes) before using

  auto connectedPixelSize_host_buf = allocBufWrapper<unsigned int>(cms::alpakatools::host(), nInnerSegments, queue_);
  auto connectedPixelIndex_host_buf = allocBufWrapper<unsigned int>(cms::alpakatools::host(), nInnerSegments, queue_);
  auto connectedPixelSize_dev_buf = allocBufWrapper<unsigned int>(devAcc_, nInnerSegments, queue_);
  auto connectedPixelIndex_dev_buf = allocBufWrapper<unsigned int>(devAcc_, nInnerSegments, queue_);

  unsigned int* connectedPixelSize_host = connectedPixelSize_host_buf.data();
  unsigned int* connectedPixelIndex_host = connectedPixelIndex_host_buf.data();

  int pixelIndexOffsetPos =
      pixelMapping_.connectedPixelsIndex[size_superbins - 1] + pixelMapping_.connectedPixelsSizes[size_superbins - 1];
  int pixelIndexOffsetNeg = pixelMapping_.connectedPixelsIndexPos[size_superbins - 1] +
                            pixelMapping_.connectedPixelsSizesPos[size_superbins - 1] + pixelIndexOffsetPos;

  // TODO: check if a map/reduction to just eligible pLSs would speed up the kernel
  // the current selection still leaves a significant fraction of unmatchable pLSs
  for (unsigned int i = 0; i < nInnerSegments; i++) {  // loop over # pLS
    PixelType pixelType = pixelTypes[i];               // Get pixel type for this pLS
    int superbin = superbins[i];                       // Get superbin for this pixel
    if ((superbin < 0) or (superbin >= (int)size_superbins) or
        ((pixelType != PixelType::kHighPt) and (pixelType != PixelType::kLowPtPosCurv) and
         (pixelType != PixelType::kLowPtNegCurv))) {
      connectedPixelSize_host[i] = 0;
      connectedPixelIndex_host[i] = 0;
      continue;
    }

    // Used pixel type to select correct size-index arrays
    switch (pixelType) {
      case PixelType::kInvalid:
        break;
      case PixelType::kHighPt:
        // number of connected modules to this pixel
        connectedPixelSize_host[i] = pixelMapping_.connectedPixelsSizes[superbin];
        // index to get start of connected modules for this superbin in map
        connectedPixelIndex_host[i] = pixelMapping_.connectedPixelsIndex[superbin];
        break;
      case PixelType::kLowPtPosCurv:
        // number of connected modules to this pixel
        connectedPixelSize_host[i] = pixelMapping_.connectedPixelsSizesPos[superbin];
        // index to get start of connected modules for this superbin in map
        connectedPixelIndex_host[i] = pixelMapping_.connectedPixelsIndexPos[superbin] + pixelIndexOffsetPos;
        break;
      case PixelType::kLowPtNegCurv:
        // number of connected modules to this pixel
        connectedPixelSize_host[i] = pixelMapping_.connectedPixelsSizesNeg[superbin];
        // index to get start of connected modules for this superbin in map
        connectedPixelIndex_host[i] = pixelMapping_.connectedPixelsIndexNeg[superbin] + pixelIndexOffsetNeg;
        break;
    }
  }

  alpaka::memcpy(queue_, connectedPixelSize_dev_buf, connectedPixelSize_host_buf, nInnerSegments);
  alpaka::memcpy(queue_, connectedPixelIndex_dev_buf, connectedPixelIndex_host_buf, nInnerSegments);

  Vec3D const threadsPerBlock{1, 4, 32};
  Vec3D const blocksPerGrid{16 /* above median of connected modules*/, 4096, 1};
  WorkDiv3D const createPixelTripletsFromMap_workDiv = createWorkDiv(blocksPerGrid, threadsPerBlock, elementsPerThread);

  alpaka::exec<Acc3D>(queue_,
                      createPixelTripletsFromMap_workDiv,
                      CreatePixelTripletsFromMap{},
                      *modulesBuffers_.data(),
                      *rangesInGPU_,
                      miniDoubletsDC_->const_view<MiniDoubletsSoA>(),
                      segmentsDC_->const_view<SegmentsSoA>(),
                      segmentsDC_->const_view<SegmentsPixelSoA>(),
                      tripletsDC_->view<TripletsSoA>(),
                      tripletsDC_->const_view<TripletsOccupancySoA>(),
                      pixelTripletsDC_->view(),
                      connectedPixelSize_dev_buf.data(),
                      connectedPixelIndex_dev_buf.data(),
                      nInnerSegments);

#ifdef WARNINGS
  auto nPixelTriplets_buf = allocBufWrapper<unsigned int>(cms::alpakatools::host(), 1, queue_);

  alpaka::memcpy(queue_, nPixelTriplets_buf, alpaka::createView(devAcc_, &(*pixelTripletsDC_)->nPixelTriplets(), 1u));
  alpaka::wait(queue_);  // wait to get the value before using it

  std::cout << "number of pixel triplets = " << *nPixelTriplets_buf.data() << std::endl;
#endif

  //pT3s can be cleaned here because they're not used in making pT5s!
  Vec3D const threadsPerBlockDupPixTrip{1, 16, 16};
  //seems like more blocks lead to conflicting writes
  Vec3D const blocksPerGridDupPixTrip{1, 40, 1};
  WorkDiv3D const removeDupPixelTripletsFromMap_workDiv =
      createWorkDiv(blocksPerGridDupPixTrip, threadsPerBlockDupPixTrip, elementsPerThread);

  alpaka::exec<Acc3D>(
      queue_, removeDupPixelTripletsFromMap_workDiv, RemoveDupPixelTripletsFromMap{}, pixelTripletsDC_->view());
}

void Event::createQuintuplets() {
  WorkDiv1D const createEligibleModulesListForQuintuplets_workDiv = createWorkDiv<Vec1D>({1}, {1024}, {1});

  alpaka::exec<Acc1D>(queue_,
                      createEligibleModulesListForQuintuplets_workDiv,
                      CreateEligibleModulesListForQuintuplets{},
                      *modulesBuffers_.data(),
                      tripletsDC_->const_view<TripletsOccupancySoA>(),
                      *rangesInGPU_);

  auto nEligibleT5Modules_buf = allocBufWrapper<uint16_t>(cms::alpakatools::host(), 1, queue_);
  auto nTotalQuintuplets_buf = allocBufWrapper<unsigned int>(cms::alpakatools::host(), 1, queue_);

  alpaka::memcpy(queue_, nEligibleT5Modules_buf, rangesBuffers_->nEligibleT5Modules_buf);
  alpaka::memcpy(queue_, nTotalQuintuplets_buf, rangesBuffers_->device_nTotalQuints_buf);
  alpaka::wait(queue_);  // wait for the values before using them

  auto nEligibleT5Modules = *nEligibleT5Modules_buf.data();
  auto nTotalQuintuplets = *nTotalQuintuplets_buf.data();

  if (!quintupletsDC_) {
    std::array<int, 2> const quintuplets_sizes{{static_cast<int>(nTotalQuintuplets), static_cast<int>(nLowerModules_)}};
    quintupletsDC_.emplace(quintuplets_sizes, queue_);
    auto quintupletsOccupancy = quintupletsDC_->view<QuintupletsOccupancySoA>();
    auto nQuintuplets_view =
        alpaka::createView(devAcc_, quintupletsOccupancy.nQuintuplets(), quintupletsOccupancy.metadata().size());
    alpaka::memset(queue_, nQuintuplets_view, 0u);
    auto totOccupancyQuintuplets_view = alpaka::createView(
        devAcc_, quintupletsOccupancy.totOccupancyQuintuplets(), quintupletsOccupancy.metadata().size());
    alpaka::memset(queue_, totOccupancyQuintuplets_view, 0u);
    auto quintuplets = quintupletsDC_->view<QuintupletsSoA>();
    auto isDup_view = alpaka::createView(devAcc_, quintuplets.isDup(), quintuplets.metadata().size());
    alpaka::memset(queue_, isDup_view, 0u);
    auto tightCutFlag_view = alpaka::createView(devAcc_, quintuplets.tightCutFlag(), quintuplets.metadata().size());
    alpaka::memset(queue_, tightCutFlag_view, 0u);
    auto partOfPT5_view = alpaka::createView(devAcc_, quintuplets.partOfPT5(), quintuplets.metadata().size());
    alpaka::memset(queue_, partOfPT5_view, 0u);
  }

  Vec3D const threadsPerBlockQuints{1, 8, 32};
  Vec3D const blocksPerGridQuints{std::max((int)nEligibleT5Modules, 1), 1, 1};
  WorkDiv3D const createQuintuplets_workDiv =
      createWorkDiv(blocksPerGridQuints, threadsPerBlockQuints, elementsPerThread);

  alpaka::exec<Acc3D>(queue_,
                      createQuintuplets_workDiv,
                      CreateQuintuplets{},
                      *modulesBuffers_.data(),
                      miniDoubletsDC_->const_view<MiniDoubletsSoA>(),
                      segmentsDC_->const_view<SegmentsSoA>(),
                      tripletsDC_->view<TripletsSoA>(),
                      tripletsDC_->const_view<TripletsOccupancySoA>(),
                      quintupletsDC_->view<QuintupletsSoA>(),
                      quintupletsDC_->view<QuintupletsOccupancySoA>(),
                      *rangesInGPU_,
                      nEligibleT5Modules);

  Vec3D const threadsPerBlockDupQuint{1, 16, 16};
  Vec3D const blocksPerGridDupQuint{max_blocks, 1, 1};
  WorkDiv3D const removeDupQuintupletsAfterBuild_workDiv =
      createWorkDiv(blocksPerGridDupQuint, threadsPerBlockDupQuint, elementsPerThread);

  alpaka::exec<Acc3D>(queue_,
                      removeDupQuintupletsAfterBuild_workDiv,
                      RemoveDupQuintupletsAfterBuild{},
                      *modulesBuffers_.data(),
                      quintupletsDC_->view<QuintupletsSoA>(),
                      quintupletsDC_->const_view<QuintupletsOccupancySoA>(),
                      *rangesInGPU_);

  WorkDiv1D const addQuintupletRangesToEventExplicit_workDiv = createWorkDiv<Vec1D>({1}, {1024}, {1});

  alpaka::exec<Acc1D>(queue_,
                      addQuintupletRangesToEventExplicit_workDiv,
                      AddQuintupletRangesToEventExplicit{},
                      *modulesBuffers_.data(),
                      quintupletsDC_->const_view<QuintupletsOccupancySoA>(),
                      *rangesInGPU_);

  if (addObjects_) {
    addQuintupletsToEventExplicit();
  }
}

void Event::pixelLineSegmentCleaning(bool no_pls_dupclean) {
  if (!no_pls_dupclean) {
    Vec3D const threadsPerBlockCheckHitspLS{1, 16, 16};
    Vec3D const blocksPerGridCheckHitspLS{1, max_blocks * 4, max_blocks / 4};
    WorkDiv3D const checkHitspLS_workDiv =
        createWorkDiv(blocksPerGridCheckHitspLS, threadsPerBlockCheckHitspLS, elementsPerThread);

    alpaka::exec<Acc3D>(queue_,
                        checkHitspLS_workDiv,
                        CheckHitspLS{},
                        *modulesBuffers_.data(),
                        segmentsDC_->const_view<SegmentsOccupancySoA>(),
                        segmentsDC_->view<SegmentsPixelSoA>(),
                        false);
  }
}

void Event::createPixelQuintuplets() {
  if (!pixelQuintupletsDC_) {
    pixelQuintupletsDC_.emplace(n_max_pixel_quintuplets, queue_);
    auto nPixelQuintuplets_view = alpaka::createView(devAcc_, &(*pixelQuintupletsDC_)->nPixelQuintuplets(), 1u);
    alpaka::memset(queue_, nPixelQuintuplets_view, 0u);
    auto totOccupancyPixelQuintuplets_view =
        alpaka::createView(devAcc_, &(*pixelQuintupletsDC_)->totOccupancyPixelQuintuplets(), 1u);
    alpaka::memset(queue_, totOccupancyPixelQuintuplets_view, 0u);
  }
  if (!trackCandidatesDC_) {
    trackCandidatesDC_.emplace(n_max_nonpixel_track_candidates + n_max_pixel_track_candidates, queue_);
    auto buf = trackCandidatesDC_->buffer();
    alpaka::memset(queue_, buf, 0u);
  }
  SegmentsOccupancy segmentsOccupancy = segmentsDC_->view<SegmentsOccupancySoA>();
  SegmentsPixelConst segmentsPixel = segmentsDC_->view<SegmentsPixelSoA>();

  auto superbins_buf = allocBufWrapper<int>(cms::alpakatools::host(), n_max_pixel_segments_per_module, queue_);
  auto pixelTypes_buf = allocBufWrapper<PixelType>(cms::alpakatools::host(), n_max_pixel_segments_per_module, queue_);

  alpaka::memcpy(
      queue_, superbins_buf, alpaka::createView(devAcc_, segmentsPixel.superbin(), n_max_pixel_segments_per_module));
  alpaka::memcpy(
      queue_, pixelTypes_buf, alpaka::createView(devAcc_, segmentsPixel.pixelType(), n_max_pixel_segments_per_module));
  auto const* superbins = superbins_buf.data();
  auto const* pixelTypes = pixelTypes_buf.data();

  unsigned int nInnerSegments;
  auto nInnerSegments_src_view = alpaka::createView(cms::alpakatools::host(), &nInnerSegments, (size_t)1u);

  // Create a sub-view for the device buffer
  unsigned int totalModules = nLowerModules_ + 1;
  auto dev_view_nSegments_buf = alpaka::createView(devAcc_, segmentsOccupancy.nSegments(), totalModules);
  auto dev_view_nSegments = alpaka::createSubView(dev_view_nSegments_buf, (Idx)1u, (Idx)nLowerModules_);

  alpaka::memcpy(queue_, nInnerSegments_src_view, dev_view_nSegments);
  alpaka::wait(queue_);  // wait to get nInnerSegments (also superbins and pixelTypes) before using

  auto connectedPixelSize_host_buf = allocBufWrapper<unsigned int>(cms::alpakatools::host(), nInnerSegments, queue_);
  auto connectedPixelIndex_host_buf = allocBufWrapper<unsigned int>(cms::alpakatools::host(), nInnerSegments, queue_);
  auto connectedPixelSize_dev_buf = allocBufWrapper<unsigned int>(devAcc_, nInnerSegments, queue_);
  auto connectedPixelIndex_dev_buf = allocBufWrapper<unsigned int>(devAcc_, nInnerSegments, queue_);

  auto* connectedPixelSize_host = connectedPixelSize_host_buf.data();
  auto* connectedPixelIndex_host = connectedPixelIndex_host_buf.data();

  int pixelIndexOffsetPos = pixelMapping_.connectedPixelsIndex[::size_superbins - 1] +
                            pixelMapping_.connectedPixelsSizes[::size_superbins - 1];
  int pixelIndexOffsetNeg = pixelMapping_.connectedPixelsIndexPos[::size_superbins - 1] +
                            pixelMapping_.connectedPixelsSizesPos[::size_superbins - 1] + pixelIndexOffsetPos;

  // Loop over # pLS
  for (unsigned int i = 0; i < nInnerSegments; i++) {
    PixelType pixelType = pixelTypes[i];  // Get pixel type for this pLS
    int superbin = superbins[i];          // Get superbin for this pixel
    if ((superbin < 0) or (superbin >= (int)size_superbins) or
        ((pixelType != PixelType::kHighPt) and (pixelType != PixelType::kLowPtPosCurv) and
         (pixelType != PixelType::kLowPtNegCurv))) {
      connectedPixelSize_host[i] = 0;
      connectedPixelIndex_host[i] = 0;
      continue;
    }

    // Used pixel type to select correct size-index arrays
    switch (pixelType) {
      case PixelType::kInvalid:
        break;
      case PixelType::kHighPt:
        // number of connected modules to this pixel
        connectedPixelSize_host[i] = pixelMapping_.connectedPixelsSizes[superbin];
        // index to get start of connected modules for this superbin in map
        connectedPixelIndex_host[i] = pixelMapping_.connectedPixelsIndex[superbin];
        break;
      case PixelType::kLowPtPosCurv:
        // number of connected modules to this pixel
        connectedPixelSize_host[i] = pixelMapping_.connectedPixelsSizesPos[superbin];
        // index to get start of connected modules for this superbin in map
        connectedPixelIndex_host[i] = pixelMapping_.connectedPixelsIndexPos[superbin] + pixelIndexOffsetPos;
        break;
      case PixelType::kLowPtNegCurv:
        // number of connected modules to this pixel
        connectedPixelSize_host[i] = pixelMapping_.connectedPixelsSizesNeg[superbin];
        // index to get start of connected modules for this superbin in map
        connectedPixelIndex_host[i] = pixelMapping_.connectedPixelsIndexNeg[superbin] + pixelIndexOffsetNeg;
        break;
    }
  }

  alpaka::memcpy(queue_, connectedPixelSize_dev_buf, connectedPixelSize_host_buf, nInnerSegments);
  alpaka::memcpy(queue_, connectedPixelIndex_dev_buf, connectedPixelIndex_host_buf, nInnerSegments);

  Vec3D const threadsPerBlockCreatePixQuints{1, 16, 16};
  Vec3D const blocksPerGridCreatePixQuints{16, max_blocks, 1};
  WorkDiv3D const createPixelQuintupletsFromMap_workDiv =
      createWorkDiv(blocksPerGridCreatePixQuints, threadsPerBlockCreatePixQuints, elementsPerThread);

  alpaka::exec<Acc3D>(queue_,
                      createPixelQuintupletsFromMap_workDiv,
                      CreatePixelQuintupletsFromMap{},
                      *modulesBuffers_.data(),
                      miniDoubletsDC_->const_view<MiniDoubletsSoA>(),
                      segmentsDC_->const_view<SegmentsSoA>(),
                      segmentsDC_->view<SegmentsPixelSoA>(),
                      tripletsDC_->view<TripletsSoA>(),
                      quintupletsDC_->view<QuintupletsSoA>(),
                      quintupletsDC_->const_view<QuintupletsOccupancySoA>(),
                      pixelQuintupletsDC_->view(),
                      connectedPixelSize_dev_buf.data(),
                      connectedPixelIndex_dev_buf.data(),
                      nInnerSegments,
                      *rangesInGPU_);

  Vec3D const threadsPerBlockDupPix{1, 16, 16};
  Vec3D const blocksPerGridDupPix{1, max_blocks, 1};
  WorkDiv3D const removeDupPixelQuintupletsFromMap_workDiv =
      createWorkDiv(blocksPerGridDupPix, threadsPerBlockDupPix, elementsPerThread);

  alpaka::exec<Acc3D>(queue_,
                      removeDupPixelQuintupletsFromMap_workDiv,
                      RemoveDupPixelQuintupletsFromMap{},
                      pixelQuintupletsDC_->view());

  WorkDiv1D const addpT5asTrackCandidate_workDiv = createWorkDiv<Vec1D>({1}, {256}, {1});

  alpaka::exec<Acc1D>(queue_,
                      addpT5asTrackCandidate_workDiv,
                      AddpT5asTrackCandidate{},
                      nLowerModules_,
                      pixelQuintupletsDC_->const_view(),
                      trackCandidatesDC_->view(),
                      segmentsDC_->const_view<SegmentsPixelSoA>(),
                      *rangesInGPU_);

#ifdef WARNINGS
  auto nPixelQuintuplets_buf = allocBufWrapper<unsigned int>(cms::alpakatools::host(), 1, queue_);

  alpaka::memcpy(
      queue_, nPixelQuintuplets_buf, alpaka::createView(devAcc_, &(*pixelQuintupletsDC_)->nPixelQuintuplets(), 1u));
  alpaka::wait(queue_);  // wait to get the value before using it

  std::cout << "number of pixel quintuplets = " << *nPixelQuintuplets_buf.data() << std::endl;
#endif
}

void Event::addMiniDoubletsToEventExplicit() {
  auto nMDsCPU_buf = allocBufWrapper<unsigned int>(cms::alpakatools::host(), nLowerModules_, queue_);
  auto mdsOccupancy = miniDoubletsDC_->const_view<MiniDoubletsOccupancySoA>();
  auto nMDs_view = alpaka::createView(devAcc_, mdsOccupancy.nMDs(), nLowerModules_);  // exclude pixel part
  alpaka::memcpy(queue_, nMDsCPU_buf, nMDs_view, nLowerModules_);

  // FIXME: replace by ES host data
  auto module_subdets_buf = allocBufWrapper<short>(cms::alpakatools::host(), nLowerModules_, queue_);
  alpaka::memcpy(queue_, module_subdets_buf, modulesBuffers_.subdets_buf, nLowerModules_);

  auto module_layers_buf = allocBufWrapper<short>(cms::alpakatools::host(), nLowerModules_, queue_);
  alpaka::memcpy(queue_, module_layers_buf, modulesBuffers_.layers_buf, nLowerModules_);

  auto module_hitRanges_buf = allocBufWrapper<int>(cms::alpakatools::host(), nLowerModules_ * 2, queue_);
  alpaka::memcpy(queue_, module_hitRanges_buf, hitsBuffers_->hitRanges_buf, nLowerModules_ * 2u);

  alpaka::wait(queue_);  // wait for inputs before using them

  auto const* nMDsCPU = nMDsCPU_buf.data();
  auto const* module_subdets = module_subdets_buf.data();
  auto const* module_layers = module_layers_buf.data();
  auto const* module_hitRanges = module_hitRanges_buf.data();

  for (unsigned int i = 0; i < nLowerModules_; i++) {
    if (!(nMDsCPU[i] == 0 or module_hitRanges[i * 2] == -1)) {
      if (module_subdets[i] == Barrel) {
        n_minidoublets_by_layer_barrel_[module_layers[i] - 1] += nMDsCPU[i];
      } else {
        n_minidoublets_by_layer_endcap_[module_layers[i] - 1] += nMDsCPU[i];
      }
    }
  }
}

void Event::addSegmentsToEventExplicit() {
  auto nSegmentsCPU_buf = allocBufWrapper<unsigned int>(cms::alpakatools::host(), nLowerModules_, queue_);
  auto nSegments_buf =
      alpaka::createView(devAcc_, segmentsDC_->const_view<SegmentsOccupancySoA>().nSegments(), nLowerModules_);
  alpaka::memcpy(queue_, nSegmentsCPU_buf, nSegments_buf, nLowerModules_);

  // FIXME: replace by ES host data
  auto module_subdets_buf = allocBufWrapper<short>(cms::alpakatools::host(), nLowerModules_, queue_);
  alpaka::memcpy(queue_, module_subdets_buf, modulesBuffers_.subdets_buf, nLowerModules_);

  auto module_layers_buf = allocBufWrapper<short>(cms::alpakatools::host(), nLowerModules_, queue_);
  alpaka::memcpy(queue_, module_layers_buf, modulesBuffers_.layers_buf, nLowerModules_);

  alpaka::wait(queue_);  // wait for inputs before using them

  auto const* nSegmentsCPU = nSegmentsCPU_buf.data();
  auto const* module_subdets = module_subdets_buf.data();
  auto const* module_layers = module_layers_buf.data();

  for (unsigned int i = 0; i < nLowerModules_; i++) {
    if (!(nSegmentsCPU[i] == 0)) {
      if (module_subdets[i] == Barrel) {
        n_segments_by_layer_barrel_[module_layers[i] - 1] += nSegmentsCPU[i];
      } else {
        n_segments_by_layer_endcap_[module_layers[i] - 1] += nSegmentsCPU[i];
      }
    }
  }
}

void Event::addQuintupletsToEventExplicit() {
  auto quintupletsOccupancy = quintupletsDC_->const_view<QuintupletsOccupancySoA>();
  auto nQuintuplets_view = alpaka::createView(devAcc_, quintupletsOccupancy.nQuintuplets(), nLowerModules_);
  auto nQuintupletsCPU_buf = allocBufWrapper<unsigned int>(cms::alpakatools::host(), nLowerModules_, queue_);
  alpaka::memcpy(queue_, nQuintupletsCPU_buf, nQuintuplets_view);

  // FIXME: replace by ES host data
  auto module_subdets_buf = allocBufWrapper<short>(cms::alpakatools::host(), nModules_, queue_);
  alpaka::memcpy(queue_, module_subdets_buf, modulesBuffers_.subdets_buf, nModules_);

  auto module_layers_buf = allocBufWrapper<short>(cms::alpakatools::host(), nLowerModules_, queue_);
  alpaka::memcpy(queue_, module_layers_buf, modulesBuffers_.layers_buf, nLowerModules_);

  auto module_quintupletModuleIndices_buf = allocBufWrapper<int>(cms::alpakatools::host(), nLowerModules_, queue_);
  alpaka::memcpy(queue_, module_quintupletModuleIndices_buf, rangesBuffers_->quintupletModuleIndices_buf);

  alpaka::wait(queue_);  // wait for inputs before using them

  auto const* nQuintupletsCPU = nQuintupletsCPU_buf.data();
  auto const* module_subdets = module_subdets_buf.data();
  auto const* module_layers = module_layers_buf.data();
  auto const* module_quintupletModuleIndices = module_quintupletModuleIndices_buf.data();

  for (uint16_t i = 0; i < nLowerModules_; i++) {
    if (!(nQuintupletsCPU[i] == 0 or module_quintupletModuleIndices[i] == -1)) {
      if (module_subdets[i] == Barrel) {
        n_quintuplets_by_layer_barrel_[module_layers[i] - 1] += nQuintupletsCPU[i];
      } else {
        n_quintuplets_by_layer_endcap_[module_layers[i] - 1] += nQuintupletsCPU[i];
      }
    }
  }
}

void Event::addTripletsToEventExplicit() {
  auto tripletsOccupancy = tripletsDC_->const_view<TripletsOccupancySoA>();
  auto nTriplets_view = alpaka::createView(devAcc_, tripletsOccupancy.nTriplets(), nLowerModules_);
  auto nTripletsCPU_buf = allocBufWrapper<unsigned int>(cms::alpakatools::host(), nLowerModules_, queue_);
  alpaka::memcpy(queue_, nTripletsCPU_buf, nTriplets_view);

  // FIXME: replace by ES host data
  auto module_subdets_buf = allocBufWrapper<short>(cms::alpakatools::host(), nLowerModules_, queue_);
  alpaka::memcpy(queue_, module_subdets_buf, modulesBuffers_.subdets_buf, nLowerModules_);

  auto module_layers_buf = allocBufWrapper<short>(cms::alpakatools::host(), nLowerModules_, queue_);
  alpaka::memcpy(queue_, module_layers_buf, modulesBuffers_.layers_buf, nLowerModules_);

  alpaka::wait(queue_);  // wait for inputs before using them

  auto const* nTripletsCPU = nTripletsCPU_buf.data();
  auto const* module_subdets = module_subdets_buf.data();
  auto const* module_layers = module_layers_buf.data();

  for (uint16_t i = 0; i < nLowerModules_; i++) {
    if (nTripletsCPU[i] != 0) {
      if (module_subdets[i] == Barrel) {
        n_triplets_by_layer_barrel_[module_layers[i] - 1] += nTripletsCPU[i];
      } else {
        n_triplets_by_layer_endcap_[module_layers[i] - 1] += nTripletsCPU[i];
      }
    }
  }
}

unsigned int Event::getNumberOfHits() {
  unsigned int hits = 0;
  for (auto& it : n_hits_by_layer_barrel_) {
    hits += it;
  }
  for (auto& it : n_hits_by_layer_endcap_) {
    hits += it;
  }

  return hits;
}

unsigned int Event::getNumberOfHitsByLayer(unsigned int layer) {
  if (layer == 6)
    return n_hits_by_layer_barrel_[layer];
  else
    return n_hits_by_layer_barrel_[layer] + n_hits_by_layer_endcap_[layer];
}

unsigned int Event::getNumberOfHitsByLayerBarrel(unsigned int layer) { return n_hits_by_layer_barrel_[layer]; }

unsigned int Event::getNumberOfHitsByLayerEndcap(unsigned int layer) { return n_hits_by_layer_endcap_[layer]; }

unsigned int Event::getNumberOfMiniDoublets() {
  unsigned int miniDoublets = 0;
  for (auto& it : n_minidoublets_by_layer_barrel_) {
    miniDoublets += it;
  }
  for (auto& it : n_minidoublets_by_layer_endcap_) {
    miniDoublets += it;
  }

  return miniDoublets;
}

unsigned int Event::getNumberOfMiniDoubletsByLayer(unsigned int layer) {
  if (layer == 6)
    return n_minidoublets_by_layer_barrel_[layer];
  else
    return n_minidoublets_by_layer_barrel_[layer] + n_minidoublets_by_layer_endcap_[layer];
}

unsigned int Event::getNumberOfMiniDoubletsByLayerBarrel(unsigned int layer) {
  return n_minidoublets_by_layer_barrel_[layer];
}

unsigned int Event::getNumberOfMiniDoubletsByLayerEndcap(unsigned int layer) {
  return n_minidoublets_by_layer_endcap_[layer];
}

unsigned int Event::getNumberOfSegments() {
  unsigned int segments = 0;
  for (auto& it : n_segments_by_layer_barrel_) {
    segments += it;
  }
  for (auto& it : n_segments_by_layer_endcap_) {
    segments += it;
  }

  return segments;
}

unsigned int Event::getNumberOfSegmentsByLayer(unsigned int layer) {
  if (layer == 6)
    return n_segments_by_layer_barrel_[layer];
  else
    return n_segments_by_layer_barrel_[layer] + n_segments_by_layer_endcap_[layer];
}

unsigned int Event::getNumberOfSegmentsByLayerBarrel(unsigned int layer) { return n_segments_by_layer_barrel_[layer]; }

unsigned int Event::getNumberOfSegmentsByLayerEndcap(unsigned int layer) { return n_segments_by_layer_endcap_[layer]; }

unsigned int Event::getNumberOfTriplets() {
  unsigned int triplets = 0;
  for (auto& it : n_triplets_by_layer_barrel_) {
    triplets += it;
  }
  for (auto& it : n_triplets_by_layer_endcap_) {
    triplets += it;
  }

  return triplets;
}

unsigned int Event::getNumberOfTripletsByLayer(unsigned int layer) {
  if (layer == 6)
    return n_triplets_by_layer_barrel_[layer];
  else
    return n_triplets_by_layer_barrel_[layer] + n_triplets_by_layer_endcap_[layer];
}

unsigned int Event::getNumberOfTripletsByLayerBarrel(unsigned int layer) { return n_triplets_by_layer_barrel_[layer]; }

unsigned int Event::getNumberOfTripletsByLayerEndcap(unsigned int layer) { return n_triplets_by_layer_endcap_[layer]; }

int Event::getNumberOfPixelTriplets() {
  auto nPixelTriplets_buf_h = cms::alpakatools::make_host_buffer<unsigned int[]>(queue_, 1u);

  alpaka::memcpy(queue_, nPixelTriplets_buf_h, alpaka::createView(devAcc_, &(*pixelTripletsDC_)->nPixelTriplets(), 1u));
  alpaka::wait(queue_);

  return *nPixelTriplets_buf_h.data();
}

int Event::getNumberOfPixelQuintuplets() {
  auto nPixelQuintuplets_buf_h = cms::alpakatools::make_host_buffer<unsigned int[]>(queue_, 1u);

  alpaka::memcpy(
      queue_, nPixelQuintuplets_buf_h, alpaka::createView(devAcc_, &(*pixelQuintupletsDC_)->nPixelQuintuplets(), 1u));
  alpaka::wait(queue_);

  return *nPixelQuintuplets_buf_h.data();
}

unsigned int Event::getNumberOfQuintuplets() {
  unsigned int quintuplets = 0;
  for (auto& it : n_quintuplets_by_layer_barrel_) {
    quintuplets += it;
  }
  for (auto& it : n_quintuplets_by_layer_endcap_) {
    quintuplets += it;
  }

  return quintuplets;
}

unsigned int Event::getNumberOfQuintupletsByLayer(unsigned int layer) {
  if (layer == 6)
    return n_quintuplets_by_layer_barrel_[layer];
  else
    return n_quintuplets_by_layer_barrel_[layer] + n_quintuplets_by_layer_endcap_[layer];
}

unsigned int Event::getNumberOfQuintupletsByLayerBarrel(unsigned int layer) {
  return n_quintuplets_by_layer_barrel_[layer];
}

unsigned int Event::getNumberOfQuintupletsByLayerEndcap(unsigned int layer) {
  return n_quintuplets_by_layer_endcap_[layer];
}

int Event::getNumberOfTrackCandidates() {
  auto nTrackCandidates_buf_h = cms::alpakatools::make_host_buffer<unsigned int[]>(queue_, 1u);

  alpaka::memcpy(
      queue_, nTrackCandidates_buf_h, alpaka::createView(devAcc_, &(*trackCandidatesDC_)->nTrackCandidates(), 1u));
  alpaka::wait(queue_);

  return *nTrackCandidates_buf_h.data();
}

int Event::getNumberOfPT5TrackCandidates() {
  auto nTrackCandidatesPT5_buf_h = cms::alpakatools::make_host_buffer<unsigned int[]>(queue_, 1u);

  alpaka::memcpy(queue_,
                 nTrackCandidatesPT5_buf_h,
                 alpaka::createView(devAcc_, &(*trackCandidatesDC_)->nTrackCandidatespT5(), 1u));
  alpaka::wait(queue_);

  return *nTrackCandidatesPT5_buf_h.data();
}

int Event::getNumberOfPT3TrackCandidates() {
  auto nTrackCandidatesPT3_buf_h = cms::alpakatools::make_host_buffer<unsigned int[]>(queue_, 1u);

  alpaka::memcpy(queue_,
                 nTrackCandidatesPT3_buf_h,
                 alpaka::createView(devAcc_, &(*trackCandidatesDC_)->nTrackCandidatespT3(), 1u));
  alpaka::wait(queue_);

  return *nTrackCandidatesPT3_buf_h.data();
}

int Event::getNumberOfPLSTrackCandidates() {
  auto nTrackCandidatesPLS_buf_h = cms::alpakatools::make_host_buffer<unsigned int[]>(queue_, 1u);

  alpaka::memcpy(queue_,
                 nTrackCandidatesPLS_buf_h,
                 alpaka::createView(devAcc_, &(*trackCandidatesDC_)->nTrackCandidatespLS(), 1u));
  alpaka::wait(queue_);

  return *nTrackCandidatesPLS_buf_h.data();
}

int Event::getNumberOfPixelTrackCandidates() {
  auto nTrackCandidates_buf_h = cms::alpakatools::make_host_buffer<unsigned int[]>(queue_, 1u);
  auto nTrackCandidatesT5_buf_h = cms::alpakatools::make_host_buffer<unsigned int[]>(queue_, 1u);

  alpaka::memcpy(
      queue_, nTrackCandidates_buf_h, alpaka::createView(devAcc_, &(*trackCandidatesDC_)->nTrackCandidates(), 1u));
  alpaka::memcpy(
      queue_, nTrackCandidatesT5_buf_h, alpaka::createView(devAcc_, &(*trackCandidatesDC_)->nTrackCandidatesT5(), 1u));
  alpaka::wait(queue_);

  return (*nTrackCandidates_buf_h.data()) - (*nTrackCandidatesT5_buf_h.data());
}

int Event::getNumberOfT5TrackCandidates() {
  auto nTrackCandidatesT5_buf_h = cms::alpakatools::make_host_buffer<unsigned int[]>(queue_, 1u);

  alpaka::memcpy(
      queue_, nTrackCandidatesT5_buf_h, alpaka::createView(devAcc_, &(*trackCandidatesDC_)->nTrackCandidatesT5(), 1u));
  alpaka::wait(queue_);

  return *nTrackCandidatesT5_buf_h.data();
}

HitsBuffer<alpaka_common::DevHost>& Event::getHits(bool sync)  //std::shared_ptr should take care of garbage collection
{
  if (!hitsInCPU_) {
    auto nHits_buf_h = cms::alpakatools::make_host_buffer<unsigned int[]>(queue_, 1u);
    alpaka::memcpy(queue_, nHits_buf_h, hitsBuffers_->nHits_buf);
    alpaka::wait(queue_);  // wait for the value before using

    auto const nHits = *nHits_buf_h.data();
    hitsInCPU_.emplace(nModules_, nHits, cms::alpakatools::host(), queue_);
    hitsInCPU_->setData(*hitsInCPU_);

    alpaka::memcpy(queue_, hitsInCPU_->nHits_buf, hitsBuffers_->nHits_buf);
    alpaka::memcpy(queue_, hitsInCPU_->idxs_buf, hitsBuffers_->idxs_buf, nHits);
    alpaka::memcpy(queue_, hitsInCPU_->detid_buf, hitsBuffers_->detid_buf, nHits);
    alpaka::memcpy(queue_, hitsInCPU_->xs_buf, hitsBuffers_->xs_buf, nHits);
    alpaka::memcpy(queue_, hitsInCPU_->ys_buf, hitsBuffers_->ys_buf, nHits);
    alpaka::memcpy(queue_, hitsInCPU_->zs_buf, hitsBuffers_->zs_buf, nHits);
    alpaka::memcpy(queue_, hitsInCPU_->moduleIndices_buf, hitsBuffers_->moduleIndices_buf, nHits);
    if (sync)
      alpaka::wait(queue_);  // host consumers expect filled data
  }
  return hitsInCPU_.value();
}

HitsBuffer<alpaka_common::DevHost>& Event::getHitsInCMSSW(bool sync) {
  if (!hitsInCPU_) {
    auto nHits_buf_h = cms::alpakatools::make_host_buffer<unsigned int[]>(queue_, 1u);
    alpaka::memcpy(queue_, nHits_buf_h, hitsBuffers_->nHits_buf);
    alpaka::wait(queue_);  // wait for the value before using

    auto const nHits = *nHits_buf_h.data();
    hitsInCPU_.emplace(nModules_, nHits, cms::alpakatools::host(), queue_);
    hitsInCPU_->setData(*hitsInCPU_);

    alpaka::memcpy(queue_, hitsInCPU_->nHits_buf, hitsBuffers_->nHits_buf);
    alpaka::memcpy(queue_, hitsInCPU_->idxs_buf, hitsBuffers_->idxs_buf, nHits);
    if (sync)
      alpaka::wait(queue_);  // host consumers expect filled data
  }
  return hitsInCPU_.value();
}

ObjectRangesBuffer<alpaka_common::DevHost>& Event::getRanges(bool sync) {
  if (!rangesInCPU_) {
    rangesInCPU_.emplace(nModules_, nLowerModules_, cms::alpakatools::host(), queue_);
    rangesInCPU_->setData(*rangesInCPU_);

    alpaka::memcpy(queue_, rangesInCPU_->hitRanges_buf, rangesBuffers_->hitRanges_buf);
    alpaka::memcpy(queue_, rangesInCPU_->quintupletModuleIndices_buf, rangesBuffers_->quintupletModuleIndices_buf);
    alpaka::memcpy(queue_, rangesInCPU_->miniDoubletModuleIndices_buf, rangesBuffers_->miniDoubletModuleIndices_buf);
    alpaka::memcpy(queue_, rangesInCPU_->segmentModuleIndices_buf, rangesBuffers_->segmentModuleIndices_buf);
    alpaka::memcpy(queue_, rangesInCPU_->tripletModuleIndices_buf, rangesBuffers_->tripletModuleIndices_buf);
    if (sync)
      alpaka::wait(queue_);  // wait to get completed host data
  }
  return rangesInCPU_.value();
}

template <typename TSoA, typename TDev>
typename TSoA::ConstView Event::getMiniDoublets(bool sync) {
  if constexpr (std::is_same_v<TDev, DevHost>) {
    return miniDoubletsDC_->const_view<TSoA>();
  } else {
    if (!miniDoubletsHC_) {
      miniDoubletsHC_.emplace(
          cms::alpakatools::CopyToHost<
              PortableMultiCollection<TDev, MiniDoubletsSoA, MiniDoubletsOccupancySoA>>::copyAsync(queue_,
                                                                                                   *miniDoubletsDC_));
      if (sync)
        alpaka::wait(queue_);  // host consumers expect filled data
    }
    return miniDoubletsHC_->const_view<TSoA>();
  }
}
template MiniDoubletsConst Event::getMiniDoublets<MiniDoubletsSoA>(bool);
template MiniDoubletsOccupancyConst Event::getMiniDoublets<MiniDoubletsOccupancySoA>(bool);

template <typename TSoA, typename TDev>
typename TSoA::ConstView Event::getSegments(bool sync) {
  if constexpr (std::is_same_v<TDev, DevHost>) {
    return segmentsDC_->const_view<TSoA>();
  } else {
    if (!segmentsHC_) {
      segmentsHC_.emplace(
          cms::alpakatools::
              CopyToHost<PortableMultiCollection<TDev, SegmentsSoA, SegmentsOccupancySoA, SegmentsPixelSoA>>::copyAsync(
                  queue_, *segmentsDC_));
      if (sync)
        alpaka::wait(queue_);  // host consumers expect filled data
    }
    return segmentsHC_->const_view<TSoA>();
  }
}
template SegmentsConst Event::getSegments<SegmentsSoA>(bool);
template SegmentsOccupancyConst Event::getSegments<SegmentsOccupancySoA>(bool);
template SegmentsPixelConst Event::getSegments<SegmentsPixelSoA>(bool);

template <typename TSoA, typename TDev>
typename TSoA::ConstView Event::getTriplets(bool sync) {
  if constexpr (std::is_same_v<TDev, DevHost>) {
    return tripletsDC_->const_view<TSoA>();
  } else {
    if (!tripletsHC_) {
      tripletsHC_.emplace(
          cms::alpakatools::CopyToHost<PortableMultiCollection<TDev, TripletsSoA, TripletsOccupancySoA>>::copyAsync(
              queue_, *tripletsDC_));

      if (sync)
        alpaka::wait(queue_);  // host consumers expect filled data
    }
  }
  return tripletsHC_->const_view<TSoA>();
}
template TripletsConst Event::getTriplets<TripletsSoA>(bool);
template TripletsOccupancyConst Event::getTriplets<TripletsOccupancySoA>(bool);

template <typename TSoA, typename TDev>
typename TSoA::ConstView Event::getQuintuplets(bool sync) {
  if constexpr (std::is_same_v<TDev, DevHost>) {
    return quintupletsDC_->const_view<TSoA>();
  } else {
    if (!quintupletsHC_) {
      quintupletsHC_.emplace(
          cms::alpakatools::CopyToHost<PortableMultiCollection<TDev, QuintupletsSoA, QuintupletsOccupancySoA>>::copyAsync(
              queue_, *quintupletsDC_));

      if (sync)
        alpaka::wait(queue_);  // host consumers expect filled data
    }
  }
  return quintupletsHC_->const_view<TSoA>();
}
template QuintupletsConst Event::getQuintuplets<QuintupletsSoA>(bool);
template QuintupletsOccupancyConst Event::getQuintuplets<QuintupletsOccupancySoA>(bool);

template <typename TDev>
PixelTripletsConst Event::getPixelTriplets(bool sync) {
  if constexpr (std::is_same_v<TDev, DevHost>) {
    return pixelTripletsDC_->const_view();
  } else {
    if (!pixelTripletsHC_) {
      pixelTripletsHC_.emplace(cms::alpakatools::CopyToHost<::PortableCollection<PixelTripletsSoA, TDev>>::copyAsync(
          queue_, *pixelTripletsDC_));

      if (sync)
        alpaka::wait(queue_);  // host consumers expect filled data
    }
  }
  return pixelTripletsHC_->const_view();
}
template PixelTripletsConst Event::getPixelTriplets<>(bool);

template <typename TDev>
PixelQuintupletsConst Event::getPixelQuintuplets(bool sync) {
  if constexpr (std::is_same_v<TDev, DevHost>) {
    return pixelQuintupletsDC_->const_view();
  } else {
    if (!pixelQuintupletsHC_) {
      pixelQuintupletsHC_.emplace(
          cms::alpakatools::CopyToHost<::PortableCollection<PixelQuintupletsSoA, TDev>>::copyAsync(
              queue_, *pixelQuintupletsDC_));

      if (sync)
        alpaka::wait(queue_);  // host consumers expect filled data
    }
  }
  return pixelQuintupletsHC_->const_view();
}
template PixelQuintupletsConst Event::getPixelQuintuplets<>(bool);

const TrackCandidatesConst& Event::getTrackCandidatesWithSelection(bool inCMSSW, bool sync) {
  if (!trackCandidatesHC_) {
    // Get nTrackCanHost parameter to initialize host based instance
    auto nTrackCanHost_buf_h = cms::alpakatools::make_host_buffer<unsigned int[]>(queue_, 1u);
    alpaka::memcpy(
        queue_, nTrackCanHost_buf_h, alpaka::createView(devAcc_, &(*trackCandidatesDC_)->nTrackCandidates(), 1u));
    alpaka::wait(queue_);  // wait here before we get nTrackCanHost and trackCandidatesInCPU becomes usable

    auto const nTrackCanHost = *nTrackCanHost_buf_h.data();
    trackCandidatesHC_.emplace(nTrackCanHost, queue_);

    (*trackCandidatesHC_)->nTrackCandidates() = nTrackCanHost;
    alpaka::memcpy(
        queue_,
        alpaka::createView(
            cms::alpakatools::host(), (*trackCandidatesHC_)->hitIndices()->data(), Params_pT5::kHits * nTrackCanHost),
        alpaka::createView(devAcc_, (*trackCandidatesDC_)->hitIndices()->data(), Params_pT5::kHits * nTrackCanHost));
    alpaka::memcpy(queue_,
                   alpaka::createView(cms::alpakatools::host(), (*trackCandidatesHC_)->pixelSeedIndex(), nTrackCanHost),
                   alpaka::createView(devAcc_, (*trackCandidatesDC_)->pixelSeedIndex(), nTrackCanHost));
    if (not inCMSSW) {
      alpaka::memcpy(queue_,
                     alpaka::createView(cms::alpakatools::host(),
                                        (*trackCandidatesHC_)->logicalLayers()->data(),
                                        Params_pT5::kLayers * nTrackCanHost),
                     alpaka::createView(
                         devAcc_, (*trackCandidatesDC_)->logicalLayers()->data(), Params_pT5::kLayers * nTrackCanHost));
      alpaka::memcpy(
          queue_,
          alpaka::createView(cms::alpakatools::host(), (*trackCandidatesHC_)->directObjectIndices(), nTrackCanHost),
          alpaka::createView(devAcc_, (*trackCandidatesDC_)->directObjectIndices(), nTrackCanHost));
      alpaka::memcpy(queue_,
                     alpaka::createView(
                         cms::alpakatools::host(), (*trackCandidatesHC_)->objectIndices()->data(), 2 * nTrackCanHost),
                     alpaka::createView(devAcc_, (*trackCandidatesDC_)->objectIndices()->data(), 2 * nTrackCanHost));
    }
    alpaka::memcpy(
        queue_,
        alpaka::createView(cms::alpakatools::host(), (*trackCandidatesHC_)->trackCandidateType(), nTrackCanHost),
        alpaka::createView(devAcc_, (*trackCandidatesDC_)->trackCandidateType(), nTrackCanHost));
    if (sync)
      alpaka::wait(queue_);  // host consumers expect filled data
  }
  return trackCandidatesHC_.value().const_view();
}

ModulesBuffer<alpaka_common::DevHost>& Event::getModules(bool isFull, bool sync) {
  if (!modulesInCPU_) {
    // The last input here is just a small placeholder for the allocation.
    modulesInCPU_.emplace(cms::alpakatools::host(), nModules_, nPixels_);

    modulesInCPU_->copyFromSrc(queue_, modulesBuffers_, isFull);
    if (sync)
      alpaka::wait(queue_);  // host consumers expect filled data
  }
  return modulesInCPU_.value();
}
