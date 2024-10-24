#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"

#include "Event.h"

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
  hitsDC_.reset();
  miniDoubletsDC_.reset();
  rangesDC_.reset();
  segmentsDC_.reset();
  tripletsInGPU_.reset();
  tripletsBuffers_.reset();
  quintupletsInGPU_.reset();
  quintupletsBuffers_.reset();
  trackCandidatesDC_.reset();
  pixelTripletsInGPU_.reset();
  pixelTripletsBuffers_.reset();
  pixelQuintupletsInGPU_.reset();
  pixelQuintupletsBuffers_.reset();

  hitsHC_.reset();
  rangesHC_.reset();
  miniDoubletsHC_.reset();
  segmentsHC_.reset();
  tripletsInCPU_.reset();
  quintupletsInCPU_.reset();
  pixelTripletsInCPU_.reset();
  pixelQuintupletsInCPU_.reset();
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
  if (!hitsDC_) {
    std::array<int, 2> const hits_sizes{{static_cast<int>(nHits), static_cast<int>(nModules_)}};
    hitsDC_.emplace(hits_sizes, queue_);

    auto hitsOccupancy = hitsDC_->view<HitsOccupancySoA>();
    auto hitRanges_view = alpaka::createView(devAcc_, hitsOccupancy.hitRanges(), hitsOccupancy.metadata().size());
    auto hitRangesLower_view =
        alpaka::createView(devAcc_, hitsOccupancy.hitRangesLower(), hitsOccupancy.metadata().size());
    auto hitRangesUpper_view =
        alpaka::createView(devAcc_, hitsOccupancy.hitRangesUpper(), hitsOccupancy.metadata().size());
    auto hitRangesnLower_view =
        alpaka::createView(devAcc_, hitsOccupancy.hitRangesnLower(), hitsOccupancy.metadata().size());
    auto hitRangesnUpper_view =
        alpaka::createView(devAcc_, hitsOccupancy.hitRangesnUpper(), hitsOccupancy.metadata().size());
    alpaka::memset(queue_, hitRanges_view, 0xff);
    alpaka::memset(queue_, hitRangesLower_view, 0xff);
    alpaka::memset(queue_, hitRangesUpper_view, 0xff);
    alpaka::memset(queue_, hitRangesnLower_view, 0xff);
    alpaka::memset(queue_, hitRangesnUpper_view, 0xff);
  }

  if (!rangesDC_) {
    std::array<int, 2> const ranges_sizes{{static_cast<int>(nModules_), static_cast<int>(nLowerModules_ + 1)}};
    rangesDC_.emplace(ranges_sizes, queue_);
    auto buf = rangesDC_->buffer();
    alpaka::memset(queue_, buf, 0xff);
  }

  // Copy the host arrays to the GPU.
  auto hits = hitsDC_->view<HitsSoA>();
  auto xs_d = alpaka::createView(devAcc_, hits.xs(), (Idx)hits.metadata().size());
  auto ys_d = alpaka::createView(devAcc_, hits.ys(), (Idx)hits.metadata().size());
  auto zs_d = alpaka::createView(devAcc_, hits.zs(), (Idx)hits.metadata().size());
  auto detId_d = alpaka::createView(devAcc_, hits.detid(), (Idx)hits.metadata().size());
  auto idxs_d = alpaka::createView(devAcc_, hits.idxs(), (Idx)hits.metadata().size());
  alpaka::memcpy(queue_, xs_d, x, (Idx)nHits);
  alpaka::memcpy(queue_, ys_d, y, (Idx)nHits);
  alpaka::memcpy(queue_, zs_d, z, (Idx)nHits);
  alpaka::memcpy(queue_, detId_d, detId, (Idx)nHits);
  alpaka::memcpy(queue_, idxs_d, idxInNtuple, (Idx)nHits);
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
                      endcapGeometry_.const_view(),
                      *modulesBuffers_.data(),
                      hitsDC_->view<HitsSoA>(),
                      hitsDC_->view<HitsOccupancySoA>(),
                      nHits);

  Vec3D const threadsPerBlock2{1, 1, 256};
  Vec3D const blocksPerGrid2{1, 1, max_blocks};
  WorkDiv3D const module_ranges_workdiv = createWorkDiv(blocksPerGrid2, threadsPerBlock2, elementsPerThread);

  alpaka::exec<Acc3D>(queue_,
                      module_ranges_workdiv,
                      ModuleRangesKernel{},
                      *modulesBuffers_.data(),
                      hitsDC_->view<HitsOccupancySoA>(),
                      nLowerModules_);
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
    // Create a view for the element nLowerModules_ inside rangesOccupancy->miniDoubletModuleOccupancy
    auto rangesOccupancy = rangesDC_->view<ObjectOccupancySoA>();
    auto miniDoubletModuleOccupancy_view = alpaka::createView(
        devAcc_, rangesOccupancy.miniDoubletModuleOccupancy(), (Idx)rangesOccupancy.metadata().size());
    auto dst_view_miniDoubletModuleOccupancy =
        alpaka::createSubView(miniDoubletModuleOccupancy_view, (Idx)1u, (Idx)nLowerModules_);

    // Create a host buffer for a value to be passed to the device
    auto pixelMaxMDs_buf_h = cms::alpakatools::make_host_buffer<int[]>(queue_, (Idx)1u);
    *pixelMaxMDs_buf_h.data() = n_max_pixel_md_per_modules;

    alpaka::memcpy(queue_, dst_view_miniDoubletModuleOccupancy, pixelMaxMDs_buf_h);

    WorkDiv1D const createMDArrayRangesGPU_workDiv = createWorkDiv<Vec1D>({1}, {1024}, {1});

    alpaka::exec<Acc1D>(queue_,
                        createMDArrayRangesGPU_workDiv,
                        CreateMDArrayRangesGPU{},
                        *modulesBuffers_.data(),
                        rangesDC_->view<ObjectOccupancySoA>());

    auto nTotalMDs_buf_h = cms::alpakatools::make_host_buffer<unsigned int[]>(queue_, (Idx)1u);
    auto nTotalMDs_buf_d = alpaka::createView(devAcc_, &rangesOccupancy.nTotalMDs(), (Idx)1u);
    alpaka::memcpy(queue_, nTotalMDs_buf_h, nTotalMDs_buf_d);
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
                        rangesDC_->view<ObjectOccupancySoA>(),
                        miniDoubletsDC_->const_view<MiniDoubletsSoA>());

    auto rangesOccupancy = rangesDC_->view<ObjectOccupancySoA>();
    auto nTotalSegments_view_h = alpaka::createView(cms::alpakatools::host(), &nTotalSegments_, (Idx)1u);
    auto nTotalSegments_view_d = alpaka::createView(devAcc_, &rangesOccupancy.nTotalSegs(), (Idx)1u);
    alpaka::memcpy(queue_, nTotalSegments_view_h, nTotalSegments_view_d);
    alpaka::wait(queue_);  // wait to get the value before manipulation

    nTotalSegments_ += n_max_pixel_segments_per_module;

    std::array<int, 3> const segments_sizes{{static_cast<int>(nTotalSegments_),
                                             static_cast<int>(nLowerModules_ + 1),
                                             static_cast<int>(n_max_pixel_segments_per_module)}};
    segmentsDC_.emplace(segments_sizes, queue_);

    auto segmentsOccupancy = segmentsDC_->view<SegmentsOccupancySoA>();
    auto nSegments_view =
        alpaka::createView(devAcc_, segmentsOccupancy.nSegments(), segmentsOccupancy.metadata().size());
    auto totOccupancySegments_view =
        alpaka::createView(devAcc_, segmentsOccupancy.totOccupancySegments(), segmentsOccupancy.metadata().size());
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

  auto segmentsOccupancy = segmentsDC_->view<SegmentsOccupancySoA>();
  auto nSegments_view =
      alpaka::createView(devAcc_, segmentsOccupancy.nSegments(), (Idx)segmentsOccupancy.metadata().size());
  auto dst_view_segments = alpaka::createSubView(nSegments_view, (Idx)1u, (Idx)pixelModuleIndex);
  alpaka::memcpy(queue_, dst_view_segments, src_view_size);

  auto totOccupancySegments_view =
      alpaka::createView(devAcc_, segmentsOccupancy.totOccupancySegments(), (Idx)segmentsOccupancy.metadata().size());
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
                      rangesDC_->const_view<ObjectOccupancySoA>(),
                      hitsDC_->view<HitsSoA>(),
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
  // Create a view for the element nLowerModules_ inside rangesOccupancy->miniDoubletModuleOccupancy
  auto rangesOccupancy = rangesDC_->view<ObjectOccupancySoA>();
  auto miniDoubletModuleOccupancy_view =
      alpaka::createView(devAcc_, rangesOccupancy.miniDoubletModuleOccupancy(), (Idx)rangesOccupancy.metadata().size());
  auto dst_view_miniDoubletModuleOccupancy =
      alpaka::createSubView(miniDoubletModuleOccupancy_view, (Idx)1u, (Idx)nLowerModules_);

  // Create a host buffer for a value to be passed to the device
  auto pixelMaxMDs_buf_h = cms::alpakatools::make_host_buffer<int[]>(queue_, (Idx)1u);
  *pixelMaxMDs_buf_h.data() = n_max_pixel_md_per_modules;

  alpaka::memcpy(queue_, dst_view_miniDoubletModuleOccupancy, pixelMaxMDs_buf_h);

  WorkDiv1D const createMDArrayRangesGPU_workDiv = createWorkDiv<Vec1D>({1}, {1024}, {1});

  alpaka::exec<Acc1D>(queue_,
                      createMDArrayRangesGPU_workDiv,
                      CreateMDArrayRangesGPU{},
                      *modulesBuffers_.data(),
                      rangesDC_->view<ObjectOccupancySoA>());

  auto nTotalMDs_buf_h = cms::alpakatools::make_host_buffer<unsigned int[]>(queue_, (Idx)1u);
  auto nTotalMDs_buf_d = alpaka::createView(devAcc_, &rangesOccupancy.nTotalMDs(), (Idx)1u);
  alpaka::memcpy(queue_, nTotalMDs_buf_h, nTotalMDs_buf_d);
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
                      hitsDC_->const_view<HitsSoA>(),
                      hitsDC_->const_view<HitsOccupancySoA>(),
                      miniDoubletsDC_->view<MiniDoubletsSoA>(),
                      miniDoubletsDC_->view<MiniDoubletsOccupancySoA>(),
                      rangesDC_->const_view<ObjectOccupancySoA>());

  WorkDiv1D const addMiniDoubletRangesToEventExplicit_workDiv = createWorkDiv<Vec1D>({1}, {1024}, {1});

  alpaka::exec<Acc1D>(queue_,
                      addMiniDoubletRangesToEventExplicit_workDiv,
                      AddMiniDoubletRangesToEventExplicit{},
                      *modulesBuffers_.data(),
                      miniDoubletsDC_->view<MiniDoubletsOccupancySoA>(),
                      rangesDC_->view<ObjectRangesSoA>(),
                      rangesDC_->const_view<ObjectOccupancySoA>(),
                      hitsDC_->const_view<HitsOccupancySoA>());

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

    auto segmentsOccupancy = segmentsDC_->view<SegmentsOccupancySoA>();
    auto nSegments_view =
        alpaka::createView(devAcc_, segmentsOccupancy.nSegments(), segmentsOccupancy.metadata().size());
    auto totOccupancySegments_view =
        alpaka::createView(devAcc_, segmentsOccupancy.totOccupancySegments(), segmentsOccupancy.metadata().size());
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
                      rangesDC_->const_view<ObjectRangesSoA>(),
                      rangesDC_->const_view<ObjectOccupancySoA>());

  WorkDiv1D const addSegmentRangesToEventExplicit_workDiv = createWorkDiv<Vec1D>({1}, {1024}, {1});

  alpaka::exec<Acc1D>(queue_,
                      addSegmentRangesToEventExplicit_workDiv,
                      AddSegmentRangesToEventExplicit{},
                      *modulesBuffers_.data(),
                      segmentsDC_->view<SegmentsOccupancySoA>(),
                      rangesDC_->view<ObjectRangesSoA>(),
                      rangesDC_->const_view<ObjectOccupancySoA>());

  if (addObjects_) {
    addSegmentsToEventExplicit();
  }
}

void Event::createTriplets() {
  if (!tripletsInGPU_) {
    WorkDiv1D const createTripletArrayRanges_workDiv = createWorkDiv<Vec1D>({1}, {1024}, {1});

    alpaka::exec<Acc1D>(queue_,
                        createTripletArrayRanges_workDiv,
                        CreateTripletArrayRanges{},
                        *modulesBuffers_.data(),
                        rangesDC_->view<ObjectOccupancySoA>(),
                        segmentsDC_->const_view<SegmentsOccupancySoA>());

    // TODO: Why are we pulling this back down only to put it back on the device in a new struct?
    auto rangesOccupancy = rangesDC_->view<ObjectOccupancySoA>();
    auto maxTriplets_buf_h = cms::alpakatools::make_host_buffer<unsigned int[]>(queue_, (Idx)1u);
    auto maxTriplets_buf_d = alpaka::createView(devAcc_, &rangesOccupancy.nTotalTrips(), (Idx)1u);
    alpaka::memcpy(queue_, maxTriplets_buf_h, maxTriplets_buf_d);
    alpaka::wait(queue_);  // wait to get the value before using it

    tripletsInGPU_.emplace();
    tripletsBuffers_.emplace(*maxTriplets_buf_h.data(), nLowerModules_, devAcc_, queue_);
    tripletsInGPU_->setData(*tripletsBuffers_);

    alpaka::memcpy(queue_, tripletsBuffers_->nMemoryLocations_buf, maxTriplets_buf_h);
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
  WorkDiv3D const createTripletsInGPUv2_workDiv =
      createWorkDiv(blocksPerGridCreateTrip, threadsPerBlockCreateTrip, elementsPerThread);

  alpaka::exec<Acc3D>(queue_,
                      createTripletsInGPUv2_workDiv,
                      CreateTripletsInGPUv2{},
                      *modulesBuffers_.data(),
                      miniDoubletsDC_->const_view<MiniDoubletsSoA>(),
                      segmentsDC_->const_view<SegmentsSoA>(),
                      segmentsDC_->const_view<SegmentsOccupancySoA>(),
                      *tripletsInGPU_,
                      rangesDC_->const_view<ObjectRangesSoA>(),
                      rangesDC_->const_view<ObjectOccupancySoA>(),
                      index_gpu_buf.data(),
                      nonZeroModules);

  WorkDiv1D const addTripletRangesToEventExplicit_workDiv = createWorkDiv<Vec1D>({1}, {1024}, {1});

  alpaka::exec<Acc1D>(queue_,
                      addTripletRangesToEventExplicit_workDiv,
                      AddTripletRangesToEventExplicit{},
                      *modulesBuffers_.data(),
                      *tripletsInGPU_,
                      rangesDC_->view<ObjectRangesSoA>(),
                      rangesDC_->const_view<ObjectOccupancySoA>());

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
                      rangesDC_->const_view<ObjectOccupancySoA>(),
                      *pixelTripletsInGPU_,
                      segmentsDC_->const_view<SegmentsPixelSoA>(),
                      *pixelQuintupletsInGPU_);

  WorkDiv1D const addpT3asTrackCandidatesInGPU_workDiv = createWorkDiv<Vec1D>({1}, {512}, {1});

  alpaka::exec<Acc1D>(queue_,
                      addpT3asTrackCandidatesInGPU_workDiv,
                      AddpT3asTrackCandidatesInGPU{},
                      nLowerModules_,
                      *pixelTripletsInGPU_,
                      trackCandidatesDC_->view(),
                      segmentsDC_->const_view<SegmentsPixelSoA>(),
                      rangesDC_->const_view<ObjectOccupancySoA>());

  // Pull nEligibleT5Modules from the device.
  auto rangesOccupancy = rangesDC_->view<ObjectOccupancySoA>();
  auto nEligibleModules_buf_h = cms::alpakatools::make_host_buffer<uint16_t[]>(queue_, 1u);
  auto nEligibleModules_buf_d = alpaka::createView(devAcc_, &rangesOccupancy.nEligibleT5Modules(), (Idx)1u);
  alpaka::memcpy(queue_, nEligibleModules_buf_h, nEligibleModules_buf_d);
  alpaka::wait(queue_);  // wait to get the value before using
  auto const nEligibleModules = *nEligibleModules_buf_h.data();

  Vec3D const threadsPerBlockRemoveDupQuints{1, 16, 32};
  Vec3D const blocksPerGridRemoveDupQuints{1, std::max(nEligibleModules / 16, 1), std::max(nEligibleModules / 32, 1)};
  WorkDiv3D const removeDupQuintupletsInGPUBeforeTC_workDiv =
      createWorkDiv(blocksPerGridRemoveDupQuints, threadsPerBlockRemoveDupQuints, elementsPerThread);

  alpaka::exec<Acc3D>(queue_,
                      removeDupQuintupletsInGPUBeforeTC_workDiv,
                      RemoveDupQuintupletsInGPUBeforeTC{},
                      *quintupletsInGPU_,
                      rangesDC_->const_view<ObjectOccupancySoA>());

  Vec3D const threadsPerBlock_crossCleanT5{32, 1, 32};
  Vec3D const blocksPerGrid_crossCleanT5{(13296 / 32) + 1, 1, max_blocks};
  WorkDiv3D const crossCleanT5_workDiv =
      createWorkDiv(blocksPerGrid_crossCleanT5, threadsPerBlock_crossCleanT5, elementsPerThread);

  alpaka::exec<Acc3D>(queue_,
                      crossCleanT5_workDiv,
                      CrossCleanT5{},
                      *modulesBuffers_.data(),
                      *quintupletsInGPU_,
                      *pixelQuintupletsInGPU_,
                      *pixelTripletsInGPU_,
                      rangesDC_->const_view<ObjectOccupancySoA>());

  Vec3D const threadsPerBlock_addT5asTrackCandidateInGPU{1, 8, 128};
  Vec3D const blocksPerGrid_addT5asTrackCandidateInGPU{1, 8, 10};
  WorkDiv3D const addT5asTrackCandidateInGPU_workDiv = createWorkDiv(
      blocksPerGrid_addT5asTrackCandidateInGPU, threadsPerBlock_addT5asTrackCandidateInGPU, elementsPerThread);

  alpaka::exec<Acc3D>(queue_,
                      addT5asTrackCandidateInGPU_workDiv,
                      AddT5asTrackCandidateInGPU{},
                      nLowerModules_,
                      *quintupletsInGPU_,
                      trackCandidatesDC_->view(),
                      rangesDC_->const_view<ObjectOccupancySoA>());

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
                      rangesDC_->const_view<ObjectOccupancySoA>(),
                      *pixelTripletsInGPU_,
                      trackCandidatesDC_->view(),
                      segmentsDC_->const_view<SegmentsSoA>(),
                      segmentsDC_->const_view<SegmentsOccupancySoA>(),
                      segmentsDC_->view<SegmentsPixelSoA>(),
                      miniDoubletsDC_->const_view<MiniDoubletsSoA>(),
                      hitsDC_->const_view<HitsSoA>(),
                      *quintupletsInGPU_);

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
  if (!pixelTripletsInGPU_) {
    pixelTripletsInGPU_.emplace();
    pixelTripletsBuffers_.emplace(n_max_pixel_triplets, devAcc_, queue_);
    pixelTripletsInGPU_->setData(*pixelTripletsBuffers_);
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
  WorkDiv3D const createPixelTripletsInGPUFromMapv2_workDiv =
      createWorkDiv(blocksPerGrid, threadsPerBlock, elementsPerThread);

  alpaka::exec<Acc3D>(queue_,
                      createPixelTripletsInGPUFromMapv2_workDiv,
                      CreatePixelTripletsInGPUFromMapv2{},
                      *modulesBuffers_.data(),
                      rangesDC_->const_view<ObjectOccupancySoA>(),
                      miniDoubletsDC_->const_view<MiniDoubletsSoA>(),
                      segmentsDC_->const_view<SegmentsSoA>(),
                      segmentsDC_->const_view<SegmentsPixelSoA>(),
                      *tripletsInGPU_,
                      *pixelTripletsInGPU_,
                      connectedPixelSize_dev_buf.data(),
                      connectedPixelIndex_dev_buf.data(),
                      nInnerSegments);

#ifdef WARNINGS
  auto nPixelTriplets_buf = allocBufWrapper<unsigned int>(cms::alpakatools::host(), 1, queue_);

  alpaka::memcpy(queue_, nPixelTriplets_buf, pixelTripletsBuffers_->nPixelTriplets_buf);
  alpaka::wait(queue_);  // wait to get the value before using it

  std::cout << "number of pixel triplets = " << *nPixelTriplets_buf.data() << std::endl;
#endif

  //pT3s can be cleaned here because they're not used in making pT5s!
  Vec3D const threadsPerBlockDupPixTrip{1, 16, 16};
  //seems like more blocks lead to conflicting writes
  Vec3D const blocksPerGridDupPixTrip{1, 40, 1};
  WorkDiv3D const removeDupPixelTripletsInGPUFromMap_workDiv =
      createWorkDiv(blocksPerGridDupPixTrip, threadsPerBlockDupPixTrip, elementsPerThread);

  alpaka::exec<Acc3D>(
      queue_, removeDupPixelTripletsInGPUFromMap_workDiv, RemoveDupPixelTripletsInGPUFromMap{}, *pixelTripletsInGPU_);
}

void Event::createQuintuplets() {
  WorkDiv1D const createEligibleModulesListForQuintupletsGPU_workDiv = createWorkDiv<Vec1D>({1}, {1024}, {1});

  alpaka::exec<Acc1D>(queue_,
                      createEligibleModulesListForQuintupletsGPU_workDiv,
                      CreateEligibleModulesListForQuintupletsGPU{},
                      *modulesBuffers_.data(),
                      *tripletsInGPU_,
                      rangesDC_->view<ObjectOccupancySoA>());

  auto nEligibleT5Modules_buf = allocBufWrapper<uint16_t>(cms::alpakatools::host(), 1, queue_);
  auto nTotalQuintuplets_buf = allocBufWrapper<unsigned int>(cms::alpakatools::host(), 1, queue_);
  auto rangesOccupancy = rangesDC_->view<ObjectOccupancySoA>();
  auto nEligibleT5Modules_view_d = alpaka::createView(devAcc_, &rangesOccupancy.nEligibleT5Modules(), (Idx)1u);
  auto nTotalQuintuplets_view_d = alpaka::createView(devAcc_, &rangesOccupancy.nTotalQuints(), (Idx)1u);
  alpaka::memcpy(queue_, nEligibleT5Modules_buf, nEligibleT5Modules_view_d);
  alpaka::memcpy(queue_, nTotalQuintuplets_buf, nTotalQuintuplets_view_d);
  alpaka::wait(queue_);  // wait for the values before using them

  auto nEligibleT5Modules = *nEligibleT5Modules_buf.data();
  auto nTotalQuintuplets = *nTotalQuintuplets_buf.data();

  if (!quintupletsInGPU_) {
    quintupletsInGPU_.emplace();
    quintupletsBuffers_.emplace(nTotalQuintuplets, nLowerModules_, devAcc_, queue_);
    quintupletsInGPU_->setData(*quintupletsBuffers_);

    alpaka::memcpy(queue_, quintupletsBuffers_->nMemoryLocations_buf, nTotalQuintuplets_buf);
  }

  Vec3D const threadsPerBlockQuints{1, 8, 32};
  Vec3D const blocksPerGridQuints{std::max((int)nEligibleT5Modules, 1), 1, 1};
  WorkDiv3D const createQuintupletsInGPUv2_workDiv =
      createWorkDiv(blocksPerGridQuints, threadsPerBlockQuints, elementsPerThread);

  alpaka::exec<Acc3D>(queue_,
                      createQuintupletsInGPUv2_workDiv,
                      CreateQuintupletsInGPUv2{},
                      *modulesBuffers_.data(),
                      miniDoubletsDC_->const_view<MiniDoubletsSoA>(),
                      segmentsDC_->const_view<SegmentsSoA>(),
                      *tripletsInGPU_,
                      *quintupletsInGPU_,
                      rangesDC_->const_view<ObjectOccupancySoA>(),
                      nEligibleT5Modules);

  Vec3D const threadsPerBlockDupQuint{1, 16, 16};
  Vec3D const blocksPerGridDupQuint{max_blocks, 1, 1};
  WorkDiv3D const removeDupQuintupletsInGPUAfterBuild_workDiv =
      createWorkDiv(blocksPerGridDupQuint, threadsPerBlockDupQuint, elementsPerThread);

  alpaka::exec<Acc3D>(queue_,
                      removeDupQuintupletsInGPUAfterBuild_workDiv,
                      RemoveDupQuintupletsInGPUAfterBuild{},
                      *modulesBuffers_.data(),
                      *quintupletsInGPU_,
                      rangesDC_->const_view<ObjectOccupancySoA>());

  WorkDiv1D const addQuintupletRangesToEventExplicit_workDiv = createWorkDiv<Vec1D>({1}, {1024}, {1});

  alpaka::exec<Acc1D>(queue_,
                      addQuintupletRangesToEventExplicit_workDiv,
                      AddQuintupletRangesToEventExplicit{},
                      *modulesBuffers_.data(),
                      *quintupletsInGPU_,
                      rangesDC_->view<ObjectRangesSoA>(),
                      rangesDC_->const_view<ObjectOccupancySoA>());

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
  if (!pixelQuintupletsInGPU_) {
    pixelQuintupletsInGPU_.emplace();
    pixelQuintupletsBuffers_.emplace(n_max_pixel_quintuplets, devAcc_, queue_);
    pixelQuintupletsInGPU_->setData(*pixelQuintupletsBuffers_);
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
  WorkDiv3D const createPixelQuintupletsInGPUFromMapv2_workDiv =
      createWorkDiv(blocksPerGridCreatePixQuints, threadsPerBlockCreatePixQuints, elementsPerThread);

  alpaka::exec<Acc3D>(queue_,
                      createPixelQuintupletsInGPUFromMapv2_workDiv,
                      CreatePixelQuintupletsInGPUFromMapv2{},
                      *modulesBuffers_.data(),
                      miniDoubletsDC_->const_view<MiniDoubletsSoA>(),
                      segmentsDC_->const_view<SegmentsSoA>(),
                      segmentsDC_->view<SegmentsPixelSoA>(),
                      *tripletsInGPU_,
                      *quintupletsInGPU_,
                      *pixelQuintupletsInGPU_,
                      connectedPixelSize_dev_buf.data(),
                      connectedPixelIndex_dev_buf.data(),
                      nInnerSegments,
                      rangesDC_->const_view<ObjectOccupancySoA>());

  Vec3D const threadsPerBlockDupPix{1, 16, 16};
  Vec3D const blocksPerGridDupPix{1, max_blocks, 1};
  WorkDiv3D const removeDupPixelQuintupletsInGPUFromMap_workDiv =
      createWorkDiv(blocksPerGridDupPix, threadsPerBlockDupPix, elementsPerThread);

  alpaka::exec<Acc3D>(queue_,
                      removeDupPixelQuintupletsInGPUFromMap_workDiv,
                      RemoveDupPixelQuintupletsInGPUFromMap{},
                      *pixelQuintupletsInGPU_);

  WorkDiv1D const addpT5asTrackCandidateInGPU_workDiv = createWorkDiv<Vec1D>({1}, {256}, {1});

  alpaka::exec<Acc1D>(queue_,
                      addpT5asTrackCandidateInGPU_workDiv,
                      AddpT5asTrackCandidateInGPU{},
                      nLowerModules_,
                      *pixelQuintupletsInGPU_,
                      trackCandidatesDC_->view(),
                      segmentsDC_->const_view<SegmentsPixelSoA>(),
                      rangesDC_->const_view<ObjectOccupancySoA>());

#ifdef WARNINGS
  auto nPixelQuintuplets_buf = allocBufWrapper<unsigned int>(cms::alpakatools::host(), 1, queue_);

  alpaka::memcpy(queue_, nPixelQuintuplets_buf, pixelQuintupletsBuffers_->nPixelQuintuplets_buf);
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

  auto module_hitRanges_buf = allocBufWrapper<ArrayIx2>(cms::alpakatools::host(), nLowerModules_, queue_);
  auto hits = hitsDC_->view<HitsOccupancySoA>();
  auto hitRanges_view = alpaka::createView(devAcc_, hits.hitRanges(), nLowerModules_);  // only lower modules
  alpaka::memcpy(queue_, module_hitRanges_buf, hitRanges_view, nLowerModules_);

  alpaka::wait(queue_);  // wait for inputs before using them

  auto const* nMDsCPU = nMDsCPU_buf.data();
  auto const* module_subdets = module_subdets_buf.data();
  auto const* module_layers = module_layers_buf.data();
  auto const* module_hitRanges = module_hitRanges_buf.data();

  for (unsigned int i = 0; i < nLowerModules_; i++) {
    if (!(nMDsCPU[i] == 0 or module_hitRanges[i][0] == -1)) {
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
  auto nQuintupletsCPU_buf = allocBufWrapper<unsigned int>(cms::alpakatools::host(), nLowerModules_, queue_);
  alpaka::memcpy(queue_, nQuintupletsCPU_buf, quintupletsBuffers_->nQuintuplets_buf);

  // FIXME: replace by ES host data
  auto module_subdets_buf = allocBufWrapper<short>(cms::alpakatools::host(), nModules_, queue_);
  alpaka::memcpy(queue_, module_subdets_buf, modulesBuffers_.subdets_buf, nModules_);

  auto module_layers_buf = allocBufWrapper<short>(cms::alpakatools::host(), nLowerModules_, queue_);
  alpaka::memcpy(queue_, module_layers_buf, modulesBuffers_.layers_buf, nLowerModules_);

  auto module_quintupletModuleIndices_buf = allocBufWrapper<int>(cms::alpakatools::host(), nLowerModules_, queue_);
  auto rangesOccupancy = rangesDC_->view<ObjectOccupancySoA>();
  auto quintupletModuleIndices_view_d =
      alpaka::createView(devAcc_, rangesOccupancy.quintupletModuleIndices(), nLowerModules_);
  alpaka::memcpy(queue_, module_quintupletModuleIndices_buf, quintupletModuleIndices_view_d);

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
  auto nTripletsCPU_buf = allocBufWrapper<unsigned int>(cms::alpakatools::host(), nLowerModules_, queue_);
  alpaka::memcpy(queue_, nTripletsCPU_buf, tripletsBuffers_->nTriplets_buf);

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

  alpaka::memcpy(queue_, nPixelTriplets_buf_h, pixelTripletsBuffers_->nPixelTriplets_buf);
  alpaka::wait(queue_);

  return *nPixelTriplets_buf_h.data();
}

int Event::getNumberOfPixelQuintuplets() {
  auto nPixelQuintuplets_buf_h = cms::alpakatools::make_host_buffer<unsigned int[]>(queue_, 1u);

  alpaka::memcpy(queue_, nPixelQuintuplets_buf_h, pixelQuintupletsBuffers_->nPixelQuintuplets_buf);
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

template <typename TSoA, typename TDev>
typename TSoA::ConstView Event::getHits(bool sync)  //std::shared_ptr should take care of garbage collection
{
  if constexpr (std::is_same_v<TDev, DevHost>) {
    return hitsDC_->const_view<TSoA>();
  } else {
    if (!hitsHC_) {
      hitsHC_.emplace(cms::alpakatools::CopyToHost<PortableMultiCollection<TDev, HitsSoA, HitsOccupancySoA>>::copyAsync(
          queue_, *hitsDC_));
      if (sync)
        alpaka::wait(queue_);  // host consumers expect filled data
    }
    return hitsHC_->const_view<TSoA>();
  }
}
template HitsConst Event::getHits<HitsSoA>(bool);
template HitsOccupancyConst Event::getHits<HitsOccupancySoA>(bool);

template <typename TSoA, typename TDev>
typename TSoA::ConstView Event::getHitsInCMSSW(bool sync) {
  if constexpr (std::is_same_v<TDev, DevHost>) {
    return hitsDC_->const_view<TSoA>();
  } else {
    if (!hitsHC_) {
      auto hits_d = hitsDC_->view<HitsSoA>();
      auto nHits = hits_d.metadata().size();
      std::array<int, 2> const hits_sizes{{static_cast<int>(nHits), static_cast<int>(nModules_)}};
      hitsHC_.emplace(hits_sizes, queue_);
      auto hits_h = hitsHC_->view<HitsSoA>();
      auto idxs_h = alpaka::createView(cms::alpakatools::host(), hits_h.idxs(), nHits);
      auto idxs_d = alpaka::createView(devAcc_, hits_d.idxs(), nHits);
      alpaka::memcpy(queue_, idxs_h, idxs_d);
      if (sync)
        alpaka::wait(queue_);  // host consumers expect filled data
    }
    return hitsHC_->const_view<TSoA>();
  }
}
template HitsConst Event::getHitsInCMSSW<HitsSoA>(bool);
template HitsOccupancyConst Event::getHitsInCMSSW<HitsOccupancySoA>(bool);

template <typename TSoA, typename TDev>
typename TSoA::ConstView Event::getRanges(bool sync) {
  if constexpr (std::is_same_v<TDev, DevHost>) {
    return rangesDC_->const_view<TSoA>();
  } else {
    if (!rangesHC_) {
      rangesHC_.emplace(
          cms::alpakatools::CopyToHost<PortableMultiCollection<TDev, ObjectRangesSoA, ObjectOccupancySoA>>::copyAsync(
              queue_, *rangesDC_));
      if (sync)
        alpaka::wait(queue_);  // host consumers expect filled data
    }
    return rangesHC_->const_view<TSoA>();
  }
}
template ObjectRangesConst Event::getRanges<ObjectRangesSoA>(bool);
template ObjectOccupancyConst Event::getRanges<ObjectOccupancySoA>(bool);

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

TripletsBuffer<alpaka_common::DevHost>& Event::getTriplets(bool sync) {
  if (!tripletsInCPU_) {
    // Get nMemoryLocations parameter to initialize host based tripletsInCPU_
    auto nMemHost_buf_h = cms::alpakatools::make_host_buffer<unsigned int[]>(queue_, 1u);
    alpaka::memcpy(queue_, nMemHost_buf_h, tripletsBuffers_->nMemoryLocations_buf);
    alpaka::wait(queue_);  // wait for the value before using

    auto const nMemHost = *nMemHost_buf_h.data();
    tripletsInCPU_.emplace(nMemHost, nLowerModules_, cms::alpakatools::host(), queue_);
    tripletsInCPU_->setData(*tripletsInCPU_);

    alpaka::memcpy(queue_, tripletsInCPU_->nMemoryLocations_buf, tripletsBuffers_->nMemoryLocations_buf);
#ifdef CUT_VALUE_DEBUG
    alpaka::memcpy(queue_, tripletsInCPU_->zOut_buf, tripletsBuffers_->zOut_buf, nMemHost);
    alpaka::memcpy(queue_, tripletsInCPU_->zLo_buf, tripletsBuffers_->zLo_buf, nMemHost);
    alpaka::memcpy(queue_, tripletsInCPU_->zHi_buf, tripletsBuffers_->zHi_buf, nMemHost);
    alpaka::memcpy(queue_, tripletsInCPU_->zLoPointed_buf, tripletsBuffers_->zLoPointed_buf, nMemHost);
    alpaka::memcpy(queue_, tripletsInCPU_->zHiPointed_buf, tripletsBuffers_->zHiPointed_buf, nMemHost);
    alpaka::memcpy(queue_, tripletsInCPU_->dPhiCut_buf, tripletsBuffers_->dPhiCut_buf, nMemHost);
    alpaka::memcpy(queue_, tripletsInCPU_->betaInCut_buf, tripletsBuffers_->betaInCut_buf, nMemHost);
    alpaka::memcpy(queue_, tripletsInCPU_->rtLo_buf, tripletsBuffers_->rtLo_buf, nMemHost);
    alpaka::memcpy(queue_, tripletsInCPU_->rtHi_buf, tripletsBuffers_->rtHi_buf, nMemHost);
#endif
    alpaka::memcpy(
        queue_, tripletsInCPU_->hitIndices_buf, tripletsBuffers_->hitIndices_buf, Params_T3::kHits * nMemHost);
    alpaka::memcpy(
        queue_, tripletsInCPU_->logicalLayers_buf, tripletsBuffers_->logicalLayers_buf, Params_T3::kLayers * nMemHost);
    alpaka::memcpy(queue_, tripletsInCPU_->segmentIndices_buf, tripletsBuffers_->segmentIndices_buf, 2 * nMemHost);
    alpaka::memcpy(queue_, tripletsInCPU_->betaIn_buf, tripletsBuffers_->betaIn_buf, nMemHost);
    alpaka::memcpy(queue_, tripletsInCPU_->circleRadius_buf, tripletsBuffers_->circleRadius_buf, nMemHost);
    alpaka::memcpy(queue_, tripletsInCPU_->nTriplets_buf, tripletsBuffers_->nTriplets_buf);
    alpaka::memcpy(queue_, tripletsInCPU_->totOccupancyTriplets_buf, tripletsBuffers_->totOccupancyTriplets_buf);
    if (sync)
      alpaka::wait(queue_);  // host consumers expect filled data
  }
  return tripletsInCPU_.value();
}

QuintupletsBuffer<alpaka_common::DevHost>& Event::getQuintuplets(bool sync) {
  if (!quintupletsInCPU_) {
    // Get nMemoryLocations parameter to initialize host based quintupletsInCPU_
    auto nMemHost_buf_h = cms::alpakatools::make_host_buffer<unsigned int[]>(queue_, 1u);
    alpaka::memcpy(queue_, nMemHost_buf_h, quintupletsBuffers_->nMemoryLocations_buf);
    alpaka::wait(queue_);  // wait for the value before using

    auto const nMemHost = *nMemHost_buf_h.data();
    quintupletsInCPU_.emplace(nMemHost, nLowerModules_, cms::alpakatools::host(), queue_);
    quintupletsInCPU_->setData(*quintupletsInCPU_);

    alpaka::memcpy(queue_, quintupletsInCPU_->nMemoryLocations_buf, quintupletsBuffers_->nMemoryLocations_buf);
    alpaka::memcpy(queue_, quintupletsInCPU_->nQuintuplets_buf, quintupletsBuffers_->nQuintuplets_buf);
    alpaka::memcpy(
        queue_, quintupletsInCPU_->totOccupancyQuintuplets_buf, quintupletsBuffers_->totOccupancyQuintuplets_buf);
    alpaka::memcpy(
        queue_, quintupletsInCPU_->tripletIndices_buf, quintupletsBuffers_->tripletIndices_buf, 2 * nMemHost);
    alpaka::memcpy(queue_,
                   quintupletsInCPU_->lowerModuleIndices_buf,
                   quintupletsBuffers_->lowerModuleIndices_buf,
                   Params_T5::kLayers * nMemHost);
    alpaka::memcpy(queue_, quintupletsInCPU_->innerRadius_buf, quintupletsBuffers_->innerRadius_buf, nMemHost);
    alpaka::memcpy(queue_, quintupletsInCPU_->bridgeRadius_buf, quintupletsBuffers_->bridgeRadius_buf, nMemHost);
    alpaka::memcpy(queue_, quintupletsInCPU_->outerRadius_buf, quintupletsBuffers_->outerRadius_buf, nMemHost);
    alpaka::memcpy(queue_, quintupletsInCPU_->isDup_buf, quintupletsBuffers_->isDup_buf, nMemHost);
    alpaka::memcpy(queue_, quintupletsInCPU_->score_rphisum_buf, quintupletsBuffers_->score_rphisum_buf, nMemHost);
    alpaka::memcpy(queue_, quintupletsInCPU_->eta_buf, quintupletsBuffers_->eta_buf, nMemHost);
    alpaka::memcpy(queue_, quintupletsInCPU_->phi_buf, quintupletsBuffers_->phi_buf, nMemHost);
    alpaka::memcpy(queue_, quintupletsInCPU_->chiSquared_buf, quintupletsBuffers_->chiSquared_buf, nMemHost);
    alpaka::memcpy(queue_, quintupletsInCPU_->rzChiSquared_buf, quintupletsBuffers_->rzChiSquared_buf, nMemHost);
    alpaka::memcpy(
        queue_, quintupletsInCPU_->nonAnchorChiSquared_buf, quintupletsBuffers_->nonAnchorChiSquared_buf, nMemHost);
    if (sync)
      alpaka::wait(queue_);  // host consumers expect filled data
  }
  return quintupletsInCPU_.value();
}

PixelTripletsBuffer<alpaka_common::DevHost>& Event::getPixelTriplets(bool sync) {
  if (!pixelTripletsInCPU_) {
    // Get nPixelTriplets parameter to initialize host based quintupletsInCPU_
    auto nPixelTriplets_buf_h = cms::alpakatools::make_host_buffer<unsigned int[]>(queue_, 1u);
    alpaka::memcpy(queue_, nPixelTriplets_buf_h, pixelTripletsBuffers_->nPixelTriplets_buf);
    alpaka::wait(queue_);  // wait for the value before using

    auto const nPixelTriplets = *nPixelTriplets_buf_h.data();
    pixelTripletsInCPU_.emplace(nPixelTriplets, cms::alpakatools::host(), queue_);
    pixelTripletsInCPU_->setData(*pixelTripletsInCPU_);

    alpaka::memcpy(queue_, pixelTripletsInCPU_->nPixelTriplets_buf, pixelTripletsBuffers_->nPixelTriplets_buf);
    alpaka::memcpy(queue_,
                   pixelTripletsInCPU_->totOccupancyPixelTriplets_buf,
                   pixelTripletsBuffers_->totOccupancyPixelTriplets_buf);
    alpaka::memcpy(
        queue_, pixelTripletsInCPU_->rzChiSquared_buf, pixelTripletsBuffers_->rzChiSquared_buf, nPixelTriplets);
    alpaka::memcpy(
        queue_, pixelTripletsInCPU_->rPhiChiSquared_buf, pixelTripletsBuffers_->rPhiChiSquared_buf, nPixelTriplets);
    alpaka::memcpy(queue_,
                   pixelTripletsInCPU_->rPhiChiSquaredInwards_buf,
                   pixelTripletsBuffers_->rPhiChiSquaredInwards_buf,
                   nPixelTriplets);
    alpaka::memcpy(
        queue_, pixelTripletsInCPU_->tripletIndices_buf, pixelTripletsBuffers_->tripletIndices_buf, nPixelTriplets);
    alpaka::memcpy(queue_,
                   pixelTripletsInCPU_->pixelSegmentIndices_buf,
                   pixelTripletsBuffers_->pixelSegmentIndices_buf,
                   nPixelTriplets);
    alpaka::memcpy(
        queue_, pixelTripletsInCPU_->pixelRadius_buf, pixelTripletsBuffers_->pixelRadius_buf, nPixelTriplets);
    alpaka::memcpy(
        queue_, pixelTripletsInCPU_->tripletRadius_buf, pixelTripletsBuffers_->tripletRadius_buf, nPixelTriplets);
    alpaka::memcpy(queue_, pixelTripletsInCPU_->isDup_buf, pixelTripletsBuffers_->isDup_buf, nPixelTriplets);
    alpaka::memcpy(queue_, pixelTripletsInCPU_->eta_buf, pixelTripletsBuffers_->eta_buf, nPixelTriplets);
    alpaka::memcpy(queue_, pixelTripletsInCPU_->phi_buf, pixelTripletsBuffers_->phi_buf, nPixelTriplets);
    alpaka::memcpy(queue_, pixelTripletsInCPU_->score_buf, pixelTripletsBuffers_->score_buf, nPixelTriplets);
    if (sync)
      alpaka::wait(queue_);  // host consumers expect filled data
  }
  return pixelTripletsInCPU_.value();
}

PixelQuintupletsBuffer<alpaka_common::DevHost>& Event::getPixelQuintuplets(bool sync) {
  if (!pixelQuintupletsInCPU_) {
    // Get nPixelQuintuplets parameter to initialize host based quintupletsInCPU_
    auto nPixelQuintuplets_buf_h = cms::alpakatools::make_host_buffer<unsigned int[]>(queue_, 1u);
    alpaka::memcpy(queue_, nPixelQuintuplets_buf_h, pixelQuintupletsBuffers_->nPixelQuintuplets_buf);
    alpaka::wait(queue_);  // wait for the value before using

    auto const nPixelQuintuplets = *nPixelQuintuplets_buf_h.data();
    pixelQuintupletsInCPU_.emplace(nPixelQuintuplets, cms::alpakatools::host(), queue_);
    pixelQuintupletsInCPU_->setData(*pixelQuintupletsInCPU_);

    alpaka::memcpy(
        queue_, pixelQuintupletsInCPU_->nPixelQuintuplets_buf, pixelQuintupletsBuffers_->nPixelQuintuplets_buf);
    alpaka::memcpy(queue_,
                   pixelQuintupletsInCPU_->totOccupancyPixelQuintuplets_buf,
                   pixelQuintupletsBuffers_->totOccupancyPixelQuintuplets_buf);
    alpaka::memcpy(queue_,
                   pixelQuintupletsInCPU_->rzChiSquared_buf,
                   pixelQuintupletsBuffers_->rzChiSquared_buf,
                   nPixelQuintuplets);
    alpaka::memcpy(queue_,
                   pixelQuintupletsInCPU_->rPhiChiSquared_buf,
                   pixelQuintupletsBuffers_->rPhiChiSquared_buf,
                   nPixelQuintuplets);
    alpaka::memcpy(queue_,
                   pixelQuintupletsInCPU_->rPhiChiSquaredInwards_buf,
                   pixelQuintupletsBuffers_->rPhiChiSquaredInwards_buf,
                   nPixelQuintuplets);
    alpaka::memcpy(queue_,
                   pixelQuintupletsInCPU_->pixelIndices_buf,
                   pixelQuintupletsBuffers_->pixelIndices_buf,
                   nPixelQuintuplets);
    alpaka::memcpy(
        queue_, pixelQuintupletsInCPU_->T5Indices_buf, pixelQuintupletsBuffers_->T5Indices_buf, nPixelQuintuplets);
    alpaka::memcpy(queue_, pixelQuintupletsInCPU_->isDup_buf, pixelQuintupletsBuffers_->isDup_buf, nPixelQuintuplets);
    alpaka::memcpy(queue_, pixelQuintupletsInCPU_->score_buf, pixelQuintupletsBuffers_->score_buf, nPixelQuintuplets);
    if (sync)
      alpaka::wait(queue_);  // host consumers expect filled data
  }
  return pixelQuintupletsInCPU_.value();
}

const TrackCandidatesHostCollection& Event::getTrackCandidates(bool sync) {
  if (!trackCandidatesHC_) {
    // Get nTrackCanHost parameter to initialize host based instance
    auto nTrackCanHost_buf_h = cms::alpakatools::make_host_buffer<unsigned int[]>(queue_, 1u);
    alpaka::memcpy(
        queue_, nTrackCanHost_buf_h, alpaka::createView(devAcc_, &(*trackCandidatesDC_)->nTrackCandidates(), 1u));
    trackCandidatesHC_.emplace(n_max_nonpixel_track_candidates + n_max_pixel_track_candidates, queue_);
    alpaka::wait(queue_);  // wait here before we get nTrackCanHost and trackCandidatesInCPU becomes usable

    auto const nTrackCanHost = *nTrackCanHost_buf_h.data();

    (*trackCandidatesHC_)->nTrackCandidates() = nTrackCanHost;
    alpaka::memcpy(
        queue_,
        alpaka::createView(
            cms::alpakatools::host(), (*trackCandidatesHC_)->hitIndices()->data(), Params_pT5::kHits * nTrackCanHost),
        alpaka::createView(devAcc_, (*trackCandidatesDC_)->hitIndices()->data(), Params_pT5::kHits * nTrackCanHost));
    alpaka::memcpy(queue_,
                   alpaka::createView(cms::alpakatools::host(), (*trackCandidatesHC_)->pixelSeedIndex(), nTrackCanHost),
                   alpaka::createView(devAcc_, (*trackCandidatesDC_)->pixelSeedIndex(), nTrackCanHost));
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
    alpaka::memcpy(
        queue_,
        alpaka::createView(cms::alpakatools::host(), (*trackCandidatesHC_)->objectIndices()->data(), 2 * nTrackCanHost),
        alpaka::createView(devAcc_, (*trackCandidatesDC_)->objectIndices()->data(), 2 * nTrackCanHost));
    alpaka::memcpy(
        queue_,
        alpaka::createView(cms::alpakatools::host(), (*trackCandidatesHC_)->trackCandidateType(), nTrackCanHost),
        alpaka::createView(devAcc_, (*trackCandidatesDC_)->trackCandidateType(), nTrackCanHost));
    if (sync)
      alpaka::wait(queue_);  // host consumers expect filled data
  }
  return trackCandidatesHC_.value();
}

const TrackCandidatesHostCollection& Event::getTrackCandidatesInCMSSW(bool sync) {
  if (!trackCandidatesHC_) {
    // Get nTrackCanHost parameter to initialize host based instance
    auto nTrackCanHost_buf_h = cms::alpakatools::make_host_buffer<unsigned int[]>(queue_, 1u);
    alpaka::memcpy(
        queue_, nTrackCanHost_buf_h, alpaka::createView(devAcc_, &(*trackCandidatesDC_)->nTrackCandidates(), 1u));
    trackCandidatesHC_.emplace(n_max_nonpixel_track_candidates + n_max_pixel_track_candidates, queue_);
    alpaka::wait(queue_);  // wait for the value before using and trackCandidatesInCPU becomes usable

    auto const nTrackCanHost = *nTrackCanHost_buf_h.data();

    (*trackCandidatesHC_)->nTrackCandidates() = nTrackCanHost;
    alpaka::memcpy(
        queue_,
        alpaka::createView(
            cms::alpakatools::host(), (*trackCandidatesHC_)->hitIndices()->data(), Params_pT5::kHits * nTrackCanHost),
        alpaka::createView(devAcc_, (*trackCandidatesDC_)->hitIndices()->data(), Params_pT5::kHits * nTrackCanHost));
    alpaka::memcpy(queue_,
                   alpaka::createView(cms::alpakatools::host(), (*trackCandidatesHC_)->pixelSeedIndex(), nTrackCanHost),
                   alpaka::createView(devAcc_, (*trackCandidatesDC_)->pixelSeedIndex(), nTrackCanHost));
    alpaka::memcpy(
        queue_,
        alpaka::createView(cms::alpakatools::host(), (*trackCandidatesHC_)->trackCandidateType(), nTrackCanHost),
        alpaka::createView(devAcc_, (*trackCandidatesDC_)->trackCandidateType(), nTrackCanHost));
    if (sync)
      alpaka::wait(queue_);  // host consumers expect filled data
  }
  return trackCandidatesHC_.value();
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
