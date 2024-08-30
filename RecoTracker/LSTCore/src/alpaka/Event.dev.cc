#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"

#include "Event.h"

using Device = ALPAKA_ACCELERATOR_NAMESPACE::Device;
using Queue = ALPAKA_ACCELERATOR_NAMESPACE::Queue;
using Acc1D = ALPAKA_ACCELERATOR_NAMESPACE::Acc1D;
using Acc3D = ALPAKA_ACCELERATOR_NAMESPACE::Acc3D;

using namespace ALPAKA_ACCELERATOR_NAMESPACE::lst;

void Event::initSync(bool verbose) {
  alpaka::wait(queue);  // other calls can be asynchronous
  addObjects = verbose;
  hitsInGPU = std::nullopt;
  mdsInGPU = std::nullopt;
  segmentsInGPU = std::nullopt;
  tripletsInGPU = std::nullopt;
  quintupletsInGPU = std::nullopt;
  trackCandidatesInGPU = std::nullopt;
  pixelTripletsInGPU = std::nullopt;
  pixelQuintupletsInGPU = std::nullopt;
  rangesInGPU = std::nullopt;

  hitsInCPU = std::nullopt;
  rangesInCPU = std::nullopt;
  mdsInCPU = std::nullopt;
  segmentsInCPU = std::nullopt;
  tripletsInCPU = std::nullopt;
  trackCandidatesInCPU = std::nullopt;
  modulesInCPU = std::nullopt;
  quintupletsInCPU = std::nullopt;
  pixelTripletsInCPU = std::nullopt;
  pixelQuintupletsInCPU = std::nullopt;

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
  alpaka::wait(queue);  // synchronize to reset consistently
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
  if (hitsInGPU) {
    hitsInGPU = std::nullopt;
    hitsBuffers = std::nullopt;
  }
  if (mdsInGPU) {
    mdsInGPU = std::nullopt;
    miniDoubletsBuffers = std::nullopt;
  }
  if (rangesInGPU) {
    rangesInGPU = std::nullopt;
    rangesBuffers = std::nullopt;
  }
  if (segmentsInGPU) {
    segmentsInGPU = std::nullopt;
    segmentsBuffers = std::nullopt;
  }
  if (tripletsInGPU) {
    tripletsInGPU = std::nullopt;
    tripletsBuffers = std::nullopt;
  }
  if (quintupletsInGPU) {
    quintupletsInGPU = std::nullopt;
    quintupletsBuffers = std::nullopt;
  }
  if (trackCandidatesInGPU) {
    trackCandidatesInGPU = std::nullopt;
    trackCandidatesBuffers = std::nullopt;
  }
  if (pixelTripletsInGPU) {
    pixelTripletsInGPU = std::nullopt;
    pixelTripletsBuffers = std::nullopt;
  }
  if (pixelQuintupletsInGPU) {
    pixelQuintupletsInGPU = std::nullopt;
    pixelQuintupletsBuffers = std::nullopt;
  }

  if (hitsInCPU) {
    hitsInCPU = std::nullopt;
  }
  if (rangesInCPU) {
    rangesInCPU = std::nullopt;
  }
  if (mdsInCPU) {
    mdsInCPU = std::nullopt;
  }
  if (segmentsInCPU) {
    segmentsInCPU = std::nullopt;
  }
  if (tripletsInCPU) {
    tripletsInCPU = std::nullopt;
  }
  if (quintupletsInCPU) {
    quintupletsInCPU = std::nullopt;
  }
  if (pixelTripletsInCPU) {
    pixelTripletsInCPU = std::nullopt;
  }
  if (pixelQuintupletsInCPU) {
    pixelQuintupletsInCPU = std::nullopt;
  }
  if (trackCandidatesInCPU) {
    trackCandidatesInCPU = std::nullopt;
  }
  if (modulesInCPU) {
    modulesInCPU = std::nullopt;
  }
}

void Event::addHitToEvent(std::vector<float> const& x,
                          std::vector<float> const& y,
                          std::vector<float> const& z,
                          std::vector<unsigned int> const& detId,
                          std::vector<unsigned int> const& idxInNtuple) {
  // Use the actual number of hits instead of a max.
  unsigned int nHits = x.size();

  // Initialize space on device/host for next event.
  if (!hitsInGPU) {
    hitsInGPU = Hits();
    hitsBuffers = HitsBuffer<Device>(nModules_, nHits, devAcc, queue);
    hitsInGPU->setData(*hitsBuffers);
  }

  if (!rangesInGPU) {
    rangesInGPU = ObjectRanges();
    rangesBuffers = ObjectRangesBuffer<Device>(nModules_, nLowerModules_, devAcc, queue);
    rangesInGPU->setData(*rangesBuffers);
  }

  // Need a view here before transferring to the device.
  auto nHits_view = alpaka::createView(devHost, &nHits, (Idx)1u);

  // Copy the host arrays to the GPU.
  alpaka::memcpy(queue, hitsBuffers->xs_buf, x, nHits);
  alpaka::memcpy(queue, hitsBuffers->ys_buf, y, nHits);
  alpaka::memcpy(queue, hitsBuffers->zs_buf, z, nHits);
  alpaka::memcpy(queue, hitsBuffers->detid_buf, detId, nHits);
  alpaka::memcpy(queue, hitsBuffers->idxs_buf, idxInNtuple, nHits);
  alpaka::memcpy(queue, hitsBuffers->nHits_buf, nHits_view);
  alpaka::wait(queue);  // FIXME: remove synch after inputs refactored to be in pinned memory

  Vec3D const threadsPerBlock1{1, 1, 256};
  Vec3D const blocksPerGrid1{1, 1, max_blocks};
  WorkDiv3D const hit_loop_workdiv = createWorkDiv(blocksPerGrid1, threadsPerBlock1, elementsPerThread);

  alpaka::exec<Acc3D>(queue,
                      hit_loop_workdiv,
                      HitLoopKernel{},
                      Endcap,
                      TwoS,
                      nModules_,
                      nEndCapMap_,
                      endcapGeometryBuffers_.geoMapDetId_buf.data(),
                      endcapGeometryBuffers_.geoMapPhi_buf.data(),
                      *modulesBuffers_.data(),
                      *hitsInGPU,
                      nHits);

  Vec3D const threadsPerBlock2{1, 1, 256};
  Vec3D const blocksPerGrid2{1, 1, max_blocks};
  WorkDiv3D const module_ranges_workdiv = createWorkDiv(blocksPerGrid2, threadsPerBlock2, elementsPerThread);

  alpaka::exec<Acc3D>(
      queue, module_ranges_workdiv, ModuleRangesKernel{}, *modulesBuffers_.data(), *hitsInGPU, nLowerModules_);
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

  if (!mdsInGPU) {
    // Create a view for the element nLowerModules_ inside rangesBuffers->miniDoubletModuleOccupancy
    auto dst_view_miniDoubletModuleOccupancy =
        alpaka::createSubView(rangesBuffers->miniDoubletModuleOccupancy_buf, (Idx)1u, (Idx)nLowerModules_);

    // Create a host buffer for a value to be passed to the device
    auto pixelMaxMDs_buf_h = cms::alpakatools::make_host_buffer<int[]>(queue, (Idx)1u);
    *pixelMaxMDs_buf_h.data() = n_max_pixel_md_per_modules;

    alpaka::memcpy(queue, dst_view_miniDoubletModuleOccupancy, pixelMaxMDs_buf_h);

    WorkDiv1D const createMDArrayRangesGPU_workDiv = createWorkDiv<Vec1D>({1}, {1024}, {1});

    alpaka::exec<Acc1D>(
        queue, createMDArrayRangesGPU_workDiv, CreateMDArrayRangesGPU{}, *modulesBuffers_.data(), *rangesInGPU);

    auto nTotalMDs_buf_h = cms::alpakatools::make_host_buffer<unsigned int[]>(queue, (Idx)1u);
    alpaka::memcpy(queue, nTotalMDs_buf_h, rangesBuffers->device_nTotalMDs_buf);
    alpaka::wait(queue);  // wait to get the data before manipulation

    *nTotalMDs_buf_h.data() += n_max_pixel_md_per_modules;
    unsigned int nTotalMDs = *nTotalMDs_buf_h.data();

    mdsInGPU = MiniDoublets();
    miniDoubletsBuffers = MiniDoubletsBuffer<Device>(nTotalMDs, nLowerModules_, devAcc, queue);
    mdsInGPU->setData(*miniDoubletsBuffers);

    alpaka::memcpy(queue, miniDoubletsBuffers->nMemoryLocations_buf, nTotalMDs_buf_h);
  }
  if (!segmentsInGPU) {
    // can be optimized here: because we didn't distinguish pixel segments and outer-tracker segments and call them both "segments", so they use the index continuously.
    // If we want to further study the memory footprint in detail, we can separate the two and allocate different memories to them

    WorkDiv1D const createSegmentArrayRanges_workDiv = createWorkDiv<Vec1D>({1}, {1024}, {1});

    alpaka::exec<Acc1D>(queue,
                        createSegmentArrayRanges_workDiv,
                        CreateSegmentArrayRanges{},
                        *modulesBuffers_.data(),
                        *rangesInGPU,
                        *mdsInGPU);

    auto nTotalSegments_view = alpaka::createView(devHost, &nTotalSegments_, (Idx)1u);

    alpaka::memcpy(queue, nTotalSegments_view, rangesBuffers->device_nTotalSegs_buf);
    alpaka::wait(queue);  // wait to get the value before manipulation

    nTotalSegments_ += n_max_pixel_segments_per_module;

    segmentsInGPU = Segments();
    segmentsBuffers =
        SegmentsBuffer<Device>(nTotalSegments_, nLowerModules_, n_max_pixel_segments_per_module, devAcc, queue);
    segmentsInGPU->setData(*segmentsBuffers);

    alpaka::memcpy(queue, segmentsBuffers->nMemoryLocations_buf, nTotalSegments_view);
  }

  auto hitIndices0_dev = allocBufWrapper<unsigned int>(devAcc, size, queue);
  auto hitIndices1_dev = allocBufWrapper<unsigned int>(devAcc, size, queue);
  auto hitIndices2_dev = allocBufWrapper<unsigned int>(devAcc, size, queue);
  auto hitIndices3_dev = allocBufWrapper<unsigned int>(devAcc, size, queue);
  auto dPhiChange_dev = allocBufWrapper<float>(devAcc, size, queue);

  alpaka::memcpy(queue, hitIndices0_dev, hitIndices0, size);
  alpaka::memcpy(queue, hitIndices1_dev, hitIndices1, size);
  alpaka::memcpy(queue, hitIndices2_dev, hitIndices2, size);
  alpaka::memcpy(queue, hitIndices3_dev, hitIndices3, size);
  alpaka::memcpy(queue, dPhiChange_dev, dPhiChange, size);

  alpaka::memcpy(queue, segmentsBuffers->ptIn_buf, ptIn, size);
  alpaka::memcpy(queue, segmentsBuffers->ptErr_buf, ptErr, size);
  alpaka::memcpy(queue, segmentsBuffers->px_buf, px, size);
  alpaka::memcpy(queue, segmentsBuffers->py_buf, py, size);
  alpaka::memcpy(queue, segmentsBuffers->pz_buf, pz, size);
  alpaka::memcpy(queue, segmentsBuffers->etaErr_buf, etaErr, size);
  alpaka::memcpy(queue, segmentsBuffers->isQuad_buf, isQuad, size);
  alpaka::memcpy(queue, segmentsBuffers->eta_buf, eta, size);
  alpaka::memcpy(queue, segmentsBuffers->phi_buf, phi, size);
  alpaka::memcpy(queue, segmentsBuffers->charge_buf, charge, size);
  alpaka::memcpy(queue, segmentsBuffers->seedIdx_buf, seedIdx, size);
  alpaka::memcpy(queue, segmentsBuffers->superbin_buf, superbin, size);
  alpaka::memcpy(queue, segmentsBuffers->pixelType_buf, pixelType, size);

  // Create source views for size and mdSize
  auto src_view_size = alpaka::createView(devHost, &size, (Idx)1u);
  auto src_view_mdSize = alpaka::createView(devHost, &mdSize, (Idx)1u);

  auto dst_view_segments = alpaka::createSubView(segmentsBuffers->nSegments_buf, (Idx)1u, (Idx)pixelModuleIndex);
  alpaka::memcpy(queue, dst_view_segments, src_view_size);

  auto dst_view_totOccupancySegments =
      alpaka::createSubView(segmentsBuffers->totOccupancySegments_buf, (Idx)1u, (Idx)pixelModuleIndex);
  alpaka::memcpy(queue, dst_view_totOccupancySegments, src_view_size);

  auto dst_view_nMDs = alpaka::createSubView(miniDoubletsBuffers->nMDs_buf, (Idx)1u, (Idx)pixelModuleIndex);
  alpaka::memcpy(queue, dst_view_nMDs, src_view_mdSize);

  auto dst_view_totOccupancyMDs =
      alpaka::createSubView(miniDoubletsBuffers->totOccupancyMDs_buf, (Idx)1u, (Idx)pixelModuleIndex);
  alpaka::memcpy(queue, dst_view_totOccupancyMDs, src_view_mdSize);

  alpaka::wait(queue);  // FIXME: remove synch after inputs refactored to be in pinned memory

  Vec3D const threadsPerBlock{1, 1, 256};
  Vec3D const blocksPerGrid{1, 1, max_blocks};
  WorkDiv3D const addPixelSegmentToEvent_workdiv = createWorkDiv(blocksPerGrid, threadsPerBlock, elementsPerThread);

  alpaka::exec<Acc3D>(queue,
                      addPixelSegmentToEvent_workdiv,
                      AddPixelSegmentToEventKernel{},
                      *modulesBuffers_.data(),
                      *rangesInGPU,
                      *hitsInGPU,
                      *mdsInGPU,
                      *segmentsInGPU,
                      hitIndices0_dev.data(),
                      hitIndices1_dev.data(),
                      hitIndices2_dev.data(),
                      hitIndices3_dev.data(),
                      dPhiChange_dev.data(),
                      pixelModuleIndex,
                      size);
}

void Event::createMiniDoublets() {
  // Create a view for the element nLowerModules_ inside rangesBuffers->miniDoubletModuleOccupancy
  auto dst_view_miniDoubletModuleOccupancy =
      alpaka::createSubView(rangesBuffers->miniDoubletModuleOccupancy_buf, (Idx)1u, (Idx)nLowerModules_);

  // Create a host buffer for a value to be passed to the device
  auto pixelMaxMDs_buf_h = cms::alpakatools::make_host_buffer<int[]>(queue, (Idx)1u);
  *pixelMaxMDs_buf_h.data() = n_max_pixel_md_per_modules;

  alpaka::memcpy(queue, dst_view_miniDoubletModuleOccupancy, pixelMaxMDs_buf_h);

  WorkDiv1D const createMDArrayRangesGPU_workDiv = createWorkDiv<Vec1D>({1}, {1024}, {1});

  alpaka::exec<Acc1D>(
      queue, createMDArrayRangesGPU_workDiv, CreateMDArrayRangesGPU{}, *modulesBuffers_.data(), *rangesInGPU);

  auto nTotalMDs_buf_h = cms::alpakatools::make_host_buffer<unsigned int[]>(queue, (Idx)1u);
  alpaka::memcpy(queue, nTotalMDs_buf_h, rangesBuffers->device_nTotalMDs_buf);
  alpaka::wait(queue);  // wait to get the data before manipulation

  *nTotalMDs_buf_h.data() += n_max_pixel_md_per_modules;
  unsigned int nTotalMDs = *nTotalMDs_buf_h.data();

  if (!mdsInGPU) {
    mdsInGPU = MiniDoublets();
    miniDoubletsBuffers = MiniDoubletsBuffer<Device>(nTotalMDs, nLowerModules_, devAcc, queue);
    mdsInGPU->setData(*miniDoubletsBuffers);
  }

  Vec3D const threadsPerBlockCreateMDInGPU{1, 16, 32};
  Vec3D const blocksPerGridCreateMDInGPU{1, nLowerModules_ / threadsPerBlockCreateMDInGPU[1], 1};
  WorkDiv3D const createMiniDoubletsInGPUv2_workDiv =
      createWorkDiv(blocksPerGridCreateMDInGPU, threadsPerBlockCreateMDInGPU, elementsPerThread);

  alpaka::exec<Acc3D>(queue,
                      createMiniDoubletsInGPUv2_workDiv,
                      CreateMiniDoubletsInGPUv2{},
                      *modulesBuffers_.data(),
                      *hitsInGPU,
                      *mdsInGPU,
                      *rangesInGPU);

  WorkDiv1D const addMiniDoubletRangesToEventExplicit_workDiv = createWorkDiv<Vec1D>({1}, {1024}, {1});

  alpaka::exec<Acc1D>(queue,
                      addMiniDoubletRangesToEventExplicit_workDiv,
                      AddMiniDoubletRangesToEventExplicit{},
                      *modulesBuffers_.data(),
                      *mdsInGPU,
                      *rangesInGPU,
                      *hitsInGPU);

  if (addObjects) {
    addMiniDoubletsToEventExplicit();
  }
}

void Event::createSegmentsWithModuleMap() {
  if (!segmentsInGPU) {
    segmentsInGPU = Segments();
    segmentsBuffers =
        SegmentsBuffer<Device>(nTotalSegments_, nLowerModules_, n_max_pixel_segments_per_module, devAcc, queue);
    segmentsInGPU->setData(*segmentsBuffers);
  }

  Vec3D const threadsPerBlockCreateSeg{1, 1, 64};
  Vec3D const blocksPerGridCreateSeg{1, 1, nLowerModules_};
  WorkDiv3D const createSegmentsInGPUv2_workDiv =
      createWorkDiv(blocksPerGridCreateSeg, threadsPerBlockCreateSeg, elementsPerThread);

  alpaka::exec<Acc3D>(queue,
                      createSegmentsInGPUv2_workDiv,
                      CreateSegmentsInGPUv2{},
                      *modulesBuffers_.data(),
                      *mdsInGPU,
                      *segmentsInGPU,
                      *rangesInGPU);

  WorkDiv1D const addSegmentRangesToEventExplicit_workDiv = createWorkDiv<Vec1D>({1}, {1024}, {1});

  alpaka::exec<Acc1D>(queue,
                      addSegmentRangesToEventExplicit_workDiv,
                      AddSegmentRangesToEventExplicit{},
                      *modulesBuffers_.data(),
                      *segmentsInGPU,
                      *rangesInGPU);

  if (addObjects) {
    addSegmentsToEventExplicit();
  }
}

void Event::createTriplets() {
  if (!tripletsInGPU) {
    WorkDiv1D const createTripletArrayRanges_workDiv = createWorkDiv<Vec1D>({1}, {1024}, {1});

    alpaka::exec<Acc1D>(queue,
                        createTripletArrayRanges_workDiv,
                        CreateTripletArrayRanges{},
                        *modulesBuffers_.data(),
                        *rangesInGPU,
                        *segmentsInGPU);

    // TODO: Why are we pulling this back down only to put it back on the device in a new struct?
    auto maxTriplets_buf_h = cms::alpakatools::make_host_buffer<unsigned int[]>(queue, (Idx)1u);

    alpaka::memcpy(queue, maxTriplets_buf_h, rangesBuffers->device_nTotalTrips_buf);
    alpaka::wait(queue);  // wait to get the value before using it

    tripletsInGPU = Triplets();
    tripletsBuffers = TripletsBuffer<Device>(*maxTriplets_buf_h.data(), nLowerModules_, devAcc, queue);
    tripletsInGPU->setData(*tripletsBuffers);

    alpaka::memcpy(queue, tripletsBuffers->nMemoryLocations_buf, maxTriplets_buf_h);
  }

  uint16_t nonZeroModules = 0;
  unsigned int max_InnerSeg = 0;

  // Allocate and copy nSegments from device to host (only nLowerModules in OT, not the +1 with pLSs)
  auto nSegments_buf_h = cms::alpakatools::make_host_buffer<unsigned int[]>(queue, nLowerModules_);
  alpaka::memcpy(queue, nSegments_buf_h, segmentsBuffers->nSegments_buf, nLowerModules_);

  // ... same for module_nConnectedModules
  // FIXME: replace by ES host data
  auto module_nConnectedModules_buf_h = cms::alpakatools::make_host_buffer<uint16_t[]>(queue, nLowerModules_);
  alpaka::memcpy(queue, module_nConnectedModules_buf_h, modulesBuffers_.nConnectedModules_buf, nLowerModules_);

  alpaka::wait(queue);  // wait for nSegments and module_nConnectedModules before using

  auto const* nSegments = nSegments_buf_h.data();
  auto const* module_nConnectedModules = module_nConnectedModules_buf_h.data();

  // Allocate host index and fill it directly
  auto index_buf_h = cms::alpakatools::make_host_buffer<uint16_t[]>(queue, nLowerModules_);
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
  auto index_gpu_buf = allocBufWrapper<uint16_t>(devAcc, nLowerModules_, queue);
  alpaka::memcpy(queue, index_gpu_buf, index_buf_h, nonZeroModules);

  Vec3D const threadsPerBlockCreateTrip{1, 16, 16};
  Vec3D const blocksPerGridCreateTrip{max_blocks, 1, 1};
  WorkDiv3D const createTripletsInGPUv2_workDiv =
      createWorkDiv(blocksPerGridCreateTrip, threadsPerBlockCreateTrip, elementsPerThread);

  alpaka::exec<Acc3D>(queue,
                      createTripletsInGPUv2_workDiv,
                      CreateTripletsInGPUv2{},
                      *modulesBuffers_.data(),
                      *mdsInGPU,
                      *segmentsInGPU,
                      *tripletsInGPU,
                      *rangesInGPU,
                      index_gpu_buf.data(),
                      nonZeroModules);

  WorkDiv1D const addTripletRangesToEventExplicit_workDiv = createWorkDiv<Vec1D>({1}, {1024}, {1});

  alpaka::exec<Acc1D>(queue,
                      addTripletRangesToEventExplicit_workDiv,
                      AddTripletRangesToEventExplicit{},
                      *modulesBuffers_.data(),
                      *tripletsInGPU,
                      *rangesInGPU);

  if (addObjects) {
    addTripletsToEventExplicit();
  }
}

void Event::createTrackCandidates(bool no_pls_dupclean, bool tc_pls_triplets) {
  if (!trackCandidatesInGPU) {
    trackCandidatesInGPU = TrackCandidates();
    trackCandidatesBuffers =
        TrackCandidatesBuffer<Device>(n_max_nonpixel_track_candidates + n_max_pixel_track_candidates, devAcc, queue);
    trackCandidatesInGPU->setData(*trackCandidatesBuffers);
  }

  Vec3D const threadsPerBlock_crossCleanpT3{1, 16, 64};
  Vec3D const blocksPerGrid_crossCleanpT3{1, 4, 20};
  WorkDiv3D const crossCleanpT3_workDiv =
      createWorkDiv(blocksPerGrid_crossCleanpT3, threadsPerBlock_crossCleanpT3, elementsPerThread);

  alpaka::exec<Acc3D>(queue,
                      crossCleanpT3_workDiv,
                      CrossCleanpT3{},
                      *modulesBuffers_.data(),
                      *rangesInGPU,
                      *pixelTripletsInGPU,
                      *segmentsInGPU,
                      *pixelQuintupletsInGPU);

  WorkDiv1D const addpT3asTrackCandidatesInGPU_workDiv = createWorkDiv<Vec1D>({1}, {512}, {1});

  alpaka::exec<Acc1D>(queue,
                      addpT3asTrackCandidatesInGPU_workDiv,
                      AddpT3asTrackCandidatesInGPU{},
                      nLowerModules_,
                      *pixelTripletsInGPU,
                      *trackCandidatesInGPU,
                      *segmentsInGPU,
                      *rangesInGPU);

  // Pull nEligibleT5Modules from the device.
  auto nEligibleModules_buf_h = cms::alpakatools::make_host_buffer<uint16_t[]>(queue, 1u);
  alpaka::memcpy(queue, nEligibleModules_buf_h, rangesBuffers->nEligibleT5Modules_buf);
  alpaka::wait(queue);  // wait to get the value before using
  auto const nEligibleModules = *nEligibleModules_buf_h.data();

  Vec3D const threadsPerBlockRemoveDupQuints{1, 16, 32};
  Vec3D const blocksPerGridRemoveDupQuints{1, std::max(nEligibleModules / 16, 1), std::max(nEligibleModules / 32, 1)};
  WorkDiv3D const removeDupQuintupletsInGPUBeforeTC_workDiv =
      createWorkDiv(blocksPerGridRemoveDupQuints, threadsPerBlockRemoveDupQuints, elementsPerThread);

  alpaka::exec<Acc3D>(queue,
                      removeDupQuintupletsInGPUBeforeTC_workDiv,
                      RemoveDupQuintupletsInGPUBeforeTC{},
                      *quintupletsInGPU,
                      *rangesInGPU);

  Vec3D const threadsPerBlock_crossCleanT5{32, 1, 32};
  Vec3D const blocksPerGrid_crossCleanT5{(13296 / 32) + 1, 1, max_blocks};
  WorkDiv3D const crossCleanT5_workDiv =
      createWorkDiv(blocksPerGrid_crossCleanT5, threadsPerBlock_crossCleanT5, elementsPerThread);

  alpaka::exec<Acc3D>(queue,
                      crossCleanT5_workDiv,
                      CrossCleanT5{},
                      *modulesBuffers_.data(),
                      *quintupletsInGPU,
                      *pixelQuintupletsInGPU,
                      *pixelTripletsInGPU,
                      *rangesInGPU);

  Vec3D const threadsPerBlock_addT5asTrackCandidateInGPU{1, 8, 128};
  Vec3D const blocksPerGrid_addT5asTrackCandidateInGPU{1, 8, 10};
  WorkDiv3D const addT5asTrackCandidateInGPU_workDiv = createWorkDiv(
      blocksPerGrid_addT5asTrackCandidateInGPU, threadsPerBlock_addT5asTrackCandidateInGPU, elementsPerThread);

  alpaka::exec<Acc3D>(queue,
                      addT5asTrackCandidateInGPU_workDiv,
                      AddT5asTrackCandidateInGPU{},
                      nLowerModules_,
                      *quintupletsInGPU,
                      *trackCandidatesInGPU,
                      *rangesInGPU);

  if (!no_pls_dupclean) {
    Vec3D const threadsPerBlockCheckHitspLS{1, 16, 16};
    Vec3D const blocksPerGridCheckHitspLS{1, max_blocks * 4, max_blocks / 4};
    WorkDiv3D const checkHitspLS_workDiv =
        createWorkDiv(blocksPerGridCheckHitspLS, threadsPerBlockCheckHitspLS, elementsPerThread);

    alpaka::exec<Acc3D>(queue, checkHitspLS_workDiv, CheckHitspLS{}, *modulesBuffers_.data(), *segmentsInGPU, true);
  }

  Vec3D const threadsPerBlock_crossCleanpLS{1, 16, 32};
  Vec3D const blocksPerGrid_crossCleanpLS{1, 4, 20};
  WorkDiv3D const crossCleanpLS_workDiv =
      createWorkDiv(blocksPerGrid_crossCleanpLS, threadsPerBlock_crossCleanpLS, elementsPerThread);

  alpaka::exec<Acc3D>(queue,
                      crossCleanpLS_workDiv,
                      CrossCleanpLS{},
                      *modulesBuffers_.data(),
                      *rangesInGPU,
                      *pixelTripletsInGPU,
                      *trackCandidatesInGPU,
                      *segmentsInGPU,
                      *mdsInGPU,
                      *hitsInGPU,
                      *quintupletsInGPU);

  Vec3D const threadsPerBlock_addpLSasTrackCandidateInGPU{1, 1, 384};
  Vec3D const blocksPerGrid_addpLSasTrackCandidateInGPU{1, 1, max_blocks};
  WorkDiv3D const addpLSasTrackCandidateInGPU_workDiv = createWorkDiv(
      blocksPerGrid_addpLSasTrackCandidateInGPU, threadsPerBlock_addpLSasTrackCandidateInGPU, elementsPerThread);

  alpaka::exec<Acc3D>(queue,
                      addpLSasTrackCandidateInGPU_workDiv,
                      AddpLSasTrackCandidateInGPU{},
                      nLowerModules_,
                      *trackCandidatesInGPU,
                      *segmentsInGPU,
                      tc_pls_triplets);

  // Check if either n_max_pixel_track_candidates or n_max_nonpixel_track_candidates was reached
  auto nTrackCanpT5Host_buf = allocBufWrapper<unsigned int>(devHost, 1, queue);
  auto nTrackCanpT3Host_buf = allocBufWrapper<unsigned int>(devHost, 1, queue);
  auto nTrackCanpLSHost_buf = allocBufWrapper<unsigned int>(devHost, 1, queue);
  auto nTrackCanT5Host_buf = allocBufWrapper<unsigned int>(devHost, 1, queue);
  alpaka::memcpy(queue, nTrackCanpT5Host_buf, trackCandidatesBuffers->nTrackCandidatespT5_buf);
  alpaka::memcpy(queue, nTrackCanpT3Host_buf, trackCandidatesBuffers->nTrackCandidatespT3_buf);
  alpaka::memcpy(queue, nTrackCanpLSHost_buf, trackCandidatesBuffers->nTrackCandidatespLS_buf);
  alpaka::memcpy(queue, nTrackCanT5Host_buf, trackCandidatesBuffers->nTrackCandidatesT5_buf);
  alpaka::wait(queue);  // wait to get the values before using them

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
  if (!pixelTripletsInGPU) {
    pixelTripletsInGPU = PixelTriplets();
    pixelTripletsBuffers = PixelTripletsBuffer<Device>(n_max_pixel_triplets, devAcc, queue);
    pixelTripletsInGPU->setData(*pixelTripletsBuffers);
  }

  auto superbins_buf = allocBufWrapper<int>(devHost, n_max_pixel_segments_per_module, queue);
  auto pixelTypes_buf = allocBufWrapper<PixelType>(devHost, n_max_pixel_segments_per_module, queue);

  alpaka::memcpy(queue, superbins_buf, segmentsBuffers->superbin_buf);
  alpaka::memcpy(queue, pixelTypes_buf, segmentsBuffers->pixelType_buf);
  auto const* superbins = superbins_buf.data();
  auto const* pixelTypes = pixelTypes_buf.data();

  unsigned int nInnerSegments;
  auto nInnerSegments_src_view = alpaka::createView(devHost, &nInnerSegments, (size_t)1u);

  // Create a sub-view for the device buffer
  auto dev_view_nSegments = alpaka::createSubView(segmentsBuffers->nSegments_buf, (Idx)1u, (Idx)nLowerModules_);

  alpaka::memcpy(queue, nInnerSegments_src_view, dev_view_nSegments);
  alpaka::wait(queue);  // wait to get nInnerSegments (also superbins and pixelTypes) before using

  auto connectedPixelSize_host_buf = allocBufWrapper<unsigned int>(devHost, nInnerSegments, queue);
  auto connectedPixelIndex_host_buf = allocBufWrapper<unsigned int>(devHost, nInnerSegments, queue);
  auto connectedPixelSize_dev_buf = allocBufWrapper<unsigned int>(devAcc, nInnerSegments, queue);
  auto connectedPixelIndex_dev_buf = allocBufWrapper<unsigned int>(devAcc, nInnerSegments, queue);

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

  alpaka::memcpy(queue, connectedPixelSize_dev_buf, connectedPixelSize_host_buf, nInnerSegments);
  alpaka::memcpy(queue, connectedPixelIndex_dev_buf, connectedPixelIndex_host_buf, nInnerSegments);

  Vec3D const threadsPerBlock{1, 4, 32};
  Vec3D const blocksPerGrid{16 /* above median of connected modules*/, 4096, 1};
  WorkDiv3D const createPixelTripletsInGPUFromMapv2_workDiv =
      createWorkDiv(blocksPerGrid, threadsPerBlock, elementsPerThread);

  alpaka::exec<Acc3D>(queue,
                      createPixelTripletsInGPUFromMapv2_workDiv,
                      CreatePixelTripletsInGPUFromMapv2{},
                      *modulesBuffers_.data(),
                      *rangesInGPU,
                      *mdsInGPU,
                      *segmentsInGPU,
                      *tripletsInGPU,
                      *pixelTripletsInGPU,
                      connectedPixelSize_dev_buf.data(),
                      connectedPixelIndex_dev_buf.data(),
                      nInnerSegments);

#ifdef WARNINGS
  auto nPixelTriplets_buf = allocBufWrapper<unsigned int>(devHost, 1, queue);

  alpaka::memcpy(queue, nPixelTriplets_buf, pixelTripletsBuffers->nPixelTriplets_buf);
  alpaka::wait(queue);  // wait to get the value before using it

  std::cout << "number of pixel triplets = " << *nPixelTriplets_buf.data() << std::endl;
#endif

  //pT3s can be cleaned here because they're not used in making pT5s!
  Vec3D const threadsPerBlockDupPixTrip{1, 16, 16};
  //seems like more blocks lead to conflicting writes
  Vec3D const blocksPerGridDupPixTrip{1, 40, 1};
  WorkDiv3D const removeDupPixelTripletsInGPUFromMap_workDiv =
      createWorkDiv(blocksPerGridDupPixTrip, threadsPerBlockDupPixTrip, elementsPerThread);

  alpaka::exec<Acc3D>(
      queue, removeDupPixelTripletsInGPUFromMap_workDiv, RemoveDupPixelTripletsInGPUFromMap{}, *pixelTripletsInGPU);
}

void Event::createQuintuplets() {
  WorkDiv1D const createEligibleModulesListForQuintupletsGPU_workDiv = createWorkDiv<Vec1D>({1}, {1024}, {1});

  alpaka::exec<Acc1D>(queue,
                      createEligibleModulesListForQuintupletsGPU_workDiv,
                      CreateEligibleModulesListForQuintupletsGPU{},
                      *modulesBuffers_.data(),
                      *tripletsInGPU,
                      *rangesInGPU);

  auto nEligibleT5Modules_buf = allocBufWrapper<uint16_t>(devHost, 1, queue);
  auto nTotalQuintuplets_buf = allocBufWrapper<unsigned int>(devHost, 1, queue);

  alpaka::memcpy(queue, nEligibleT5Modules_buf, rangesBuffers->nEligibleT5Modules_buf);
  alpaka::memcpy(queue, nTotalQuintuplets_buf, rangesBuffers->device_nTotalQuints_buf);
  alpaka::wait(queue);  // wait for the values before using them

  auto nEligibleT5Modules = *nEligibleT5Modules_buf.data();
  auto nTotalQuintuplets = *nTotalQuintuplets_buf.data();

  if (!quintupletsInGPU) {
    quintupletsInGPU = Quintuplets();
    quintupletsBuffers = QuintupletsBuffer<Device>(nTotalQuintuplets, nLowerModules_, devAcc, queue);
    quintupletsInGPU->setData(*quintupletsBuffers);

    alpaka::memcpy(queue, quintupletsBuffers->nMemoryLocations_buf, nTotalQuintuplets_buf);
  }

  Vec3D const threadsPerBlockQuints{1, 8, 32};
  Vec3D const blocksPerGridQuints{std::max((int)nEligibleT5Modules, 1), 1, 1};
  WorkDiv3D const createQuintupletsInGPUv2_workDiv =
      createWorkDiv(blocksPerGridQuints, threadsPerBlockQuints, elementsPerThread);

  alpaka::exec<Acc3D>(queue,
                      createQuintupletsInGPUv2_workDiv,
                      CreateQuintupletsInGPUv2{},
                      *modulesBuffers_.data(),
                      *mdsInGPU,
                      *segmentsInGPU,
                      *tripletsInGPU,
                      *quintupletsInGPU,
                      *rangesInGPU,
                      nEligibleT5Modules);

  Vec3D const threadsPerBlockDupQuint{1, 16, 16};
  Vec3D const blocksPerGridDupQuint{max_blocks, 1, 1};
  WorkDiv3D const removeDupQuintupletsInGPUAfterBuild_workDiv =
      createWorkDiv(blocksPerGridDupQuint, threadsPerBlockDupQuint, elementsPerThread);

  alpaka::exec<Acc3D>(queue,
                      removeDupQuintupletsInGPUAfterBuild_workDiv,
                      RemoveDupQuintupletsInGPUAfterBuild{},
                      *modulesBuffers_.data(),
                      *quintupletsInGPU,
                      *rangesInGPU);

  WorkDiv1D const addQuintupletRangesToEventExplicit_workDiv = createWorkDiv<Vec1D>({1}, {1024}, {1});

  alpaka::exec<Acc1D>(queue,
                      addQuintupletRangesToEventExplicit_workDiv,
                      AddQuintupletRangesToEventExplicit{},
                      *modulesBuffers_.data(),
                      *quintupletsInGPU,
                      *rangesInGPU);

  if (addObjects) {
    addQuintupletsToEventExplicit();
  }
}

void Event::pixelLineSegmentCleaning(bool no_pls_dupclean) {
  if (!no_pls_dupclean) {
    Vec3D const threadsPerBlockCheckHitspLS{1, 16, 16};
    Vec3D const blocksPerGridCheckHitspLS{1, max_blocks * 4, max_blocks / 4};
    WorkDiv3D const checkHitspLS_workDiv =
        createWorkDiv(blocksPerGridCheckHitspLS, threadsPerBlockCheckHitspLS, elementsPerThread);

    alpaka::exec<Acc3D>(queue, checkHitspLS_workDiv, CheckHitspLS{}, *modulesBuffers_.data(), *segmentsInGPU, false);
  }
}

void Event::createPixelQuintuplets() {
  if (!pixelQuintupletsInGPU) {
    pixelQuintupletsInGPU = PixelQuintuplets();
    pixelQuintupletsBuffers = PixelQuintupletsBuffer<Device>(n_max_pixel_quintuplets, devAcc, queue);
    pixelQuintupletsInGPU->setData(*pixelQuintupletsBuffers);
  }
  if (!trackCandidatesInGPU) {
    trackCandidatesInGPU = TrackCandidates();
    trackCandidatesBuffers =
        TrackCandidatesBuffer<Device>(n_max_nonpixel_track_candidates + n_max_pixel_track_candidates, devAcc, queue);
    trackCandidatesInGPU->setData(*trackCandidatesBuffers);
  }

  auto superbins_buf = allocBufWrapper<int>(devHost, n_max_pixel_segments_per_module, queue);
  auto pixelTypes_buf = allocBufWrapper<PixelType>(devHost, n_max_pixel_segments_per_module, queue);

  alpaka::memcpy(queue, superbins_buf, segmentsBuffers->superbin_buf);
  alpaka::memcpy(queue, pixelTypes_buf, segmentsBuffers->pixelType_buf);
  auto const* superbins = superbins_buf.data();
  auto const* pixelTypes = pixelTypes_buf.data();

  unsigned int nInnerSegments;
  auto nInnerSegments_src_view = alpaka::createView(devHost, &nInnerSegments, (size_t)1u);

  // Create a sub-view for the device buffer
  auto dev_view_nSegments = alpaka::createSubView(segmentsBuffers->nSegments_buf, (Idx)1u, (Idx)nLowerModules_);

  alpaka::memcpy(queue, nInnerSegments_src_view, dev_view_nSegments);
  alpaka::wait(queue);  // wait to get nInnerSegments (also superbins and pixelTypes) before using

  auto connectedPixelSize_host_buf = allocBufWrapper<unsigned int>(devHost, nInnerSegments, queue);
  auto connectedPixelIndex_host_buf = allocBufWrapper<unsigned int>(devHost, nInnerSegments, queue);
  auto connectedPixelSize_dev_buf = allocBufWrapper<unsigned int>(devAcc, nInnerSegments, queue);
  auto connectedPixelIndex_dev_buf = allocBufWrapper<unsigned int>(devAcc, nInnerSegments, queue);

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

  alpaka::memcpy(queue, connectedPixelSize_dev_buf, connectedPixelSize_host_buf, nInnerSegments);
  alpaka::memcpy(queue, connectedPixelIndex_dev_buf, connectedPixelIndex_host_buf, nInnerSegments);

  Vec3D const threadsPerBlockCreatePixQuints{1, 16, 16};
  Vec3D const blocksPerGridCreatePixQuints{16, max_blocks, 1};
  WorkDiv3D const createPixelQuintupletsInGPUFromMapv2_workDiv =
      createWorkDiv(blocksPerGridCreatePixQuints, threadsPerBlockCreatePixQuints, elementsPerThread);

  alpaka::exec<Acc3D>(queue,
                      createPixelQuintupletsInGPUFromMapv2_workDiv,
                      CreatePixelQuintupletsInGPUFromMapv2{},
                      *modulesBuffers_.data(),
                      *mdsInGPU,
                      *segmentsInGPU,
                      *tripletsInGPU,
                      *quintupletsInGPU,
                      *pixelQuintupletsInGPU,
                      connectedPixelSize_dev_buf.data(),
                      connectedPixelIndex_dev_buf.data(),
                      nInnerSegments,
                      *rangesInGPU);

  Vec3D const threadsPerBlockDupPix{1, 16, 16};
  Vec3D const blocksPerGridDupPix{1, max_blocks, 1};
  WorkDiv3D const removeDupPixelQuintupletsInGPUFromMap_workDiv =
      createWorkDiv(blocksPerGridDupPix, threadsPerBlockDupPix, elementsPerThread);

  alpaka::exec<Acc3D>(queue,
                      removeDupPixelQuintupletsInGPUFromMap_workDiv,
                      RemoveDupPixelQuintupletsInGPUFromMap{},
                      *pixelQuintupletsInGPU);

  WorkDiv1D const addpT5asTrackCandidateInGPU_workDiv = createWorkDiv<Vec1D>({1}, {256}, {1});

  alpaka::exec<Acc1D>(queue,
                      addpT5asTrackCandidateInGPU_workDiv,
                      AddpT5asTrackCandidateInGPU{},
                      nLowerModules_,
                      *pixelQuintupletsInGPU,
                      *trackCandidatesInGPU,
                      *segmentsInGPU,
                      *rangesInGPU);

#ifdef WARNINGS
  auto nPixelQuintuplets_buf = allocBufWrapper<unsigned int>(devHost, 1, queue);

  alpaka::memcpy(queue, nPixelQuintuplets_buf, pixelQuintupletsBuffers->nPixelQuintuplets_buf);
  alpaka::wait(queue);  // wait to get the value before using it

  std::cout << "number of pixel quintuplets = " << *nPixelQuintuplets_buf.data() << std::endl;
#endif
}

void Event::addMiniDoubletsToEventExplicit() {
  auto nMDsCPU_buf = allocBufWrapper<unsigned int>(devHost, nLowerModules_, queue);
  alpaka::memcpy(queue, nMDsCPU_buf, miniDoubletsBuffers->nMDs_buf, nLowerModules_);

  // FIXME: replace by ES host data
  auto module_subdets_buf = allocBufWrapper<short>(devHost, nLowerModules_, queue);
  alpaka::memcpy(queue, module_subdets_buf, modulesBuffers_.subdets_buf, nLowerModules_);

  auto module_layers_buf = allocBufWrapper<short>(devHost, nLowerModules_, queue);
  alpaka::memcpy(queue, module_layers_buf, modulesBuffers_.layers_buf, nLowerModules_);

  auto module_hitRanges_buf = allocBufWrapper<int>(devHost, nLowerModules_ * 2, queue);
  alpaka::memcpy(queue, module_hitRanges_buf, hitsBuffers->hitRanges_buf, nLowerModules_ * 2u);

  alpaka::wait(queue);  // wait for inputs before using them

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
  auto nSegmentsCPU_buf = allocBufWrapper<unsigned int>(devHost, nLowerModules_, queue);
  alpaka::memcpy(queue, nSegmentsCPU_buf, segmentsBuffers->nSegments_buf, nLowerModules_);

  // FIXME: replace by ES host data
  auto module_subdets_buf = allocBufWrapper<short>(devHost, nLowerModules_, queue);
  alpaka::memcpy(queue, module_subdets_buf, modulesBuffers_.subdets_buf, nLowerModules_);

  auto module_layers_buf = allocBufWrapper<short>(devHost, nLowerModules_, queue);
  alpaka::memcpy(queue, module_layers_buf, modulesBuffers_.layers_buf, nLowerModules_);

  alpaka::wait(queue);  // wait for inputs before using them

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
  auto nQuintupletsCPU_buf = allocBufWrapper<unsigned int>(devHost, nLowerModules_, queue);
  alpaka::memcpy(queue, nQuintupletsCPU_buf, quintupletsBuffers->nQuintuplets_buf);

  // FIXME: replace by ES host data
  auto module_subdets_buf = allocBufWrapper<short>(devHost, nModules_, queue);
  alpaka::memcpy(queue, module_subdets_buf, modulesBuffers_.subdets_buf, nModules_);

  auto module_layers_buf = allocBufWrapper<short>(devHost, nLowerModules_, queue);
  alpaka::memcpy(queue, module_layers_buf, modulesBuffers_.layers_buf, nLowerModules_);

  auto module_quintupletModuleIndices_buf = allocBufWrapper<int>(devHost, nLowerModules_, queue);
  alpaka::memcpy(queue, module_quintupletModuleIndices_buf, rangesBuffers->quintupletModuleIndices_buf);

  alpaka::wait(queue);  // wait for inputs before using them

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
  auto nTripletsCPU_buf = allocBufWrapper<unsigned int>(devHost, nLowerModules_, queue);
  alpaka::memcpy(queue, nTripletsCPU_buf, tripletsBuffers->nTriplets_buf);

  // FIXME: replace by ES host data
  auto module_subdets_buf = allocBufWrapper<short>(devHost, nLowerModules_, queue);
  alpaka::memcpy(queue, module_subdets_buf, modulesBuffers_.subdets_buf, nLowerModules_);

  auto module_layers_buf = allocBufWrapper<short>(devHost, nLowerModules_, queue);
  alpaka::memcpy(queue, module_layers_buf, modulesBuffers_.layers_buf, nLowerModules_);

  alpaka::wait(queue);  // wait for inputs before using them

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
  auto nPixelTriplets_buf_h = cms::alpakatools::make_host_buffer<unsigned int[]>(queue, 1u);

  alpaka::memcpy(queue, nPixelTriplets_buf_h, pixelTripletsBuffers->nPixelTriplets_buf);
  alpaka::wait(queue);

  return *nPixelTriplets_buf_h.data();
}

int Event::getNumberOfPixelQuintuplets() {
  auto nPixelQuintuplets_buf_h = cms::alpakatools::make_host_buffer<unsigned int[]>(queue, 1u);

  alpaka::memcpy(queue, nPixelQuintuplets_buf_h, pixelQuintupletsBuffers->nPixelQuintuplets_buf);
  alpaka::wait(queue);

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
  auto nTrackCandidates_buf_h = cms::alpakatools::make_host_buffer<unsigned int[]>(queue, 1u);

  alpaka::memcpy(queue, nTrackCandidates_buf_h, trackCandidatesBuffers->nTrackCandidates_buf);
  alpaka::wait(queue);

  return *nTrackCandidates_buf_h.data();
}

int Event::getNumberOfPT5TrackCandidates() {
  auto nTrackCandidatesPT5_buf_h = cms::alpakatools::make_host_buffer<unsigned int[]>(queue, 1u);

  alpaka::memcpy(queue, nTrackCandidatesPT5_buf_h, trackCandidatesBuffers->nTrackCandidatespT5_buf);
  alpaka::wait(queue);

  return *nTrackCandidatesPT5_buf_h.data();
}

int Event::getNumberOfPT3TrackCandidates() {
  auto nTrackCandidatesPT3_buf_h = cms::alpakatools::make_host_buffer<unsigned int[]>(queue, 1u);

  alpaka::memcpy(queue, nTrackCandidatesPT3_buf_h, trackCandidatesBuffers->nTrackCandidatespT3_buf);
  alpaka::wait(queue);

  return *nTrackCandidatesPT3_buf_h.data();
}

int Event::getNumberOfPLSTrackCandidates() {
  auto nTrackCandidatesPLS_buf_h = cms::alpakatools::make_host_buffer<unsigned int[]>(queue, 1u);

  alpaka::memcpy(queue, nTrackCandidatesPLS_buf_h, trackCandidatesBuffers->nTrackCandidatespLS_buf);
  alpaka::wait(queue);

  return *nTrackCandidatesPLS_buf_h.data();
}

int Event::getNumberOfPixelTrackCandidates() {
  auto nTrackCandidates_buf_h = cms::alpakatools::make_host_buffer<unsigned int[]>(queue, 1u);
  auto nTrackCandidatesT5_buf_h = cms::alpakatools::make_host_buffer<unsigned int[]>(queue, 1u);

  alpaka::memcpy(queue, nTrackCandidates_buf_h, trackCandidatesBuffers->nTrackCandidates_buf);
  alpaka::memcpy(queue, nTrackCandidatesT5_buf_h, trackCandidatesBuffers->nTrackCandidatesT5_buf);
  alpaka::wait(queue);

  return (*nTrackCandidates_buf_h.data()) - (*nTrackCandidatesT5_buf_h.data());
}

int Event::getNumberOfT5TrackCandidates() {
  auto nTrackCandidatesT5_buf_h = cms::alpakatools::make_host_buffer<unsigned int[]>(queue, 1u);

  alpaka::memcpy(queue, nTrackCandidatesT5_buf_h, trackCandidatesBuffers->nTrackCandidatesT5_buf);
  alpaka::wait(queue);

  return *nTrackCandidatesT5_buf_h.data();
}

HitsBuffer<alpaka_common::DevHost>& Event::getHits(bool sync)  //std::shared_ptr should take care of garbage collection
{
  if (!hitsInCPU) {
    auto nHits_buf_h = cms::alpakatools::make_host_buffer<unsigned int[]>(queue, 1u);
    alpaka::memcpy(queue, nHits_buf_h, hitsBuffers->nHits_buf);
    alpaka::wait(queue);  // wait for the value before using

    auto const nHits = *nHits_buf_h.data();
    hitsInCPU = HitsBuffer<DevHost>(nModules_, nHits, devHost, queue);
    hitsInCPU->setData(*hitsInCPU);

    alpaka::memcpy(queue, hitsInCPU->nHits_buf, hitsBuffers->nHits_buf);
    alpaka::memcpy(queue, hitsInCPU->idxs_buf, hitsBuffers->idxs_buf, nHits);
    alpaka::memcpy(queue, hitsInCPU->detid_buf, hitsBuffers->detid_buf, nHits);
    alpaka::memcpy(queue, hitsInCPU->xs_buf, hitsBuffers->xs_buf, nHits);
    alpaka::memcpy(queue, hitsInCPU->ys_buf, hitsBuffers->ys_buf, nHits);
    alpaka::memcpy(queue, hitsInCPU->zs_buf, hitsBuffers->zs_buf, nHits);
    alpaka::memcpy(queue, hitsInCPU->moduleIndices_buf, hitsBuffers->moduleIndices_buf, nHits);
    if (sync)
      alpaka::wait(queue);  // host consumers expect filled data
  }
  return hitsInCPU.value();
}

HitsBuffer<alpaka_common::DevHost>& Event::getHitsInCMSSW(bool sync) {
  if (!hitsInCPU) {
    auto nHits_buf_h = cms::alpakatools::make_host_buffer<unsigned int[]>(queue, 1u);
    alpaka::memcpy(queue, nHits_buf_h, hitsBuffers->nHits_buf);
    alpaka::wait(queue);  // wait for the value before using

    auto const nHits = *nHits_buf_h.data();
    hitsInCPU = HitsBuffer<DevHost>(nModules_, nHits, devHost, queue);
    hitsInCPU->setData(*hitsInCPU);

    alpaka::memcpy(queue, hitsInCPU->nHits_buf, hitsBuffers->nHits_buf);
    alpaka::memcpy(queue, hitsInCPU->idxs_buf, hitsBuffers->idxs_buf, nHits);
    if (sync)
      alpaka::wait(queue);  // host consumers expect filled data
  }
  return hitsInCPU.value();
}

ObjectRangesBuffer<alpaka_common::DevHost>& Event::getRanges(bool sync) {
  if (!rangesInCPU) {
    rangesInCPU = ObjectRangesBuffer<DevHost>(nModules_, nLowerModules_, devHost, queue);
    rangesInCPU->setData(*rangesInCPU);

    alpaka::memcpy(queue, rangesInCPU->hitRanges_buf, rangesBuffers->hitRanges_buf);
    alpaka::memcpy(queue, rangesInCPU->quintupletModuleIndices_buf, rangesBuffers->quintupletModuleIndices_buf);
    alpaka::memcpy(queue, rangesInCPU->miniDoubletModuleIndices_buf, rangesBuffers->miniDoubletModuleIndices_buf);
    alpaka::memcpy(queue, rangesInCPU->segmentModuleIndices_buf, rangesBuffers->segmentModuleIndices_buf);
    alpaka::memcpy(queue, rangesInCPU->tripletModuleIndices_buf, rangesBuffers->tripletModuleIndices_buf);
    if (sync)
      alpaka::wait(queue);  // wait to get completed host data
  }
  return rangesInCPU.value();
}

MiniDoubletsBuffer<alpaka_common::DevHost>& Event::getMiniDoublets(bool sync) {
  if (!mdsInCPU) {
    // Get nMemoryLocations parameter to initialize host based mdsInCPU
    auto nMemHost_buf_h = cms::alpakatools::make_host_buffer<unsigned int[]>(queue, 1u);
    alpaka::memcpy(queue, nMemHost_buf_h, miniDoubletsBuffers->nMemoryLocations_buf);
    alpaka::wait(queue);  // wait for the value before using

    auto const nMemHost = *nMemHost_buf_h.data();
    mdsInCPU = MiniDoubletsBuffer<DevHost>(nMemHost, nLowerModules_, devHost, queue);
    mdsInCPU->setData(*mdsInCPU);

    alpaka::memcpy(queue, mdsInCPU->nMemoryLocations_buf, miniDoubletsBuffers->nMemoryLocations_buf);
    alpaka::memcpy(queue, mdsInCPU->anchorHitIndices_buf, miniDoubletsBuffers->anchorHitIndices_buf, nMemHost);
    alpaka::memcpy(queue, mdsInCPU->outerHitIndices_buf, miniDoubletsBuffers->outerHitIndices_buf, nMemHost);
    alpaka::memcpy(queue, mdsInCPU->dphichanges_buf, miniDoubletsBuffers->dphichanges_buf, nMemHost);
    alpaka::memcpy(queue, mdsInCPU->nMDs_buf, miniDoubletsBuffers->nMDs_buf);
    alpaka::memcpy(queue, mdsInCPU->totOccupancyMDs_buf, miniDoubletsBuffers->totOccupancyMDs_buf);
    if (sync)
      alpaka::wait(queue);  // host consumers expect filled data
  }
  return mdsInCPU.value();
}

SegmentsBuffer<alpaka_common::DevHost>& Event::getSegments(bool sync) {
  if (!segmentsInCPU) {
    // Get nMemoryLocations parameter to initialize host based segmentsInCPU
    auto nMemHost_buf_h = cms::alpakatools::make_host_buffer<unsigned int[]>(queue, 1u);
    alpaka::memcpy(queue, nMemHost_buf_h, segmentsBuffers->nMemoryLocations_buf);
    alpaka::wait(queue);  // wait for the value before using

    auto const nMemHost = *nMemHost_buf_h.data();
    segmentsInCPU = SegmentsBuffer<DevHost>(nMemHost, nLowerModules_, n_max_pixel_segments_per_module, devHost, queue);
    segmentsInCPU->setData(*segmentsInCPU);

    alpaka::memcpy(queue, segmentsInCPU->nMemoryLocations_buf, segmentsBuffers->nMemoryLocations_buf);
    alpaka::memcpy(queue, segmentsInCPU->nSegments_buf, segmentsBuffers->nSegments_buf);
    alpaka::memcpy(queue, segmentsInCPU->mdIndices_buf, segmentsBuffers->mdIndices_buf, 2u * nMemHost);
    alpaka::memcpy(queue,
                   segmentsInCPU->innerMiniDoubletAnchorHitIndices_buf,
                   segmentsBuffers->innerMiniDoubletAnchorHitIndices_buf,
                   nMemHost);
    alpaka::memcpy(queue,
                   segmentsInCPU->outerMiniDoubletAnchorHitIndices_buf,
                   segmentsBuffers->outerMiniDoubletAnchorHitIndices_buf,
                   nMemHost);
    alpaka::memcpy(queue, segmentsInCPU->totOccupancySegments_buf, segmentsBuffers->totOccupancySegments_buf);
    alpaka::memcpy(queue, segmentsInCPU->ptIn_buf, segmentsBuffers->ptIn_buf);
    alpaka::memcpy(queue, segmentsInCPU->eta_buf, segmentsBuffers->eta_buf);
    alpaka::memcpy(queue, segmentsInCPU->phi_buf, segmentsBuffers->phi_buf);
    alpaka::memcpy(queue, segmentsInCPU->seedIdx_buf, segmentsBuffers->seedIdx_buf);
    alpaka::memcpy(queue, segmentsInCPU->isDup_buf, segmentsBuffers->isDup_buf);
    alpaka::memcpy(queue, segmentsInCPU->isQuad_buf, segmentsBuffers->isQuad_buf);
    alpaka::memcpy(queue, segmentsInCPU->score_buf, segmentsBuffers->score_buf);
    if (sync)
      alpaka::wait(queue);  // host consumers expect filled data
  }
  return segmentsInCPU.value();
}

TripletsBuffer<alpaka_common::DevHost>& Event::getTriplets(bool sync) {
  if (!tripletsInCPU) {
    // Get nMemoryLocations parameter to initialize host based tripletsInCPU
    auto nMemHost_buf_h = cms::alpakatools::make_host_buffer<unsigned int[]>(queue, 1u);
    alpaka::memcpy(queue, nMemHost_buf_h, tripletsBuffers->nMemoryLocations_buf);
    alpaka::wait(queue);  // wait for the value before using

    auto const nMemHost = *nMemHost_buf_h.data();
    tripletsInCPU = TripletsBuffer<DevHost>(nMemHost, nLowerModules_, devHost, queue);
    tripletsInCPU->setData(*tripletsInCPU);

    alpaka::memcpy(queue, tripletsInCPU->nMemoryLocations_buf, tripletsBuffers->nMemoryLocations_buf);
#ifdef CUT_VALUE_DEBUG
    alpaka::memcpy(queue, tripletsInCPU->zOut_buf, tripletsBuffers->zOut_buf, nMemHost);
    alpaka::memcpy(queue, tripletsInCPU->zLo_buf, tripletsBuffers->zLo_buf, nMemHost);
    alpaka::memcpy(queue, tripletsInCPU->zHi_buf, tripletsBuffers->zHi_buf, nMemHost);
    alpaka::memcpy(queue, tripletsInCPU->zLoPointed_buf, tripletsBuffers->zLoPointed_buf, nMemHost);
    alpaka::memcpy(queue, tripletsInCPU->zHiPointed_buf, tripletsBuffers->zHiPointed_buf, nMemHost);
    alpaka::memcpy(queue, tripletsInCPU->dPhiCut_buf, tripletsBuffers->dPhiCut_buf, nMemHost);
    alpaka::memcpy(queue, tripletsInCPU->betaInCut_buf, tripletsBuffers->betaInCut_buf, nMemHost);
    alpaka::memcpy(queue, tripletsInCPU->rtLo_buf, tripletsBuffers->rtLo_buf, nMemHost);
    alpaka::memcpy(queue, tripletsInCPU->rtHi_buf, tripletsBuffers->rtHi_buf, nMemHost);
#endif
    alpaka::memcpy(queue, tripletsInCPU->hitIndices_buf, tripletsBuffers->hitIndices_buf, Params_T3::kHits * nMemHost);
    alpaka::memcpy(
        queue, tripletsInCPU->logicalLayers_buf, tripletsBuffers->logicalLayers_buf, Params_T3::kLayers * nMemHost);
    alpaka::memcpy(queue, tripletsInCPU->segmentIndices_buf, tripletsBuffers->segmentIndices_buf, 2 * nMemHost);
    alpaka::memcpy(queue, tripletsInCPU->betaIn_buf, tripletsBuffers->betaIn_buf, nMemHost);
    alpaka::memcpy(queue, tripletsInCPU->circleRadius_buf, tripletsBuffers->circleRadius_buf, nMemHost);
    alpaka::memcpy(queue, tripletsInCPU->nTriplets_buf, tripletsBuffers->nTriplets_buf);
    alpaka::memcpy(queue, tripletsInCPU->totOccupancyTriplets_buf, tripletsBuffers->totOccupancyTriplets_buf);
    if (sync)
      alpaka::wait(queue);  // host consumers expect filled data
  }
  return tripletsInCPU.value();
}

QuintupletsBuffer<alpaka_common::DevHost>& Event::getQuintuplets(bool sync) {
  if (!quintupletsInCPU) {
    // Get nMemoryLocations parameter to initialize host based quintupletsInCPU
    auto nMemHost_buf_h = cms::alpakatools::make_host_buffer<unsigned int[]>(queue, 1u);
    alpaka::memcpy(queue, nMemHost_buf_h, quintupletsBuffers->nMemoryLocations_buf);
    alpaka::wait(queue);  // wait for the value before using

    auto const nMemHost = *nMemHost_buf_h.data();
    quintupletsInCPU = QuintupletsBuffer<DevHost>(nMemHost, nLowerModules_, devHost, queue);
    quintupletsInCPU->setData(*quintupletsInCPU);

    alpaka::memcpy(queue, quintupletsInCPU->nMemoryLocations_buf, quintupletsBuffers->nMemoryLocations_buf);
    alpaka::memcpy(queue, quintupletsInCPU->nQuintuplets_buf, quintupletsBuffers->nQuintuplets_buf);
    alpaka::memcpy(
        queue, quintupletsInCPU->totOccupancyQuintuplets_buf, quintupletsBuffers->totOccupancyQuintuplets_buf);
    alpaka::memcpy(queue, quintupletsInCPU->tripletIndices_buf, quintupletsBuffers->tripletIndices_buf, 2 * nMemHost);
    alpaka::memcpy(queue,
                   quintupletsInCPU->lowerModuleIndices_buf,
                   quintupletsBuffers->lowerModuleIndices_buf,
                   Params_T5::kLayers * nMemHost);
    alpaka::memcpy(queue, quintupletsInCPU->innerRadius_buf, quintupletsBuffers->innerRadius_buf, nMemHost);
    alpaka::memcpy(queue, quintupletsInCPU->bridgeRadius_buf, quintupletsBuffers->bridgeRadius_buf, nMemHost);
    alpaka::memcpy(queue, quintupletsInCPU->outerRadius_buf, quintupletsBuffers->outerRadius_buf, nMemHost);
    alpaka::memcpy(queue, quintupletsInCPU->isDup_buf, quintupletsBuffers->isDup_buf, nMemHost);
    alpaka::memcpy(queue, quintupletsInCPU->score_rphisum_buf, quintupletsBuffers->score_rphisum_buf, nMemHost);
    alpaka::memcpy(queue, quintupletsInCPU->eta_buf, quintupletsBuffers->eta_buf, nMemHost);
    alpaka::memcpy(queue, quintupletsInCPU->phi_buf, quintupletsBuffers->phi_buf, nMemHost);
    alpaka::memcpy(queue, quintupletsInCPU->chiSquared_buf, quintupletsBuffers->chiSquared_buf, nMemHost);
    alpaka::memcpy(queue, quintupletsInCPU->rzChiSquared_buf, quintupletsBuffers->rzChiSquared_buf, nMemHost);
    alpaka::memcpy(
        queue, quintupletsInCPU->nonAnchorChiSquared_buf, quintupletsBuffers->nonAnchorChiSquared_buf, nMemHost);
    if (sync)
      alpaka::wait(queue);  // host consumers expect filled data
  }
  return quintupletsInCPU.value();
}

PixelTripletsBuffer<alpaka_common::DevHost>& Event::getPixelTriplets(bool sync) {
  if (!pixelTripletsInCPU) {
    // Get nPixelTriplets parameter to initialize host based quintupletsInCPU
    auto nPixelTriplets_buf_h = cms::alpakatools::make_host_buffer<unsigned int[]>(queue, 1u);
    alpaka::memcpy(queue, nPixelTriplets_buf_h, pixelTripletsBuffers->nPixelTriplets_buf);
    alpaka::wait(queue);  // wait for the value before using

    auto const nPixelTriplets = *nPixelTriplets_buf_h.data();
    pixelTripletsInCPU = PixelTripletsBuffer<DevHost>(nPixelTriplets, devHost, queue);
    pixelTripletsInCPU->setData(*pixelTripletsInCPU);

    alpaka::memcpy(queue, pixelTripletsInCPU->nPixelTriplets_buf, pixelTripletsBuffers->nPixelTriplets_buf);
    alpaka::memcpy(
        queue, pixelTripletsInCPU->totOccupancyPixelTriplets_buf, pixelTripletsBuffers->totOccupancyPixelTriplets_buf);
    alpaka::memcpy(queue, pixelTripletsInCPU->rzChiSquared_buf, pixelTripletsBuffers->rzChiSquared_buf, nPixelTriplets);
    alpaka::memcpy(
        queue, pixelTripletsInCPU->rPhiChiSquared_buf, pixelTripletsBuffers->rPhiChiSquared_buf, nPixelTriplets);
    alpaka::memcpy(queue,
                   pixelTripletsInCPU->rPhiChiSquaredInwards_buf,
                   pixelTripletsBuffers->rPhiChiSquaredInwards_buf,
                   nPixelTriplets);
    alpaka::memcpy(
        queue, pixelTripletsInCPU->tripletIndices_buf, pixelTripletsBuffers->tripletIndices_buf, nPixelTriplets);
    alpaka::memcpy(queue,
                   pixelTripletsInCPU->pixelSegmentIndices_buf,
                   pixelTripletsBuffers->pixelSegmentIndices_buf,
                   nPixelTriplets);
    alpaka::memcpy(queue, pixelTripletsInCPU->pixelRadius_buf, pixelTripletsBuffers->pixelRadius_buf, nPixelTriplets);
    alpaka::memcpy(
        queue, pixelTripletsInCPU->tripletRadius_buf, pixelTripletsBuffers->tripletRadius_buf, nPixelTriplets);
    alpaka::memcpy(queue, pixelTripletsInCPU->isDup_buf, pixelTripletsBuffers->isDup_buf, nPixelTriplets);
    alpaka::memcpy(queue, pixelTripletsInCPU->eta_buf, pixelTripletsBuffers->eta_buf, nPixelTriplets);
    alpaka::memcpy(queue, pixelTripletsInCPU->phi_buf, pixelTripletsBuffers->phi_buf, nPixelTriplets);
    alpaka::memcpy(queue, pixelTripletsInCPU->score_buf, pixelTripletsBuffers->score_buf, nPixelTriplets);
    if (sync)
      alpaka::wait(queue);  // host consumers expect filled data
  }
  return pixelTripletsInCPU.value();
}

PixelQuintupletsBuffer<alpaka_common::DevHost>& Event::getPixelQuintuplets(bool sync) {
  if (!pixelQuintupletsInCPU) {
    // Get nPixelQuintuplets parameter to initialize host based quintupletsInCPU
    auto nPixelQuintuplets_buf_h = cms::alpakatools::make_host_buffer<unsigned int[]>(queue, 1u);
    alpaka::memcpy(queue, nPixelQuintuplets_buf_h, pixelQuintupletsBuffers->nPixelQuintuplets_buf);
    alpaka::wait(queue);  // wait for the value before using

    auto const nPixelQuintuplets = *nPixelQuintuplets_buf_h.data();
    pixelQuintupletsInCPU = PixelQuintupletsBuffer<DevHost>(nPixelQuintuplets, devHost, queue);
    pixelQuintupletsInCPU->setData(*pixelQuintupletsInCPU);

    alpaka::memcpy(queue, pixelQuintupletsInCPU->nPixelQuintuplets_buf, pixelQuintupletsBuffers->nPixelQuintuplets_buf);
    alpaka::memcpy(queue,
                   pixelQuintupletsInCPU->totOccupancyPixelQuintuplets_buf,
                   pixelQuintupletsBuffers->totOccupancyPixelQuintuplets_buf);
    alpaka::memcpy(
        queue, pixelQuintupletsInCPU->rzChiSquared_buf, pixelQuintupletsBuffers->rzChiSquared_buf, nPixelQuintuplets);
    alpaka::memcpy(queue,
                   pixelQuintupletsInCPU->rPhiChiSquared_buf,
                   pixelQuintupletsBuffers->rPhiChiSquared_buf,
                   nPixelQuintuplets);
    alpaka::memcpy(queue,
                   pixelQuintupletsInCPU->rPhiChiSquaredInwards_buf,
                   pixelQuintupletsBuffers->rPhiChiSquaredInwards_buf,
                   nPixelQuintuplets);
    alpaka::memcpy(
        queue, pixelQuintupletsInCPU->pixelIndices_buf, pixelQuintupletsBuffers->pixelIndices_buf, nPixelQuintuplets);
    alpaka::memcpy(
        queue, pixelQuintupletsInCPU->T5Indices_buf, pixelQuintupletsBuffers->T5Indices_buf, nPixelQuintuplets);
    alpaka::memcpy(queue, pixelQuintupletsInCPU->isDup_buf, pixelQuintupletsBuffers->isDup_buf, nPixelQuintuplets);
    alpaka::memcpy(queue, pixelQuintupletsInCPU->score_buf, pixelQuintupletsBuffers->score_buf, nPixelQuintuplets);
    if (sync)
      alpaka::wait(queue);  // host consumers expect filled data
  }
  return pixelQuintupletsInCPU.value();
}

TrackCandidatesBuffer<alpaka_common::DevHost>& Event::getTrackCandidates(bool sync) {
  if (!trackCandidatesInCPU) {
    // Get nTrackCanHost parameter to initialize host based trackCandidatesInCPU
    auto nTrackCanHost_buf_h = cms::alpakatools::make_host_buffer<unsigned int[]>(queue, 1u);
    alpaka::memcpy(queue, nTrackCanHost_buf_h, trackCandidatesBuffers->nTrackCandidates_buf);
    trackCandidatesInCPU =
        TrackCandidatesBuffer<DevHost>(n_max_nonpixel_track_candidates + n_max_pixel_track_candidates, devHost, queue);
    trackCandidatesInCPU->setData(*trackCandidatesInCPU);
    alpaka::wait(queue);  // wait here before we get nTrackCanHost and trackCandidatesInCPU becomes usable

    auto const nTrackCanHost = *nTrackCanHost_buf_h.data();

    *trackCandidatesInCPU->nTrackCandidates_buf.data() = nTrackCanHost;
    alpaka::memcpy(queue,
                   trackCandidatesInCPU->hitIndices_buf,
                   trackCandidatesBuffers->hitIndices_buf,
                   Params_pT5::kHits * nTrackCanHost);
    alpaka::memcpy(
        queue, trackCandidatesInCPU->pixelSeedIndex_buf, trackCandidatesBuffers->pixelSeedIndex_buf, nTrackCanHost);
    alpaka::memcpy(queue,
                   trackCandidatesInCPU->logicalLayers_buf,
                   trackCandidatesBuffers->logicalLayers_buf,
                   Params_pT5::kLayers * nTrackCanHost);
    alpaka::memcpy(queue,
                   trackCandidatesInCPU->directObjectIndices_buf,
                   trackCandidatesBuffers->directObjectIndices_buf,
                   nTrackCanHost);
    alpaka::memcpy(
        queue, trackCandidatesInCPU->objectIndices_buf, trackCandidatesBuffers->objectIndices_buf, 2 * nTrackCanHost);
    alpaka::memcpy(queue,
                   trackCandidatesInCPU->trackCandidateType_buf,
                   trackCandidatesBuffers->trackCandidateType_buf,
                   nTrackCanHost);
    if (sync)
      alpaka::wait(queue);  // host consumers expect filled data
  }
  return trackCandidatesInCPU.value();
}

TrackCandidatesBuffer<alpaka_common::DevHost>& Event::getTrackCandidatesInCMSSW(bool sync) {
  if (!trackCandidatesInCPU) {
    // Get nTrackCanHost parameter to initialize host based trackCandidatesInCPU
    auto nTrackCanHost_buf_h = cms::alpakatools::make_host_buffer<unsigned int[]>(queue, 1u);
    alpaka::memcpy(queue, nTrackCanHost_buf_h, trackCandidatesBuffers->nTrackCandidates_buf);
    trackCandidatesInCPU =
        TrackCandidatesBuffer<DevHost>(n_max_nonpixel_track_candidates + n_max_pixel_track_candidates, devHost, queue);
    trackCandidatesInCPU->setData(*trackCandidatesInCPU);
    alpaka::wait(queue);  // wait for the value before using and trackCandidatesInCPU becomes usable

    auto const nTrackCanHost = *nTrackCanHost_buf_h.data();

    *trackCandidatesInCPU->nTrackCandidates_buf.data() = nTrackCanHost;
    alpaka::memcpy(queue,
                   trackCandidatesInCPU->hitIndices_buf,
                   trackCandidatesBuffers->hitIndices_buf,
                   Params_pT5::kHits * nTrackCanHost);
    alpaka::memcpy(
        queue, trackCandidatesInCPU->pixelSeedIndex_buf, trackCandidatesBuffers->pixelSeedIndex_buf, nTrackCanHost);
    alpaka::memcpy(queue,
                   trackCandidatesInCPU->trackCandidateType_buf,
                   trackCandidatesBuffers->trackCandidateType_buf,
                   nTrackCanHost);
    if (sync)
      alpaka::wait(queue);  // host consumers expect filled data
  }
  return trackCandidatesInCPU.value();
}

ModulesBuffer<alpaka_common::DevHost>& Event::getModules(bool isFull, bool sync) {
  if (!modulesInCPU) {
    // The last input here is just a small placeholder for the allocation.
    modulesInCPU = ModulesBuffer<DevHost>(devHost, nModules_, nPixels_);

    modulesInCPU->copyFromSrc(queue, modulesBuffers_, isFull);
    if (sync)
      alpaka::wait(queue);  // host consumers expect filled data
  }
  return modulesInCPU.value();
}
