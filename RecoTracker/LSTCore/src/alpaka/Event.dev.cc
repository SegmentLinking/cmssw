#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"

#include "Event.h"

using namespace ALPAKA_ACCELERATOR_NAMESPACE;

void lst::Event<Acc3D>::init(bool verbose) {
  addObjects = verbose;
  hitsInGPU = nullptr;
  mdsInGPU = nullptr;
  segmentsInGPU = nullptr;
  tripletsInGPU = nullptr;
  quintupletsInGPU = nullptr;
  trackCandidatesInGPU = nullptr;
  pixelTripletsInGPU = nullptr;
  pixelQuintupletsInGPU = nullptr;
  rangesInGPU = nullptr;

  hitsInCPU = nullptr;
  rangesInCPU = nullptr;
  mdsInCPU = nullptr;
  segmentsInCPU = nullptr;
  tripletsInCPU = nullptr;
  trackCandidatesInCPU = nullptr;
  modulesInCPU = nullptr;
  quintupletsInCPU = nullptr;
  pixelTripletsInCPU = nullptr;
  pixelQuintupletsInCPU = nullptr;

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

void lst::Event<Acc3D>::resetEvent() {
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
    delete hitsInGPU;
    delete hitsBuffers;
    hitsInGPU = nullptr;
  }
  if (mdsInGPU) {
    delete mdsInGPU;
    delete miniDoubletsBuffers;
    mdsInGPU = nullptr;
  }
  if (rangesInGPU) {
    delete rangesInGPU;
    delete rangesBuffers;
    rangesInGPU = nullptr;
  }
  if (segmentsInGPU) {
    delete segmentsInGPU;
    delete segmentsBuffers;
    segmentsInGPU = nullptr;
  }
  if (tripletsInGPU) {
    delete tripletsInGPU;
    delete tripletsBuffers;
    tripletsInGPU = nullptr;
  }
  if (quintupletsInGPU) {
    delete quintupletsInGPU;
    delete quintupletsBuffers;
    quintupletsInGPU = nullptr;
  }
  if (trackCandidatesInGPU) {
    delete trackCandidatesInGPU;
    delete trackCandidatesBuffers;
    trackCandidatesInGPU = nullptr;
  }
  if (pixelTripletsInGPU) {
    delete pixelTripletsInGPU;
    delete pixelTripletsBuffers;
    pixelTripletsInGPU = nullptr;
  }
  if (pixelQuintupletsInGPU) {
    delete pixelQuintupletsInGPU;
    delete pixelQuintupletsBuffers;
    pixelQuintupletsInGPU = nullptr;
  }

  if (hitsInCPU != nullptr) {
    delete hitsInCPU;
    hitsInCPU = nullptr;
  }
  if (rangesInCPU != nullptr) {
    delete rangesInCPU;
    rangesInCPU = nullptr;
  }
  if (mdsInCPU != nullptr) {
    delete mdsInCPU;
    mdsInCPU = nullptr;
  }
  if (segmentsInCPU != nullptr) {
    delete segmentsInCPU;
    segmentsInCPU = nullptr;
  }
  if (tripletsInCPU != nullptr) {
    delete tripletsInCPU;
    tripletsInCPU = nullptr;
  }
  if (quintupletsInCPU != nullptr) {
    delete quintupletsInCPU;
    quintupletsInCPU = nullptr;
  }
  if (pixelTripletsInCPU != nullptr) {
    delete pixelTripletsInCPU;
    pixelTripletsInCPU = nullptr;
  }
  if (pixelQuintupletsInCPU != nullptr) {
    delete pixelQuintupletsInCPU;
    pixelQuintupletsInCPU = nullptr;
  }
  if (trackCandidatesInCPU != nullptr) {
    delete trackCandidatesInCPU;
    trackCandidatesInCPU = nullptr;
  }
  if (modulesInCPU != nullptr) {
    delete modulesInCPU;
    modulesInCPU = nullptr;
  }
}

void lst::Event<Acc3D>::addHitToEvent(std::vector<float> const& x,
                                      std::vector<float> const& y,
                                      std::vector<float> const& z,
                                      std::vector<unsigned int> const& detId,
                                      std::vector<unsigned int> const& idxInNtuple) {
  // Use the actual number of hits instead of a max.
  unsigned int nHits = x.size();

  // Initialize space on device/host for next event.
  if (hitsInGPU == nullptr) {
    hitsInGPU = new lst::Hits();
    hitsBuffers = new lst::HitsBuffer<Device>(nModules_, nHits, devAcc, queue);
    hitsInGPU->setData(*hitsBuffers);
  }

  if (rangesInGPU == nullptr) {
    rangesInGPU = new lst::ObjectRanges();
    rangesBuffers = new lst::ObjectRangesBuffer<Device>(nModules_, nLowerModules_, devAcc, queue);
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

  hitLoopKernel hit_loop_kernel;
  alpaka::exec<Acc3D>(queue,
                      hit_loop_workdiv,
                      hit_loop_kernel,
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

  moduleRangesKernel module_ranges_kernel;
  alpaka::exec<Acc3D>(
      queue, module_ranges_workdiv, module_ranges_kernel, *modulesBuffers_.data(), *hitsInGPU, nLowerModules_);
}

void lst::Event<Acc3D>::addPixelSegmentToEvent(std::vector<unsigned int> const& hitIndices0,
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
                                               std::vector<int8_t> const& pixelType,
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

  if (mdsInGPU == nullptr) {
    // Create a view for the element nLowerModules_ inside rangesBuffers->miniDoubletModuleOccupancy
    auto dst_view_miniDoubletModuleOccupancy =
        alpaka::createSubView(rangesBuffers->miniDoubletModuleOccupancy_buf, (Idx)1u, (Idx)nLowerModules_);

    // Create a host buffer for a value to be passed to the device
    auto pixelMaxMDs_buf_h = cms::alpakatools::make_host_buffer<int[]>(queue, (Idx)1u);
    *pixelMaxMDs_buf_h.data() = n_max_pixel_md_per_modules;

    alpaka::memcpy(queue, dst_view_miniDoubletModuleOccupancy, pixelMaxMDs_buf_h);

    Vec3D const threadsPerBlockCreateMD{1, 1, 1024};
    Vec3D const blocksPerGridCreateMD{1, 1, 1};
    WorkDiv3D const createMDArrayRangesGPU_workDiv =
        createWorkDiv(blocksPerGridCreateMD, threadsPerBlockCreateMD, elementsPerThread);

    lst::createMDArrayRangesGPU createMDArrayRangesGPU_kernel;
    alpaka::exec<Acc3D>(
        queue, createMDArrayRangesGPU_workDiv, createMDArrayRangesGPU_kernel, *modulesBuffers_.data(), *rangesInGPU);

    auto nTotalMDs_buf_h = cms::alpakatools::make_host_buffer<unsigned int[]>(queue, (Idx)1u);
    alpaka::memcpy(queue, nTotalMDs_buf_h, rangesBuffers->device_nTotalMDs_buf);
    alpaka::wait(queue);  // wait to get the data before manipulation

    *nTotalMDs_buf_h.data() += n_max_pixel_md_per_modules;
    unsigned int nTotalMDs = *nTotalMDs_buf_h.data();

    mdsInGPU = new lst::MiniDoublets();
    miniDoubletsBuffers = new lst::MiniDoubletsBuffer<Device>(nTotalMDs, nLowerModules_, devAcc, queue);
    mdsInGPU->setData(*miniDoubletsBuffers);

    alpaka::memcpy(queue, miniDoubletsBuffers->nMemoryLocations_buf, nTotalMDs_buf_h);
  }
  if (segmentsInGPU == nullptr) {
    // can be optimized here: because we didn't distinguish pixel segments and outer-tracker segments and call them both "segments", so they use the index continuously.
    // If we want to further study the memory footprint in detail, we can separate the two and allocate different memories to them

    Vec3D const threadsPerBlockCreateSeg{1, 1, 1024};
    Vec3D const blocksPerGridCreateSeg{1, 1, 1};
    WorkDiv3D const createSegmentArrayRanges_workDiv =
        createWorkDiv(blocksPerGridCreateSeg, threadsPerBlockCreateSeg, elementsPerThread);

    lst::createSegmentArrayRanges createSegmentArrayRanges_kernel;
    alpaka::exec<Acc3D>(queue,
                        createSegmentArrayRanges_workDiv,
                        createSegmentArrayRanges_kernel,
                        *modulesBuffers_.data(),
                        *rangesInGPU,
                        *mdsInGPU);

    auto nTotalSegments_view = alpaka::createView(devHost, &nTotalSegments_, (Idx)1u);

    alpaka::memcpy(queue, nTotalSegments_view, rangesBuffers->device_nTotalSegs_buf);
    alpaka::wait(queue);  // wait to get the value before manipulation

    nTotalSegments_ += n_max_pixel_segments_per_module;

    segmentsInGPU = new lst::Segments();
    segmentsBuffers = new lst::SegmentsBuffer<Device>(
        nTotalSegments_, nLowerModules_, n_max_pixel_segments_per_module, devAcc, queue);
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

  addPixelSegmentToEventKernel addPixelSegmentToEvent_kernel;
  alpaka::exec<Acc3D>(queue,
                      addPixelSegmentToEvent_workdiv,
                      addPixelSegmentToEvent_kernel,
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

void lst::Event<Acc3D>::createMiniDoublets() {
  // Create a view for the element nLowerModules_ inside rangesBuffers->miniDoubletModuleOccupancy
  auto dst_view_miniDoubletModuleOccupancy =
      alpaka::createSubView(rangesBuffers->miniDoubletModuleOccupancy_buf, (Idx)1u, (Idx)nLowerModules_);

  // Create a host buffer for a value to be passed to the device
  auto pixelMaxMDs_buf_h = cms::alpakatools::make_host_buffer<int[]>(queue, (Idx)1u);
  *pixelMaxMDs_buf_h.data() = n_max_pixel_md_per_modules;

  alpaka::memcpy(queue, dst_view_miniDoubletModuleOccupancy, pixelMaxMDs_buf_h);

  Vec3D const threadsPerBlockCreateMD{1, 1, 1024};
  Vec3D const blocksPerGridCreateMD{1, 1, 1};
  WorkDiv3D const createMDArrayRangesGPU_workDiv =
      createWorkDiv(blocksPerGridCreateMD, threadsPerBlockCreateMD, elementsPerThread);

  lst::createMDArrayRangesGPU createMDArrayRangesGPU_kernel;
  alpaka::exec<Acc3D>(
      queue, createMDArrayRangesGPU_workDiv, createMDArrayRangesGPU_kernel, *modulesBuffers_.data(), *rangesInGPU);

  auto nTotalMDs_buf_h = cms::alpakatools::make_host_buffer<unsigned int[]>(queue, (Idx)1u);
  alpaka::memcpy(queue, nTotalMDs_buf_h, rangesBuffers->device_nTotalMDs_buf);
  alpaka::wait(queue);  // wait to get the data before manipulation

  *nTotalMDs_buf_h.data() += n_max_pixel_md_per_modules;
  unsigned int nTotalMDs = *nTotalMDs_buf_h.data();

  if (mdsInGPU == nullptr) {
    mdsInGPU = new lst::MiniDoublets();
    miniDoubletsBuffers = new lst::MiniDoubletsBuffer<Device>(nTotalMDs, nLowerModules_, devAcc, queue);
    mdsInGPU->setData(*miniDoubletsBuffers);
  }

  Vec3D const threadsPerBlockCreateMDInGPU{1, 16, 32};
  Vec3D const blocksPerGridCreateMDInGPU{1, nLowerModules_ / threadsPerBlockCreateMDInGPU[1], 1};
  WorkDiv3D const createMiniDoubletsInGPUv2_workDiv =
      createWorkDiv(blocksPerGridCreateMDInGPU, threadsPerBlockCreateMDInGPU, elementsPerThread);

  lst::createMiniDoubletsInGPUv2 createMiniDoubletsInGPUv2_kernel;
  alpaka::exec<Acc3D>(queue,
                      createMiniDoubletsInGPUv2_workDiv,
                      createMiniDoubletsInGPUv2_kernel,
                      *modulesBuffers_.data(),
                      *hitsInGPU,
                      *mdsInGPU,
                      *rangesInGPU);

  Vec3D const threadsPerBlockAddMD{1, 1, 1024};
  Vec3D const blocksPerGridAddMD{1, 1, 1};
  WorkDiv3D const addMiniDoubletRangesToEventExplicit_workDiv =
      createWorkDiv(blocksPerGridAddMD, threadsPerBlockAddMD, elementsPerThread);

  lst::addMiniDoubletRangesToEventExplicit addMiniDoubletRangesToEventExplicit_kernel;
  alpaka::exec<Acc3D>(queue,
                      addMiniDoubletRangesToEventExplicit_workDiv,
                      addMiniDoubletRangesToEventExplicit_kernel,
                      *modulesBuffers_.data(),
                      *mdsInGPU,
                      *rangesInGPU,
                      *hitsInGPU);

  if (addObjects) {
    addMiniDoubletsToEventExplicit();
  }
}

void lst::Event<Acc3D>::createSegmentsWithModuleMap() {
  if (segmentsInGPU == nullptr) {
    segmentsInGPU = new lst::Segments();
    segmentsBuffers = new lst::SegmentsBuffer<Device>(
        nTotalSegments_, nLowerModules_, n_max_pixel_segments_per_module, devAcc, queue);
    segmentsInGPU->setData(*segmentsBuffers);
  }

  Vec3D const threadsPerBlockCreateSeg{1, 1, 64};
  Vec3D const blocksPerGridCreateSeg{1, 1, nLowerModules_};
  WorkDiv3D const createSegmentsInGPUv2_workDiv =
      createWorkDiv(blocksPerGridCreateSeg, threadsPerBlockCreateSeg, elementsPerThread);

  lst::createSegmentsInGPUv2 createSegmentsInGPUv2_kernel;
  alpaka::exec<Acc3D>(queue,
                      createSegmentsInGPUv2_workDiv,
                      createSegmentsInGPUv2_kernel,
                      *modulesBuffers_.data(),
                      *mdsInGPU,
                      *segmentsInGPU,
                      *rangesInGPU);

  Vec3D const threadsPerBlockAddSeg{1, 1, 1024};
  Vec3D const blocksPerGridAddSeg{1, 1, 1};
  WorkDiv3D const addSegmentRangesToEventExplicit_workDiv =
      createWorkDiv(blocksPerGridAddSeg, threadsPerBlockAddSeg, elementsPerThread);

  lst::addSegmentRangesToEventExplicit addSegmentRangesToEventExplicit_kernel;
  alpaka::exec<Acc3D>(queue,
                      addSegmentRangesToEventExplicit_workDiv,
                      addSegmentRangesToEventExplicit_kernel,
                      *modulesBuffers_.data(),
                      *segmentsInGPU,
                      *rangesInGPU);

  if (addObjects) {
    addSegmentsToEventExplicit();
  }
}

void lst::Event<Acc3D>::createTriplets() {
  if (tripletsInGPU == nullptr) {
    Vec3D const threadsPerBlockCreateTrip{1, 1, 1024};
    Vec3D const blocksPerGridCreateTrip{1, 1, 1};
    WorkDiv3D const createTripletArrayRanges_workDiv =
        createWorkDiv(blocksPerGridCreateTrip, threadsPerBlockCreateTrip, elementsPerThread);

    lst::createTripletArrayRanges createTripletArrayRanges_kernel;
    alpaka::exec<Acc3D>(queue,
                        createTripletArrayRanges_workDiv,
                        createTripletArrayRanges_kernel,
                        *modulesBuffers_.data(),
                        *rangesInGPU,
                        *segmentsInGPU);

    // TODO: Why are we pulling this back down only to put it back on the device in a new struct?
    auto maxTriplets_buf_h = cms::alpakatools::make_host_buffer<unsigned int[]>(queue, (Idx)1u);

    alpaka::memcpy(queue, maxTriplets_buf_h, rangesBuffers->device_nTotalTrips_buf);
    alpaka::wait(queue);  // wait to get the value before using it

    tripletsInGPU = new lst::Triplets();
    tripletsBuffers = new lst::TripletsBuffer<Device>(*maxTriplets_buf_h.data(), nLowerModules_, devAcc, queue);
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

  lst::createTripletsInGPUv2 createTripletsInGPUv2_kernel;
  alpaka::exec<Acc3D>(queue,
                      createTripletsInGPUv2_workDiv,
                      createTripletsInGPUv2_kernel,
                      *modulesBuffers_.data(),
                      *mdsInGPU,
                      *segmentsInGPU,
                      *tripletsInGPU,
                      *rangesInGPU,
                      index_gpu_buf.data(),
                      nonZeroModules);

  Vec3D const threadsPerBlockAddTrip{1, 1, 1024};
  Vec3D const blocksPerGridAddTrip{1, 1, 1};
  WorkDiv3D const addTripletRangesToEventExplicit_workDiv =
      createWorkDiv(blocksPerGridAddTrip, threadsPerBlockAddTrip, elementsPerThread);

  lst::addTripletRangesToEventExplicit addTripletRangesToEventExplicit_kernel;
  alpaka::exec<Acc3D>(queue,
                      addTripletRangesToEventExplicit_workDiv,
                      addTripletRangesToEventExplicit_kernel,
                      *modulesBuffers_.data(),
                      *tripletsInGPU,
                      *rangesInGPU);

  if (addObjects) {
    addTripletsToEventExplicit();
  }
}

void lst::Event<Acc3D>::createTrackCandidates(bool no_pls_dupclean, bool tc_pls_triplets) {
  if (trackCandidatesInGPU == nullptr) {
    trackCandidatesInGPU = new lst::TrackCandidates();
    trackCandidatesBuffers = new lst::TrackCandidatesBuffer<Device>(
        n_max_nonpixel_track_candidates + n_max_pixel_track_candidates, devAcc, queue);
    trackCandidatesInGPU->setData(*trackCandidatesBuffers);
  }

  Vec3D const threadsPerBlock_crossCleanpT3{1, 16, 64};
  Vec3D const blocksPerGrid_crossCleanpT3{1, 4, 20};
  WorkDiv3D const crossCleanpT3_workDiv =
      createWorkDiv(blocksPerGrid_crossCleanpT3, threadsPerBlock_crossCleanpT3, elementsPerThread);

  lst::crossCleanpT3 crossCleanpT3_kernel;
  alpaka::exec<Acc3D>(queue,
                      crossCleanpT3_workDiv,
                      crossCleanpT3_kernel,
                      *modulesBuffers_.data(),
                      *rangesInGPU,
                      *pixelTripletsInGPU,
                      *segmentsInGPU,
                      *pixelQuintupletsInGPU);

  Vec3D const threadsPerBlock_addpT3asTrackCandidatesInGPU{1, 1, 512};
  Vec3D const blocksPerGrid_addpT3asTrackCandidatesInGPU{1, 1, 1};
  WorkDiv3D const addpT3asTrackCandidatesInGPU_workDiv = createWorkDiv(
      blocksPerGrid_addpT3asTrackCandidatesInGPU, threadsPerBlock_addpT3asTrackCandidatesInGPU, elementsPerThread);

  lst::addpT3asTrackCandidatesInGPU addpT3asTrackCandidatesInGPU_kernel;
  alpaka::exec<Acc3D>(queue,
                      addpT3asTrackCandidatesInGPU_workDiv,
                      addpT3asTrackCandidatesInGPU_kernel,
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

  lst::removeDupQuintupletsInGPUBeforeTC removeDupQuintupletsInGPUBeforeTC_kernel;
  alpaka::exec<Acc3D>(queue,
                      removeDupQuintupletsInGPUBeforeTC_workDiv,
                      removeDupQuintupletsInGPUBeforeTC_kernel,
                      *quintupletsInGPU,
                      *rangesInGPU);

  Vec3D const threadsPerBlock_crossCleanT5{32, 1, 32};
  Vec3D const blocksPerGrid_crossCleanT5{(13296 / 32) + 1, 1, max_blocks};
  WorkDiv3D const crossCleanT5_workDiv =
      createWorkDiv(blocksPerGrid_crossCleanT5, threadsPerBlock_crossCleanT5, elementsPerThread);

  lst::crossCleanT5 crossCleanT5_kernel;
  alpaka::exec<Acc3D>(queue,
                      crossCleanT5_workDiv,
                      crossCleanT5_kernel,
                      *modulesBuffers_.data(),
                      *quintupletsInGPU,
                      *pixelQuintupletsInGPU,
                      *pixelTripletsInGPU,
                      *rangesInGPU);

  Vec3D const threadsPerBlock_addT5asTrackCandidateInGPU{1, 8, 128};
  Vec3D const blocksPerGrid_addT5asTrackCandidateInGPU{1, 8, 10};
  WorkDiv3D const addT5asTrackCandidateInGPU_workDiv = createWorkDiv(
      blocksPerGrid_addT5asTrackCandidateInGPU, threadsPerBlock_addT5asTrackCandidateInGPU, elementsPerThread);

  lst::addT5asTrackCandidateInGPU addT5asTrackCandidateInGPU_kernel;
  alpaka::exec<Acc3D>(queue,
                      addT5asTrackCandidateInGPU_workDiv,
                      addT5asTrackCandidateInGPU_kernel,
                      nLowerModules_,
                      *quintupletsInGPU,
                      *trackCandidatesInGPU,
                      *rangesInGPU);

  if (!no_pls_dupclean) {
    Vec3D const threadsPerBlockCheckHitspLS{1, 16, 16};
    Vec3D const blocksPerGridCheckHitspLS{1, max_blocks * 4, max_blocks / 4};
    WorkDiv3D const checkHitspLS_workDiv =
        createWorkDiv(blocksPerGridCheckHitspLS, threadsPerBlockCheckHitspLS, elementsPerThread);

    lst::checkHitspLS checkHitspLS_kernel;
    alpaka::exec<Acc3D>(
        queue, checkHitspLS_workDiv, checkHitspLS_kernel, *modulesBuffers_.data(), *segmentsInGPU, true);
  }

  Vec3D const threadsPerBlock_crossCleanpLS{1, 16, 32};
  Vec3D const blocksPerGrid_crossCleanpLS{1, 4, 20};
  WorkDiv3D const crossCleanpLS_workDiv =
      createWorkDiv(blocksPerGrid_crossCleanpLS, threadsPerBlock_crossCleanpLS, elementsPerThread);

  lst::crossCleanpLS crossCleanpLS_kernel;
  alpaka::exec<Acc3D>(queue,
                      crossCleanpLS_workDiv,
                      crossCleanpLS_kernel,
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

  lst::addpLSasTrackCandidateInGPU addpLSasTrackCandidateInGPU_kernel;
  alpaka::exec<Acc3D>(queue,
                      addpLSasTrackCandidateInGPU_workDiv,
                      addpLSasTrackCandidateInGPU_kernel,
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

void lst::Event<Acc3D>::createPixelTriplets() {
  if (pixelTripletsInGPU == nullptr) {
    pixelTripletsInGPU = new lst::PixelTriplets();
    pixelTripletsBuffers = new lst::PixelTripletsBuffer<Device>(n_max_pixel_triplets, devAcc, queue);
    pixelTripletsInGPU->setData(*pixelTripletsBuffers);
  }

  auto superbins_buf = allocBufWrapper<int>(devHost, n_max_pixel_segments_per_module, queue);
  auto pixelTypes_buf = allocBufWrapper<int8_t>(devHost, n_max_pixel_segments_per_module, queue);

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
    int8_t pixelType = pixelTypes[i];                  // Get pixel type for this pLS
    int superbin = superbins[i];                       // Get superbin for this pixel
    if ((superbin < 0) or (superbin >= (int)size_superbins) or (pixelType > 2) or (pixelType < 0)) {
      connectedPixelSize_host[i] = 0;
      connectedPixelIndex_host[i] = 0;
      continue;
    }

    // Used pixel type to select correct size-index arrays
    if (pixelType == 0) {
      connectedPixelSize_host[i] =
          pixelMapping_.connectedPixelsSizes[superbin];  // number of connected modules to this pixel
      auto connectedIdxBase = pixelMapping_.connectedPixelsIndex[superbin];
      connectedPixelIndex_host[i] =
          connectedIdxBase;  // index to get start of connected modules for this superbin in map
    } else if (pixelType == 1) {
      connectedPixelSize_host[i] =
          pixelMapping_.connectedPixelsSizesPos[superbin];  // number of pixel connected modules
      auto connectedIdxBase = pixelMapping_.connectedPixelsIndexPos[superbin] + pixelIndexOffsetPos;
      connectedPixelIndex_host[i] = connectedIdxBase;  // index to get start of connected pixel modules
    } else if (pixelType == 2) {
      connectedPixelSize_host[i] =
          pixelMapping_.connectedPixelsSizesNeg[superbin];  // number of pixel connected modules
      auto connectedIdxBase = pixelMapping_.connectedPixelsIndexNeg[superbin] + pixelIndexOffsetNeg;
      connectedPixelIndex_host[i] = connectedIdxBase;  // index to get start of connected pixel modules
    }
  }

  alpaka::memcpy(queue, connectedPixelSize_dev_buf, connectedPixelSize_host_buf, nInnerSegments);
  alpaka::memcpy(queue, connectedPixelIndex_dev_buf, connectedPixelIndex_host_buf, nInnerSegments);

  Vec3D const threadsPerBlock{1, 4, 32};
  Vec3D const blocksPerGrid{16 /* above median of connected modules*/, 4096, 1};
  WorkDiv3D const createPixelTripletsInGPUFromMapv2_workDiv =
      createWorkDiv(blocksPerGrid, threadsPerBlock, elementsPerThread);

  lst::createPixelTripletsInGPUFromMapv2 createPixelTripletsInGPUFromMapv2_kernel;
  alpaka::exec<Acc3D>(queue,
                      createPixelTripletsInGPUFromMapv2_workDiv,
                      createPixelTripletsInGPUFromMapv2_kernel,
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

  lst::removeDupPixelTripletsInGPUFromMap removeDupPixelTripletsInGPUFromMap_kernel;
  alpaka::exec<Acc3D>(queue,
                      removeDupPixelTripletsInGPUFromMap_workDiv,
                      removeDupPixelTripletsInGPUFromMap_kernel,
                      *pixelTripletsInGPU);
}

void lst::Event<Acc3D>::createQuintuplets() {
  Vec3D const threadsPerBlockCreateQuints{1, 1, 1024};
  Vec3D const blocksPerGridCreateQuints{1, 1, 1};
  WorkDiv3D const createEligibleModulesListForQuintupletsGPU_workDiv =
      createWorkDiv(blocksPerGridCreateQuints, threadsPerBlockCreateQuints, elementsPerThread);

  lst::createEligibleModulesListForQuintupletsGPU createEligibleModulesListForQuintupletsGPU_kernel;
  alpaka::exec<Acc3D>(queue,
                      createEligibleModulesListForQuintupletsGPU_workDiv,
                      createEligibleModulesListForQuintupletsGPU_kernel,
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

  if (quintupletsInGPU == nullptr) {
    quintupletsInGPU = new lst::Quintuplets();
    quintupletsBuffers = new lst::QuintupletsBuffer<Device>(nTotalQuintuplets, nLowerModules_, devAcc, queue);
    quintupletsInGPU->setData(*quintupletsBuffers);

    alpaka::memcpy(queue, quintupletsBuffers->nMemoryLocations_buf, nTotalQuintuplets_buf);
  }

  Vec3D const threadsPerBlockQuints{1, 8, 32};
  Vec3D const blocksPerGridQuints{std::max((int)nEligibleT5Modules, 1), 1, 1};
  WorkDiv3D const createQuintupletsInGPUv2_workDiv =
      createWorkDiv(blocksPerGridQuints, threadsPerBlockQuints, elementsPerThread);

  lst::createQuintupletsInGPUv2 createQuintupletsInGPUv2_kernel;
  alpaka::exec<Acc3D>(queue,
                      createQuintupletsInGPUv2_workDiv,
                      createQuintupletsInGPUv2_kernel,
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

  lst::removeDupQuintupletsInGPUAfterBuild removeDupQuintupletsInGPUAfterBuild_kernel;
  alpaka::exec<Acc3D>(queue,
                      removeDupQuintupletsInGPUAfterBuild_workDiv,
                      removeDupQuintupletsInGPUAfterBuild_kernel,
                      *modulesBuffers_.data(),
                      *quintupletsInGPU,
                      *rangesInGPU);

  Vec3D const threadsPerBlockAddQuint{1, 1, 1024};
  Vec3D const blocksPerGridAddQuint{1, 1, 1};
  WorkDiv3D const addQuintupletRangesToEventExplicit_workDiv =
      createWorkDiv(blocksPerGridAddQuint, threadsPerBlockAddQuint, elementsPerThread);

  lst::addQuintupletRangesToEventExplicit addQuintupletRangesToEventExplicit_kernel;
  alpaka::exec<Acc3D>(queue,
                      addQuintupletRangesToEventExplicit_workDiv,
                      addQuintupletRangesToEventExplicit_kernel,
                      *modulesBuffers_.data(),
                      *quintupletsInGPU,
                      *rangesInGPU);

  if (addObjects) {
    addQuintupletsToEventExplicit();
  }
}

void lst::Event<Acc3D>::pixelLineSegmentCleaning(bool no_pls_dupclean) {
  if (!no_pls_dupclean) {
    Vec3D const threadsPerBlockCheckHitspLS{1, 16, 16};
    Vec3D const blocksPerGridCheckHitspLS{1, max_blocks * 4, max_blocks / 4};
    WorkDiv3D const checkHitspLS_workDiv =
        createWorkDiv(blocksPerGridCheckHitspLS, threadsPerBlockCheckHitspLS, elementsPerThread);

    lst::checkHitspLS checkHitspLS_kernel;
    alpaka::exec<Acc3D>(
        queue, checkHitspLS_workDiv, checkHitspLS_kernel, *modulesBuffers_.data(), *segmentsInGPU, false);
  }
}

void lst::Event<Acc3D>::createPixelQuintuplets() {
  if (pixelQuintupletsInGPU == nullptr) {
    pixelQuintupletsInGPU = new lst::PixelQuintuplets();
    pixelQuintupletsBuffers = new lst::PixelQuintupletsBuffer<Device>(n_max_pixel_quintuplets, devAcc, queue);
    pixelQuintupletsInGPU->setData(*pixelQuintupletsBuffers);
  }
  if (trackCandidatesInGPU == nullptr) {
    trackCandidatesInGPU = new lst::TrackCandidates();
    trackCandidatesBuffers = new lst::TrackCandidatesBuffer<Device>(
        n_max_nonpixel_track_candidates + n_max_pixel_track_candidates, devAcc, queue);
    trackCandidatesInGPU->setData(*trackCandidatesBuffers);
  }

  auto superbins_buf = allocBufWrapper<int>(devHost, n_max_pixel_segments_per_module, queue);
  auto pixelTypes_buf = allocBufWrapper<int8_t>(devHost, n_max_pixel_segments_per_module, queue);

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

  int pixelIndexOffsetPos =
      pixelMapping_.connectedPixelsIndex[size_superbins - 1] + pixelMapping_.connectedPixelsSizes[size_superbins - 1];
  int pixelIndexOffsetNeg = pixelMapping_.connectedPixelsIndexPos[size_superbins - 1] +
                            pixelMapping_.connectedPixelsSizesPos[size_superbins - 1] + pixelIndexOffsetPos;

  // Loop over # pLS
  for (unsigned int i = 0; i < nInnerSegments; i++) {
    int8_t pixelType = pixelTypes[i];  // Get pixel type for this pLS
    int superbin = superbins[i];       // Get superbin for this pixel
    if ((superbin < 0) or (superbin >= (int)size_superbins) or (pixelType > 2) or (pixelType < 0)) {
      connectedPixelIndex_host[i] = 0;
      connectedPixelSize_host[i] = 0;
      continue;
    }
    // Used pixel type to select correct size-index arrays
    if (pixelType == 0) {
      connectedPixelSize_host[i] =
          pixelMapping_.connectedPixelsSizes[superbin];  //number of connected modules to this pixel
      unsigned int connectedIdxBase = pixelMapping_.connectedPixelsIndex[superbin];
      connectedPixelIndex_host[i] = connectedIdxBase;
    } else if (pixelType == 1) {
      connectedPixelSize_host[i] = pixelMapping_.connectedPixelsSizesPos[superbin];  //number of pixel connected modules
      unsigned int connectedIdxBase = pixelMapping_.connectedPixelsIndexPos[superbin] + pixelIndexOffsetPos;
      connectedPixelIndex_host[i] = connectedIdxBase;
    } else if (pixelType == 2) {
      connectedPixelSize_host[i] = pixelMapping_.connectedPixelsSizesNeg[superbin];  //number of pixel connected modules
      unsigned int connectedIdxBase = pixelMapping_.connectedPixelsIndexNeg[superbin] + pixelIndexOffsetNeg;
      connectedPixelIndex_host[i] = connectedIdxBase;
    }
  }

  alpaka::memcpy(queue, connectedPixelSize_dev_buf, connectedPixelSize_host_buf, nInnerSegments);
  alpaka::memcpy(queue, connectedPixelIndex_dev_buf, connectedPixelIndex_host_buf, nInnerSegments);

  Vec3D const threadsPerBlockCreatePixQuints{1, 16, 16};
  Vec3D const blocksPerGridCreatePixQuints{16, max_blocks, 1};
  WorkDiv3D const createPixelQuintupletsInGPUFromMapv2_workDiv =
      createWorkDiv(blocksPerGridCreatePixQuints, threadsPerBlockCreatePixQuints, elementsPerThread);

  lst::createPixelQuintupletsInGPUFromMapv2 createPixelQuintupletsInGPUFromMapv2_kernel;
  alpaka::exec<Acc3D>(queue,
                      createPixelQuintupletsInGPUFromMapv2_workDiv,
                      createPixelQuintupletsInGPUFromMapv2_kernel,
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

  lst::removeDupPixelQuintupletsInGPUFromMap removeDupPixelQuintupletsInGPUFromMap_kernel;
  alpaka::exec<Acc3D>(queue,
                      removeDupPixelQuintupletsInGPUFromMap_workDiv,
                      removeDupPixelQuintupletsInGPUFromMap_kernel,
                      *pixelQuintupletsInGPU);

  Vec3D const threadsPerBlockAddpT5asTrackCan{1, 1, 256};
  Vec3D const blocksPerGridAddpT5asTrackCan{1, 1, 1};
  WorkDiv3D const addpT5asTrackCandidateInGPU_workDiv =
      createWorkDiv(blocksPerGridAddpT5asTrackCan, threadsPerBlockAddpT5asTrackCan, elementsPerThread);

  lst::addpT5asTrackCandidateInGPU addpT5asTrackCandidateInGPU_kernel;
  alpaka::exec<Acc3D>(queue,
                      addpT5asTrackCandidateInGPU_workDiv,
                      addpT5asTrackCandidateInGPU_kernel,
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

void lst::Event<Acc3D>::addMiniDoubletsToEventExplicit() {
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

void lst::Event<Acc3D>::addSegmentsToEventExplicit() {
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

void lst::Event<Acc3D>::addQuintupletsToEventExplicit() {
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

void lst::Event<Acc3D>::addTripletsToEventExplicit() {
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

unsigned int lst::Event<Acc3D>::getNumberOfHits() {
  unsigned int hits = 0;
  for (auto& it : n_hits_by_layer_barrel_) {
    hits += it;
  }
  for (auto& it : n_hits_by_layer_endcap_) {
    hits += it;
  }

  return hits;
}

unsigned int lst::Event<Acc3D>::getNumberOfHitsByLayer(unsigned int layer) {
  if (layer == 6)
    return n_hits_by_layer_barrel_[layer];
  else
    return n_hits_by_layer_barrel_[layer] + n_hits_by_layer_endcap_[layer];
}

unsigned int lst::Event<Acc3D>::getNumberOfHitsByLayerBarrel(unsigned int layer) {
  return n_hits_by_layer_barrel_[layer];
}

unsigned int lst::Event<Acc3D>::getNumberOfHitsByLayerEndcap(unsigned int layer) {
  return n_hits_by_layer_endcap_[layer];
}

unsigned int lst::Event<Acc3D>::getNumberOfMiniDoublets() {
  unsigned int miniDoublets = 0;
  for (auto& it : n_minidoublets_by_layer_barrel_) {
    miniDoublets += it;
  }
  for (auto& it : n_minidoublets_by_layer_endcap_) {
    miniDoublets += it;
  }

  return miniDoublets;
}

unsigned int lst::Event<Acc3D>::getNumberOfMiniDoubletsByLayer(unsigned int layer) {
  if (layer == 6)
    return n_minidoublets_by_layer_barrel_[layer];
  else
    return n_minidoublets_by_layer_barrel_[layer] + n_minidoublets_by_layer_endcap_[layer];
}

unsigned int lst::Event<Acc3D>::getNumberOfMiniDoubletsByLayerBarrel(unsigned int layer) {
  return n_minidoublets_by_layer_barrel_[layer];
}

unsigned int lst::Event<Acc3D>::getNumberOfMiniDoubletsByLayerEndcap(unsigned int layer) {
  return n_minidoublets_by_layer_endcap_[layer];
}

unsigned int lst::Event<Acc3D>::getNumberOfSegments() {
  unsigned int segments = 0;
  for (auto& it : n_segments_by_layer_barrel_) {
    segments += it;
  }
  for (auto& it : n_segments_by_layer_endcap_) {
    segments += it;
  }

  return segments;
}

unsigned int lst::Event<Acc3D>::getNumberOfSegmentsByLayer(unsigned int layer) {
  if (layer == 6)
    return n_segments_by_layer_barrel_[layer];
  else
    return n_segments_by_layer_barrel_[layer] + n_segments_by_layer_endcap_[layer];
}

unsigned int lst::Event<Acc3D>::getNumberOfSegmentsByLayerBarrel(unsigned int layer) {
  return n_segments_by_layer_barrel_[layer];
}

unsigned int lst::Event<Acc3D>::getNumberOfSegmentsByLayerEndcap(unsigned int layer) {
  return n_segments_by_layer_endcap_[layer];
}

unsigned int lst::Event<Acc3D>::getNumberOfTriplets() {
  unsigned int triplets = 0;
  for (auto& it : n_triplets_by_layer_barrel_) {
    triplets += it;
  }
  for (auto& it : n_triplets_by_layer_endcap_) {
    triplets += it;
  }

  return triplets;
}

unsigned int lst::Event<Acc3D>::getNumberOfTripletsByLayer(unsigned int layer) {
  if (layer == 6)
    return n_triplets_by_layer_barrel_[layer];
  else
    return n_triplets_by_layer_barrel_[layer] + n_triplets_by_layer_endcap_[layer];
}

unsigned int lst::Event<Acc3D>::getNumberOfTripletsByLayerBarrel(unsigned int layer) {
  return n_triplets_by_layer_barrel_[layer];
}

unsigned int lst::Event<Acc3D>::getNumberOfTripletsByLayerEndcap(unsigned int layer) {
  return n_triplets_by_layer_endcap_[layer];
}

int lst::Event<Acc3D>::getNumberOfPixelTriplets() {
  auto nPixelTriplets_buf_h = cms::alpakatools::make_host_buffer<unsigned int[]>(queue, 1u);

  alpaka::memcpy(queue, nPixelTriplets_buf_h, pixelTripletsBuffers->nPixelTriplets_buf);

  return *nPixelTriplets_buf_h.data();
}

int lst::Event<Acc3D>::getNumberOfPixelQuintuplets() {
  auto nPixelQuintuplets_buf_h = cms::alpakatools::make_host_buffer<unsigned int[]>(queue, 1u);

  alpaka::memcpy(queue, nPixelQuintuplets_buf_h, pixelQuintupletsBuffers->nPixelQuintuplets_buf);

  return *nPixelQuintuplets_buf_h.data();
}

unsigned int lst::Event<Acc3D>::getNumberOfQuintuplets() {
  unsigned int quintuplets = 0;
  for (auto& it : n_quintuplets_by_layer_barrel_) {
    quintuplets += it;
  }
  for (auto& it : n_quintuplets_by_layer_endcap_) {
    quintuplets += it;
  }

  return quintuplets;
}

unsigned int lst::Event<Acc3D>::getNumberOfQuintupletsByLayer(unsigned int layer) {
  if (layer == 6)
    return n_quintuplets_by_layer_barrel_[layer];
  else
    return n_quintuplets_by_layer_barrel_[layer] + n_quintuplets_by_layer_endcap_[layer];
}

unsigned int lst::Event<Acc3D>::getNumberOfQuintupletsByLayerBarrel(unsigned int layer) {
  return n_quintuplets_by_layer_barrel_[layer];
}

unsigned int lst::Event<Acc3D>::getNumberOfQuintupletsByLayerEndcap(unsigned int layer) {
  return n_quintuplets_by_layer_endcap_[layer];
}

int lst::Event<Acc3D>::getNumberOfTrackCandidates() {
  auto nTrackCandidates_buf_h = cms::alpakatools::make_host_buffer<unsigned int[]>(queue, 1u);

  alpaka::memcpy(queue, nTrackCandidates_buf_h, trackCandidatesBuffers->nTrackCandidates_buf);

  return *nTrackCandidates_buf_h.data();
}

int lst::Event<Acc3D>::getNumberOfPT5TrackCandidates() {
  auto nTrackCandidatesPT5_buf_h = cms::alpakatools::make_host_buffer<unsigned int[]>(queue, 1u);

  alpaka::memcpy(queue, nTrackCandidatesPT5_buf_h, trackCandidatesBuffers->nTrackCandidatespT5_buf);
  alpaka::wait(queue);

  return *nTrackCandidatesPT5_buf_h.data();
}

int lst::Event<Acc3D>::getNumberOfPT3TrackCandidates() {
  auto nTrackCandidatesPT3_buf_h = cms::alpakatools::make_host_buffer<unsigned int[]>(queue, 1u);

  alpaka::memcpy(queue, nTrackCandidatesPT3_buf_h, trackCandidatesBuffers->nTrackCandidatespT3_buf);

  return *nTrackCandidatesPT3_buf_h.data();
}

int lst::Event<Acc3D>::getNumberOfPLSTrackCandidates() {
  auto nTrackCandidatesPLS_buf_h = cms::alpakatools::make_host_buffer<unsigned int[]>(queue, 1u);

  alpaka::memcpy(queue, nTrackCandidatesPLS_buf_h, trackCandidatesBuffers->nTrackCandidatespLS_buf);

  return *nTrackCandidatesPLS_buf_h.data();
}

int lst::Event<Acc3D>::getNumberOfPixelTrackCandidates() {
  auto nTrackCandidates_buf_h = cms::alpakatools::make_host_buffer<unsigned int[]>(queue, 1u);
  auto nTrackCandidatesT5_buf_h = cms::alpakatools::make_host_buffer<unsigned int[]>(queue, 1u);

  alpaka::memcpy(queue, nTrackCandidates_buf_h, trackCandidatesBuffers->nTrackCandidates_buf);
  alpaka::memcpy(queue, nTrackCandidatesT5_buf_h, trackCandidatesBuffers->nTrackCandidatesT5_buf);

  return (*nTrackCandidates_buf_h.data()) - (*nTrackCandidatesT5_buf_h.data());
}

int lst::Event<Acc3D>::getNumberOfT5TrackCandidates() {
  auto nTrackCandidatesT5_buf_h = cms::alpakatools::make_host_buffer<unsigned int[]>(queue, 1u);

  alpaka::memcpy(queue, nTrackCandidatesT5_buf_h, trackCandidatesBuffers->nTrackCandidatesT5_buf);

  return *nTrackCandidatesT5_buf_h.data();
}

lst::HitsBuffer<DevHost>* lst::Event<Acc3D>::getHits()  //std::shared_ptr should take care of garbage collection
{
  if (hitsInCPU == nullptr) {
    auto nHits_buf_h = cms::alpakatools::make_host_buffer<unsigned int[]>(queue, 1u);
    alpaka::memcpy(queue, nHits_buf_h, hitsBuffers->nHits_buf);
    alpaka::wait(queue);  // wait for the value before using

    auto const nHits = *nHits_buf_h.data();
    hitsInCPU = new lst::HitsBuffer<DevHost>(nModules_, nHits, devHost, queue);
    hitsInCPU->setData(*hitsInCPU);

    *hitsInCPU->nHits_buf.data() = nHits;
    alpaka::memcpy(queue, hitsInCPU->idxs_buf, hitsBuffers->idxs_buf, nHits);
    alpaka::memcpy(queue, hitsInCPU->detid_buf, hitsBuffers->detid_buf, nHits);
    alpaka::memcpy(queue, hitsInCPU->xs_buf, hitsBuffers->xs_buf, nHits);
    alpaka::memcpy(queue, hitsInCPU->ys_buf, hitsBuffers->ys_buf, nHits);
    alpaka::memcpy(queue, hitsInCPU->zs_buf, hitsBuffers->zs_buf, nHits);
    alpaka::memcpy(queue, hitsInCPU->moduleIndices_buf, hitsBuffers->moduleIndices_buf, nHits);
  }
  return hitsInCPU;
}

lst::HitsBuffer<DevHost>* lst::Event<Acc3D>::getHitsInCMSSW() {
  if (hitsInCPU == nullptr) {
    auto nHits_buf_h = cms::alpakatools::make_host_buffer<unsigned int[]>(queue, 1u);
    alpaka::memcpy(queue, nHits_buf_h, hitsBuffers->nHits_buf);
    alpaka::wait(queue);  // wait for the value before using

    auto const nHits = *nHits_buf_h.data();
    hitsInCPU = new lst::HitsBuffer<DevHost>(nModules_, nHits, devHost, queue);
    hitsInCPU->setData(*hitsInCPU);

    *hitsInCPU->nHits_buf.data() = nHits;
    alpaka::memcpy(queue, hitsInCPU->idxs_buf, hitsBuffers->idxs_buf, nHits);
  }
  return hitsInCPU;
}

lst::ObjectRangesBuffer<DevHost>* lst::Event<Acc3D>::getRanges() {
  if (rangesInCPU == nullptr) {
    rangesInCPU = new lst::ObjectRangesBuffer<DevHost>(nModules_, nLowerModules_, devHost, queue);
    rangesInCPU->setData(*rangesInCPU);

    alpaka::memcpy(queue, rangesInCPU->hitRanges_buf, rangesBuffers->hitRanges_buf);
    alpaka::memcpy(queue, rangesInCPU->quintupletModuleIndices_buf, rangesBuffers->quintupletModuleIndices_buf);
    alpaka::memcpy(queue, rangesInCPU->miniDoubletModuleIndices_buf, rangesBuffers->miniDoubletModuleIndices_buf);
    alpaka::memcpy(queue, rangesInCPU->segmentModuleIndices_buf, rangesBuffers->segmentModuleIndices_buf);
    alpaka::memcpy(queue, rangesInCPU->tripletModuleIndices_buf, rangesBuffers->tripletModuleIndices_buf);
    alpaka::wait(queue);  // wait to get completed host data
  }
  return rangesInCPU;
}

lst::MiniDoubletsBuffer<DevHost>* lst::Event<Acc3D>::getMiniDoublets() {
  if (mdsInCPU == nullptr) {
    // Get nMemoryLocations parameter to initialize host based mdsInCPU
    auto nMemHost_buf_h = cms::alpakatools::make_host_buffer<unsigned int[]>(queue, 1u);
    alpaka::memcpy(queue, nMemHost_buf_h, miniDoubletsBuffers->nMemoryLocations_buf);
    alpaka::wait(queue);  // wait for the value before using

    auto const nMemHost = *nMemHost_buf_h.data();
    mdsInCPU = new lst::MiniDoubletsBuffer<DevHost>(nMemHost, nLowerModules_, devHost, queue);
    mdsInCPU->setData(*mdsInCPU);

    *mdsInCPU->nMemoryLocations_buf.data() = nMemHost;
    alpaka::memcpy(queue, mdsInCPU->anchorHitIndices_buf, miniDoubletsBuffers->anchorHitIndices_buf, nMemHost);
    alpaka::memcpy(queue, mdsInCPU->outerHitIndices_buf, miniDoubletsBuffers->outerHitIndices_buf, nMemHost);
    alpaka::memcpy(queue, mdsInCPU->dphichanges_buf, miniDoubletsBuffers->dphichanges_buf, nMemHost);
    alpaka::memcpy(queue, mdsInCPU->nMDs_buf, miniDoubletsBuffers->nMDs_buf);
    alpaka::memcpy(queue, mdsInCPU->totOccupancyMDs_buf, miniDoubletsBuffers->totOccupancyMDs_buf);
  }
  return mdsInCPU;
}

lst::SegmentsBuffer<DevHost>* lst::Event<Acc3D>::getSegments() {
  if (segmentsInCPU == nullptr) {
    // Get nMemoryLocations parameter to initialize host based segmentsInCPU
    auto nMemHost_buf_h = cms::alpakatools::make_host_buffer<unsigned int[]>(queue, 1u);
    alpaka::memcpy(queue, nMemHost_buf_h, segmentsBuffers->nMemoryLocations_buf);
    alpaka::wait(queue);  // wait for the value before using

    auto const nMemHost = *nMemHost_buf_h.data();
    segmentsInCPU =
        new lst::SegmentsBuffer<DevHost>(nMemHost, nLowerModules_, n_max_pixel_segments_per_module, devHost, queue);
    segmentsInCPU->setData(*segmentsInCPU);

    *segmentsInCPU->nMemoryLocations_buf.data() = nMemHost;
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
  }
  return segmentsInCPU;
}

lst::TripletsBuffer<DevHost>* lst::Event<Acc3D>::getTriplets() {
  if (tripletsInCPU == nullptr) {
    // Get nMemoryLocations parameter to initialize host based tripletsInCPU
    auto nMemHost_buf_h = cms::alpakatools::make_host_buffer<unsigned int[]>(queue, 1u);
    alpaka::memcpy(queue, nMemHost_buf_h, tripletsBuffers->nMemoryLocations_buf);
    alpaka::wait(queue);  // wait for the value before using

    auto const nMemHost = *nMemHost_buf_h.data();
    tripletsInCPU = new lst::TripletsBuffer<DevHost>(nMemHost, nLowerModules_, devHost, queue);
    tripletsInCPU->setData(*tripletsInCPU);

    *tripletsInCPU->nMemoryLocations_buf.data() = nMemHost;
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
  }
  return tripletsInCPU;
}

lst::QuintupletsBuffer<DevHost>* lst::Event<Acc3D>::getQuintuplets() {
  if (quintupletsInCPU == nullptr) {
    // Get nMemoryLocations parameter to initialize host based quintupletsInCPU
    auto nMemHost_buf_h = cms::alpakatools::make_host_buffer<unsigned int[]>(queue, 1u);
    alpaka::memcpy(queue, nMemHost_buf_h, quintupletsBuffers->nMemoryLocations_buf);
    alpaka::wait(queue);  // wait for the value before using

    auto const nMemHost = *nMemHost_buf_h.data();
    quintupletsInCPU = new lst::QuintupletsBuffer<DevHost>(nMemHost, nLowerModules_, devHost, queue);
    quintupletsInCPU->setData(*quintupletsInCPU);

    *quintupletsInCPU->nMemoryLocations_buf.data() = nMemHost;
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
  }
  return quintupletsInCPU;
}

lst::PixelTripletsBuffer<DevHost>* lst::Event<Acc3D>::getPixelTriplets() {
  if (pixelTripletsInCPU == nullptr) {
    // Get nPixelTriplets parameter to initialize host based quintupletsInCPU
    auto nPixelTriplets_buf_h = cms::alpakatools::make_host_buffer<unsigned int[]>(queue, 1u);
    alpaka::memcpy(queue, nPixelTriplets_buf_h, pixelTripletsBuffers->nPixelTriplets_buf);
    alpaka::wait(queue);  // wait for the value before using

    auto const nPixelTriplets = *nPixelTriplets_buf_h.data();
    pixelTripletsInCPU = new lst::PixelTripletsBuffer<DevHost>(nPixelTriplets, devHost, queue);
    pixelTripletsInCPU->setData(*pixelTripletsInCPU);

    *pixelTripletsInCPU->nPixelTriplets_buf.data() = nPixelTriplets;
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
  }
  return pixelTripletsInCPU;
}

lst::PixelQuintupletsBuffer<DevHost>* lst::Event<Acc3D>::getPixelQuintuplets() {
  if (pixelQuintupletsInCPU == nullptr) {
    // Get nPixelQuintuplets parameter to initialize host based quintupletsInCPU
    auto nPixelQuintuplets_buf_h = cms::alpakatools::make_host_buffer<unsigned int[]>(queue, 1u);
    alpaka::memcpy(queue, nPixelQuintuplets_buf_h, pixelQuintupletsBuffers->nPixelQuintuplets_buf);
    alpaka::wait(queue);  // wait for the value before using

    auto const nPixelQuintuplets = *nPixelQuintuplets_buf_h.data();
    pixelQuintupletsInCPU = new lst::PixelQuintupletsBuffer<DevHost>(nPixelQuintuplets, devHost, queue);
    pixelQuintupletsInCPU->setData(*pixelQuintupletsInCPU);

    *pixelQuintupletsInCPU->nPixelQuintuplets_buf.data() = nPixelQuintuplets;
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
  }
  return pixelQuintupletsInCPU;
}

lst::TrackCandidatesBuffer<DevHost>* lst::Event<Acc3D>::getTrackCandidates() {
  if (trackCandidatesInCPU == nullptr) {
    // Get nTrackCanHost parameter to initialize host based trackCandidatesInCPU
    auto nTrackCanHost_buf_h = cms::alpakatools::make_host_buffer<unsigned int[]>(queue, 1u);
    alpaka::memcpy(queue, nTrackCanHost_buf_h, trackCandidatesBuffers->nTrackCandidates_buf);
    alpaka::wait(queue);

    auto const nTrackCanHost = *nTrackCanHost_buf_h.data();
    trackCandidatesInCPU = new lst::TrackCandidatesBuffer<DevHost>(
        n_max_nonpixel_track_candidates + n_max_pixel_track_candidates, devHost, queue);
    trackCandidatesInCPU->setData(*trackCandidatesInCPU);

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
  }
  return trackCandidatesInCPU;
}

lst::TrackCandidatesBuffer<DevHost>* lst::Event<Acc3D>::getTrackCandidatesInCMSSW() {
  if (trackCandidatesInCPU == nullptr) {
    // Get nTrackCanHost parameter to initialize host based trackCandidatesInCPU
    auto nTrackCanHost_buf_h = cms::alpakatools::make_host_buffer<unsigned int[]>(queue, 1u);
    alpaka::memcpy(queue, nTrackCanHost_buf_h, trackCandidatesBuffers->nTrackCandidates_buf);
    alpaka::wait(queue);  // wait for the value before using

    auto const nTrackCanHost = *nTrackCanHost_buf_h.data();
    trackCandidatesInCPU = new lst::TrackCandidatesBuffer<DevHost>(
        n_max_nonpixel_track_candidates + n_max_pixel_track_candidates, devHost, queue);
    trackCandidatesInCPU->setData(*trackCandidatesInCPU);

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
  }
  return trackCandidatesInCPU;
}

lst::ModulesBuffer<DevHost>* lst::Event<Acc3D>::getModules(bool isFull) {
  if (modulesInCPU == nullptr) {
    // The last input here is just a small placeholder for the allocation.
    modulesInCPU = new lst::ModulesBuffer<DevHost>(devHost, nModules_, nPixels_);

    modulesInCPU->copyFromSrc(queue, modulesBuffers_, isFull);
  }
  return modulesInCPU;
}
