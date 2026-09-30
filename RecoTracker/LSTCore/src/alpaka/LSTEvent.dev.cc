#include "HeterogeneousCore/AlpakaInterface/interface/memory.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"
#include "HeterogeneousCore/AlpakaInterface/interface/CopyToDevice.h"

#include "LSTEvent.h"

#include "Hit.h"
#include "Kernels.h"
#include "MiniDoublet.h"
#include "ObjectCompaction.h"
#include "PixelQuintuplet.h"
#include "PixelTriplet.h"
#include "Quintuplet.h"
#include "Segment.h"
#include "TrackCandidate.h"
#include "Triplet.h"
#include "Quadruplet.h"

#include <format>

using Device = ALPAKA_ACCELERATOR_NAMESPACE::Device;
using Queue = ALPAKA_ACCELERATOR_NAMESPACE::Queue;
using Acc1D = ALPAKA_ACCELERATOR_NAMESPACE::Acc1D;
using Acc2D = ALPAKA_ACCELERATOR_NAMESPACE::Acc2D;
using Acc3D = ALPAKA_ACCELERATOR_NAMESPACE::Acc3D;

using namespace ALPAKA_ACCELERATOR_NAMESPACE::lst;

namespace {
  // One block per lower module on a GPU; on a serial backend one block loops over the modules in order.
  auto perModuleWorkDiv(uint16_t nLowerModules) {
    constexpr bool kSerial = cms::alpakatools::requires_single_thread_per_block_v<Acc2D>;
    return cms::alpakatools::make_workdiv<Acc2D>({kSerial ? 1u : nLowerModules, 1u}, {1u, 32u});
  }
}  // namespace

void LSTEvent::initSync() {
  alpaka::wait(queue_);  // other calls can be asynchronous

  //reset the arrays
  for (int i = 0; i < 6; i++) {
    n_minidoublets_by_layer_barrel_[i] = 0;
    n_segments_by_layer_barrel_[i] = 0;
    n_triplets_by_layer_barrel_[i] = 0;
    n_quintuplets_by_layer_barrel_[i] = 0;
    n_quadruplets_by_layer_barrel_[i] = 0;
    if (i < 5) {
      n_minidoublets_by_layer_endcap_[i] = 0;
      n_segments_by_layer_endcap_[i] = 0;
      n_triplets_by_layer_endcap_[i] = 0;
      n_quintuplets_by_layer_endcap_[i] = 0;
      n_quadruplets_by_layer_endcap_[i] = 0;
    }
  }
}

void LSTEvent::resetEventSync() {
  alpaka::wait(queue_);  // synchronize to reset consistently
  //reset the arrays
  for (int i = 0; i < 6; i++) {
    n_minidoublets_by_layer_barrel_[i] = 0;
    n_segments_by_layer_barrel_[i] = 0;
    n_triplets_by_layer_barrel_[i] = 0;
    n_quintuplets_by_layer_barrel_[i] = 0;
    n_quadruplets_by_layer_barrel_[i] = 0;
    if (i < 5) {
      n_minidoublets_by_layer_endcap_[i] = 0;
      n_segments_by_layer_endcap_[i] = 0;
      n_triplets_by_layer_endcap_[i] = 0;
      n_quintuplets_by_layer_endcap_[i] = 0;
      n_quadruplets_by_layer_endcap_[i] = 0;
    }
  }
  memoryAllocatedMB_ = 0;
  memoryLiveMB_ = 0;
  nT4Compact_ = 0;
  superbinsHost_.reset();
  pixelTypesHost_.reset();
  connectedPixelSize_.reset();
  connectedPixelIndex_.reset();
  memoryPeakLiveMB_ = 0;
  lstInputDC_ = nullptr;
  hitsDC_.reset();
  rangesDC_.reset();
  miniDoubletsDC_.reset();
  miniDoubletsBuildDC_.reset();
  segmentsT3CountsDC_.reset();
  miniDoubletsT5BuildDC_.reset();
  segmentsDC_.reset();
  pixelSegmentsDC_.reset();
  tripletsDC_.reset();
  tripletsListRangesDC_.reset();
  quintupletsDC_.reset();
  trackCandidatesBaseDC_.reset();
  trackCandidatesExtendedDC_.reset();
  pixelTripletsDC_.reset();
  pixelQuintupletsDC_.reset();
  quadrupletsDC_.reset();

  lstInputHC_.reset();
  hitsHC_.reset();
  rangesHC_.reset();
  miniDoubletsHC_.reset();
  miniDoubletsBuildHC_.reset();
  segmentsHC_.reset();
  pixelSegmentsHC_.reset();
  tripletsHC_.reset();
  quintupletsHC_.reset();
  pixelTripletsHC_.reset();
  pixelQuintupletsHC_.reset();
  trackCandidatesBaseHC_.reset();
  trackCandidatesExtendedHC_.reset();
  modulesHC_.reset();
  quadrupletsHC_.reset();
}

void LSTEvent::trackAllocatedMB(double mb) {
  memoryAllocatedMB_ += mb;
  trackTransientMB(mb);
}

void LSTEvent::trackTransientMB(double mb) {
  memoryLiveMB_ += mb;
  memoryPeakLiveMB_ = std::max(memoryPeakLiveMB_, memoryLiveMB_);
}

template <typename TDC, typename THC>
void LSTEvent::releaseDeviceCollection(std::optional<TDC>& dc, std::optional<THC>& hc) {
  if (!dc)
    return;
  if (objectsStatistics_)
    memoryLiveMB_ -= alpaka::getExtentProduct(dc->buffer()) / 1e6;
  if (keepHostCopies_ && !hc) {
    if constexpr (std::is_same_v<Device, DevHost>) {
      hc.emplace(std::move(*dc));  // same type on the host backend: the buffer moves to the host readers
    } else {
      hc.emplace(cms::alpakatools::CopyToHost<TDC>::copyAsync(queue_, *dc));
    }
  }
  // Queue-ordered free: the caching allocator reuses the block only after the queued work that reads it.
  dc.reset();
}

template <typename TDC>
void LSTEvent::releaseDeviceCollection(std::optional<TDC>& dc) {
  if (!dc)
    return;
  if (objectsStatistics_)
    memoryLiveMB_ -= alpaka::getExtentProduct(dc->buffer()) / 1e6;
  dc.reset();
}

void LSTEvent::addInputToEvent(LSTInputDeviceCollection const* lstInputDC) {
  lstInputDC_ = lstInputDC;

  pixelSize_ = lstInputDC_->size()[1];
  pixelModuleIndex_ = pixelMapping_.pixelModuleIndex;
}

void LSTEvent::addHitToEvent() {
  if (!hitsDC_) {
    const int32_t nHits = lstInputDC_->size()[0];
    hitsDC_.emplace(queue_, nHits, nModules_);
    auto buf = hitsDC_->buffer();
    alpaka::memset(queue_, buf, 0xff);
    if (objectsStatistics_) {
      double mb = alpaka::getExtentProduct(hitsDC_->buffer()) / 1e6;
      trackAllocatedMB(mb);
      lstWarning(std::format("[MEM] Hits: {} allocated ({:.1f} MB)", nHits, mb));
    }
  }

  if (!rangesDC_) {
    rangesDC_.emplace(queue_, nLowerModules_ + 1);
    auto buf = rangesDC_->buffer();
    alpaka::memset(queue_, buf, 0xff);
    if (objectsStatistics_) {
      double mb = alpaka::getExtentProduct(rangesDC_->buffer()) / 1e6;
      trackAllocatedMB(mb);
      lstWarning(std::format("[MEM] Ranges: {} allocated ({:.1f} MB)", nLowerModules_ + 1, mb));
    }
  }

  // One thread per hit: the per-hit module lookup is latency bound.
  const unsigned int nHitsIn = lstInputDC_->size()[0];
  auto const hit_loop_workdiv =
      cms::alpakatools::make_workdiv<Acc1D>(std::max(cms::alpakatools::divide_up_by(nHitsIn, 256u), 1u), 256);

  alpaka::exec<Acc1D>(queue_,
                      hit_loop_workdiv,
                      HitLoopKernel{},
                      nModules_,
                      modules_.const_view().modules(),
                      lstInputDC_->const_view().hits(),
                      hitsDC_->view().extended(),
                      hitsDC_->view().ranges());

  auto const module_ranges_workdiv = cms::alpakatools::make_workdiv<Acc1D>(max_blocks, 256);

  alpaka::exec<Acc1D>(queue_,
                      module_ranges_workdiv,
                      ModuleRangesKernel{},
                      modules_.const_view().modules(),
                      hitsDC_->view().ranges(),
                      nLowerModules_);
}

void LSTEvent::addPixelSegmentToEvent() {
  if (pixelSize_ == n_max_pixel_segments_per_module) {
    lstWarning(
        "\
          *********************************************************\n\
          * Warning: Pixel line segments may be truncated.        *\n\
          * You need to increase n_max_pixel_segments_per_module. *\n\
          *********************************************************");
  }

  if (!pixelSegmentsDC_) {
    pixelSegmentsDC_.emplace(queue_, pixelSize_);
    if (objectsStatistics_) {
      double mb = alpaka::getExtentProduct(pixelSegmentsDC_->buffer()) / 1e6;
      trackAllocatedMB(mb);
      lstWarning(std::format("[MEM] PixelSegments: {} allocated ({:.1f} MB)", pixelSize_, mb));
    }
  }

  auto const addPixelSegmentToEvent_workdiv = cms::alpakatools::make_workdiv<Acc1D>(max_blocks, 256);

  alpaka::exec<Acc1D>(queue_,
                      addPixelSegmentToEvent_workdiv,
                      AddPixelSegmentToEventKernel{},
                      rangesDC_->const_view(),
                      lstInputDC_->const_view().hits(),
                      lstInputDC_->const_view().hitsIT(),
                      lstInputDC_->const_view().pixelSeeds(),
                      miniDoubletsDC_->const_view().miniDoublets(),
                      segmentsDC_->view().segments(),
                      pixelSegmentsDC_->view(),
                      pixelModuleIndex_,
                      pixelSize_);
}

void LSTEvent::createMiniDoublets() {
  // Hit pairs accepted by CountMiniDoublets, one bit mask per lower hit (indexed by hit); transient.
  const int32_t nHits = lstInputDC_->size()[0];
  auto mdPassMask_buf = cms::alpakatools::make_device_buffer<uint64_t[]>(queue_, nHits);
  // Cut terms of the hits that set the MD cut, once per hit (indexed by hit); transient.
  auto mdHitTerms_buf = cms::alpakatools::make_device_buffer<MDHitTerms[]>(queue_, nHits);
  {
    constexpr int threadsPerBlockY = 16;
    alpaka::exec<Acc2D>(queue_,
                        cms::alpakatools::make_workdiv<Acc2D>(
                            {(nLowerModules_ + threadsPerBlockY - 1) / threadsPerBlockY, 1}, {threadsPerBlockY, 32}),
                        FillMDHitTerms{},
                        modules_.const_view().modules(),
                        hitsDC_->const_view().extended(),
                        hitsDC_->const_view().ranges(),
                        mdHitTerms_buf.data(),
                        ptCut_);
  }

  if (!miniDoubletsDC_) {
    auto rangesOccupancy = rangesDC_->view();

    // Zero the occupancy array so CountMiniDoublets's atomicAdd starts from 0.
    auto miniDoubletModuleOccupancy_view =
        cms::alpakatools::make_device_view(queue_, rangesOccupancy.miniDoubletModuleOccupancy());
    alpaka::memset(queue_, miniDoubletModuleOccupancy_view, 0u);

    // Upper hits of each lower module binned along the module (indexed by module / by hit; CPU only); transient.
    auto mdUpperBins_buf =
        cms::alpakatools::make_device_buffer<MDUpperBins[]>(queue_, kMDBinUpperHits ? nLowerModules_ : 1);
    auto mdBinStart_buf = cms::alpakatools::make_device_buffer<uint16_t[]>(queue_, kMDBinUpperHits ? nHits : 1);
    auto mdBinPerm_buf = cms::alpakatools::make_device_buffer<uint16_t[]>(queue_, kMDBinUpperHits ? nHits : 1);
    if constexpr (kMDBinUpperHits) {
      alpaka::exec<Acc1D>(queue_,
                          cms::alpakatools::make_workdiv<Acc1D>((nLowerModules_ + 127) / 128, 128),
                          FillMDUpperBins{},
                          modules_.const_view().modules(),
                          lstInputDC_->const_view().hits(),
                          hitsDC_->const_view().extended(),
                          hitsDC_->const_view().ranges(),
                          mdHitTerms_buf.data(),
                          mdUpperBins_buf.data(),
                          mdBinStart_buf.data(),
                          mdBinPerm_buf.data(),
                          ptCut_);
    }

    // 256 threads per block, as CreateMiniDoublets: up to 128 registers per thread with the CMSSW nvcc flags.
    constexpr int threadsPerBlockY = 8;
    auto const countMiniDoublets_workDiv =
        cms::alpakatools::make_workdiv<Acc2D>({nLowerModules_ / threadsPerBlockY, 1}, {threadsPerBlockY, 32});

    alpaka::exec<Acc2D>(queue_,
                        countMiniDoublets_workDiv,
                        CountMiniDoublets{},
                        modules_.const_view().modules(),
                        lstInputDC_->const_view().hits(),
                        hitsDC_->const_view().extended(),
                        hitsDC_->const_view().ranges(),
                        rangesDC_->view(),
                        mdPassMask_buf.data(),
                        mdHitTerms_buf.data(),
                        mdUpperBins_buf.data(),
                        mdBinStart_buf.data(),
                        mdBinPerm_buf.data(),
                        ptCut_,
                        clustSizeCut_);

    auto const createMDArrayRangesGPU_workDiv = cms::alpakatools::make_workdiv<Acc1D>(1, 1024);

    alpaka::exec<Acc1D>(queue_,
                        createMDArrayRangesGPU_workDiv,
                        CreateMDArrayRangesGPU{},
                        modules_.const_view().modules(),
                        rangesDC_->view());

    auto nTotalMDs_buf_h = cms::alpakatools::make_host_buffer<unsigned int>(queue_);
    auto nTotalMDs_buf_d = cms::alpakatools::make_device_view(queue_, rangesOccupancy.nTotalMDs());
    alpaka::memcpy(queue_, nTotalMDs_buf_h, nTotalMDs_buf_d);
    copyPixelSeedTypesToHost();  // read with this sync, so that the pT5 and pT3 stages need none
    alpaka::wait(queue_);        // wait to get the data before manipulation

    nTotalMDsOT_ = *nTotalMDs_buf_h.data();
    *nTotalMDs_buf_h.data() += 2 * pixelSize_;
    unsigned int nTotalMDs = *nTotalMDs_buf_h.data();

    miniDoubletsDC_.emplace(queue_, nTotalMDs, nLowerModules_ + 1);
    miniDoubletsBuildDC_.emplace(queue_, nTotalMDs);
    if (objectsStatistics_) {
      double mb = alpaka::getExtentProduct(miniDoubletsDC_->buffer()) / 1e6;
      trackAllocatedMB(mb);
      lstWarning(std::format("[MEM] MiniDoublets: {} allocated ({:.1f} MB)", nTotalMDs, mb));
      mb = alpaka::getExtentProduct(miniDoubletsBuildDC_->buffer()) / 1e6;
      trackAllocatedMB(mb);
      lstWarning(std::format("[MEM] MiniDoubletsBuild: {} allocated ({:.1f} MB)", nTotalMDs, mb));
    }

    auto mdsOccupancy = miniDoubletsDC_->view().miniDoubletsOccupancy();
    auto nMDs_view = cms::alpakatools::make_device_view(queue_, mdsOccupancy.nMDs());
    auto totOccupancyMDs_view = cms::alpakatools::make_device_view(queue_, mdsOccupancy.totOccupancyMDs());
    alpaka::memset(queue_, nMDs_view, 0u);
    alpaka::memset(queue_, totOccupancyMDs_view, 0u);
  }

  auto connView = cms::alpakatools::make_device_view(queue_, miniDoubletsBuildDC_->view().connectedMax());
  alpaka::memset(queue_, connView, 0u);

  auto mdSize_buf_h = cms::alpakatools::make_host_buffer<unsigned int>(queue_);
  *mdSize_buf_h.data() = pixelSize_ * 2;

  auto mdsOccupancy = miniDoubletsDC_->view().miniDoubletsOccupancy();
  auto dst_view_nMDs = cms::alpakatools::make_device_view(queue_, mdsOccupancy.nMDs()[pixelModuleIndex_]);
  alpaka::memcpy(queue_, dst_view_nMDs, mdSize_buf_h);

  auto dst_view_totOccupancyMDs =
      cms::alpakatools::make_device_view(queue_, mdsOccupancy.totOccupancyMDs()[pixelModuleIndex_]);
  alpaka::memcpy(queue_, dst_view_totOccupancyMDs, mdSize_buf_h);

  // 256 threads per block: with the CMSSW nvcc flags (no fast-math) this kernel needs up to 176 registers per thread,
  // so 512-thread blocks exceed the register file and fail to launch.
  constexpr int threadsPerBlockY = 8;
  auto const createMiniDoublets_workDiv =
      cms::alpakatools::make_workdiv<Acc2D>({nLowerModules_ / threadsPerBlockY, 1}, {threadsPerBlockY, 32});

  alpaka::exec<Acc2D>(queue_,
                      createMiniDoublets_workDiv,
                      CreateMiniDoublets{},
                      modules_.const_view().modules(),
                      lstInputDC_->const_view().hits(),
                      hitsDC_->const_view().extended(),
                      hitsDC_->const_view().ranges(),
                      miniDoubletsDC_->view().miniDoublets(),
                      miniDoubletsBuildDC_->view(),
                      miniDoubletsDC_->view().miniDoubletsOccupancy(),
                      rangesDC_->const_view(),
                      mdPassMask_buf.data(),
                      mdHitTerms_buf.data(),
                      ptCut_,
                      clustSizeCut_);

  // Pixel MDs (two per pLS) at the pixel-module slots reserved above.
  auto const addPixelMiniDoublets_workdiv = cms::alpakatools::make_workdiv<Acc1D>(max_blocks, 256);

  alpaka::exec<Acc1D>(queue_,
                      addPixelMiniDoublets_workdiv,
                      AddPixelMiniDoubletsToEventKernel{},
                      modules_.const_view().modules(),
                      rangesDC_->const_view(),
                      lstInputDC_->const_view().hits(),
                      hitsDC_->const_view().extended(),
                      lstInputDC_->const_view().pixelSeeds(),
                      miniDoubletsDC_->view().miniDoublets(),
                      miniDoubletsBuildDC_->view(),
                      pixelModuleIndex_,
                      pixelSize_);

  auto const addMiniDoubletRangesToEventExplicit_workDiv =
      cms::alpakatools::make_workdiv<Acc1D>(cms::alpakatools::divide_up_by(nLowerModules_, 256), 256);

  alpaka::exec<Acc1D>(queue_,
                      addMiniDoubletRangesToEventExplicit_workDiv,
                      AddMiniDoubletRangesToEventExplicit{},
                      modules_.const_view().modules(),
                      miniDoubletsDC_->view().miniDoubletsOccupancy(),
                      rangesDC_->view(),
                      hitsDC_->const_view().ranges());

  if (objectsStatistics_) {
    addMiniDoubletsToEventExplicit();
  }

  // Last reader of the Hits collection.
  releaseDeviceCollection(hitsDC_, hitsHC_);
}

void LSTEvent::createSegmentsWithModuleMap() {
  // Called once per event: the segments are created into a count-sized candidate scratch (MD pair + outer
  // module only), then written at exact, module-ordered slots by compactSegments(); the scratch is freed here.
  // MD pairs accepted by CountMiniDoubletConnections, kSegPassMaskWords words per inner OT MD; transient.
  auto segPassMask_buf = cms::alpakatools::make_device_buffer<uint64_t[]>(queue_, nTotalMDsOT_ * kSegPassMaskWords);
  alpaka::memset(queue_, segPassMask_buf, 0u);

  // Outer-MD terms of the segment selection (asin of the anchor rt), once per OT MD; transient.
  auto outerMDTerms_buf = cms::alpakatools::make_device_buffer<SegOuterMDTerms[]>(queue_, nTotalMDsOT_);
  alpaka::exec<Acc1D>(queue_,
                      cms::alpakatools::make_workdiv<Acc1D>(std::max((nTotalMDsOT_ + 255) / 256, 1u), 256),
                      FillSegOuterMDTerms{},
                      miniDoubletsDC_->const_view().miniDoublets(),
                      nTotalMDsOT_,
                      outerMDTerms_buf.data(),
                      ptCut_);

  auto const countMDConn_wd = cms::alpakatools::make_workdiv<Acc3D>({nLowerModules_, 1, 1}, {1, 8, 32});

  alpaka::exec<Acc3D>(queue_,
                      countMDConn_wd,
                      CountMiniDoubletConnections{},
                      modules_.const_view().modules(),
                      miniDoubletsDC_->const_view().miniDoublets(),
                      miniDoubletsBuildDC_->view(),
                      miniDoubletsDC_->const_view().miniDoubletsOccupancy(),
                      rangesDC_->const_view(),
                      segPassMask_buf.data(),
                      outerMDTerms_buf.data(),
                      ptCut_);

  // The totals are summed over the module blocks.
  auto rangesOccupancy = rangesDC_->view();
  auto nTotalSegs_view = cms::alpakatools::make_device_view(queue_, rangesOccupancy.nTotalSegs());
  alpaka::memset(queue_, nTotalSegs_view, 0u);
  auto nSegmentOverflows_view = cms::alpakatools::make_device_view(queue_, rangesOccupancy.nSegmentOverflows());
  alpaka::memset(queue_, nSegmentOverflows_view, 0u);
  auto const createSegmentArrayRanges_workDiv = perModuleWorkDiv(nLowerModules_);

  alpaka::exec<Acc2D>(queue_,
                      createSegmentArrayRanges_workDiv,
                      CreateSegmentArrayRanges{},
                      modules_.const_view().modules(),
                      rangesDC_->view(),
                      miniDoubletsBuildDC_->const_view(),
                      miniDoubletsDC_->const_view().miniDoubletsOccupancy());

  auto nTotalSegments_view_h = cms::alpakatools::make_host_view(nTotalSegmentsOT_);
  auto nTotalSegments_view_d = cms::alpakatools::make_device_view(queue_, rangesOccupancy.nTotalSegs());
  alpaka::memcpy(queue_, nTotalSegments_view_h, nTotalSegments_view_d);
  alpaka::wait(queue_);  // wait to get the value before manipulation

  SegmentCandidatesDeviceCollection candidatesDC(queue_, nTotalSegmentsOT_, nLowerModules_ + 1);
  double candidatesMB = 0;
  if (objectsStatistics_) {
    // Transient: live until the end of this function, not added to the allocated total.
    candidatesMB = alpaka::getExtentProduct(candidatesDC.buffer()) / 1e6;
    trackTransientMB(candidatesMB);
    lstWarning(
        std::format("[MEM] (transient) SegmentCandidates: {} allocated ({:.1f} MB)", nTotalSegmentsOT_, candidatesMB));
  }

  auto segmentsOccupancy = candidatesDC.view().segmentsOccupancy();
  auto nSegments_view = cms::alpakatools::make_device_view(queue_, segmentsOccupancy.nSegments());
  auto totOccupancySegments_view = cms::alpakatools::make_device_view(queue_, segmentsOccupancy.totOccupancySegments());
  alpaka::memset(queue_, nSegments_view, 0u);
  alpaka::memset(queue_, totOccupancySegments_view, 0u);

  auto src_view_size = cms::alpakatools::make_host_view(pixelSize_);

  auto dst_view_segments = cms::alpakatools::make_device_view(queue_, segmentsOccupancy.nSegments()[pixelModuleIndex_]);
  alpaka::memcpy(queue_, dst_view_segments, src_view_size);

  auto dst_view_totOccupancySegments =
      cms::alpakatools::make_device_view(queue_, segmentsOccupancy.totOccupancySegments()[pixelModuleIndex_]);
  alpaka::memcpy(queue_, dst_view_totOccupancySegments, src_view_size);

  auto const createSegments_workDiv = cms::alpakatools::make_workdiv<Acc3D>({nLowerModules_, 1, 1}, {1, 8, 32});

  alpaka::exec<Acc3D>(queue_,
                      createSegments_workDiv,
                      CreateSegments{},
                      modules_.const_view().modules(),
                      miniDoubletsDC_->const_view().miniDoublets(),
                      miniDoubletsBuildDC_->const_view(),
                      miniDoubletsDC_->const_view().miniDoubletsOccupancy(),
                      candidatesDC.view().candidates(),
                      candidatesDC.view().segmentsOccupancy(),
                      rangesDC_->view(),
                      segPassMask_buf.data(),
                      outerMDTerms_buf.data(),
                      ptCut_);

  compactSegments(candidatesDC);
  if (objectsStatistics_)
    memoryLiveMB_ -= candidatesMB;

  auto const addSegmentRangesToEventExplicit_workDiv =
      cms::alpakatools::make_workdiv<Acc1D>(cms::alpakatools::divide_up_by(nLowerModules_, 256), 256);

  alpaka::exec<Acc1D>(queue_,
                      addSegmentRangesToEventExplicit_workDiv,
                      AddSegmentRangesToEventExplicit{},
                      modules_.const_view().modules(),
                      segmentsDC_->view().segmentsOccupancy(),
                      rangesDC_->view());

  if (objectsStatistics_) {
    addSegmentsToEventExplicit();
  }

  // Last reader of the build-only MD columns (FillCompactSegments in compactSegments reads them).
  releaseDeviceCollection(miniDoubletsBuildDC_, miniDoubletsBuildHC_);
}

void LSTEvent::compactSegments(SegmentCandidatesDeviceCollection const& candidatesDC) {
  // Compact segment collection: OT segments at module-ordered offsets (prefix sum of nSegments), then the pLS
  // slots (filled later by addPixelSegmentToEvent at segmentModuleIndices[pixel] = nCompactOT).
  auto compactOffsets_buf = cms::alpakatools::make_device_buffer<int[]>(queue_, nLowerModules_ + 1);

  alpaka::exec<Acc1D>(queue_,
                      cms::alpakatools::make_workdiv<Acc1D>(1, 1024),
                      ComputeCompactSegmentOffsets{},
                      modules_.const_view().modules(),
                      candidatesDC.const_view().segmentsOccupancy(),
                      rangesDC_->view(),
                      compactOffsets_buf.data());

  auto nCompactOT_view_h = cms::alpakatools::make_host_view(nTotalSegmentsOT_);
  auto nCompactOT_view_d = cms::alpakatools::make_device_view(queue_, rangesDC_->view().nTotalSegs());
  alpaka::memcpy(queue_, nCompactOT_view_h, nCompactOT_view_d);
  alpaka::wait(queue_);
  nTotalSegments_ = nTotalSegmentsOT_ + pixelSize_;

  segmentsDC_.emplace(queue_, nTotalSegments_, nLowerModules_ + 1);

  alpaka::exec<Acc1D>(queue_,
                      cms::alpakatools::make_workdiv<Acc1D>(nLowerModules_, 128),
                      FillCompactSegments{},
                      modules_.const_view().modules(),
                      miniDoubletsDC_->const_view().miniDoublets(),
                      miniDoubletsBuildDC_->const_view(),
                      miniDoubletsDC_->const_view().miniDoubletsOccupancy(),
                      rangesDC_->const_view(),
                      candidatesDC.const_view().candidates(),
                      candidatesDC.const_view().segmentsOccupancy(),
                      compactOffsets_buf.data(),
                      segmentsDC_->view().segments(),
                      ptCut_);

  alpaka::exec<Acc1D>(
      queue_,
      cms::alpakatools::make_workdiv<Acc1D>(cms::alpakatools::divide_up_by(nLowerModules_ + 1, 256), 256),
      SetCompactSegmentModuleIndices{},
      modules_.const_view().modules(),
      candidatesDC.const_view().segmentsOccupancy(),
      compactOffsets_buf.data(),
      segmentsDC_->view().segmentsOccupancy(),
      rangesDC_->view());

  if (objectsStatistics_) {
    double mb = alpaka::getExtentProduct(segmentsDC_->buffer()) / 1e6;
    trackAllocatedMB(mb);
    lstWarning(std::format("[MEM] Segments: {} allocated ({:.1f} MB)", nTotalSegments_, mb));
  }
}

void LSTEvent::createTriplets() {
  // per-segment T3 counter, OT segments only (the pLSs follow the OT segments); sizes the creation buffer
  segmentsT3CountsDC_.emplace(queue_, nTotalSegmentsOT_);
  if (objectsStatistics_) {
    double mb = alpaka::getExtentProduct(segmentsT3CountsDC_->buffer()) / 1e6;
    trackAllocatedMB(mb);
    lstWarning(std::format("[MEM] SegmentsT3Counts: {} allocated ({:.1f} MB)", nTotalSegmentsOT_, mb));
  }
  auto segConnView = cms::alpakatools::make_device_view(queue_, segmentsT3CountsDC_->view().connectedMax());
  alpaka::memset(queue_, segConnView, 0u);

  // Segments grouped by inner MD, live only in this stage: the T3 kernels visit only the segments at the shared MD.
  using SegByMDBuf = cms::alpakatools::device_buffer<Device, unsigned int[]>;
  std::optional<SegByMDBuf> nSegByMD_buf, segByMDOffset_buf, segByMD_buf;
  nSegByMD_buf.emplace(cms::alpakatools::make_device_buffer<unsigned int[]>(queue_, nTotalMDsOT_));
  segByMDOffset_buf.emplace(cms::alpakatools::make_device_buffer<unsigned int[]>(queue_, nTotalMDsOT_));
  segByMD_buf.emplace(cms::alpakatools::make_device_buffer<unsigned int[]>(queue_, nTotalSegmentsOT_));
  std::optional<cms::alpakatools::device_buffer<Device, uint16_t[]>> segInnerModule_buf;
  segInnerModule_buf.emplace(cms::alpakatools::make_device_buffer<uint16_t[]>(queue_, nTotalSegmentsOT_));
  // Step-1 decisions of the count (pass / loose-pointing bits per inner segment), read by CreateTripletsT.
  using T3MaskBuf = cms::alpakatools::device_buffer<Device, uint32_t[]>;
  std::optional<T3MaskBuf> t3PassMask_buf, t3LooseMask_buf;
  t3PassMask_buf.emplace(cms::alpakatools::make_device_buffer<uint32_t[]>(queue_, nTotalSegmentsOT_));
  t3LooseMask_buf.emplace(cms::alpakatools::make_device_buffer<uint32_t[]>(queue_, nTotalSegmentsOT_));
  alpaka::memset(queue_, *t3PassMask_buf, 0u);
  alpaka::memset(queue_, *t3LooseMask_buf, 0u);
  double segByMDMB = 0;
  if (objectsStatistics_) {
    segByMDMB =
        ((2. * nTotalMDsOT_ + 3. * nTotalSegmentsOT_) * sizeof(unsigned int) + nTotalSegmentsOT_ * sizeof(uint16_t)) /
        1e6;
    trackTransientMB(segByMDMB);
    lstWarning(
        std::format("[MEM] (transient) SegmentsByInnerMD: {} segments ({:.1f} MB)", nTotalSegmentsOT_, segByMDMB));
  }
  alpaka::memset(queue_, *nSegByMD_buf, 0u);
  alpaka::exec<Acc1D>(queue_,
                      cms::alpakatools::make_workdiv<Acc1D>(nLowerModules_, 128),
                      FillSegmentsByInnerMD{},
                      modules_.const_view().modules(),
                      miniDoubletsDC_->const_view().miniDoubletsOccupancy(),
                      segmentsDC_->const_view().segments(),
                      segmentsDC_->const_view().segmentsOccupancy(),
                      rangesDC_->const_view(),
                      nSegByMD_buf->data(),
                      segByMDOffset_buf->data(),
                      segByMD_buf->data(),
                      segInnerModule_buf->data());

  // 16 inner segments per block (not one module per block).
  auto const countSegConn_wd = cms::alpakatools::make_workdiv<Acc2D>(
      {std::max(cms::alpakatools::divide_up_by(nTotalSegmentsOT_, 16u), 1u), 1u}, {16, 16});

  alpaka::exec<Acc2D>(queue_,
                      countSegConn_wd,
                      CountSegmentConnections{},
                      modules_.const_view().modules(),
                      miniDoubletsDC_->const_view().miniDoublets(),
                      segmentsT3CountsDC_->view(),
                      segmentsDC_->view().segments(),
                      ptCut_,
                      nSegByMD_buf->data(),
                      segByMDOffset_buf->data(),
                      segByMD_buf->data(),
                      segInnerModule_buf->data(),
                      nTotalSegmentsOT_,
                      t3PassMask_buf->data(),
                      t3LooseMask_buf->data());

  // The totals are summed over the module blocks.
  auto rangesOccupancy = rangesDC_->view();
  auto nTotalTrips_view = cms::alpakatools::make_device_view(queue_, rangesOccupancy.nTotalTrips());
  alpaka::memset(queue_, nTotalTrips_view, 0u);
  auto nTripletOverflows_view = cms::alpakatools::make_device_view(queue_, rangesOccupancy.nTripletOverflows());
  alpaka::memset(queue_, nTripletOverflows_view, 0u);
  alpaka::exec<Acc2D>(queue_,
                      perModuleWorkDiv(nLowerModules_),
                      CreateTripletArrayRanges{},
                      modules_.const_view().modules(),
                      rangesDC_->view(),
                      segmentsT3CountsDC_->const_view(),
                      segmentsDC_->const_view().segmentsOccupancy());

  auto maxTriplets_buf_h = cms::alpakatools::make_host_buffer<unsigned int>(queue_);
  auto maxTriplets_buf_d = cms::alpakatools::make_device_view(queue_, rangesOccupancy.nTotalTrips());
  alpaka::memcpy(queue_, maxTriplets_buf_h, maxTriplets_buf_d);

  // Allocate and copy nSegments from device to host (only nLowerModules in OT, not the +1 with pLSs)
  auto nSegments_buf_h = cms::alpakatools::make_host_buffer<unsigned int[]>(queue_, nLowerModules_);
  auto nSegments_buf_d = cms::alpakatools::make_device_view(
      queue_, segmentsDC_->const_view().segmentsOccupancy().nSegments(), nLowerModules_);
  alpaka::memcpy(queue_, nSegments_buf_h, nSegments_buf_d, nLowerModules_);

  // ... same for module_nConnectedModules
  // FIXME: replace by ES host data
  auto modules = modules_.const_view().modules();
  auto module_nConnectedModules_buf_h = cms::alpakatools::make_host_buffer<uint16_t[]>(queue_, nLowerModules_);
  auto module_nConnectedModules_buf_d =
      cms::alpakatools::make_device_view(queue_, modules.nConnectedModules(), nLowerModules_);  // only lower modules
  alpaka::memcpy(queue_, module_nConnectedModules_buf_h, module_nConnectedModules_buf_d, nLowerModules_);

  alpaka::wait(queue_);  // wait for the loose count, nSegments and module_nConnectedModules before using them

  tripletsListRangesDC_.emplace(queue_, nTotalSegmentsOT_, nTotalMDsOT_);
  double listRangesMB = 0;
  if (objectsStatistics_) {
    listRangesMB = alpaka::getExtentProduct(tripletsListRangesDC_->buffer()) / 1e6;
    trackAllocatedMB(listRangesMB);
  }
  auto nBySegment_view = cms::alpakatools::make_device_view(
      queue_, tripletsListRangesDC_->view().tripletsRangesBySegment().n(), nTotalSegmentsOT_);
  alpaka::memset(queue_, nBySegment_view, 0u);
  auto nByMD_view =
      cms::alpakatools::make_device_view(queue_, tripletsListRangesDC_->view().tripletsRangesByMD().n(), nTotalMDsOT_);
  alpaka::memset(queue_, nByMD_view, 0u);

  // Creation buffer, sized by the loose count; its triplets are compacted into tripletsDC_ below.
  unsigned int nLooseTriplets = *maxTriplets_buf_h.data();
  TripletsBuildDeviceCollection looseTripletsDC(queue_, nLooseTriplets, nLowerModules_, nLooseTriplets);
  double looseMB = 0;
  if (objectsStatistics_) {
    // Transient: live until the end of this function, not added to the allocated total.
    looseMB = alpaka::getExtentProduct(looseTripletsDC.buffer()) / 1e6;
    trackTransientMB(looseMB);
  }
  auto looseOccupancy = looseTripletsDC.view().tripletsOccupancy();
  auto nTriplets_view = cms::alpakatools::make_device_view(queue_, looseOccupancy.nTriplets());
  alpaka::memset(queue_, nTriplets_view, 0u);
  auto totOccupancyTriplets_view = cms::alpakatools::make_device_view(queue_, looseOccupancy.totOccupancyTriplets());
  alpaka::memset(queue_, totOccupancyTriplets_view, 0u);

  uint16_t nonZeroModules = 0;
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
  }

  // Allocate and copy to device index
  auto index_gpu_buf = cms::alpakatools::make_device_buffer<uint16_t[]>(queue_, nLowerModules_);
  auto const createTriplets_workDiv =
      cms::alpakatools::make_workdiv<Acc3D>({std::max<int>(nonZeroModules, 1), 1, 1}, {1, 16, 16});

  if (nonZeroModules > 0) {
    alpaka::memcpy(queue_, index_gpu_buf, index_buf_h, nonZeroModules);
    // Slots of the creation buffer that receive no match keep this flag (skipped by CreateTripletsFromMatches).
    auto scratchFlags_view =
        cms::alpakatools::make_device_view(queue_, looseTripletsDC.view().scratch().flags(), nLooseTriplets);
    alpaka::memset(queue_, scratchFlags_view, kT3EmptySlot);

    // Per-module cursor of the stored matches (step 1 is flat over the segments).
    auto matchCount_buf = cms::alpakatools::make_device_buffer<unsigned int[]>(queue_, nLowerModules_);
    alpaka::memset(queue_, matchCount_buf, 0u);
    alpaka::exec<Acc2D>(queue_,
                        countSegConn_wd,
                        CreateTriplets{},
                        modules_.const_view().modules(),
                        miniDoubletsDC_->const_view().miniDoublets(),
                        segmentsT3CountsDC_->const_view(),
                        segmentsDC_->const_view().segments(),
                        looseTripletsDC.view().triplets(),
                        looseTripletsDC.view().tripletsOccupancy(),
                        looseTripletsDC.view().scratch(),
                        tripletsListRangesDC_->view().tripletsRangesBySegment(),
                        tripletsListRangesDC_->view().tripletsRangesByMD(),
                        rangesDC_->view(),
                        ptCut_,
                        nSegByMD_buf->data(),
                        segByMDOffset_buf->data(),
                        segByMD_buf->data(),
                        segInnerModule_buf->data(),
                        nTotalSegmentsOT_,
                        matchCount_buf.data(),
                        t3PassMask_buf->data(),
                        t3LooseMask_buf->data());
    alpaka::exec<Acc1D>(
        queue_,
        cms::alpakatools::make_workdiv<Acc1D>(std::max(cms::alpakatools::divide_up_by(nLooseTriplets, 256u), 1u), 256),
        CreateTripletsFromMatches{},
        modules_.const_view().modules(),
        miniDoubletsDC_->const_view().miniDoublets(),
        segmentsDC_->const_view().segments(),
        looseTripletsDC.view().triplets(),
        looseTripletsDC.view().tripletsOccupancy(),
        looseTripletsDC.view().scratch(),
        tripletsListRangesDC_->view().tripletsRangesBySegment(),
        tripletsListRangesDC_->view().tripletsRangesByMD(),
        rangesDC_->view(),
        segInnerModule_buf->data(),
        nLooseTriplets,
        ptCut_);
  }

  // Last reader of the per-segment T3 counter and of the segments-by-MD list.
  releaseDeviceCollection(segmentsT3CountsDC_);
  nSegByMD_buf.reset();
  segByMDOffset_buf.reset();
  segByMD_buf.reset();
  segInnerModule_buf.reset();
  t3PassMask_buf.reset();
  t3LooseMask_buf.reset();
  if (objectsStatistics_)
    memoryLiveMB_ -= segByMDMB;

  // Compact module offsets (module order on the serial backend, as the loose ones), then the compact buffer.
  auto looseModuleIndices_buf = cms::alpakatools::make_device_buffer<int[]>(queue_, nLowerModules_);
  alpaka::exec<Acc1D>(queue_,
                      cms::alpakatools::make_workdiv<Acc1D>(1, 1024),
                      SetCompactTripletModuleIndices{},
                      modules_.const_view().modules(),
                      looseTripletsDC.const_view().tripletsOccupancy(),
                      rangesDC_->view(),
                      looseModuleIndices_buf.data());
  alpaka::memcpy(queue_, maxTriplets_buf_h, maxTriplets_buf_d);
  alpaka::wait(queue_);  // wait for the exact count before using it

  unsigned int nTotalTriplets = *maxTriplets_buf_h.data();
  tripletsDC_.emplace(queue_, nTotalTriplets, nLowerModules_, nTotalTriplets, nTotalTriplets);
  if (objectsStatistics_) {
    double mb = alpaka::getExtentProduct(tripletsDC_->buffer()) / 1e6;
    trackAllocatedMB(mb);
    lstWarning(std::format("[MEM] Triplets: {} allocated ({:.1f} MB)", nTotalTriplets, mb + listRangesMB));
    lstWarning(
        std::format("[MEM] (transient) Triplets creation buffer: {} slots ({:.1f} MB)", nLooseTriplets, looseMB));
  }

  auto tripletsOccupancy = tripletsDC_->view().tripletsOccupancy();
  alpaka::memcpy(queue_,
                 cms::alpakatools::make_device_view(queue_, tripletsOccupancy.nTriplets(), nLowerModules_),
                 cms::alpakatools::make_device_view(queue_, looseOccupancy.nTriplets(), nLowerModules_));
  alpaka::memcpy(queue_,
                 cms::alpakatools::make_device_view(queue_, tripletsOccupancy.totOccupancyTriplets(), nLowerModules_),
                 cms::alpakatools::make_device_view(queue_, looseOccupancy.totOccupancyTriplets(), nLowerModules_));

  if (nonZeroModules > 0) {
    alpaka::exec<Acc3D>(queue_,
                        createTriplets_workDiv,
                        CompactTriplets{},
                        miniDoubletsDC_->const_view().miniDoubletsOccupancy(),
                        segmentsDC_->const_view().segments(),
                        segmentsDC_->const_view().segmentsOccupancy(),
                        looseTripletsDC.const_view().triplets(),
                        looseTripletsDC.const_view().tripletsOccupancy(),
                        tripletsDC_->view().triplets(),
                        tripletsDC_->view().tripletsBySegment(),
                        tripletsListRangesDC_->view().tripletsRangesBySegment(),
                        tripletsDC_->view().tripletsByMD(),
                        tripletsListRangesDC_->view().tripletsRangesByMD(),
                        rangesDC_->const_view(),
                        looseModuleIndices_buf.data(),
                        index_gpu_buf.data(),
                        nonZeroModules);
  }
  if (objectsStatistics_)
    memoryLiveMB_ -= looseMB;

  if (objectsStatistics_) {
    addTripletsToEventExplicit();
  }
}

void LSTEvent::compactPixelQuintuplets(Queue& queue) {
  auto nPT5_buf_h = cms::alpakatools::make_host_buffer<unsigned int>(queue);
  alpaka::memcpy(
      queue, nPT5_buf_h, cms::alpakatools::make_device_view(queue, (*pixelQuintupletsDC_)->nPixelQuintuplets()));
  auto triedPT5_buf_h = cms::alpakatools::make_host_buffer<unsigned int>(queue);
  alpaka::memcpy(queue,
                 triedPT5_buf_h,
                 cms::alpakatools::make_device_view(queue, (*pixelQuintupletsDC_)->totOccupancyPixelQuintuplets()));
  alpaka::wait(queue);  // the compact size is needed on the host
  const unsigned int nPT5 = *nPT5_buf_h.data();
  if (*triedPT5_buf_h.data() > nPT5)
    lstWarning(std::format("Capacity overflow, objects dropped: {} pT5", *triedPT5_buf_h.data() - nPT5));

  if (objectsStatistics_)
    memoryLiveMB_ -= alpaka::getExtentProduct(pixelQuintupletsDC_->buffer()) / 1e6;
  PixelQuintupletsDeviceCollection compactPT5(queue, nPT5);
  if (nPT5 > 0)
    alpaka::exec<Acc1D>(queue,
                        cms::alpakatools::make_workdiv<Acc1D>(cms::alpakatools::divide_up_by(nPT5, 128), 128),
                        CompactPrefixObjects{},
                        nPT5,
                        pixelQuintupletsDC_->const_view(),
                        compactPT5.view());
  alpaka::memcpy(queue,
                 cms::alpakatools::make_device_view(queue, compactPT5->nPixelQuintuplets()),
                 cms::alpakatools::make_device_view(queue, (*pixelQuintupletsDC_)->nPixelQuintuplets()));
  alpaka::memcpy(queue,
                 cms::alpakatools::make_device_view(queue, compactPT5->totOccupancyPixelQuintuplets()),
                 cms::alpakatools::make_device_view(queue, (*pixelQuintupletsDC_)->totOccupancyPixelQuintuplets()));
  pixelQuintupletsDC_.emplace(std::move(compactPT5));
  if (objectsStatistics_) {
    double mb = alpaka::getExtentProduct(pixelQuintupletsDC_->buffer()) / 1e6;
    trackAllocatedMB(mb);
    lstWarning(std::format("[MEM] PixelQuintuplets: {} allocated ({:.1f} MB)", nPT5, mb));
  }
}

void LSTEvent::compactPixelTriplets(Queue& queue) {
  auto nPT3_buf_h = cms::alpakatools::make_host_buffer<unsigned int>(queue);
  alpaka::memcpy(queue, nPT3_buf_h, cms::alpakatools::make_device_view(queue, (*pixelTripletsDC_)->nPixelTriplets()));
  auto triedPT3_buf_h = cms::alpakatools::make_host_buffer<unsigned int>(queue);
  alpaka::memcpy(queue,
                 triedPT3_buf_h,
                 cms::alpakatools::make_device_view(queue, (*pixelTripletsDC_)->totOccupancyPixelTriplets()));
  alpaka::wait(queue);  // the compact size is needed on the host
  const unsigned int nPT3 = *nPT3_buf_h.data();
  if (*triedPT3_buf_h.data() > nPT3)
    lstWarning(std::format("Capacity overflow, objects dropped: {} pT3", *triedPT3_buf_h.data() - nPT3));

  if (objectsStatistics_)
    memoryLiveMB_ -= alpaka::getExtentProduct(pixelTripletsDC_->buffer()) / 1e6;
  PixelTripletsDeviceCollection compactPT3(queue, nPT3);
  if (nPT3 > 0)
    alpaka::exec<Acc1D>(queue,
                        cms::alpakatools::make_workdiv<Acc1D>(cms::alpakatools::divide_up_by(nPT3, 128), 128),
                        CompactPrefixObjects{},
                        nPT3,
                        pixelTripletsDC_->const_view(),
                        compactPT3.view());
  alpaka::memcpy(queue,
                 cms::alpakatools::make_device_view(queue, compactPT3->nPixelTriplets()),
                 cms::alpakatools::make_device_view(queue, (*pixelTripletsDC_)->nPixelTriplets()));
  alpaka::memcpy(queue,
                 cms::alpakatools::make_device_view(queue, compactPT3->totOccupancyPixelTriplets()),
                 cms::alpakatools::make_device_view(queue, (*pixelTripletsDC_)->totOccupancyPixelTriplets()));
  pixelTripletsDC_.emplace(std::move(compactPT3));
  if (objectsStatistics_) {
    double mb = alpaka::getExtentProduct(pixelTripletsDC_->buffer()) / 1e6;
    trackAllocatedMB(mb);
    lstWarning(std::format("[MEM] PixelTriplets: {} allocated ({:.1f} MB)", nPT3, mb));
  }
}

void LSTEvent::compactQuadruplets(Queue& queue, uint16_t nEligibleT4Modules) {
  auto rangesOccupancy = rangesDC_->view();
  auto quadrupletsOccupancyLoose = quadrupletsDC_->const_view().quadrupletsOccupancy();
  // Compact quadruplet offsets (module order) and their total.
  auto compactT4Indices_buf = cms::alpakatools::make_device_buffer<int[]>(queue, nLowerModules_ + 1);
  alpaka::exec<Acc1D>(queue,
                      cms::alpakatools::make_workdiv<Acc1D>(1, 1024),
                      CompactModuleOffsetsKernel{},
                      static_cast<unsigned int>(nLowerModules_),
                      quadrupletsOccupancyLoose.nQuadruplets().data(),
                      rangesOccupancy.quadrupletModuleIndices().data(),
                      compactT4Indices_buf.data());
  auto nT4_buf_h = cms::alpakatools::make_host_buffer<int>(queue);
  alpaka::memcpy(
      queue, nT4_buf_h, cms::alpakatools::make_device_view(queue, compactT4Indices_buf.data()[nLowerModules_]));
  // Same sum over the attempts, for the capacity-overflow check.
  auto triedT4Indices_buf = cms::alpakatools::make_device_buffer<int[]>(queue, nLowerModules_ + 1);
  alpaka::exec<Acc1D>(queue,
                      cms::alpakatools::make_workdiv<Acc1D>(1, 1024),
                      CompactModuleOffsetsKernel{},
                      static_cast<unsigned int>(nLowerModules_),
                      quadrupletsOccupancyLoose.totOccupancyQuadruplets().data(),
                      rangesOccupancy.quadrupletModuleIndices().data(),
                      triedT4Indices_buf.data());
  auto triedT4_buf_h = cms::alpakatools::make_host_buffer<int>(queue);
  alpaka::memcpy(
      queue, triedT4_buf_h, cms::alpakatools::make_device_view(queue, triedT4Indices_buf.data()[nLowerModules_]));
  alpaka::wait(queue);  // the compact size is needed on the host
  const unsigned int nT4 = *nT4_buf_h.data();
  nT4Compact_ = nT4;
  if (*triedT4_buf_h.data() > *nT4_buf_h.data())
    lstWarning(std::format("Capacity overflow, objects dropped: {} T4", *triedT4_buf_h.data() - *nT4_buf_h.data()));

  if (objectsStatistics_)
    memoryLiveMB_ -= alpaka::getExtentProduct(quadrupletsDC_->buffer()) / 1e6;
  QuadrupletsDeviceCollection compactT4(queue, nT4, nLowerModules_);
  alpaka::exec<Acc1D>(queue,
                      cms::alpakatools::make_workdiv<Acc1D>(std::max((int)nEligibleT4Modules, 1), 128),
                      CompactModuleObjects{},
                      static_cast<unsigned int>(nLowerModules_),
                      rangesOccupancy.indicesOfEligibleT4Modules().data(),
                      static_cast<unsigned int>(nEligibleT4Modules),
                      quadrupletsOccupancyLoose.nQuadruplets().data(),
                      rangesOccupancy.quadrupletModuleIndices().data(),
                      compactT4Indices_buf.data(),
                      quadrupletsDC_->const_view().quadruplets(),
                      compactT4.view().quadruplets(),
                      quadrupletsOccupancyLoose,
                      compactT4.view().quadrupletsOccupancy());
  alpaka::exec<Acc1D>(queue,
                      cms::alpakatools::make_workdiv<Acc1D>(cms::alpakatools::divide_up_by(nLowerModules_, 256), 256),
                      SetCompactModuleIndices{},
                      static_cast<unsigned int>(nLowerModules_),
                      quadrupletsOccupancyLoose.nQuadruplets().data(),
                      compactT4Indices_buf.data(),
                      rangesOccupancy.quadrupletModuleIndices().data(),
                      rangesOccupancy.quadrupletModuleOccupancy().data());
  quadrupletsDC_.emplace(std::move(compactT4));
  if (objectsStatistics_) {
    double mb = alpaka::getExtentProduct(quadrupletsDC_->buffer()) / 1e6;
    trackAllocatedMB(mb);
    lstWarning(std::format("[MEM] Quadruplets: {} allocated ({:.1f} MB)", nT4, mb));
  }
}

void LSTEvent::createTrackCandidates(bool no_pls_dupclean, bool tc_pls_triplets) {
  {
    // pT3 x pT5 cross cleaning through an eta-phi grid of the pT5 seeds (count, scan, fill, then 3x3 neighbourhood).
    auto cellCount_buf = cms::alpakatools::make_device_buffer<unsigned int[]>(queue_, t5DupGrid::kNCells);
    auto cellStart_buf = cms::alpakatools::make_device_buffer<unsigned int[]>(queue_, t5DupGrid::kNCells + 1);
    auto cellEntries_buf = cms::alpakatools::make_device_buffer<unsigned int[]>(queue_, n_max_pixel_quintuplets);
    alpaka::memset(queue_, cellCount_buf, 0u);
    auto const pT5Grid_workDiv = cms::alpakatools::make_workdiv<Acc1D>(max_blocks, 128);

    alpaka::exec<Acc1D>(queue_,
                        pT5Grid_workDiv,
                        CountPixelQuintupletSeedGrid{},
                        modules_.const_view().modules(),
                        rangesDC_->const_view(),
                        lstInputDC_->const_view().pixelSeeds(),
                        pixelQuintupletsDC_->const_view(),
                        cellCount_buf.data());
    alpaka::exec<Acc1D>(queue_,
                        cms::alpakatools::make_workdiv<Acc1D>(1, 1024),
                        ScanCellCounts{},
                        cellCount_buf.data(),
                        cellStart_buf.data(),
                        static_cast<unsigned int>(t5DupGrid::kNCells));
    alpaka::exec<Acc1D>(queue_,
                        pT5Grid_workDiv,
                        FillPixelQuintupletSeedGrid{},
                        modules_.const_view().modules(),
                        rangesDC_->const_view(),
                        lstInputDC_->const_view().pixelSeeds(),
                        pixelQuintupletsDC_->const_view(),
                        cellCount_buf.data(),
                        cellEntries_buf.data());
    alpaka::exec<Acc2D>(queue_,
                        cms::alpakatools::make_workdiv<Acc2D>({64, 1}, {8, 32}),
                        CrossCleanpT3{},
                        modules_.const_view().modules(),
                        rangesDC_->const_view(),
                        pixelTripletsDC_->view(),
                        lstInputDC_->const_view().pixelSeeds(),
                        pixelQuintupletsDC_->const_view(),
                        cellStart_buf.data(),
                        cellEntries_buf.data());
  }

  // Pull nEligibleT5Modules from the device.
  auto rangesOccupancy = rangesDC_->view();
  auto nEligibleModules_buf_h = cms::alpakatools::make_host_buffer<uint16_t>(queue_);
  auto nEligibleModules_buf_d = cms::alpakatools::make_device_view(queue_, rangesOccupancy.nEligibleT5Modules());
  alpaka::memcpy(queue_, nEligibleModules_buf_h, nEligibleModules_buf_d);
  alpaka::wait(queue_);  // wait to get the values before using
  auto const nEligibleModules = *nEligibleModules_buf_h.data();

  {
    // T5 duplicate removal on an eta-phi grid of the alive T5s (count, scan, fill, then 3x3 neighbourhood).
    auto const nT5Slots = quintupletsDC_->const_view().quintuplets().metadata().size();
    auto cellCount_buf = cms::alpakatools::make_device_buffer<unsigned int[]>(queue_, t5DupGrid::kNCells);
    auto cellStart_buf = cms::alpakatools::make_device_buffer<unsigned int[]>(queue_, t5DupGrid::kNCells + 1);
    auto cellEntries_buf = cms::alpakatools::make_device_buffer<unsigned int[]>(queue_, nT5Slots);
    alpaka::memset(queue_, cellCount_buf, 0u);
    auto const t5Grid_workDiv = cms::alpakatools::make_workdiv<Acc2D>({std::max<int>(nEligibleModules, 1), 1}, {1, 32});

    alpaka::exec<Acc2D>(queue_,
                        t5Grid_workDiv,
                        CountT5DupGrid{},
                        quintupletsDC_->const_view().quintuplets(),
                        quintupletsDC_->const_view().quintupletsOccupancy(),
                        rangesDC_->const_view(),
                        cellCount_buf.data());
    alpaka::exec<Acc1D>(queue_,
                        cms::alpakatools::make_workdiv<Acc1D>(1, 1024),
                        ScanCellCounts{},
                        cellCount_buf.data(),
                        cellStart_buf.data(),
                        static_cast<unsigned int>(t5DupGrid::kNCells));
    alpaka::exec<Acc2D>(queue_,
                        t5Grid_workDiv,
                        FillT5DupGrid{},
                        quintupletsDC_->const_view().quintuplets(),
                        quintupletsDC_->const_view().quintupletsOccupancy(),
                        rangesDC_->const_view(),
                        cellCount_buf.data(),
                        cellEntries_buf.data());
    alpaka::exec<Acc3D>(queue_,
                        cms::alpakatools::make_workdiv<Acc3D>({std::max<int>(nEligibleModules, 1), 1, 1}, {1, 4, 32}),
                        RemoveDupQuintupletsBeforeTC{},
                        quintupletsDC_->view().quintuplets(),
                        quintupletsDC_->const_view().quintupletsOccupancy(),
                        rangesDC_->const_view(),
                        cellStart_buf.data(),
                        cellEntries_buf.data());
  }

  // T5 x promoted pixel object cross-cleaning on an eta-phi grid (window 0.15) of the pT5s and pT3s.
  auto const pixelGrid = EtaPhiGrid::make(0.15f, 3.f);
  auto pixelCellCount_buf = cms::alpakatools::make_device_buffer<unsigned int[]>(queue_, pixelGrid.nCells());
  auto pixelCellStart_buf = cms::alpakatools::make_device_buffer<unsigned int[]>(queue_, pixelGrid.nCells() + 1);
  auto pixelCellItems_buf =
      cms::alpakatools::make_device_buffer<unsigned int[]>(queue_, n_max_pixel_quintuplets + n_max_pixel_triplets);
  alpaka::memset(queue_, pixelCellCount_buf, 0u);
  auto const pixelGridFill_workDiv = cms::alpakatools::make_workdiv<Acc1D>(
      cms::alpakatools::divide_up_by(n_max_pixel_quintuplets + n_max_pixel_triplets, 256u), 256u);
  alpaka::exec<Acc1D>(queue_,
                      pixelGridFill_workDiv,
                      FillPixelObjectGrid{},
                      pixelQuintupletsDC_->const_view(),
                      pixelTripletsDC_->const_view(),
                      pixelGrid,
                      pixelCellCount_buf.data(),
                      static_cast<unsigned int*>(nullptr));
  alpaka::exec<Acc1D>(queue_,
                      cms::alpakatools::make_workdiv<Acc1D>(1, 1024),
                      EtaPhiGridPrefix{},
                      pixelGrid.nCells(),
                      pixelCellCount_buf.data(),
                      pixelCellStart_buf.data());
  alpaka::exec<Acc1D>(queue_,
                      pixelGridFill_workDiv,
                      FillPixelObjectGrid{},
                      pixelQuintupletsDC_->const_view(),
                      pixelTripletsDC_->const_view(),
                      pixelGrid,
                      pixelCellCount_buf.data(),
                      pixelCellItems_buf.data());

  auto const crossCleanT5_workDiv = cms::alpakatools::make_workdiv<Acc3D>({nLowerModules_, 1, 1}, {1, 8, 32});

  alpaka::exec<Acc3D>(queue_,
                      crossCleanT5_workDiv,
                      CrossCleanT5{},
                      modules_.const_view().modules(),
                      quintupletsDC_->view().quintuplets(),
                      quintupletsDC_->const_view().quintupletsOccupancy(),
                      pixelQuintupletsDC_->const_view(),
                      pixelTripletsDC_->const_view(),
                      rangesDC_->const_view(),
                      pixelGrid,
                      pixelCellStart_buf.data(),
                      pixelCellItems_buf.data());

  // T4s are compacted (dense, module order) by compactQuadruplets() at the end of their stage.
  const unsigned int nT4 = nT4Compact_;
  auto const removeDupQuadrupletsBeforeTC_workDiv =
      cms::alpakatools::make_workdiv<Acc2D>({std::max(cms::alpakatools::divide_up_by(nT4, 16u), 1u), 1u}, {16, 32});

  alpaka::exec<Acc2D>(queue_,
                      removeDupQuadrupletsBeforeTC_workDiv,
                      RemoveDupQuadrupletsBeforeTC{},
                      quadrupletsDC_->view().quadruplets(),
                      rangesDC_->const_view(),
                      nT4);

  if (!no_pls_dupclean) {
    auto const checkHitspLS_workDiv = cms::alpakatools::make_workdiv<Acc2D>({max_blocks * 4, max_blocks / 4}, {16, 16});

    alpaka::exec<Acc2D>(queue_,
                        checkHitspLS_workDiv,
                        CheckHitspLS{},
                        modules_.const_view().modules(),
                        segmentsDC_->const_view().segmentsOccupancy(),
                        lstInputDC_->const_view().pixelSeeds(),
                        pixelSegmentsDC_->view(),
                        true);
  }

  // Counting kernel
  // [0..4] surviving pT5/pT3/T5/T4/pLS, [5..6] pixel-hit key max IT/OT, [7..11] overflow counters: one readback.
  constexpr unsigned int kNTCCounts = 12;
  auto nSurvivingTCs_dev = cms::alpakatools::make_device_buffer<unsigned int[]>(queue_, kNTCCounts);
  alpaka::memset(queue_, nSurvivingTCs_dev, 0u);

  auto const countSurvivingTCs_workDiv = cms::alpakatools::make_workdiv<Acc1D>(max_blocks, 256);

  alpaka::exec<Acc1D>(queue_,
                      countSurvivingTCs_workDiv,
                      CountSurvivingTCs{},
                      nLowerModules_,
                      pixelQuintupletsDC_->const_view(),
                      pixelTripletsDC_->const_view(),
                      quintupletsDC_->const_view().quintuplets(),
                      quintupletsDC_->const_view().quintupletsOccupancy(),
                      quadrupletsDC_->const_view().quadruplets(),
                      quadrupletsDC_->const_view().quadrupletsOccupancy(),
                      segmentsDC_->const_view().segmentsOccupancy(),
                      lstInputDC_->const_view().pixelSeeds(),
                      pixelSegmentsDC_->const_view(),
                      rangesDC_->const_view(),
                      nSurvivingTCs_dev.data(),
                      tc_pls_triplets);

  // Largest pLS hit indices, to size the pixel-hit bit set of CrossCleanpLS.
  alpaka::exec<Acc1D>(queue_,
                      cms::alpakatools::make_workdiv<Acc1D>(1, 1024),
                      PixelHitKeyMax{},
                      nLowerModules_,
                      segmentsDC_->const_view().segmentsOccupancy(),
                      pixelSegmentsDC_->const_view(),
                      nSurvivingTCs_dev.data() + 5);
  auto nSurvivingTCs_host = cms::alpakatools::make_host_buffer<unsigned int[]>(queue_, kNTCCounts);
  alpaka::memcpy(queue_, nSurvivingTCs_host, nSurvivingTCs_dev);
  alpaka::wait(queue_);  // wait to get counts before allocation
  // Objects the creation kernels found no reserved slot for (the counting kernels must be a superset).
  auto const* counts = nSurvivingTCs_host.data();
  nSegmentOverflows_ = counts[7];
  nTripletOverflows_ = counts[8];
  nQuintupletOverflows_ = counts[9];
  nT5byMDOverflows_ = counts[10];
  nT5CapDrops_ = counts[11];
  if (nSegmentOverflows_ + nTripletOverflows_ + nQuintupletOverflows_ > 0)
    lstWarning(std::format("Counting-kernel overflow, objects dropped: {} segments, {} triplets, {} quintuplets",
                           nSegmentOverflows_,
                           nTripletOverflows_,
                           nQuintupletOverflows_));
  if (objectsStatistics_ && nT5byMDOverflows_ > 0)
    lstWarning(std::format("T5 by-MD list full: {} quintuplets kept but not listed for the by-MD duplicate search",
                           nT5byMDOverflows_));
  if (objectsStatistics_ && nT5CapDrops_ > 0)
    lstWarning(std::format("T5 per-module cap reached: {} quintuplets dropped (fixed cap, not a counting shortfall)",
                           nT5CapDrops_));
  if (objectsStatistics_)
    lstWarning(std::format("[CNT] overflows: {} {} {}", nSegmentOverflows_, nTripletOverflows_, nQuintupletOverflows_));

  // The TC adders stop at nTotal <= nMaxTC, so truncation shows in the counts, not in nTrackCandidates.
  constexpr unsigned int nMaxTC = n_max_nonpixel_track_candidates + n_max_pixel_track_candidates;
  const unsigned int nCounted = counts[0] + counts[1] + counts[2] + counts[3] + counts[4];
  if (nCounted > nMaxTC) {
    lstWarning(
        "\
        ****************************************************************************************************\n\
        * Track candidates were possibly truncated.                                                        *\n\
        * The dynamically allocated TC buffer was fully used.                                              *\n\
        * Run the code with the WARNINGS flag activated for more details.                                  *\n\
        ****************************************************************************************************");
  }
  unsigned int nTotal = std::min(nCounted, nMaxTC);
  if (nTotal == 0)
    nTotal = 1;  // avoid zero-size allocation

  // TC allocation
  trackCandidatesBaseDC_.emplace(queue_, nTotal);
  trackCandidatesBaseDC_->zeroInitialise(queue_);
  trackCandidatesExtendedDC_.emplace(queue_, nTotal);
  trackCandidatesExtendedDC_->zeroInitialise(queue_);
  if (objectsStatistics_) {
    double mb = (alpaka::getExtentProduct(trackCandidatesBaseDC_->buffer()) +
                 alpaka::getExtentProduct(trackCandidatesExtendedDC_->buffer())) /
                1e6;
    trackAllocatedMB(mb);
    lstWarning(std::format(
        "[MEM] TrackCandidates: {} allocated ({:.1f} MB) [dynamic: {} pT5 + {} pT3 + {} T5 + {} T4 + {} pLS]",
        nTotal,
        mb,
        counts[0],
        counts[1],
        counts[2],
        counts[3],
        counts[4]));
  }

  // Blocks cover the fixed capacity: on a serial backend each block is one contiguous chunk, so the TC order is unchanged.
  auto const addpT5asTrackCandidate_workDiv =
      cms::alpakatools::make_workdiv<Acc1D>(cms::alpakatools::divide_up_by(n_max_pixel_quintuplets, 256u), 256);

  alpaka::exec<Acc1D>(queue_,
                      addpT5asTrackCandidate_workDiv,
                      AddpT5asTrackCandidate{},
                      nLowerModules_,
                      miniDoubletsDC_->const_view().miniDoublets(),
                      segmentsDC_->const_view().segments(),
                      quintupletsDC_->const_view().quintuplets(),
                      pixelQuintupletsDC_->const_view(),
                      trackCandidatesBaseDC_->view(),
                      trackCandidatesExtendedDC_->view(),
                      lstInputDC_->const_view().pixelSeeds(),
                      rangesDC_->const_view(),
                      nTotal);

  auto const addpT3asTrackCandidates_workDiv =
      cms::alpakatools::make_workdiv<Acc1D>(cms::alpakatools::divide_up_by(n_max_pixel_triplets, 256u), 256);

  alpaka::exec<Acc1D>(queue_,
                      addpT3asTrackCandidates_workDiv,
                      AddpT3asTrackCandidates{},
                      nLowerModules_,
                      pixelTripletsDC_->const_view(),
                      trackCandidatesBaseDC_->view(),
                      trackCandidatesExtendedDC_->view(),
                      lstInputDC_->const_view().pixelSeeds(),
                      rangesDC_->const_view(),
                      nTotal);

  // GPU: one warp per module. A serial backend keeps the old division, whose block order sets the TC order.
  constexpr bool kSerialTCOrder = cms::alpakatools::requires_single_thread_per_block_v<Acc2D>;
  auto const addT5asTrackCandidate_workDiv =
      kSerialTCOrder
          ? cms::alpakatools::make_workdiv<Acc2D>({8, 10}, {8, 128})
          : cms::alpakatools::make_workdiv<Acc2D>({cms::alpakatools::divide_up_by(nLowerModules_, 4u), 1u}, {4, 32});

  alpaka::exec<Acc2D>(queue_,
                      addT5asTrackCandidate_workDiv,
                      AddT5asTrackCandidate{},
                      nLowerModules_,
                      quintupletsDC_->const_view().quintuplets(),
                      quintupletsDC_->const_view().quintupletsOccupancy(),
                      trackCandidatesBaseDC_->view(),
                      trackCandidatesExtendedDC_->view(),
                      rangesDC_->const_view(),
                      nTotal);

  {
    // CrossCleanT4 through the TCs listed by hit: at most the surviving T5/pT5 (T5 hits) and pT3 (T3 hits) entries.
    const unsigned int nBucketEntries =
        std::max((counts[0] + counts[2]) * Params_T5::kHits + counts[1] * Params_T3::kHits, 1u);
    auto bucketCount_buf = cms::alpakatools::make_device_buffer<unsigned int[]>(queue_, tcHitBuckets::kNBuckets);
    auto bucketStart_buf = cms::alpakatools::make_device_buffer<unsigned int[]>(queue_, tcHitBuckets::kNBuckets + 1);
    auto bucketTCs_buf = cms::alpakatools::make_device_buffer<unsigned int[]>(queue_, nBucketEntries);
    auto bucketHits_buf = cms::alpakatools::make_device_buffer<unsigned int[]>(queue_, nBucketEntries);
    alpaka::memset(queue_, bucketCount_buf, 0u);
    auto const tcHitBuckets_workDiv =
        cms::alpakatools::make_workdiv<Acc1D>(cms::alpakatools::divide_up_by(nTotal, 256u), 256u);

    alpaka::exec<Acc1D>(queue_,
                        tcHitBuckets_workDiv,
                        FillTCHitBuckets{},
                        pixelTripletsDC_->const_view(),
                        quintupletsDC_->const_view().quintuplets(),
                        trackCandidatesBaseDC_->const_view(),
                        trackCandidatesExtendedDC_->const_view(),
                        bucketCount_buf.data(),
                        static_cast<unsigned int*>(nullptr),
                        static_cast<unsigned int*>(nullptr));
    alpaka::exec<Acc1D>(queue_,
                        cms::alpakatools::make_workdiv<Acc1D>(1, 1024),
                        ScanCellCounts{},
                        bucketCount_buf.data(),
                        bucketStart_buf.data(),
                        tcHitBuckets::kNBuckets);
    alpaka::exec<Acc1D>(queue_,
                        tcHitBuckets_workDiv,
                        FillTCHitBuckets{},
                        pixelTripletsDC_->const_view(),
                        quintupletsDC_->const_view().quintuplets(),
                        trackCandidatesBaseDC_->const_view(),
                        trackCandidatesExtendedDC_->const_view(),
                        bucketCount_buf.data(),
                        bucketTCs_buf.data(),
                        bucketHits_buf.data());

    // T4s (dense after compactQuadruplets) on y, their hits on x.
    auto const crossCleanT4_workDiv =
        cms::alpakatools::make_workdiv<Acc2D>({std::max(cms::alpakatools::divide_up_by(nT4, 16u), 1u), 1u}, {16, 8});

    alpaka::exec<Acc2D>(queue_,
                        crossCleanT4_workDiv,
                        CrossCleanT4{},
                        quadrupletsDC_->view().quadruplets(),
                        pixelTripletsDC_->const_view(),
                        quintupletsDC_->const_view().quintuplets(),
                        trackCandidatesBaseDC_->const_view(),
                        trackCandidatesExtendedDC_->const_view(),
                        nT4,
                        bucketStart_buf.data(),
                        bucketTCs_buf.data(),
                        bucketHits_buf.data());
  }

  auto const addT4asTrackCandidate_workDiv =
      kSerialTCOrder
          ? cms::alpakatools::make_workdiv<Acc2D>({8, 10}, {8, 128})
          : cms::alpakatools::make_workdiv<Acc2D>({cms::alpakatools::divide_up_by(nLowerModules_, 4u), 1u}, {4, 32});

  alpaka::exec<Acc2D>(queue_,
                      addT4asTrackCandidate_workDiv,
                      AddT4asTrackCandidate{},
                      nLowerModules_,
                      quadrupletsDC_->view().quadruplets(),
                      quadrupletsDC_->const_view().quadrupletsOccupancy(),
                      tripletsDC_->const_view().triplets(),
                      trackCandidatesBaseDC_->view(),
                      trackCandidatesExtendedDC_->view(),
                      rangesDC_->const_view(),
                      nTotal);

  // CrossCleanpLS: pixel-hit bits of the pT3/pT5 seeds and an (eta, phi) grid of the T5/pT3/pT5 TCs.
  unsigned int const nKeysIT = counts[5] + 1;
  uint64_t const nHitKeys = uint64_t(nKeysIT) + counts[6] + 1;
  auto hitKeyBits_buf = cms::alpakatools::make_device_buffer<uint32_t[]>(queue_, (nHitKeys + 31) / 32);
  alpaka::memset(queue_, hitKeyBits_buf, 0u);
  auto const grid = EtaPhiGrid::make(kCrossCleanGridWindow, kCrossCleanGridEtaMax);
  auto cellCount_buf = cms::alpakatools::make_device_buffer<unsigned int[]>(queue_, grid.nCells());
  alpaka::memset(queue_, cellCount_buf, 0u);
  auto cellStart_buf = cms::alpakatools::make_device_buffer<unsigned int[]>(queue_, grid.nCells() + 1);
  auto cellTCs_buf = cms::alpakatools::make_device_buffer<unsigned int[]>(queue_, nTotal);

  auto const crossCleanGrid_workDiv = cms::alpakatools::make_workdiv<Acc1D>(max_blocks, 256);
  alpaka::exec<Acc1D>(queue_,
                      crossCleanGrid_workDiv,
                      CountCrossCleanGrid{},
                      modules_.const_view().modules(),
                      rangesDC_->const_view(),
                      pixelTripletsDC_->const_view(),
                      trackCandidatesBaseDC_->const_view(),
                      trackCandidatesExtendedDC_->const_view(),
                      lstInputDC_->const_view().pixelSeeds(),
                      pixelSegmentsDC_->const_view(),
                      quintupletsDC_->const_view().quintuplets(),
                      grid,
                      nKeysIT,
                      hitKeyBits_buf.data(),
                      cellCount_buf.data());

  alpaka::exec<Acc1D>(queue_,
                      cms::alpakatools::make_workdiv<Acc1D>(1, 1024),
                      EtaPhiGridPrefix{},
                      grid.nCells(),
                      cellCount_buf.data(),
                      cellStart_buf.data());

  alpaka::exec<Acc1D>(queue_,
                      crossCleanGrid_workDiv,
                      FillCrossCleanGrid{},
                      modules_.const_view().modules(),
                      rangesDC_->const_view(),
                      pixelTripletsDC_->const_view(),
                      trackCandidatesBaseDC_->const_view(),
                      trackCandidatesExtendedDC_->const_view(),
                      lstInputDC_->const_view().pixelSeeds(),
                      quintupletsDC_->const_view().quintuplets(),
                      grid,
                      cellCount_buf.data(),
                      cellTCs_buf.data());

  auto const crossCleanpLS_workDiv = cms::alpakatools::make_workdiv<Acc1D>(max_blocks, 256);

  alpaka::exec<Acc1D>(queue_,
                      crossCleanpLS_workDiv,
                      CrossCleanpLS{},
                      modules_.const_view().modules(),
                      rangesDC_->const_view(),
                      pixelTripletsDC_->const_view(),
                      trackCandidatesBaseDC_->const_view(),
                      trackCandidatesExtendedDC_->const_view(),
                      segmentsDC_->const_view().segmentsOccupancy(),
                      lstInputDC_->const_view().pixelSeeds(),
                      pixelSegmentsDC_->view(),
                      quintupletsDC_->const_view().quintuplets(),
                      grid,
                      nKeysIT,
                      hitKeyBits_buf.data(),
                      cellStart_buf.data(),
                      cellTCs_buf.data());

  auto const addpLSasTrackCandidate_workDiv = cms::alpakatools::make_workdiv<Acc1D>(max_blocks, 384);

  alpaka::exec<Acc1D>(queue_,
                      addpLSasTrackCandidate_workDiv,
                      AddpLSasTrackCandidate{},
                      nLowerModules_,
                      trackCandidatesBaseDC_->view(),
                      trackCandidatesExtendedDC_->view(),
                      segmentsDC_->const_view().segmentsOccupancy(),
                      lstInputDC_->const_view().pixelSeeds(),
                      pixelSegmentsDC_->const_view(),
                      tc_pls_triplets,
                      nTotal);
}

void LSTEvent::copyPixelSeedTypesToHost() {
  auto pixelSeeds = lstInputDC_->const_view().pixelSeeds();
  superbinsHost_.emplace(cms::alpakatools::make_host_buffer<int[]>(queue_, pixelSize_));
  alpaka::memcpy(
      queue_, *superbinsHost_, cms::alpakatools::make_device_view(queue_, pixelSeeds.superbin(), pixelSize_));
  pixelTypesHost_.emplace(cms::alpakatools::make_host_buffer<PixelType[]>(queue_, pixelSize_));
  alpaka::memcpy(
      queue_, *pixelTypesHost_, cms::alpakatools::make_device_view(queue_, pixelSeeds.pixelType(), pixelSize_));
}

void LSTEvent::fillConnectedPixels() {
  // Connected-module range of each pLS in the pixel map, shared by pT5 and pT3.
  if (connectedPixelSize_)
    return;
  if (!superbinsHost_) {
    copyPixelSeedTypesToHost();
    alpaka::wait(queue_);  // wait to get superbins and pixelTypes before using them
  }
  auto const* superbins = superbinsHost_->data();
  auto const* pixelTypes = pixelTypesHost_->data();
  const unsigned int nInnerSegments = pixelSize_;

  auto connectedPixelSize_host_buf = cms::alpakatools::make_host_buffer<unsigned int[]>(queue_, nInnerSegments);
  auto connectedPixelIndex_host_buf = cms::alpakatools::make_host_buffer<unsigned int[]>(queue_, nInnerSegments);
  unsigned int* connectedPixelSize_host = connectedPixelSize_host_buf.data();
  unsigned int* connectedPixelIndex_host = connectedPixelIndex_host_buf.data();

  int pixelIndexOffsetPos =
      pixelMapping_.connectedPixelsIndex[size_superbins - 1] + pixelMapping_.connectedPixelsSizes[size_superbins - 1];
  int pixelIndexOffsetNeg = pixelMapping_.connectedPixelsIndexPos[size_superbins - 1] +
                            pixelMapping_.connectedPixelsSizesPos[size_superbins - 1] + pixelIndexOffsetPos;

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
  superbinsHost_.reset();
  pixelTypesHost_.reset();

  connectedPixelSize_.emplace(cms::alpakatools::make_device_buffer<unsigned int[]>(queue_, nInnerSegments));
  connectedPixelIndex_.emplace(cms::alpakatools::make_device_buffer<unsigned int[]>(queue_, nInnerSegments));
  alpaka::memcpy(queue_, *connectedPixelSize_, connectedPixelSize_host_buf, nInnerSegments);
  alpaka::memcpy(queue_, *connectedPixelIndex_, connectedPixelIndex_host_buf, nInnerSegments);
}

void LSTEvent::createPixelTriplets() {
  if (!pixelTripletsDC_) {
    pixelTripletsDC_.emplace(queue_, n_max_pixel_triplets);
    auto nPixelTriplets_view = cms::alpakatools::make_device_view(queue_, (*pixelTripletsDC_)->nPixelTriplets());
    alpaka::memset(queue_, nPixelTriplets_view, 0u);
    auto totOccupancyPixelTriplets_view =
        cms::alpakatools::make_device_view(queue_, (*pixelTripletsDC_)->totOccupancyPixelTriplets());
    alpaka::memset(queue_, totOccupancyPixelTriplets_view, 0u);
    if (objectsStatistics_) {
      double mb = alpaka::getExtentProduct(pixelTripletsDC_->buffer()) / 1e6;
      trackTransientMB(mb);  // shrunk to the produced pT3s by compactPixelTriplets
      lstWarning(std::format("[MEM-loose] PixelTriplets: {} allocated ({:.1f} MB) [fixed]", n_max_pixel_triplets, mb));
    }
  }

  fillConnectedPixels();
  // nSegments of the pixel module is pixelSize_ (set in createSegmentsWithModuleMap).
  const unsigned int nInnerSegments = pixelSize_;

  auto createPixelTripletsFromMap_workDiv =
      cms::alpakatools::make_workdiv<Acc3D>({4096, 16 /* above median of connected modules*/, 1}, {4, 1, 32});
  // On the GPU a block is one (pLS, module) row: its threads share the pLS x segment tests of the module.
  if constexpr (not cms::alpakatools::requires_single_thread_per_block_v<Acc3D>)
    createPixelTripletsFromMap_workDiv = cms::alpakatools::make_workdiv<Acc3D>(
        {std::clamp(nInnerSegments, 1u, 1024u), 16, 1}, {1, 1, kPT3SegmentsPerChunk});

  alpaka::exec<Acc3D>(queue_,
                      createPixelTripletsFromMap_workDiv,
                      CreatePixelTripletsFromMap{},
                      modules_.const_view().modules(),
                      modules_.const_view().modulesPixel(),
                      rangesDC_->const_view(),
                      miniDoubletsDC_->const_view().miniDoublets(),
                      segmentsDC_->const_view().segments(),
                      segmentsDC_->const_view().segmentsOccupancy(),
                      lstInputDC_->const_view().pixelSeeds(),
                      pixelSegmentsDC_->const_view(),
                      tripletsDC_->view().triplets(),
                      tripletsDC_->const_view().tripletsOccupancy(),
                      tripletsDC_->const_view().tripletsBySegment(),
                      tripletsListRangesDC_->const_view().tripletsRangesBySegment(),
                      pixelTripletsDC_->view(),
                      connectedPixelSize_->data(),
                      connectedPixelIndex_->data(),
                      nInnerSegments,
                      ptCut_);
  // Last reader of the pixel-map ranges.
  connectedPixelSize_.reset();
  connectedPixelIndex_.reset();

#ifdef WARNINGS
  auto nPixelTriplets_buf = cms::alpakatools::make_host_buffer<unsigned int>(queue_);

  alpaka::memcpy(
      queue_, nPixelTriplets_buf, cms::alpakatools::make_device_view(queue_, (*pixelTripletsDC_)->nPixelTriplets()));
  alpaka::wait(queue_);  // wait to get the value before using it

  std::cout << "number of pixel triplets = " << *nPixelTriplets_buf.data() << std::endl;
#endif

  //pT3s can be cleaned here because they're not used in making pT5s!
  {
    // Candidate duplicates through the T3 hits (count, scan, fill buckets), then the pairwise test.
    auto bucketCount_buf = cms::alpakatools::make_device_buffer<unsigned int[]>(queue_, pT3HitBuckets::kNBuckets);
    auto bucketStart_buf = cms::alpakatools::make_device_buffer<unsigned int[]>(queue_, pT3HitBuckets::kNBuckets + 1);
    auto bucketEntries_buf =
        cms::alpakatools::make_device_buffer<unsigned int[]>(queue_, n_max_pixel_triplets * pT3HitBuckets::kT3Hits);
    alpaka::memset(queue_, bucketCount_buf, 0u);
    auto const pT3Dup_workDiv = cms::alpakatools::make_workdiv<Acc1D>(40, 128);

    alpaka::exec<Acc1D>(
        queue_, pT3Dup_workDiv, CountPixelTripletHitBuckets{}, pixelTripletsDC_->const_view(), bucketCount_buf.data());
    alpaka::exec<Acc1D>(queue_,
                        cms::alpakatools::make_workdiv<Acc1D>(1, 1024),
                        ScanCellCounts{},
                        bucketCount_buf.data(),
                        bucketStart_buf.data(),
                        pT3HitBuckets::kNBuckets);
    alpaka::exec<Acc1D>(queue_,
                        pT3Dup_workDiv,
                        FillPixelTripletHitBuckets{},
                        pixelTripletsDC_->const_view(),
                        bucketCount_buf.data(),
                        bucketEntries_buf.data());
    alpaka::exec<Acc2D>(queue_,
                        cms::alpakatools::make_workdiv<Acc2D>({64, 1}, {8, 32}),
                        RemoveDupPixelTripletsFromMap{},
                        pixelTripletsDC_->view(),
                        bucketStart_buf.data(),
                        bucketEntries_buf.data());
  }

  compactPixelTriplets(queue_);
}

void LSTEvent::createQuintuplets() {
  // per-MD T5 counters and T5-by-MD ranges, OT MDs only (the pixel MDs follow the OT ones)
  miniDoubletsT5BuildDC_.emplace(queue_, nTotalMDsOT_, nTotalMDsOT_, nTotalMDsOT_);
  if (objectsStatistics_) {
    double mb = alpaka::getExtentProduct(miniDoubletsT5BuildDC_->buffer()) / 1e6;
    trackAllocatedMB(mb);
    lstWarning(std::format("[MEM] MiniDoubletsT5Build: {} allocated ({:.1f} MB)", nTotalMDsOT_, mb));
  }

  // Keys of the T5 builders: segments that carry triplets, by inner MD (transient, this stage only).
  auto nKeysByMD_buf = cms::alpakatools::make_device_buffer<unsigned int[]>(queue_, nTotalMDsOT_);
  alpaka::memset(queue_, nKeysByMD_buf, 0u);
  auto keyOffsetByMD_buf = cms::alpakatools::make_device_buffer<unsigned int[]>(queue_, nTotalMDsOT_);
  auto keys_buf = cms::alpakatools::make_device_buffer<unsigned int[]>(queue_, nTotalSegmentsOT_);
  alpaka::exec<Acc1D>(queue_,
                      cms::alpakatools::make_workdiv<Acc1D>(nLowerModules_, 128),
                      FillT5KeysByMD{},
                      modules_.const_view().modules(),
                      miniDoubletsDC_->const_view().miniDoubletsOccupancy(),
                      segmentsDC_->const_view().segments(),
                      segmentsDC_->const_view().segmentsOccupancy(),
                      tripletsDC_->const_view().tripletsOccupancy(),
                      tripletsListRangesDC_->const_view().tripletsRangesBySegment(),
                      rangesDC_->const_view(),
                      nKeysByMD_buf.data(),
                      keyOffsetByMD_buf.data(),
                      keys_buf.data());

  // Per-triplet bits: which of the first 64 keys of an inner triplet pass the first dBeta cut.
  const unsigned int nTripletSlots = tripletsDC_->const_view().triplets().metadata().size();
  auto dBetaPassMask_buf = cms::alpakatools::make_device_buffer<T5KeyMask[]>(queue_, nTripletSlots);
  alpaka::memset(queue_, dBetaPassMask_buf, 0u);

  // One row of threads per inner triplet (8 per block), over all triplets; on the serial backend one pass in triplet
  // order (the T5 slot order of a module), elsewhere capped at the CUDA grid-y limit.
  unsigned int countConnBlocks = std::max(cms::alpakatools::divide_up_by(nTripletSlots, 8u), 1u);
  if constexpr (not cms::alpakatools::requires_single_thread_per_block_v<Acc3D>)
    countConnBlocks = std::min(countConnBlocks, 65535u);
  auto const countConn_workDiv = cms::alpakatools::make_workdiv<Acc3D>({1, countConnBlocks, 1}, {1, 8, 32});
  alpaka::exec<Acc3D>(queue_,
                      countConn_workDiv,
                      CountT5KeyCuts{},
                      modules_.const_view().modules(),
                      miniDoubletsDC_->const_view().miniDoublets(),
                      segmentsDC_->const_view().segments(),
                      tripletsDC_->const_view().triplets(),
                      nKeysByMD_buf.data(),
                      keyOffsetByMD_buf.data(),
                      keys_buf.data(),
                      nTripletSlots,
                      ptCut_,
                      dBetaPassMask_buf.data());

  // The counting kernel selects each pair once and records the selected pairs (transient, this stage only). T5s are
  // a small fraction of the triplets; a larger list and a second run are needed only when more are selected.
  unsigned int selectedCapacity = std::max(nTripletSlots / 4, 1024u);
  auto moduleT5Count_buf = cms::alpakatools::make_device_buffer<unsigned int[]>(queue_, nLowerModules_);
  auto nSelected_d = cms::alpakatools::make_device_buffer<unsigned int[]>(queue_, 1u);
  auto nEligibleT5Modules_buf = cms::alpakatools::make_host_buffer<uint16_t>(queue_);
  auto nTotalQuintuplets_buf = cms::alpakatools::make_host_buffer<unsigned int>(queue_);
  auto nTotalQuintuplets0_buf = cms::alpakatools::make_host_buffer<unsigned int>(queue_);
  auto nTotalQuintuplets1_buf = cms::alpakatools::make_host_buffer<unsigned int>(queue_);
  auto nSelected_buf = cms::alpakatools::make_host_buffer<unsigned int[]>(queue_, 1u);
  auto rangesOccupancy = rangesDC_->const_view();
  auto nEligibleT5Modules_view_d = cms::alpakatools::make_device_view(queue_, rangesOccupancy.nEligibleT5Modules());
  auto nTotalQuintuplets_view_d = cms::alpakatools::make_device_view(queue_, rangesOccupancy.nTotalQuints());
  auto nTotalQuintuplets0_view_d = cms::alpakatools::make_device_view(queue_, rangesOccupancy.nTotalQuintsByMD0());
  auto nTotalQuintuplets1_view_d = cms::alpakatools::make_device_view(queue_, rangesOccupancy.nTotalQuintsByMD1());
  // Second dBeta cut decisions per last outer segment (see passQuintupletDBeta2Memo); valid for both counting runs.
  auto dBeta2Memo_buf = cms::alpakatools::make_device_buffer<unsigned int[]>(queue_, nTotalSegmentsOT_);
  alpaka::memset(queue_, dBeta2Memo_buf, 0xFF);
  // The selection synchronizes the threads of a row, so on the GPU a block is one row.
  auto select_workDiv = countConn_workDiv;
  if constexpr (not cms::alpakatools::requires_single_thread_per_block_v<Acc3D>)
    select_workDiv =
        cms::alpakatools::make_workdiv<Acc3D>({1, std::clamp(nTripletSlots, 1u, 65535u), 1}, {1, 1, kT5KeysPerChunk});
  auto countQuintuplets = [&](unsigned int capacity) {
    auto selectedT3s_buf = cms::alpakatools::make_device_buffer<unsigned int[]>(queue_, 2 * capacity);
    auto selectedBridgeRadius_buf = cms::alpakatools::make_device_buffer<float[]>(queue_, capacity);
    auto selectedDnnScore_buf = cms::alpakatools::make_device_buffer<float[]>(queue_, capacity);
    auto t5Counts = miniDoubletsT5BuildDC_->view().t5Counts();
    auto connT50View = cms::alpakatools::make_device_view(queue_, t5Counts.connectedT5s0Max());
    alpaka::memset(queue_, connT50View, 0u);
    auto connT51View = cms::alpakatools::make_device_view(queue_, t5Counts.connectedT5s1Max());
    alpaka::memset(queue_, connT51View, 0u);
    alpaka::memset(queue_, moduleT5Count_buf, 0u);
    alpaka::memset(queue_, nSelected_d, 0u);
    alpaka::exec<Acc3D>(queue_,
                        select_workDiv,
                        CountTripletConnections{},
                        modules_.const_view().modules(),
                        miniDoubletsDC_->const_view().miniDoublets(),
                        t5Counts,
                        segmentsDC_->const_view().segments(),
                        tripletsDC_->const_view().triplets(),
                        tripletsDC_->const_view().tripletsBySegment(),
                        tripletsListRangesDC_->const_view().tripletsRangesBySegment(),
                        tripletsListRangesDC_->const_view().tripletsRangesByMD(),
                        miniDoubletsDC_->const_view().miniDoubletsOccupancy(),
                        nKeysByMD_buf.data(),
                        keyOffsetByMD_buf.data(),
                        keys_buf.data(),
                        nTripletSlots,
                        ptCut_,
                        dBetaPassMask_buf.data(),
                        moduleT5Count_buf.data(),
                        nSelected_d.data(),
                        capacity,
                        selectedT3s_buf.data(),
                        selectedBridgeRadius_buf.data(),
                        selectedDnnScore_buf.data(),
                        dBeta2Memo_buf.data());

    alpaka::exec<Acc1D>(queue_,
                        cms::alpakatools::make_workdiv<Acc1D>(1, 1024),
                        CreateEligibleModulesListForQuintuplets{},
                        modules_.const_view().modules(),
                        tripletsDC_->const_view().tripletsOccupancy(),
                        rangesDC_->view(),
                        moduleT5Count_buf.data());

    alpaka::exec<Acc2D>(queue_,
                        perModuleWorkDiv(nLowerModules_),
                        CreateQuintupletRangesByMD{},
                        modules_.const_view().modules(),
                        rangesDC_->view(),
                        miniDoubletsT5BuildDC_->const_view().t5Counts(),
                        miniDoubletsDC_->const_view().miniDoubletsOccupancy(),
                        miniDoubletsT5BuildDC_->view().quintupletsRangesByMD0(),
                        miniDoubletsT5BuildDC_->view().quintupletsRangesByMD1());

    alpaka::memcpy(queue_, nEligibleT5Modules_buf, nEligibleT5Modules_view_d);
    alpaka::memcpy(queue_, nTotalQuintuplets_buf, nTotalQuintuplets_view_d);
    alpaka::memcpy(queue_, nTotalQuintuplets0_buf, nTotalQuintuplets0_view_d);
    alpaka::memcpy(queue_, nTotalQuintuplets1_buf, nTotalQuintuplets1_view_d);
    alpaka::memcpy(queue_, nSelected_buf, nSelected_d);
    alpaka::wait(queue_);  // wait for the values before using them
    return std::make_tuple(selectedT3s_buf, selectedBridgeRadius_buf, selectedDnnScore_buf);
  };
  auto selected = countQuintuplets(selectedCapacity);
  if (*nSelected_buf.data() > selectedCapacity) {
    selectedCapacity = *nSelected_buf.data();
    selected = countQuintuplets(selectedCapacity);
  }
  auto& [selectedT3s_buf, selectedBridgeRadius_buf, selectedDnnScore_buf] = selected;
  const unsigned int nSelected = std::min(*nSelected_buf.data(), selectedCapacity);

  auto nEligibleT5Modules = *nEligibleT5Modules_buf.data();
  auto nTotalQuintuplets = *nTotalQuintuplets_buf.data();
  auto nTotalQuintuplets0 = *nTotalQuintuplets0_buf.data();
  auto nTotalQuintuplets1 = *nTotalQuintuplets1_buf.data();

  // Build-time records at the counting-kernel size; the quintuplets themselves are written at their exact size below.
  // The last two sizes are for quintupletsByMD{0,1}, which can differ from nTotalQuintuplets due to truncation.
  QuintupletsLooseDeviceCollection looseDC(
      queue_, nTotalQuintuplets, nLowerModules_, nTotalQuintuplets0, nTotalQuintuplets1);
  double looseMB = 0;
  if (objectsStatistics_) {
    looseMB = alpaka::getExtentProduct(looseDC.buffer()) / 1e6;
    trackTransientMB(looseMB);  // live until the end of this function, not added to the allocated total
    lstWarning(std::format("[MEM-loose] Quintuplets: {} allocated ({:.1f} MB)", nTotalQuintuplets, looseMB));
  }
  auto looseOccupancy = looseDC.view().quintupletsOccupancy();
  auto nQuintuplets_view = cms::alpakatools::make_device_view(queue_, looseOccupancy.nQuintuplets());
  alpaka::memset(queue_, nQuintuplets_view, 0u);
  auto totOccupancyQuintuplets_view =
      cms::alpakatools::make_device_view(queue_, looseOccupancy.totOccupancyQuintuplets());
  alpaka::memset(queue_, totOccupancyQuintuplets_view, 0u);

  if (nSelected > 0) {
    for (bool densePass : {true, false}) {
      alpaka::exec<Acc1D>(queue_,
                          cms::alpakatools::make_workdiv<Acc1D>(cms::alpakatools::divide_up_by(nSelected, 256u), 256),
                          AddSelectedQuintuplets{},
                          modules_.const_view().modules(),
                          miniDoubletsDC_->const_view().miniDoublets(),
                          miniDoubletsT5BuildDC_->const_view().t5Counts(),
                          miniDoubletsDC_->const_view().miniDoubletsOccupancy(),
                          segmentsDC_->const_view().segments(),
                          tripletsDC_->view().triplets(),
                          tripletsDC_->const_view().tripletsOccupancy(),
                          tripletsListRangesDC_->const_view().tripletsRangesByMD(),
                          looseDC.view().quintupletsLoose(),
                          looseOccupancy,
                          miniDoubletsT5BuildDC_->view().quintupletsRangesByMD0(),
                          miniDoubletsT5BuildDC_->view().quintupletsRangesByMD1(),
                          rangesDC_->view(),
                          selectedT3s_buf.data(),
                          selectedBridgeRadius_buf.data(),
                          selectedDnnScore_buf.data(),
                          nSelected,
                          densePass,
                          ptCut_);
    }
  }

  // Compact size: the selected quintuplets, laid out in module order.
  auto ranges = rangesDC_->view();
  auto compactIndices_buf = cms::alpakatools::make_device_buffer<int[]>(queue_, nLowerModules_ + 1);
  alpaka::exec<Acc1D>(queue_,
                      cms::alpakatools::make_workdiv<Acc1D>(1, 1024),
                      CompactModuleOffsetsKernel{},
                      static_cast<unsigned int>(nLowerModules_),
                      looseOccupancy.nQuintuplets().data(),
                      ranges.quintupletModuleIndices().data(),
                      compactIndices_buf.data());
  auto nCompact_buf = cms::alpakatools::make_host_buffer<int>(queue_);
  alpaka::memcpy(
      queue_, nCompact_buf, cms::alpakatools::make_device_view(queue_, compactIndices_buf.data()[nLowerModules_]));
  alpaka::wait(queue_);  // wait for the exact size before allocating
  const unsigned int nCompact = *nCompact_buf.data();

  if (!quintupletsDC_) {
    quintupletsDC_.emplace(queue_, nCompact, nLowerModules_, 0, 0);
    if (objectsStatistics_) {
      double mb = alpaka::getExtentProduct(quintupletsDC_->buffer()) / 1e6;
      trackAllocatedMB(mb);
      lstWarning(std::format("[MEM] Quintuplets: {} allocated ({:.1f} MB)", nCompact, mb));
    }
  }

  alpaka::exec<Acc1D>(queue_,
                      cms::alpakatools::make_workdiv<Acc1D>(std::max((int)nEligibleT5Modules, 1), 64),
                      FinalizeQuintuplets{},
                      modules_.const_view().modules(),
                      miniDoubletsDC_->const_view().miniDoublets(),
                      segmentsDC_->const_view().segments(),
                      tripletsDC_->const_view().triplets(),
                      miniDoubletsDC_->const_view().miniDoubletsOccupancy(),
                      tripletsListRangesDC_->const_view().tripletsRangesByMD(),
                      looseDC.const_view().quintupletsLoose(),
                      looseDC.const_view().quintupletsOccupancy(),
                      quintupletsDC_->view().quintuplets(),
                      quintupletsDC_->view().quintupletsOccupancy(),
                      looseDC.view().quintupletsByMD0(),
                      looseDC.view().quintupletsByMD1(),
                      rangesDC_->const_view(),
                      compactIndices_buf.data(),
                      static_cast<unsigned int>(nEligibleT5Modules));

  alpaka::exec<Acc1D>(queue_,
                      cms::alpakatools::make_workdiv<Acc1D>(cms::alpakatools::divide_up_by(nLowerModules_, 256), 256),
                      SetCompactModuleIndices{},
                      static_cast<unsigned int>(nLowerModules_),
                      looseDC.const_view().quintupletsOccupancy().nQuintuplets().data(),
                      compactIndices_buf.data(),
                      ranges.quintupletModuleIndices().data(),
                      ranges.quintupletModuleOccupancy().data());

  if (nCompact > 0) {
    auto const extendT5_workDiv = cms::alpakatools::make_workdiv<Acc1D>(nCompact, 32);

    alpaka::exec<Acc1D>(queue_,
                        extendT5_workDiv,
                        ExtendT5FromDupT5ByMD{},
                        quintupletsDC_->view().quintuplets(),
                        quintupletsDC_->const_view().quintupletsOccupancy(),
                        miniDoubletsT5BuildDC_->const_view().quintupletsRangesByMD0(),
                        looseDC.const_view().quintupletsByMD0(),
                        miniDoubletsT5BuildDC_->const_view().quintupletsRangesByMD1(),
                        looseDC.const_view().quintupletsByMD1(),
                        tripletsDC_->const_view().triplets(),
                        segmentsDC_->const_view().segments());
  }

  // Last reader of the per-MD T5 counters and T5-by-MD ranges.
  releaseDeviceCollection(miniDoubletsT5BuildDC_);

  auto const removeDupQuintupletsAfterBuild_workDiv =
      cms::alpakatools::make_workdiv<Acc3D>({max_blocks, 1, 1}, {1, 16, 16});

  alpaka::exec<Acc3D>(queue_,
                      removeDupQuintupletsAfterBuild_workDiv,
                      RemoveDupQuintupletsAfterBuild{},
                      modules_.const_view().modules(),
                      quintupletsDC_->view().quintuplets(),
                      quintupletsDC_->const_view().quintupletsOccupancy(),
                      rangesDC_->const_view());

  if (objectsStatistics_) {
    memoryLiveMB_ -= looseMB;
    addQuintupletsToEventExplicit();
  }
}

void LSTEvent::pixelLineSegmentCleaning(bool no_pls_dupclean) {
  if (!no_pls_dupclean) {
    auto const checkHitspLS_workDiv = cms::alpakatools::make_workdiv<Acc2D>({max_blocks * 4, max_blocks / 4}, {16, 16});

    alpaka::exec<Acc2D>(queue_,
                        checkHitspLS_workDiv,
                        CheckHitspLS{},
                        modules_.const_view().modules(),
                        segmentsDC_->const_view().segmentsOccupancy(),
                        lstInputDC_->const_view().pixelSeeds(),
                        pixelSegmentsDC_->view(),
                        false);
  }
}

void LSTEvent::createPixelQuintuplets() {
  if (!pixelQuintupletsDC_) {
    pixelQuintupletsDC_.emplace(queue_, n_max_pixel_quintuplets);
    auto nPixelQuintuplets_view =
        cms::alpakatools::make_device_view(queue_, (*pixelQuintupletsDC_)->nPixelQuintuplets());
    alpaka::memset(queue_, nPixelQuintuplets_view, 0u);
    auto totOccupancyPixelQuintuplets_view =
        cms::alpakatools::make_device_view(queue_, (*pixelQuintupletsDC_)->totOccupancyPixelQuintuplets());
    alpaka::memset(queue_, totOccupancyPixelQuintuplets_view, 0u);
    if (objectsStatistics_) {
      double mb = alpaka::getExtentProduct(pixelQuintupletsDC_->buffer()) / 1e6;
      trackTransientMB(mb);  // shrunk to the produced pT5s by compactPixelQuintuplets
      lstWarning(
          std::format("[MEM-loose] PixelQuintuplets: {} allocated ({:.1f} MB) [fixed]", n_max_pixel_quintuplets, mb));
    }
  }

  fillConnectedPixels();
  // nSegments of the pixel module is pixelSize_ (set in createSegmentsWithModuleMap).
  const unsigned int nInnerSegments = pixelSize_;

  auto const createPixelQuintupletsFromMap_workDiv =
      cms::alpakatools::make_workdiv<Acc3D>({max_blocks, 16, 1}, {16, 1, 16});

  alpaka::exec<Acc3D>(queue_,
                      createPixelQuintupletsFromMap_workDiv,
                      CreatePixelQuintupletsFromMap{},
                      modules_.const_view().modules(),
                      modules_.const_view().modulesPixel(),
                      miniDoubletsDC_->const_view().miniDoublets(),
                      segmentsDC_->const_view().segments(),
                      lstInputDC_->const_view().pixelSeeds(),
                      pixelSegmentsDC_->view(),
                      tripletsDC_->view().triplets(),
                      quintupletsDC_->view().quintuplets(),
                      quintupletsDC_->const_view().quintupletsOccupancy(),
                      pixelQuintupletsDC_->view(),
                      connectedPixelSize_->data(),
                      connectedPixelIndex_->data(),
                      nInnerSegments,
                      rangesDC_->const_view(),
                      ptCut_);

  // pT5 duplicate removal on an eta-phi grid (window 0.2): count -> prefix -> scatter, then 3x3 cells per pT5.
  auto const pT5Grid = EtaPhiGrid::make(0.2f, 3.f);
  auto cellCount_buf = cms::alpakatools::make_device_buffer<unsigned int[]>(queue_, pT5Grid.nCells());
  auto cellStart_buf = cms::alpakatools::make_device_buffer<unsigned int[]>(queue_, pT5Grid.nCells() + 1);
  auto cellItems_buf = cms::alpakatools::make_device_buffer<unsigned int[]>(queue_, n_max_pixel_quintuplets);
  alpaka::memset(queue_, cellCount_buf, 0u);
  auto const pT5GridFill_workDiv =
      cms::alpakatools::make_workdiv<Acc1D>(cms::alpakatools::divide_up_by(n_max_pixel_quintuplets, 256u), 256u);
  alpaka::exec<Acc1D>(queue_,
                      pT5GridFill_workDiv,
                      FillPixelQuintupletGrid{},
                      pixelQuintupletsDC_->const_view(),
                      pT5Grid,
                      cellCount_buf.data(),
                      static_cast<unsigned int*>(nullptr));
  alpaka::exec<Acc1D>(queue_,
                      cms::alpakatools::make_workdiv<Acc1D>(1, 1024),
                      EtaPhiGridPrefix{},
                      pT5Grid.nCells(),
                      cellCount_buf.data(),
                      cellStart_buf.data());
  alpaka::exec<Acc1D>(queue_,
                      pT5GridFill_workDiv,
                      FillPixelQuintupletGrid{},
                      pixelQuintupletsDC_->const_view(),
                      pT5Grid,
                      cellCount_buf.data(),
                      cellItems_buf.data());

  auto const removeDupPixelQuintupletsFromMap_workDiv = cms::alpakatools::make_workdiv<Acc2D>(
      {cms::alpakatools::divide_up_by(n_max_pixel_quintuplets, 16u), 1u}, {16u, 16u});

  alpaka::exec<Acc2D>(queue_,
                      removeDupPixelQuintupletsFromMap_workDiv,
                      RemoveDupPixelQuintupletsFromMap{},
                      miniDoubletsDC_->const_view().miniDoublets(),
                      segmentsDC_->const_view().segments(),
                      quintupletsDC_->const_view().quintuplets(),
                      pixelQuintupletsDC_->view(),
                      pT5Grid,
                      cellStart_buf.data(),
                      cellItems_buf.data());

#ifdef WARNINGS
  auto nPixelQuintuplets_buf = cms::alpakatools::make_host_buffer<unsigned int>(queue_);

  alpaka::memcpy(queue_,
                 nPixelQuintuplets_buf,
                 cms::alpakatools::make_device_view(queue_, (*pixelQuintupletsDC_)->nPixelQuintuplets()));
  alpaka::wait(queue_);  // wait to get the value before using it

  std::cout << "number of pixel quintuplets = " << *nPixelQuintuplets_buf.data() << std::endl;
#endif

  compactPixelQuintuplets(queue_);
}

void LSTEvent::createQuadruplets() {
  // Per-triplet T4 counter (count -> create), live only in this stage.
  const unsigned int nTripletSlots = tripletsDC_->const_view().triplets().metadata().size();
  auto t3ConnectedLSMax_buf = cms::alpakatools::make_device_buffer<unsigned int[]>(queue_, nTripletSlots);
  alpaka::memset(queue_, t3ConnectedLSMax_buf, 0u);
  // Per-module sum of the per-triplet counters, accumulated by the counting kernel.
  auto moduleT4Max_buf = cms::alpakatools::make_device_buffer<unsigned int[]>(queue_, nLowerModules_);
  alpaka::memset(queue_, moduleT4Max_buf, 0u);
  // Dense-pair decisions of the counting kernel (count -> create), live only in this stage.
  auto t4PassMask_buf = cms::alpakatools::make_device_buffer<uint32_t[]>(queue_, nTripletSlots);
  alpaka::memset(queue_, t4PassMask_buf, 0u);

  // 32 inner triplets per block (not one module per block): dense jet modules spread over the whole device. A row of
  // 8 threads per inner triplet, as the outer triplets of its second segment are few (0-1 in ttbar, ~8 in jets).
  auto const countLSConn_workDiv = cms::alpakatools::make_workdiv<Acc2D>(
      {std::max(cms::alpakatools::divide_up_by(nTripletSlots, 32u), 1u), 1u}, {32, 8});

  alpaka::exec<Acc2D>(queue_,
                      countLSConn_workDiv,
                      CountTripletLSConnections{},
                      modules_.const_view().modules(),
                      miniDoubletsDC_->const_view().miniDoublets(),
                      segmentsDC_->const_view().segments(),
                      tripletsDC_->view().triplets(),
                      tripletsDC_->const_view().tripletsOccupancy(),
                      tripletsDC_->const_view().tripletsBySegment(),
                      tripletsListRangesDC_->const_view().tripletsRangesBySegment(),
                      miniDoubletsDC_->const_view().miniDoubletsOccupancy(),
                      tripletsListRangesDC_->const_view().tripletsRangesByMD(),
                      ptCut_,
                      t3ConnectedLSMax_buf.data(),
                      moduleT4Max_buf.data(),
                      t4PassMask_buf.data(),
                      nTripletSlots);

  auto const createEligibleModulesListForQuadruplets_workDiv = cms::alpakatools::make_workdiv<Acc1D>(1, 1024);

  alpaka::exec<Acc1D>(queue_,
                      createEligibleModulesListForQuadruplets_workDiv,
                      CreateEligibleModulesListForQuadruplets{},
                      modules_.const_view().modules(),
                      tripletsDC_->const_view().tripletsOccupancy(),
                      rangesDC_->view(),
                      moduleT4Max_buf.data());

  auto nEligibleT4Modules_buf = cms::alpakatools::make_host_buffer<uint16_t>(queue_);
  auto nTotalQuadruplets_buf = cms::alpakatools::make_host_buffer<unsigned int>(queue_);
  auto rangesOccupancy = rangesDC_->view();
  auto nEligibleT4Modules_view_d = cms::alpakatools::make_device_view(queue_, rangesOccupancy.nEligibleT4Modules());
  auto nTotalQuadruplets_view_d = cms::alpakatools::make_device_view(queue_, rangesOccupancy.nTotalQuads());
  alpaka::memcpy(queue_, nEligibleT4Modules_buf, nEligibleT4Modules_view_d);
  alpaka::memcpy(queue_, nTotalQuadruplets_buf, nTotalQuadruplets_view_d);
  alpaka::wait(queue_);  // wait for the values before using them

  auto nEligibleT4Modules = *nEligibleT4Modules_buf.data();
  auto nTotalQuadruplets = *nTotalQuadruplets_buf.data();

  if (!quadrupletsDC_) {
    quadrupletsDC_.emplace(queue_, nTotalQuadruplets, nLowerModules_);
    if (objectsStatistics_) {
      double mb = alpaka::getExtentProduct(quadrupletsDC_->buffer()) / 1e6;
      trackTransientMB(mb);  // shrunk to the produced T4s by compactQuadruplets
      lstWarning(std::format("[MEM-loose] Quadruplets: {} allocated ({:.1f} MB)", nTotalQuadruplets, mb));
    }
    auto quadrupletsOccupancy = quadrupletsDC_->view().quadrupletsOccupancy();
    auto nQuadruplets_view = cms::alpakatools::make_device_view(
        queue_, quadrupletsOccupancy.nQuadruplets(), quadrupletsOccupancy.metadata().size());
    alpaka::memset(queue_, nQuadruplets_view, 0u);
    auto totOccupancyQuadruplets_view = cms::alpakatools::make_device_view(
        queue_, quadrupletsOccupancy.totOccupancyQuadruplets(), quadrupletsOccupancy.metadata().size());
    alpaka::memset(queue_, totOccupancyQuadruplets_view, 0u);
    auto quadruplets = quadrupletsDC_->view().quadruplets();
    auto isDup_view = cms::alpakatools::make_device_view(queue_, quadruplets.isDup(), quadruplets.metadata().size());
    alpaka::memset(queue_, isDup_view, 0u);
  }

  auto const createQuadruplets_workDiv =
      cms::alpakatools::make_workdiv<Acc3D>({std::max((int)nEligibleT4Modules, 1), 1, 1}, {1, 8, 32});

  alpaka::exec<Acc3D>(queue_,
                      createQuadruplets_workDiv,
                      CreateQuadruplets{},
                      modules_.const_view().modules(),
                      miniDoubletsDC_->const_view().miniDoublets(),
                      segmentsDC_->const_view().segments(),
                      tripletsDC_->const_view().triplets(),
                      tripletsDC_->const_view().tripletsOccupancy(),
                      tripletsDC_->const_view().tripletsBySegment(),
                      tripletsListRangesDC_->const_view().tripletsRangesBySegment(),
                      miniDoubletsDC_->const_view().miniDoubletsOccupancy(),
                      tripletsListRangesDC_->const_view().tripletsRangesByMD(),
                      quadrupletsDC_->view().quadruplets(),
                      quadrupletsDC_->view().quadrupletsOccupancy(),
                      rangesDC_->const_view(),
                      nEligibleT4Modules,
                      ptCut_,
                      t3ConnectedLSMax_buf.data(),
                      t4PassMask_buf.data());
  alpaka::exec<Acc2D>(queue_,
                      countLSConn_workDiv,
                      CreateQuadrupletsDense{},
                      modules_.const_view().modules(),
                      miniDoubletsDC_->const_view().miniDoublets(),
                      segmentsDC_->const_view().segments(),
                      tripletsDC_->const_view().triplets(),
                      tripletsDC_->const_view().tripletsOccupancy(),
                      tripletsDC_->const_view().tripletsBySegment(),
                      tripletsListRangesDC_->const_view().tripletsRangesBySegment(),
                      miniDoubletsDC_->const_view().miniDoubletsOccupancy(),
                      tripletsListRangesDC_->const_view().tripletsRangesByMD(),
                      quadrupletsDC_->view().quadruplets(),
                      quadrupletsDC_->view().quadrupletsOccupancy(),
                      rangesDC_->const_view(),
                      ptCut_,
                      t3ConnectedLSMax_buf.data(),
                      t4PassMask_buf.data(),
                      nTripletSlots);

  // 16 y blocks per module: the T4s of a dense jet module are not left to one block.
  auto const removeDupQuadrupletsAfterBuild_workDiv =
      cms::alpakatools::make_workdiv<Acc3D>({std::max((unsigned int)nEligibleT4Modules, 1u), 16u, 1u}, {1, 16, 16});

  alpaka::exec<Acc3D>(queue_,
                      removeDupQuadrupletsAfterBuild_workDiv,
                      RemoveDupQuadrupletsAfterBuild{},
                      quadrupletsDC_->view().quadruplets(),
                      quadrupletsDC_->const_view().quadrupletsOccupancy(),
                      rangesDC_->const_view());

  compactQuadruplets(queue_, nEligibleT4Modules);

  if (objectsStatistics_) {
    addQuadrupletsToEventExplicit();
  }
}

void LSTEvent::addMiniDoubletsToEventExplicit() {
  auto nMDsCPU_buf = cms::alpakatools::make_host_buffer<unsigned int[]>(queue_, nLowerModules_);
  auto mdsOccupancy = miniDoubletsDC_->const_view().miniDoubletsOccupancy();
  auto nMDs_view =
      cms::alpakatools::make_device_view(queue_, mdsOccupancy.nMDs(), nLowerModules_);  // exclude pixel part
  alpaka::memcpy(queue_, nMDsCPU_buf, nMDs_view, nLowerModules_);

  auto modules = modules_.const_view().modules();

  // FIXME: replace by ES host data
  auto module_subdets_buf = cms::alpakatools::make_host_buffer<short[]>(queue_, nLowerModules_);
  auto module_subdets_view =
      cms::alpakatools::make_device_view(queue_, modules.subdets(), nLowerModules_);  // only lower modules
  alpaka::memcpy(queue_, module_subdets_buf, module_subdets_view, nLowerModules_);

  auto module_layers_buf = cms::alpakatools::make_host_buffer<short[]>(queue_, nLowerModules_);
  auto module_layers_view =
      cms::alpakatools::make_device_view(queue_, modules.layers(), nLowerModules_);  // only lower modules
  alpaka::memcpy(queue_, module_layers_buf, module_layers_view, nLowerModules_);

  alpaka::wait(queue_);  // wait for inputs before using them

  auto const* nMDsCPU = nMDsCPU_buf.data();
  auto const* module_subdets = module_subdets_buf.data();
  auto const* module_layers = module_layers_buf.data();

  for (unsigned int i = 0; i < nLowerModules_; i++) {
    if (nMDsCPU[i] != 0) {
      if (module_subdets[i] == Barrel) {
        n_minidoublets_by_layer_barrel_[module_layers[i] - 1] += nMDsCPU[i];
      } else {
        n_minidoublets_by_layer_endcap_[module_layers[i] - 1] += nMDsCPU[i];
      }
    }
  }
}

void LSTEvent::addSegmentsToEventExplicit() {
  auto nSegmentsCPU_buf = cms::alpakatools::make_host_buffer<unsigned int[]>(queue_, nLowerModules_);
  auto nSegments_buf = cms::alpakatools::make_device_view(
      queue_, segmentsDC_->const_view().segmentsOccupancy().nSegments(), nLowerModules_);
  alpaka::memcpy(queue_, nSegmentsCPU_buf, nSegments_buf, nLowerModules_);

  auto modules = modules_.const_view().modules();

  // FIXME: replace by ES host data
  auto module_subdets_buf = cms::alpakatools::make_host_buffer<short[]>(queue_, nLowerModules_);
  auto module_subdets_view =
      cms::alpakatools::make_device_view(queue_, modules.subdets(), nLowerModules_);  // only lower modules
  alpaka::memcpy(queue_, module_subdets_buf, module_subdets_view, nLowerModules_);

  auto module_layers_buf = cms::alpakatools::make_host_buffer<short[]>(queue_, nLowerModules_);
  auto module_layers_view =
      cms::alpakatools::make_device_view(queue_, modules.layers(), nLowerModules_);  // only lower modules
  alpaka::memcpy(queue_, module_layers_buf, module_layers_view, nLowerModules_);

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

void LSTEvent::addQuintupletsToEventExplicit() {
  auto quintupletsOccupancy = quintupletsDC_->const_view().quintupletsOccupancy();
  auto nQuintuplets_view =
      cms::alpakatools::make_device_view(queue_, quintupletsOccupancy.nQuintuplets(), nLowerModules_);
  auto nQuintupletsCPU_buf = cms::alpakatools::make_host_buffer<unsigned int[]>(queue_, nLowerModules_);
  alpaka::memcpy(queue_, nQuintupletsCPU_buf, nQuintuplets_view);

  auto modules = modules_.const_view().modules();

  // FIXME: replace by ES host data
  auto module_subdets_buf = cms::alpakatools::make_host_buffer<short[]>(queue_, nLowerModules_);
  auto module_subdets_view =
      cms::alpakatools::make_device_view(queue_, modules.subdets(), nLowerModules_);  // only lower modules
  alpaka::memcpy(queue_, module_subdets_buf, module_subdets_view, nLowerModules_);

  auto module_layers_buf = cms::alpakatools::make_host_buffer<short[]>(queue_, nLowerModules_);
  auto module_layers_view =
      cms::alpakatools::make_device_view(queue_, modules.layers(), nLowerModules_);  // only lower modules
  alpaka::memcpy(queue_, module_layers_buf, module_layers_view, nLowerModules_);

  auto module_quintupletModuleIndices_buf = cms::alpakatools::make_host_buffer<int[]>(queue_, nLowerModules_);
  auto rangesOccupancy = rangesDC_->view();
  auto quintupletModuleIndices_view_d =
      cms::alpakatools::make_device_view(queue_, rangesOccupancy.quintupletModuleIndices(), nLowerModules_);
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

void LSTEvent::addTripletsToEventExplicit() {
  auto tripletsOccupancy = tripletsDC_->const_view().tripletsOccupancy();
  auto nTriplets_view = cms::alpakatools::make_device_view(queue_, tripletsOccupancy.nTriplets(), nLowerModules_);
  auto nTripletsCPU_buf = cms::alpakatools::make_host_buffer<unsigned int[]>(queue_, nLowerModules_);
  alpaka::memcpy(queue_, nTripletsCPU_buf, nTriplets_view);

  auto modules = modules_.const_view().modules();

  // FIXME: replace by ES host data
  auto module_subdets_buf = cms::alpakatools::make_host_buffer<short[]>(queue_, nLowerModules_);
  auto module_subdets_view =
      cms::alpakatools::make_device_view(queue_, modules.subdets(), nLowerModules_);  // only lower modules
  alpaka::memcpy(queue_, module_subdets_buf, module_subdets_view, nLowerModules_);

  auto module_layers_buf = cms::alpakatools::make_host_buffer<short[]>(queue_, nLowerModules_);
  auto module_layers_view =
      cms::alpakatools::make_device_view(queue_, modules.layers(), nLowerModules_);  // only lower modules
  alpaka::memcpy(queue_, module_layers_buf, module_layers_view, nLowerModules_);

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

void LSTEvent::addQuadrupletsToEventExplicit() {
  auto quadrupletsOccupancy = quadrupletsDC_->const_view().quadrupletsOccupancy();
  auto nQuadruplets_view =
      cms::alpakatools::make_device_view(queue_, quadrupletsOccupancy.nQuadruplets(), nLowerModules_);
  auto nQuadrupletsCPU_buf = cms::alpakatools::make_host_buffer<unsigned int[]>(queue_, nLowerModules_);
  alpaka::memcpy(queue_, nQuadrupletsCPU_buf, nQuadruplets_view);

  auto modules = modules_.const_view().modules();

  // FIXME: replace by ES host data
  auto module_subdets_buf = cms::alpakatools::make_host_buffer<short[]>(queue_, nLowerModules_);
  auto module_subdets_view =
      cms::alpakatools::make_device_view(queue_, modules.subdets(), nLowerModules_);  // only lower modules
  alpaka::memcpy(queue_, module_subdets_buf, module_subdets_view, nLowerModules_);

  auto module_layers_buf = cms::alpakatools::make_host_buffer<short[]>(queue_, nLowerModules_);
  auto module_layers_view =
      cms::alpakatools::make_device_view(queue_, modules.layers(), nLowerModules_);  // only lower modules
  alpaka::memcpy(queue_, module_layers_buf, module_layers_view, nLowerModules_);

  alpaka::wait(queue_);  // wait for inputs before using them

  auto const* nQuadrupletsCPU = nQuadrupletsCPU_buf.data();
  auto const* module_subdets = module_subdets_buf.data();
  auto const* module_layers = module_layers_buf.data();

  for (uint16_t i = 0; i < nLowerModules_; i++) {
    if (nQuadrupletsCPU[i] != 0) {
      if (module_subdets[i] == Barrel) {
        n_quadruplets_by_layer_barrel_[module_layers[i] - 1] += nQuadrupletsCPU[i];
      } else {
        n_quadruplets_by_layer_endcap_[module_layers[i] - 1] += nQuadrupletsCPU[i];
      }
    }
  }
}

unsigned int LSTEvent::getNumberOfMiniDoublets() {
  unsigned int miniDoublets = 0;
  for (auto& it : n_minidoublets_by_layer_barrel_) {
    miniDoublets += it;
  }
  for (auto& it : n_minidoublets_by_layer_endcap_) {
    miniDoublets += it;
  }

  return miniDoublets;
}

unsigned int LSTEvent::getNumberOfMiniDoubletsByLayerBarrel(unsigned int layer) {
  return n_minidoublets_by_layer_barrel_[layer];
}

unsigned int LSTEvent::getNumberOfMiniDoubletsByLayerEndcap(unsigned int layer) {
  return n_minidoublets_by_layer_endcap_[layer];
}

unsigned int LSTEvent::getNumberOfSegments() {
  unsigned int segments = 0;
  for (auto& it : n_segments_by_layer_barrel_) {
    segments += it;
  }
  for (auto& it : n_segments_by_layer_endcap_) {
    segments += it;
  }

  return segments;
}

unsigned int LSTEvent::getNumberOfSegmentsByLayerBarrel(unsigned int layer) {
  return n_segments_by_layer_barrel_[layer];
}

unsigned int LSTEvent::getNumberOfSegmentsByLayerEndcap(unsigned int layer) {
  return n_segments_by_layer_endcap_[layer];
}

unsigned int LSTEvent::getNumberOfTriplets() {
  unsigned int triplets = 0;
  for (auto& it : n_triplets_by_layer_barrel_) {
    triplets += it;
  }
  for (auto& it : n_triplets_by_layer_endcap_) {
    triplets += it;
  }

  return triplets;
}

unsigned int LSTEvent::getNumberOfTripletsByLayerBarrel(unsigned int layer) {
  return n_triplets_by_layer_barrel_[layer];
}

unsigned int LSTEvent::getNumberOfTripletsByLayerEndcap(unsigned int layer) {
  return n_triplets_by_layer_endcap_[layer];
}

int LSTEvent::getNumberOfPixelTriplets() {
  auto nPixelTriplets_buf_h = cms::alpakatools::make_host_buffer<unsigned int>(queue_);

  alpaka::memcpy(
      queue_, nPixelTriplets_buf_h, cms::alpakatools::make_device_view(queue_, (*pixelTripletsDC_)->nPixelTriplets()));
  alpaka::wait(queue_);

  return *nPixelTriplets_buf_h.data();
}

int LSTEvent::getNumberOfPixelQuintuplets() {
  auto nPixelQuintuplets_buf_h = cms::alpakatools::make_host_buffer<unsigned int>(queue_);

  alpaka::memcpy(queue_,
                 nPixelQuintuplets_buf_h,
                 cms::alpakatools::make_device_view(queue_, (*pixelQuintupletsDC_)->nPixelQuintuplets()));
  alpaka::wait(queue_);

  return *nPixelQuintuplets_buf_h.data();
}

unsigned int LSTEvent::getNumberOfQuintuplets() {
  unsigned int quintuplets = 0;
  for (auto& it : n_quintuplets_by_layer_barrel_) {
    quintuplets += it;
  }
  for (auto& it : n_quintuplets_by_layer_endcap_) {
    quintuplets += it;
  }

  return quintuplets;
}

unsigned int LSTEvent::getNumberOfQuintupletsByLayerBarrel(unsigned int layer) {
  return n_quintuplets_by_layer_barrel_[layer];
}

unsigned int LSTEvent::getNumberOfQuintupletsByLayerEndcap(unsigned int layer) {
  return n_quintuplets_by_layer_endcap_[layer];
}

int LSTEvent::getNumberOfTrackCandidates() {
  auto nTrackCandidates_buf_h = cms::alpakatools::make_host_buffer<unsigned int>(queue_);

  alpaka::memcpy(queue_,
                 nTrackCandidates_buf_h,
                 cms::alpakatools::make_device_view(queue_, (*trackCandidatesBaseDC_)->nTrackCandidates()));
  alpaka::wait(queue_);

  return *nTrackCandidates_buf_h.data();
}

int LSTEvent::getNumberOfPT5TrackCandidates() {
  auto nTrackCandidatesPT5_buf_h = cms::alpakatools::make_host_buffer<unsigned int>(queue_);

  alpaka::memcpy(queue_,
                 nTrackCandidatesPT5_buf_h,
                 cms::alpakatools::make_device_view(queue_, (*trackCandidatesExtendedDC_)->nTrackCandidatespT5()));
  alpaka::wait(queue_);

  return *nTrackCandidatesPT5_buf_h.data();
}

int LSTEvent::getNumberOfPT3TrackCandidates() {
  auto nTrackCandidatesPT3_buf_h = cms::alpakatools::make_host_buffer<unsigned int>(queue_);

  alpaka::memcpy(queue_,
                 nTrackCandidatesPT3_buf_h,
                 cms::alpakatools::make_device_view(queue_, (*trackCandidatesExtendedDC_)->nTrackCandidatespT3()));
  alpaka::wait(queue_);

  return *nTrackCandidatesPT3_buf_h.data();
}

int LSTEvent::getNumberOfPLSTrackCandidates() {
  auto nTrackCandidatesPLS_buf_h = cms::alpakatools::make_host_buffer<unsigned int>(queue_);

  alpaka::memcpy(queue_,
                 nTrackCandidatesPLS_buf_h,
                 cms::alpakatools::make_device_view(queue_, (*trackCandidatesExtendedDC_)->nTrackCandidatespLS()));
  alpaka::wait(queue_);

  return *nTrackCandidatesPLS_buf_h.data();
}

int LSTEvent::getNumberOfPixelTrackCandidates() {
  auto nTrackCandidates_buf_h = cms::alpakatools::make_host_buffer<unsigned int>(queue_);
  auto nTrackCandidatesT5_buf_h = cms::alpakatools::make_host_buffer<unsigned int>(queue_);
  auto nTrackCandidatesT4_buf_h = cms::alpakatools::make_host_buffer<unsigned int>(queue_);

  alpaka::memcpy(queue_,
                 nTrackCandidates_buf_h,
                 cms::alpakatools::make_device_view(queue_, (*trackCandidatesBaseDC_)->nTrackCandidates()));
  alpaka::memcpy(queue_,
                 nTrackCandidatesT5_buf_h,
                 cms::alpakatools::make_device_view(queue_, (*trackCandidatesExtendedDC_)->nTrackCandidatesT5()));
  alpaka::memcpy(queue_,
                 nTrackCandidatesT4_buf_h,
                 cms::alpakatools::make_device_view(queue_, (*trackCandidatesExtendedDC_)->nTrackCandidatesT4()));
  alpaka::wait(queue_);

  return (*nTrackCandidates_buf_h.data()) - (*nTrackCandidatesT5_buf_h.data()) - (*nTrackCandidatesT4_buf_h.data());
}

int LSTEvent::getNumberOfT5TrackCandidates() {
  auto nTrackCandidatesT5_buf_h = cms::alpakatools::make_host_buffer<unsigned int>(queue_);

  alpaka::memcpy(queue_,
                 nTrackCandidatesT5_buf_h,
                 cms::alpakatools::make_device_view(queue_, (*trackCandidatesExtendedDC_)->nTrackCandidatesT5()));
  alpaka::wait(queue_);

  return *nTrackCandidatesT5_buf_h.data();
}

int LSTEvent::getNumberOfT4TrackCandidates() {
  auto nTrackCandidatesT4_buf_h = cms::alpakatools::make_host_buffer<unsigned int>(queue_);

  alpaka::memcpy(queue_,
                 nTrackCandidatesT4_buf_h,
                 cms::alpakatools::make_device_view(queue_, (*trackCandidatesExtendedDC_)->nTrackCandidatesT4()));
  alpaka::wait(queue_);

  return *nTrackCandidatesT4_buf_h.data();
}

unsigned int LSTEvent::getNumberOfQuadruplets() {
  unsigned int quadruplets = 0;
  for (auto& it : n_quadruplets_by_layer_barrel_) {
    quadruplets += it;
  }
  for (auto& it : n_quadruplets_by_layer_endcap_) {
    quadruplets += it;
  }

  return quadruplets;
}

unsigned int LSTEvent::getNumberOfQuadrupletsByLayerBarrel(unsigned int layer) {
  return n_quadruplets_by_layer_barrel_[layer];
}

unsigned int LSTEvent::getNumberOfQuadrupletsByLayerEndcap(unsigned int layer) {
  return n_quadruplets_by_layer_endcap_[layer];
}

template <typename TDev>
LSTInputConstView LSTEvent::getInput(bool sync) {
  if constexpr (std::is_same_v<TDev, DevHost>) {
    return lstInputDC_->const_view();
  } else {
    if (!lstInputHC_) {
      lstInputHC_.emplace(
          cms::alpakatools::CopyToHost<PortableDeviceCollection<TDev, LSTInputSoA>>::copyAsync(queue_, *lstInputDC_));
      if (sync)
        alpaka::wait(queue_);  // host consumers expect filled data
    }
    return lstInputHC_->const_view();
  }
}
template LSTInputConstView LSTEvent::getInput<>(bool);

template <typename TSoA, typename TDev>
typename TSoA::ConstView LSTEvent::getHits(bool sync) {
  if (!hitsDC_ && !hitsHC_)
    lstLogicError("LSTEvent::getHits: hits released after the MD stage; call setKeepHostCopies(true)");
  if constexpr (std::is_same_v<TDev, DevHost>) {
    return HitsViewAccessor<TSoA>::get(hitsDC_ ? hitsDC_->const_view() : hitsHC_->const_view());
  } else {
    if (!hitsHC_) {
      hitsHC_.emplace(
          cms::alpakatools::CopyToHost<PortableDeviceCollection<TDev, HitsSoA>>::copyAsync(queue_, *hitsDC_));
      if (sync)
        alpaka::wait(queue_);  // host consumers expect filled data
    }
    return HitsViewAccessor<TSoA>::get(hitsHC_->const_view());
  }
}
template HitsExtendedConst LSTEvent::getHits<HitsExtendedSoA>(bool);
template HitsRangesConst LSTEvent::getHits<HitsRangesSoA>(bool);

template <typename TDev>
ObjectRangesConst LSTEvent::getRanges(bool sync) {
  if constexpr (std::is_same_v<TDev, DevHost>) {
    return rangesDC_->const_view();
  } else {
    if (!rangesHC_) {
      rangesHC_.emplace(
          cms::alpakatools::CopyToHost<PortableDeviceCollection<TDev, ObjectRangesSoA>>::copyAsync(queue_, *rangesDC_));
      if (sync)
        alpaka::wait(queue_);  // host consumers expect filled data
    }
    return rangesHC_->const_view();
  }
}
template ObjectRangesConst LSTEvent::getRanges<>(bool);

template <typename TSoA, typename TDev>
typename TSoA::ConstView LSTEvent::getMiniDoublets(bool sync) {
  if constexpr (std::is_same_v<TDev, DevHost>) {
    return MiniDoubletsViewAccessor<TSoA>::get(miniDoubletsDC_->const_view());
  } else {
    if (!miniDoubletsHC_) {
      miniDoubletsHC_.emplace(
          cms::alpakatools::CopyToHost<PortableDeviceCollection<TDev, MiniDoubletsSoABlocks>>::copyAsync(
              queue_, *miniDoubletsDC_));
      if (sync)
        alpaka::wait(queue_);  // host consumers expect filled data
    }
    return MiniDoubletsViewAccessor<TSoA>::get(miniDoubletsHC_->const_view());
  }
}
template MiniDoubletsConst LSTEvent::getMiniDoublets<MiniDoubletsSoA>(bool);
template MiniDoubletsOccupancyConst LSTEvent::getMiniDoublets<MiniDoubletsOccupancySoA>(bool);

template <typename TDev>
MiniDoubletsBuildConst LSTEvent::getMiniDoubletsBuild(bool sync) {
  if (!miniDoubletsBuildDC_ && !miniDoubletsBuildHC_)
    lstLogicError("LSTEvent::getMiniDoubletsBuild: released after the LS stage; call setKeepHostCopies(true)");
  if constexpr (std::is_same_v<TDev, DevHost>) {
    return miniDoubletsBuildDC_ ? miniDoubletsBuildDC_->const_view() : miniDoubletsBuildHC_->const_view();
  } else {
    if (!miniDoubletsBuildHC_) {
      miniDoubletsBuildHC_.emplace(
          cms::alpakatools::CopyToHost<PortableDeviceCollection<TDev, MiniDoubletsBuildSoA>>::copyAsync(
              queue_, *miniDoubletsBuildDC_));
      if (sync)
        alpaka::wait(queue_);  // host consumers expect filled data
    }
    return miniDoubletsBuildHC_->const_view();
  }
}
template MiniDoubletsBuildConst LSTEvent::getMiniDoubletsBuild<>(bool);

template <typename TSoA, typename TDev>
typename TSoA::ConstView LSTEvent::getSegments(bool sync) {
  if constexpr (std::is_same_v<TDev, DevHost>) {
    return SegmentsViewAccessor<TSoA>::get(segmentsDC_->const_view());
  } else {
    if (!segmentsHC_) {
      segmentsHC_.emplace(cms::alpakatools::CopyToHost<PortableDeviceCollection<TDev, SegmentsSoABlocks>>::copyAsync(
          queue_, *segmentsDC_));
      if (sync)
        alpaka::wait(queue_);  // host consumers expect filled data
    }
    return SegmentsViewAccessor<TSoA>::get(segmentsHC_->const_view());
  }
}
template SegmentsConst LSTEvent::getSegments<SegmentsSoA>(bool);
template SegmentsOccupancyConst LSTEvent::getSegments<SegmentsOccupancySoA>(bool);

template <typename TDev>
PixelSegmentsConst LSTEvent::getPixelSegments(bool sync) {
  if constexpr (std::is_same_v<TDev, DevHost>) {
    return pixelSegmentsDC_->const_view();
  } else {
    if (!pixelSegmentsHC_) {
      pixelSegmentsHC_.emplace(cms::alpakatools::CopyToHost<::PortableCollection<TDev, PixelSegmentsSoA>>::copyAsync(
          queue_, *pixelSegmentsDC_));

      if (sync)
        alpaka::wait(queue_);  // host consumers expect filled data
    }
  }
  return pixelSegmentsHC_->const_view();
}
template PixelSegmentsConst LSTEvent::getPixelSegments<>(bool);

template <typename TSoA, typename TDev>
typename TSoA::ConstView LSTEvent::getTriplets(bool sync) {
  if constexpr (std::is_same_v<TDev, DevHost>) {
    return TripletsViewAccessor<TSoA>::get(tripletsDC_->const_view());
  } else {
    if (!tripletsHC_) {
      tripletsHC_.emplace(cms::alpakatools::CopyToHost<PortableDeviceCollection<TDev, TripletsSoABlocks>>::copyAsync(
          queue_, *tripletsDC_));
      if (sync)
        alpaka::wait(queue_);  // host consumers expect filled data
    }
  }
  return TripletsViewAccessor<TSoA>::get(tripletsHC_->const_view());
}
template TripletsConst LSTEvent::getTriplets<TripletsSoA>(bool);
template TripletsOccupancyConst LSTEvent::getTriplets<TripletsOccupancySoA>(bool);

template <typename TSoA, typename TDev>
typename TSoA::ConstView LSTEvent::getQuadruplets(bool sync) {
  if constexpr (std::is_same_v<TDev, DevHost>) {
    return QuadrupletsViewAccessor<TSoA>::get(quadrupletsDC_->const_view());
  } else {
    if (!quadrupletsHC_) {
      quadrupletsHC_.emplace(
          cms::alpakatools::CopyToHost<PortableDeviceCollection<TDev, QuadrupletsSoABlocks>>::copyAsync(
              queue_, *quadrupletsDC_));
      if (sync)
        alpaka::wait(queue_);  // host consumers expect filled data
    }
  }
  return QuadrupletsViewAccessor<TSoA>::get(quadrupletsHC_->const_view());
}
template QuadrupletsConst LSTEvent::getQuadruplets<QuadrupletsSoA>(bool);
template QuadrupletsOccupancyConst LSTEvent::getQuadruplets<QuadrupletsOccupancySoA>(bool);

template <typename TSoA, typename TDev>
typename TSoA::ConstView LSTEvent::getQuintuplets(bool sync) {
  if constexpr (std::is_same_v<TDev, DevHost>) {
    return QuintupletsViewAccessor<TSoA>::get(quintupletsDC_->const_view());
  } else {
    if (!quintupletsHC_) {
      quintupletsHC_.emplace(
          cms::alpakatools::CopyToHost<PortableDeviceCollection<TDev, QuintupletsSoABlocks>>::copyAsync(
              queue_, *quintupletsDC_));
      if (sync)
        alpaka::wait(queue_);  // host consumers expect filled data
    }
  }
  return QuintupletsViewAccessor<TSoA>::get(quintupletsHC_->const_view());
}
template QuintupletsConst LSTEvent::getQuintuplets<QuintupletsSoA>(bool);
template QuintupletsOccupancyConst LSTEvent::getQuintuplets<QuintupletsOccupancySoA>(bool);

template <typename TDev>
PixelTripletsConst LSTEvent::getPixelTriplets(bool sync) {
  if constexpr (std::is_same_v<TDev, DevHost>) {
    return pixelTripletsDC_->const_view();
  } else {
    if (!pixelTripletsHC_) {
      pixelTripletsHC_.emplace(cms::alpakatools::CopyToHost<::PortableCollection<TDev, PixelTripletsSoA>>::copyAsync(
          queue_, *pixelTripletsDC_));

      if (sync)
        alpaka::wait(queue_);  // host consumers expect filled data
    }
  }
  return pixelTripletsHC_->const_view();
}
template PixelTripletsConst LSTEvent::getPixelTriplets<>(bool);

template <typename TDev>
PixelQuintupletsConst LSTEvent::getPixelQuintuplets(bool sync) {
  if constexpr (std::is_same_v<TDev, DevHost>) {
    return pixelQuintupletsDC_->const_view();
  } else {
    if (!pixelQuintupletsHC_) {
      pixelQuintupletsHC_.emplace(
          cms::alpakatools::CopyToHost<::PortableCollection<TDev, PixelQuintupletsSoA>>::copyAsync(
              queue_, *pixelQuintupletsDC_));

      if (sync)
        alpaka::wait(queue_);  // host consumers expect filled data
    }
  }
  return pixelQuintupletsHC_->const_view();
}
template PixelQuintupletsConst LSTEvent::getPixelQuintuplets<>(bool);

template <typename TDev>
TrackCandidatesBaseConst LSTEvent::getTrackCandidatesBase(bool sync) {
  if constexpr (std::is_same_v<TDev, DevHost>) {
    return trackCandidatesBaseDC_->const_view();
  } else {
    if (!trackCandidatesBaseHC_) {
      trackCandidatesBaseHC_.emplace(
          cms::alpakatools::CopyToHost<::PortableCollection<TDev, TrackCandidatesBaseSoA>>::copyAsync(
              queue_, *trackCandidatesBaseDC_));

      if (sync)
        alpaka::wait(queue_);  // host consumers expect filled data
    }
  }
  return trackCandidatesBaseHC_->const_view();
}
template TrackCandidatesBaseConst LSTEvent::getTrackCandidatesBase<>(bool);

template <typename TDev>
TrackCandidatesExtendedConst LSTEvent::getTrackCandidatesExtended(bool sync) {
  if constexpr (std::is_same_v<TDev, DevHost>) {
    return trackCandidatesExtendedDC_->const_view();
  } else {
    if (!trackCandidatesExtendedHC_) {
      trackCandidatesExtendedHC_.emplace(
          cms::alpakatools::CopyToHost<::PortableCollection<TDev, TrackCandidatesExtendedSoA>>::copyAsync(
              queue_, *trackCandidatesExtendedDC_));

      if (sync)
        alpaka::wait(queue_);  // host consumers expect filled data
    }
  }
  return trackCandidatesExtendedHC_->const_view();
}
template TrackCandidatesExtendedConst LSTEvent::getTrackCandidatesExtended<>(bool);

std::unique_ptr<TrackCandidatesBaseDeviceCollection> LSTEvent::releaseTrackCandidatesBaseDeviceCollection() {
  return std::make_unique<TrackCandidatesBaseDeviceCollection>(std::move(trackCandidatesBaseDC_.value()));
}

template <typename TSoA, typename TDev>
typename TSoA::ConstView LSTEvent::getModules(bool sync) {
  if constexpr (std::is_same_v<TDev, DevHost>) {
    return ModulesViewAccessor<TSoA>::get(modules_.const_view());
  } else {
    if (!modulesHC_) {
      modulesHC_.emplace(
          cms::alpakatools::CopyToHost<PortableDeviceCollection<TDev, ModulesSoABlocks>>::copyAsync(queue_, modules_));
      if (sync)
        alpaka::wait(queue_);  // host consumers expect filled data
    }
    return ModulesViewAccessor<TSoA>::get(modulesHC_->const_view());
  }
}
template ModulesConst LSTEvent::getModules<ModulesSoA>(bool);
template ModulesPixelConst LSTEvent::getModules<ModulesPixelSoA>(bool);
