#ifndef RecoTracker_MkSeederAlpaka_plugins_alpaka_MkSeederQueues_h
#define RecoTracker_MkSeederAlpaka_plugins_alpaka_MkSeederQueues_h

// The chain finder's candidate pool and queues as the kernels use them (MkSeederCandidatesSoA.h): the device-side
// part of findQuads() (MkSeederDevice.dev.cc), not seen by the module.

#include <cstdint>

#include "DataFormats/Portable/interface/alpaka/PortableCollection.h"
#include "RecoTracker/MkSeederAlpaka/interface/MkSeederCandidatesSoA.h"

#include "MkSeederDeviceTypes.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::mkseeder {

  using CandidatesView = ScratchView<::mkseeder::MkSeederCandidatesSoA>;
  using CandidateLinesView = ScratchView<::mkseeder::MkSeederCandidateLinesSoA>;
  using TripletStageView = ScratchView<::mkseeder::MkSeederTripletStageSoA>;
  using QueuesView = ScratchView<::mkseeder::MkSeederQueuesSoA>;
  using ChunksView = ScratchView<::mkseeder::MkSeederChunksSoA>;

  // a candidate in the kernels (a row of MkSeederCandidatesSoA): hits in the layers' bin order, chain positions
  // (pos[2] = 0: a doublet); its r-z line (a-b for a doublet, a-c for a triplet) is recomputed from the hits
  struct Candidate {
    uint32_t hits[3];
    uint8_t positions[3], holes, stage, innerHoles;
    float score;
  };
  ALPAKA_FN_HOST_ACC inline void loadCandidate(CandidatesView pool, uint32_t slot, Candidate &c) {
    using namespace ::mkseeder;
    const MkSeederCandidate r = pool.candidate(slot);
    c.hits[0] = r.hits[0], c.hits[1] = r.hits[1], c.hits[2] = r.hits[2];
    const uint32_t f = r.positionsAndState;
    c.positions[0] = candidateField(f, kPositionA), c.positions[1] = candidateField(f, kPositionB);
    c.positions[2] = candidateField(f, kPositionC), c.holes = candidateField(f, kHoles);
    c.innerHoles = candidateField(f, kInnerHoles), c.stage = candidateField(f, kStage), c.score = pool.score(slot);
  }
  ALPAKA_FN_HOST_ACC inline void storeCandidate(CandidatesView pool, uint32_t slot, const Candidate &c) {
    pool.candidate(slot) = {{c.hits[0], c.hits[1], c.hits[2]},
                            ::mkseeder::packCandidateFields(
                                c.positions[0], c.positions[1], c.positions[2], c.holes, c.innerHoles, c.stage)};
    pool.score(slot) = c.score;
  }
  // the serial backend: a candidate's r-z line and the chain positions it crosses (a row of MkSeederCandidateLinesSoA)
  struct CandidateLine {
    float z0, cot;
    uint32_t crossedPositions, definitePositions;
  };
  ALPAKA_FN_HOST_ACC inline void loadCandidateLine(CandidateLinesView lines, uint32_t slot, CandidateLine &l) {
    const ::mkseeder::MkSeederCandidateLine r = lines.line(slot);
    l.z0 = r.z0, l.cot = r.cot, l.crossedPositions = r.crossedPositions, l.definitePositions = r.definitePositions;
  }
  ALPAKA_FN_HOST_ACC inline void storeCandidateLine(CandidateLinesView lines, uint32_t slot, const CandidateLine &l) {
    lines.line(slot) = {l.z0, l.cot, l.crossedPositions, l.definitePositions};
  }

  // The queues of one side: Q2 (doublets) and Q3 (triplets) per chain position, each a list of fixed-size chunks
  // from one pool (MkSeederCandidatesSoA.h). A queue is consumed once, at its position, after every producer for it
  // has run.
  struct Queues {
    static constexpr uint32_t kCandidateChunkSize = ::mkseeder::kCandidateChunkSize;
    // the views are mutable: a const Queues still writes the queues (as the kernels' const references do)
    mutable CandidatesView pool;
    mutable CandidateLinesView lines;       // the serial backend: per pool slot (hasLines)
    mutable TripletStageView tripletStage;  // a GPU: per candidate of the doublet queue stage c consumes
    bool hasLines;
    uint32_t poolChunks, maxChunksPerQueue;
    mutable QueuesView queues;      // per queue: count, reserve
    mutable ChunksView chunkTable;  // per queue: maxChunksPerQueue chunk ids, -1: not yet
    mutable ChunksView freeChunks;  // the chunks of consumed queues (KernelReleaseConsumedChunks), taken by push() first
    MkSeederFinderCounters *counters;
    mutable QuadsView quads;  // the found quads (counters->nQuads, at most quadCapacity)
    uint32_t quadCapacity;
  };

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::mkseeder

namespace ALPAKA_ACCELERATOR_NAMESPACE::mkseeder {
  using MkSeederCandidatePoolDeviceCollection = PortableCollection<::mkseeder::MkSeederCandidatePoolSoA>;
  using MkSeederQueueTablesDeviceCollection = PortableCollection<::mkseeder::MkSeederQueueTablesSoA>;
}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::mkseeder

#endif
