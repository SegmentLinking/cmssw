#ifndef RecoTracker_MkSeederAlpaka_interface_MkSeederCandidatesSoA_h
#define RecoTracker_MkSeederAlpaka_interface_MkSeederCandidatesSoA_h

// The chain finder's candidates (the doublets and triplets on its queues) and the queues' bookkeeping.
// The queues of one z side: a doublet queue and a triplet queue per chain position, each a list of fixed-size chunks
// of the candidate pool (kCandidateChunkSize rows); a queue is consumed once, at its position, after every producer for
// it has run, and its chunks are then free for the later pushes.

#include <cstdint>

#include "DataFormats/SoATemplate/interface/SoABlocks.h"
#include "DataFormats/SoATemplate/interface/SoALayout.h"
#include "RecoTracker/MkSeederAlpaka/interface/MkSeederCounters.h"

namespace mkseeder {

  // A candidate: its hits are the rows of its chain positions' layers (from the layer's firstHit, in bin order). Its
  // r-z line (a-b for a doublet, a-c for a triplet) is recomputed from the hits where needed. Packed in one word
  // (with the score 20 bytes per candidate: the pool is the seeder's largest buffer) are its chain positions (C = 0 for a
  // doublet), the positions it crossed without a hit (holes), the definite misses inside its start pair (a gap-map
  // start) and after it (inner holes: with MkSeederChainParameters::innerOuterTrackerOnly a candidate with one
  // continues on the outer tracker only), its state at the position of the queue it is on (stage, 2: a definite
  // crossing); 6 bits each (exact for chains of at most kMaxCandidateChainPositions positions), 2 for the stage
  // (one 16-byte load on a GPU; the score is a column of its own)
  struct alignas(16) MkSeederCandidate {
    uint32_t hits[3];
    uint32_t positionsAndState;
  };
  constexpr int kMaxCandidateChainPositions = 62;
  constexpr int kCandidateFieldBits = 6;
  constexpr uint32_t kCandidateFieldMask = (1u << kCandidateFieldBits) - 1;
  // the fields of MkSeederCandidate::positionsAndState in order
  enum MkSeederCandidateField { kPositionA = 0, kPositionB, kPositionC, kHoles, kInnerHoles, kStage };
  constexpr uint32_t packCandidateFields(
      uint32_t positionA, uint32_t positionB, uint32_t positionC, uint32_t holes, uint32_t innerHoles, uint32_t stage) {
    return positionA | positionB << kCandidateFieldBits | positionC << 2 * kCandidateFieldBits |
           holes << 3 * kCandidateFieldBits | innerHoles << 4 * kCandidateFieldBits | stage << 5 * kCandidateFieldBits;
  }
  constexpr uint32_t candidateField(uint32_t positionsAndState, MkSeederCandidateField field) {
    return field == kStage ? positionsAndState >> 5 * kCandidateFieldBits
                           : positionsAndState >> field * kCandidateFieldBits & kCandidateFieldMask;
  }

  // The serial backend: a candidate's r-z line and the chain positions it crosses (bit p of crossedPositions: state
  // > 0 at position p; of definitePositions: state 2), stored by the push so that the later stages neither recompute
  // the line nor evaluate the states again (positions < 32 only)
  struct MkSeederCandidateLine {
    float z0, cot;
    uint32_t crossedPositions, definitePositions;
  };

  // One row per pool slot (chunk id * kCandidateChunkSize + slot): the candidate, read and written whole by every
  // stage (one record: the serial backend's pushes go to many queues at once, and a column per field multiplies
  // their write streams, 10 % slower there), stage c's score of a triplet
  GENERATE_SOA_LAYOUT(MkSeederCandidatesLayout, SOA_COLUMN(MkSeederCandidate, candidate), SOA_COLUMN(float, score))

  // The serial backend, one row per pool slot: the candidate's line
  GENERATE_SOA_LAYOUT(MkSeederCandidateLinesLayout, SOA_COLUMN(MkSeederCandidateLine, line))

  // A GPU, one row per candidate of the doublet queue stage c is consuming (by its position in the queue): a triplet
  // was found from it (cleared once read by the kernel that forwards the misses)
  GENERATE_SOA_LAYOUT(MkSeederTripletStageLayout, SOA_COLUMN(uint8_t, foundTriplet))

  // the candidate pool: the candidates; their lines (the serial backend, else one row)
  GENERATE_SOA_BLOCKS(MkSeederCandidatePoolLayout,
                      SOA_BLOCK(candidates, MkSeederCandidatesLayout),
                      SOA_BLOCK(lines, MkSeederCandidateLinesLayout))

  // One row per queue (the doublet queue of chain position p is queue p, its triplet queue nPositions + p): the
  // candidates pushed
  GENERATE_SOA_LAYOUT(MkSeederQueuesLayout, SOA_COLUMN(uint32_t, count))
  // a list of chunk ids: per queue its maxChunksPerQueue chunks (-1: not yet), or the free chunks
  GENERATE_SOA_LAYOUT(MkSeederChunksLayout, SOA_COLUMN(int32_t, chunk))

  // the queues' bookkeeping: the queues, their chunk tables (queue * maxChunksPerQueue + i), the free chunks (a
  // stack of the chunks of consumed queues, taken by the pushes before fresh ones), stage c's flags (a GPU, else one
  // row; not in the pool's allocation: that stays in a smaller bin of the caching allocator)
  GENERATE_SOA_BLOCKS(MkSeederQueueTablesLayout,
                      SOA_BLOCK(queues, MkSeederQueuesLayout),
                      SOA_BLOCK(chunkTable, MkSeederChunksLayout),
                      SOA_BLOCK(freeChunks, MkSeederChunksLayout),
                      SOA_BLOCK(tripletStage, MkSeederTripletStageLayout))

  using MkSeederCandidatesSoA = MkSeederCandidatesLayout<>;
  using MkSeederCandidateLinesSoA = MkSeederCandidateLinesLayout<>;
  using MkSeederTripletStageSoA = MkSeederTripletStageLayout<>;
  using MkSeederCandidatePoolSoA = MkSeederCandidatePoolLayout<>;
  using MkSeederQueuesSoA = MkSeederQueuesLayout<>;
  using MkSeederChunksSoA = MkSeederChunksLayout<>;
  using MkSeederQueueTablesSoA = MkSeederQueueTablesLayout<>;

}  // namespace mkseeder

#endif
