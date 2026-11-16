#ifndef RecoTracker_MkSeederAlpaka_interface_MkSeederCounters_h
#define RecoTracker_MkSeederAlpaka_interface_MkSeederCounters_h

// The counters of the seeder's device steps (one PortableObject each); the host reads the chain finder's (a rerun
// with larger buffers), the cleaning's (the sizes of the next steps) and the final seeds' number

#include <cstdint>

namespace mkseeder {

  // the rows of a chain finder's queue chunk (MkSeederCandidatesSoA.h)
  constexpr uint32_t kCandidateChunkSize = 256;

  // the chain finder
  struct MkSeederFinderCounters {
    uint32_t nextFreshChunk;   // the chunks handed out fresh (= the peak of live chunks: a freed chunk is taken first)
    uint32_t nFreeChunks;      // the free chunks on the stack
    uint32_t nLost;            // the candidates and quads lost to a full pool, chunk table or quad buffer
    uint32_t nQuads;           // the quads and triplet seeds found (more than the buffer holds: lost)
    uint32_t chunksUsed[2];    // per z side its nextFreshChunk at the end
    uint32_t stageCItems;      // a GPU: stage c's items handed out to the blocks of KernelExtendToTriplets (0 between)
    uint32_t stageCNetBlocks;  // a GPU with trip_net_c: the blocks done with the network's pass (0 between)
  };

  // the quad / triplet cleaning: the items are decided in rounds; a round after one that left nothing
  // undecided does nothing (its count stays 0); what the rounds leave undecided the completion kernel decides
  struct MkSeederCleanCounters {
    static constexpr int kMaxRounds = 64;
    uint32_t nQuads, nTriplets;
    uint32_t nKept;           // the kept seeds
    uint32_t nProbes;         // the kept triplets that failed the triplet MLP (the extension's probe items)
    uint32_t nProbesWritten;  // the probe items made (KernelBuildProbeItems)
    uint32_t undecidedAfterQuadRound[kMaxRounds];
    uint32_t undecidedAfterTripletRound[kMaxRounds];
  };

  // a duplicate cleaning; the same rounds and completion
  struct MkSeederDupCounters {
    static constexpr int kMaxRounds = 96;
    uint32_t nSeeds;   // its input seeds
    uint32_t nOutput;  // the seeds kept
    uint32_t undecidedAfterRound[kMaxRounds];
  };

}  // namespace mkseeder

#endif
