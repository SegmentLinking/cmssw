#ifndef RecoTracker_LSTCore_src_alpaka_HitBuckets_h
#define RecoTracker_LSTCore_src_alpaka_HitBuckets_h

#include "RecoTracker/LSTCore/interface/alpaka/Common.h"
#include "RecoTracker/LSTCore/interface/PixelTripletsSoA.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::lst {

  // Objects listed by hit index in buckets given by the low kBucketBits bits of the hit index: two objects that share a
  // hit are in the same bucket.
  namespace hitBuckets {
    // Of the order of the hits listed in a dense event, so that a bucket holds few entries; few enough that the
    // one-block ScanCellCounts over the bucket counts stays short.
    constexpr unsigned int kBucketBits = 13;
    constexpr unsigned int kNBuckets = 1u << kBucketBits;
    ALPAKA_FN_ACC ALPAKA_FN_INLINE unsigned int bucket(unsigned int hit) { return hit & (kNBuckets - 1); }
  }  // namespace hitBuckets

  // The Params_T3::kHits hits of the T3 of a pT3, which follow its pLS hits.
  ALPAKA_FN_ACC ALPAKA_FN_INLINE unsigned int const* pixelTripletT3Hits(PixelTripletsConst pixelTriplets,
                                                                        unsigned int pixelTripletIndex) {
    return pixelTriplets.hitIndices()[pixelTripletIndex].data() + Params_pLS::kHits;
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::lst

#endif
