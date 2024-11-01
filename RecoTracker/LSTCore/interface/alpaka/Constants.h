#ifndef RecoTracker_LSTCore_interface_alpaka_Constants_h
#define RecoTracker_LSTCore_interface_alpaka_Constants_h

#include "RecoTracker/LSTCore/interface/Constants.h"

#ifdef ALPAKA_ACC_GPU_CUDA_ENABLED
#include <cuda_fp16.h>
#endif

namespace lst {

  using namespace ALPAKA_ACCELERATOR_NAMESPACE;

// Half precision wrapper functions.
#if defined(FP16_Base)
#define __F2H __float2half
#define __H2F __half2float
  typedef __half float FPX;
#else
#define __F2H
#define __H2F
  typedef float FPX;
#endif

  Vec3D constexpr elementsPerThread(Vec3D::all(static_cast<Idx>(1)));

// Needed for files that are compiled by g++ to not throw an error.
// uint4 is defined only for CUDA, so we will have to revisit this soon when running on other backends.
#if !defined(ALPAKA_ACC_GPU_CUDA_ENABLED) && !defined(ALPAKA_ACC_GPU_HIP_ENABLED)
  struct uint4 {
    unsigned int x;
    unsigned int y;
    unsigned int z;
    unsigned int w;
  };
#endif

  // Adjust grid and block sizes based on backend configuration
  template <typename Vec>
  ALPAKA_FN_HOST ALPAKA_FN_INLINE WorkDiv3D createWorkDiv(const Vec& blocksPerGrid,
                                                          const Vec& threadsPerBlock,
                                                          const Vec& elementsPerThreadArg) {
    Vec adjustedBlocks = blocksPerGrid;
    Vec adjustedThreads = threadsPerBlock;

    // Serial execution, so all launch parameters set to 1.
#if defined(ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED)
    adjustedBlocks = Vec::all(static_cast<Idx>(1));
    adjustedThreads = Vec::all(static_cast<Idx>(1));
#endif

    // Threads enabled, set number of blocks to 1.
#if defined(ALPAKA_ACC_CPU_B_SEQ_T_THREADS_ENABLED)
    adjustedBlocks = Vec::all(static_cast<Idx>(1));
#endif

    return WorkDiv3D(adjustedBlocks, adjustedThreads, elementsPerThreadArg);
  }

  // The constants below are usually used in functions like alpaka::math::min(),
  // expecting a reference (T const&) in the arguments. Hence,
  // ALPAKA_STATIC_ACC_MEM_GLOBAL needs to be used in addition to constexpr.

  // 15 MeV constant from the approximate Bethe-Bloch formula
  ALPAKA_STATIC_ACC_MEM_GLOBAL constexpr float kMulsInGeV = 0.015;
  ALPAKA_STATIC_ACC_MEM_GLOBAL constexpr float kMiniMulsPtScaleBarrel[6] = {
      0.0052, 0.0038, 0.0034, 0.0034, 0.0032, 0.0034};
  ALPAKA_STATIC_ACC_MEM_GLOBAL constexpr float kMiniMulsPtScaleEndcap[5] = {0.006, 0.006, 0.006, 0.006, 0.006};
  ALPAKA_STATIC_ACC_MEM_GLOBAL constexpr float kMiniRminMeanBarrel[6] = {
      25.007152356, 37.2186993757, 52.3104270826, 68.6658656666, 85.9770373007, 108.301772384};
  ALPAKA_STATIC_ACC_MEM_GLOBAL constexpr float kMiniRminMeanEndcap[5] = {
      130.992832231, 154.813883559, 185.352604327, 221.635123002, 265.022076742};
  ALPAKA_STATIC_ACC_MEM_GLOBAL constexpr float k2Rinv1GeVf = (2.99792458e-3 * 3.8) / 2;
  ALPAKA_STATIC_ACC_MEM_GLOBAL constexpr float kR1GeVf = 1. / (2.99792458e-3 * 3.8);
  ALPAKA_STATIC_ACC_MEM_GLOBAL constexpr float kSinAlphaMax = 0.95;
  ALPAKA_STATIC_ACC_MEM_GLOBAL constexpr float kDeltaZLum = 15.0;
  ALPAKA_STATIC_ACC_MEM_GLOBAL constexpr float kPixelPSZpitch = 0.15;
  ALPAKA_STATIC_ACC_MEM_GLOBAL constexpr float kStripPSZpitch = 2.4;
  ALPAKA_STATIC_ACC_MEM_GLOBAL constexpr float kStrip2SZpitch = 5.0;
  ALPAKA_STATIC_ACC_MEM_GLOBAL constexpr float kWidth2S = 0.009;
  ALPAKA_STATIC_ACC_MEM_GLOBAL constexpr float kWidthPS = 0.01;
  ALPAKA_STATIC_ACC_MEM_GLOBAL constexpr float kPt_betaMax = 7.0;
  // Since C++ can't represent infinity, lst_INF = 123456789 was used to represent infinity in the data table
  ALPAKA_STATIC_ACC_MEM_GLOBAL constexpr float lst_INF = 123456789.0;

  ALPAKA_STATIC_ACC_MEM_GLOBAL constexpr float kMiniDeltaTilted[3] = {0.26f, 0.26f, 0.26f};
  ALPAKA_STATIC_ACC_MEM_GLOBAL constexpr float kMiniDeltaFlat[6] = {0.26f, 0.16f, 0.16f, 0.18f, 0.18f, 0.18f};
  ALPAKA_STATIC_ACC_MEM_GLOBAL constexpr float kMiniDeltaLooseTilted[3] = {0.4f, 0.4f, 0.4f};
  ALPAKA_STATIC_ACC_MEM_GLOBAL constexpr float kMiniDeltaEndcap[5][15] = {
      {0.4f, 0.4f, 0.4f, 0.4f, 0.4f, 0.4f, 0.4f, 0.4f, 0.4f, 0.4f, /*10*/ 0.18f, 0.18f, 0.18f, 0.18f, 0.18f},
      {0.4f, 0.4f, 0.4f, 0.4f, 0.4f, 0.4f, 0.4f, 0.4f, 0.4f, 0.4f, /*10*/ 0.18f, 0.18f, 0.18f, 0.18f, 0.18f},
      {0.4f, 0.4f, 0.4f, 0.4f, 0.4f, 0.4f, 0.4f, 0.4f, 0.18f, 0.18f, /*10*/ 0.18f, 0.18f, 0.18f, 0.18f, 0.18f},
      {0.4f, 0.4f, 0.4f, 0.4f, 0.4f, 0.4f, 0.4f, 0.4f, 0.18f, 0.18f, /*10*/ 0.18f, 0.18f, 0.18f, 0.18f, 0.18f},
      {0.4f, 0.4f, 0.4f, 0.4f, 0.4f, 0.4f, 0.4f, 0.4f, 0.4f, 0.18f, /*10*/ 0.18f, 0.18f, 0.18f, 0.18f, 0.18f}};

  namespace t5dnn {
    ALPAKA_STATIC_ACC_MEM_GLOBAL constexpr float kZ_max = 267.2349854f;
    ALPAKA_STATIC_ACC_MEM_GLOBAL constexpr float kR_max = 110.1099396f;
    ALPAKA_STATIC_ACC_MEM_GLOBAL constexpr float kEta_norm = 2.5f;
    ALPAKA_STATIC_ACC_MEM_GLOBAL constexpr float kPhi_norm = M_PI;
    // pt, eta binned
    constexpr unsigned int kPtBins = 2;
    constexpr unsigned int kEtaBins = 10;
    ALPAKA_STATIC_ACC_MEM_GLOBAL constexpr float kWp[kPtBins][kEtaBins] = {
        {0.4586, 0.519, 0.5776, 0.6284, 0.5546, 0.5703, 0.7089, 0.7521, 0.8421, 0.8439},
        {0.3678, 0.4509, 0.5192, 0.581, 0.4228, 0.3227, 0.3465, 0.2802, 0.4634, 0.3555}};
  }  // namespace t5dnn

}  //namespace lst
#endif
