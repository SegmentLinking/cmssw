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
  ALPAKA_STATIC_ACC_MEM_GLOBAL constexpr float kPi = float(M_PI);
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
    // pt, eta binned
    constexpr unsigned int kPtBins = 2;
    constexpr unsigned int kEtaBins = 10;
    ALPAKA_STATIC_ACC_MEM_GLOBAL constexpr float kWp[kPtBins][kEtaBins] = {
        {0.268, 0.3172, 0.3907, 0.4653, 0.4364, 0.4696, 0.6018, 0.6487, 0.7401, 0.7146},
        {0.1654, 0.2385, 0.2935, 0.3534, 0.2455, 0.1748, 0.1565, 0.1811, 0.3435, 0.1784}};
  }  // namespace t5dnn
  namespace t4dnn {
    ALPAKA_STATIC_ACC_MEM_GLOBAL constexpr float kZ_max = 267.2349854f;
    ALPAKA_STATIC_ACC_MEM_GLOBAL constexpr float kR_max = 110.1099396f;
    ALPAKA_STATIC_ACC_MEM_GLOBAL constexpr float kEta_norm = 2.5f;
    ALPAKA_STATIC_ACC_MEM_GLOBAL constexpr float kPhi_norm = kPi;
    // pt, eta binned
    constexpr unsigned int kPtBins = 2;
    constexpr unsigned int kEtaBins = 10;
    ALPAKA_STATIC_ACC_MEM_GLOBAL constexpr float kWp[kPtBins][kEtaBins] = {
        // {0.1217, 0.1447, 0.2592, 0.4781, 0.5232, 0.6688, 0.8224, 0.8768, 0.9512, 0.9875}, //90% retention, with T3 DNN added
        // {0.0121, 0.0194, 0.0404, 0.1188, 0.0514, 0.3286, 0.5045, 0.597, 0.8681, 0.84}};
        //T4s from all T3s
        // {0.0927, 0.0897, 0.1882, 0.3076, 0.3461, 0.3585, 0.4689, 0.5765, 0.7808, 0.834}, //98% retention for all pT<5
        // {0.0702, 0.0738, 0.1644, 0.2533, 0.2952, 0.3181, 0.5181, 0.6596, 0.8174, 0.8082}, //98% retention for all pT<5, weighted high pt
        // {0.1049, 0.1091, 0.2611, 0.4626, 0.4515, 0.4862, 0.6446, 0.7168, 0.8796, 0.8844}, //98% for all pT<5, upweight high pT and displaced
        // {0.4522, 0.5232, 0.6688, 0.7258, 0.6113, 0.6554, 0.7637, 0.821, 0.9332, 0.945}, //95% for all pT<5, upweight high pT and displaced
        // {0.0979, 0.1106, 0.2722, 0.4723, 0.4681, 0.5053, 0.6933, 0.7626, 0.8639, 0.89}, //98% pt<5, 2 weighted, add chi2 to dnn
        // {0.6069, 0.6766, 0.768, 0.8011, 0.4515, 0.4862, 0.6446, 0.7168, 0.8796, 0.8844}, //93% for eta<1 pT<5, 98 rest upweight high pT and displaced  
        // {0.6196, 0.6521, 0.7417, 0.7771,0.3461, 0.3585, 0.4689, 0.5765, 0.7808, 0.834}, //93% for eta<1, 98% eta>1 pT<5
        // {0.8918, 0.9151, 0.9362, 0.9475, 0.8448, 0.8828, 0.9238, 0.9529, 0.9823, 0.9878}, //75% pT<5, upweight pt, displaced
        {0.91, 0.9293, 0.9472, 0.9565, 0.8684, 0.9, 0.9372, 0.962, 0.9851, 0.9893}, //70% pt<5, upweighted pt, displaced
        // {0.0021, 0.0102, 0.0768, 0.2252, 0.0803, 0.1378, 0.1524, 0.0755, 0.1271, 0.204}}; //98% retention for all pT>5
        // {0.0624, 0.1378, 0.6163, 0.734,  0.0803, 0.1378, 0.1524, 0.0755, 0.1271, 0.204}}; //93% for eta<1, 98% eta>1 pT>5
        // {0.1331, 0.2744, 0.7355, 0.8241, 0.5422, 0.5823, 0.6942, 0.6528, 0.6915, 0.8488}}; //93% for all pT>5, weighted
        // {0.0624, 0.1378, 0.6163, 0.734, 0.3895, 0.3878, 0.3819, 0.2771, 0.3255, 0.5547}}; //93% for all pT>5, unweighted
        // {0.0731, 0.2287, 0.6304, 0.7117, 0.3657, 0.408, 0.5527, 0.5865, 0.7565, 0.769}}; //93% for all pt>5, upweight high pT and displaced
        // {0.3163, 0.4803, 0.7272, 0.7973, 0.4898, 0.5055, 0.6198, 0.6644, 0.8367, 0.8263}}; //90% for all pt>5, upweight high pT and displaced
        // {0.0502, 0.1149, 0.5333, 0.6745, 0.3464, 0.3777, 0.4857, 0.5192, 0.6673, 0.6757}}; //93% all pt>5, 2 weighted, add chi2 to dnn
        // {0.0731, 0.2287, 0.6304, 0.7117, 0.089, 0.1468, 0.2836, 0.297, 0.3757, 0.2174}}; //93% for eta<1, pt>5, 98 rest upweight high pT and displaced
        {0.7644, 0.837, 0.8899, 0.9275, 0.7406, 0.718, 0.7857, 0.8366, 0.9292, 0.9365}}; //75% pT>5, upweight pt, displaced
  }  // namespace t4dnn
  namespace t3dnn {
    ALPAKA_STATIC_ACC_MEM_GLOBAL constexpr float kZ_max = 224.149505f;
    ALPAKA_STATIC_ACC_MEM_GLOBAL constexpr float kR_max = 98.932365f;
    ALPAKA_STATIC_ACC_MEM_GLOBAL constexpr float kEta_norm = 2.5f;
    ALPAKA_STATIC_ACC_MEM_GLOBAL constexpr float kPhi_norm = kPi;
    constexpr unsigned int kPtBins = 2;
    constexpr unsigned int kEtaBins = 10;
    ALPAKA_STATIC_ACC_MEM_GLOBAL constexpr float kWp_prompt[kPtBins][kEtaBins] = {
        {0.4805, 0.4796, 0.4868, 0.4948, 0.4148, 0.4374, 0.4664, 0.4813, 0.5375, 0.5437},
        {0.0087, 0.0158, 0.0456, 0.0795, 0.1072, 0.1662, 0.2793, 0.2937, 0.2526, 0.2738}};
    ALPAKA_STATIC_ACC_MEM_GLOBAL constexpr float kWp_displaced[kPtBins][kEtaBins] = {
        {0.0411, 0.0487, 0.0650, 0.1041, 0.1146, 0.1124, 0.1339, 0.1961, 0.1982, 0.2045},
        {0.0066, 0.0062, 0.0297, 0.0295, 0.0669, 0.0376, 0.2249, 0.2131, 0.1783, 0.0528}};
  }  // namespace t3dnn

}  //namespace lst
#endif
