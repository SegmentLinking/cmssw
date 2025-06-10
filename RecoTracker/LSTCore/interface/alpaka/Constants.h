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
  namespace t3dnn {
    ALPAKA_STATIC_ACC_MEM_GLOBAL constexpr float kZ_max = 224.149505f;
    ALPAKA_STATIC_ACC_MEM_GLOBAL constexpr float kR_max = 98.932365f;
    ALPAKA_STATIC_ACC_MEM_GLOBAL constexpr float kEta_norm = 2.5f;
    ALPAKA_STATIC_ACC_MEM_GLOBAL constexpr float kPhi_norm = kPi;
    constexpr unsigned int kPtBins = 2;
    constexpr unsigned int kEtaBins = 10;
    ALPAKA_STATIC_ACC_MEM_GLOBAL constexpr float kWp_prompt[kPtBins][kEtaBins] = {
        {0.4805, 0.4796, 0.4868, 0.4948, 0.4148, 0.4374, 0.4664, 0.4813, 0.5375, 0.5437},
        {0.0087, 0.0158, 0.0456, 0.0795, 0.1072, 0.1662, 0.2793, 0.2937, 0.2526, 0.2738}}; //90% retention
    ALPAKA_STATIC_ACC_MEM_GLOBAL constexpr float kWp_displaced[kPtBins][kEtaBins] = {
        {0.0411, 0.0487, 0.0650, 0.1041, 0.1146, 0.1124, 0.1339, 0.1961, 0.1982, 0.2045},
        {0.0066, 0.0062, 0.0297, 0.0295, 0.0669, 0.0376, 0.2249, 0.2131, 0.1783, 0.0528}}; //99% retention
  }  // namespace t3dnn

  //#ifdef USE_T4_PT4
  namespace t4dnn {
    ALPAKA_STATIC_ACC_MEM_GLOBAL constexpr float kZ_max = 267.2349854f;
    ALPAKA_STATIC_ACC_MEM_GLOBAL constexpr float kR_max = 110.1099396f;
    ALPAKA_STATIC_ACC_MEM_GLOBAL constexpr float kEta_norm = 2.5f;
    ALPAKA_STATIC_ACC_MEM_GLOBAL constexpr float kPhi_norm = kPi;
    // pt, eta binned
    constexpr unsigned int kPtBins = 2;
    constexpr unsigned int kEtaBins = 10;
    ALPAKA_STATIC_ACC_MEM_GLOBAL constexpr float kWp[kPtBins][kEtaBins] = {
    //     // {0.1217, 0.1447, 0.2592, 0.4781, 0.5232, 0.6688, 0.8224, 0.8768, 0.9512, 0.9875}, //90% retention, with T3 DNN added
    //     // {0.0121, 0.0194, 0.0404, 0.1188, 0.0514, 0.3286, 0.5045, 0.597, 0.8681, 0.84}};
    //     //T4s from all T3s
    //     // {0.0927, 0.0897, 0.1882, 0.3076, 0.3461, 0.3585, 0.4689, 0.5765, 0.7808, 0.834}, //98% retention for all pT<5
    //     // {0.0702, 0.0738, 0.1644, 0.2533, 0.2952, 0.3181, 0.5181, 0.6596, 0.8174, 0.8082}, //98% retention for all pT<5, weighted high pt
    //     // {0.1049, 0.1091, 0.2611, 0.4626, 0.4515, 0.4862, 0.6446, 0.7168, 0.8796, 0.8844}, //98% for all pT<5, upweight high pT and displaced
    //     // {0.4522, 0.5232, 0.6688, 0.7258, 0.6113, 0.6554, 0.7637, 0.821, 0.9332, 0.945}, //95% for all pT<5, upweight high pT and displaced
    //     // {0.6069, 0.6766, 0.768, 0.8011, 0.4515, 0.4862, 0.6446, 0.7168, 0.8796, 0.8844}, //93% for eta<1 pT<5, 98 rest upweight high pT and displaced  
    //     // {0.6196, 0.6521, 0.7417, 0.7771,0.3461, 0.3585, 0.4689, 0.5765, 0.7808, 0.834}, //93% for eta<1, 98% eta>1 pT<5
        {0.8918, 0.9151, 0.9362, 0.9475, 0.8448, 0.8828, 0.9238, 0.9529, 0.9823, 0.9878}, //75% pT<5, upweight pt, displaced
        // {0.91, 0.9293, 0.9472, 0.9565, 0.8684, 0.9, 0.9372, 0.962, 0.9851, 0.9893}, //70% pt<5, upweighted pt, displaced
    //     // {0.0021, 0.0102, 0.0768, 0.2252, 0.0803, 0.1378, 0.1524, 0.0755, 0.1271, 0.204}}; //98% retention for all pT>5
    //     // {0.0624, 0.1378, 0.6163, 0.734,  0.0803, 0.1378, 0.1524, 0.0755, 0.1271, 0.204}}; //93% for eta<1, 98% eta>1 pT>5
    //     // {0.1331, 0.2744, 0.7355, 0.8241, 0.5422, 0.5823, 0.6942, 0.6528, 0.6915, 0.8488}}; //93% for all pT>5, weighted
    //     // {0.0624, 0.1378, 0.6163, 0.734, 0.3895, 0.3878, 0.3819, 0.2771, 0.3255, 0.5547}}; //93% for all pT>5, unweighted
    //     // {0.0731, 0.2287, 0.6304, 0.7117, 0.3657, 0.408, 0.5527, 0.5865, 0.7565, 0.769}}; //93% for all pt>5, upweight high pT and displaced
        {0.3163, 0.4803, 0.7272, 0.7973, 0.4898, 0.5055, 0.6198, 0.6644, 0.8367, 0.8263}}; //90% for all pt>5, upweight high pT and displaced
    //     // {0.0731, 0.2287, 0.6304, 0.7117, 0.089, 0.1468, 0.2836, 0.297, 0.3757, 0.2174}}; //93% for eta<1, pt>5, 98 rest upweight high pT and displaced
        // {0.7644, 0.837, 0.8899, 0.9275, 0.7406, 0.718, 0.7857, 0.8366, 0.9292, 0.9365}}; //75% pT>5, upweight pt, displaced
        // {0.6948, 0.7923, 0.8618, 0.9099, 0.6816, 0.6759, 0.748, 0.7932, 0.9101, 0.9204}}; //80% pt>5 upweighted pt, displaced
    ALPAKA_STATIC_ACC_MEM_GLOBAL constexpr float kWp_prompt[kPtBins][kEtaBins] = {
      // {0.7332, 0.7309, 0.7231, 0.7173, 0.6362, 0.6404, 0.6453, 0.6572, 0.7257, 0.7374}, //70% low pt, with DS
      // {0.7117, 0.7102, 0.7025, 0.6972, 0.6083, 0.6180, 0.6192, 0.6338, 0.7017, 0.7123}, //75% low pt, with DS
      // {0.5218, 0.5484, 0.5622, 0.5593, 0.5475, 0.6060, 0.6310, 0.6591, 0.7207, 0.7191}, //75% low pt, with DS, uncert
      // {0.5269, 0.5406, 0.5379, 0.5432, 0.5096, 0.5429, 0.5172, 0.4907, 0.5139, 0.4761}, //75% low pt, with DS, updated for lay adj
      // {0.6517, 0.6356, 0.6338, 0.6369, 0.5520, 0.5746, 0.5863, 0.5790, 0.6127, 0.6303}, //add radii
      // {0.7045, 0.6896, 0.6754, 0.6678, 0.5622, 0.5894, 0.6036, 0.6200, 0.6775, 0.7046}, //radii, weight pmatched and class
      // {0.7355, 0.7389, 0.7479, 0.7567, 0.6939, 0.6878, 0.7075, 0.7093, 0.7428, 0.7473}, // 75% low pt radii, t3 scores
      {0.6398, 0.6448, 0.6586, 0.6875, 0.6146, 0.6181, 0.6515, 0.6528, 0.6872, 0.6870}, //85% low pt radii, t3 scores
      // {0.6375, 0.6406, 0.6441, 0.6674, 0.6248, 0.6175, 0.6351, 0.6427, 0.6924, 0.7012}, //85% low pt, radii, t3, uncert, larger
      // {0.0693, 0.0831, 0.1002, 0.1325, 0.1656, 0.1850, 0.2386, 0.2103, 0.2142, 0.1758}, //98%
      // {0.2252, 0.2498, 0.2765, 0.2948, 0.2971, 0.2957, 0.3775, 0.4306, 0.4598, 0.3966}}; //75% hight pt, with DS
      // {0.2001, 0.2230, 0.2600, 0.2829, 0.2794, 0.2715, 0.3439, 0.4018, 0.4332, 0.3548}}; //80% high pt, with DS
      // {0.2112, 0.2199, 0.2350, 0.2569, 0.2870, 0.2793, 0.3994, 0.4245, 0.3539, 0.4033}}; //80% high pt, with DS, uncert
      // {0.2277, 0.2458, 0.2715, 0.2727, 0.2802, 0.2795, 0.3441, 0.3631, 0.3218, 0.2860}}; //80% high pt, with DS, updated for lay adj
      // {0.1863, 0.2165, 0.2712, 0.3089, 0.3614, 0.3507, 0.3695, 0.3950, 0.3467, 0.3109}}; //add radii
      // {0.1958, 0.2084, 0.2605, 0.2894, 0.3142, 0.3299, 0.3303, 0.4260, 0.4377, 0.4076}}; //radii, weight pmatched and class
      // {0.2810, 0.3003, 0.3260, 0.3410, 0.3141, 0.3069, 0.4320, 0.4565, 0.3892, 0.3778}}; //80% radii, t3 scores
      // {0.2610, 0.2831, 0.3053, 0.3166, 0.2778, 0.2703, 0.3849, 0.4145, 0.3539, 0.3315}}; //85% high pt radii, t3 scores
      {0.2145, 0.2547, 0.2754, 0.2830, 0.2413, 0.2249, 0.3232, 0.3532, 0.3180, 0.2721}}; //90%
      // {0.1236, 0.1581, 0.1762, 0.1975, 0.2619, 0.2307, 0.2914, 0.2935, 0.2785, 0.3350}}; //85% high pt, radii, t3, uncert, larger
      // {0.0325, 0.0278, 0.0946, 0.1384, 0.1447, 0.1438, 0.2064, 0.1638, 0.1327, 0.2354}}; //98%
      ALPAKA_STATIC_ACC_MEM_GLOBAL constexpr float kWp_displaced[kPtBins][kEtaBins] = {
      // {0.3092, 0.3312, 0.3757, 0.4073, 0.3422, 0.3500, 0.3945, 0.4233, 0.4754, 0.5115}, //70% low pt, with DS
      // {0.3553, 0.3637, 0.4032, 0.4208, 0.3635, 0.3503, 0.3550, 0.3587, 0.3938, 0.4006}, //70% low pt, with DS, uncert
      // {0.3370, 0.3464, 0.3607, 0.3823, 0.3593, 0.3480, 0.3977, 0.4322, 0.4461, 0.4954}, //70% low pt, with DS, updated for lay adj
      // {0.4178, 0.4768, 0.4876, 0.4755, 0.3889, 0.3670, 0.3995, 0.4328, 0.4868, 0.4840}, //add radii
      // {0.4763, 0.5233, 0.5124, 0.4602, 0.4025, 0.4002, 0.4024, 0.4051, 0.4279, 0.4230}, //radii, weight pmatched and class
      // {0.5547, 0.6118, 0.6279, 0.6516, 0.5045, 0.4903, 0.4947, 0.5501, 0.6648, 0.6970}, //70% low pt radii, t3 scores
      {0.4056, 0.4600, 0.4429, 0.4547, 0.3771, 0.3657, 0.3660, 0.3992, 0.4521, 0.4764}, //75% low pt radii, t3 scores
      // {0.3598, 0.4200, 0.4260, 0.4663, 0.3512, 0.3549, 0.3821, 0.4369, 0.5027, 0.5260}, //75% low pt radii, t3, uncert, larger
      // {0.0719, 0.1007, 0.1281, 0.1422, 0.1437, 0.1429, 0.1583, 0.1712, 0.1712, 0.1672}, //98%
      // {0.4811, 0.4768, 0.5232, 0.5122, 0.4487, 0.3415, 0.3641, 0.3567, 0.4224, 0.4338}}; //75% hight pt, with DS
      // {0.4498, 0.4645, 0.4961, 0.4989, 0.4400, 0.3740, 0.3358, 0.3378, 0.3636, 0.3732}}; //75% high pt, with DS, uncert
      // {0.4606, 0.4747, 0.5040, 0.5299, 0.4366, 0.3653, 0.3994, 0.3761, 0.4413, 0.4720}}; //75% high pt, with DS, updated for lay adj
      // {0.2183, 0.2461, 0.2993, 0.3398, 0.3892, 0.3807, 0.4049, 0.4265, 0.3914, 0.3525}}; //add radii
      // {0.2558, 0.2573, 0.3044, 0.3267, 0.3527, 0.3576, 0.3708, 0.4624, 0.4701, 0.4465}}; //radii, weight pmatched and class
      // {0.5679, 0.5338, 0.5111, 0.4980, 0.4634, 0.4077, 0.3836, 0.3608, 0.3823, 0.3811}}; // 75% high pt radii, t3 scores
      // {0.5392, 0.5101, 0.4800, 0.4678, 0.4219, 0.3687, 0.3505, 0.3183, 0.3489, 0.3391}}; //80% high pt radii, t3 scores
      {0.4893, 0.4653, 0.4464, 0.4310, 0.3706, 0.3340, 0.3140, 0.2800, 0.3107, 0.3028}}; //85% high pt radii, t3 scores
      // {0.6421, 0.6100, 0.5993, 0.5550, 0.4117, 0.4345, 0.3770, 0.3765, 0.3827, 0.3756}}; //80% high pt, radii, t3, uncert, larger
      // {0.0383, 0.1442, 0.2314, 0.2787, 0.1740, 0.1892, 0.1928, 0.1094, 0.2177, 0.1538}}; //98%
    ALPAKA_STATIC_ACC_MEM_GLOBAL constexpr float kWp_prompt_tight[kPtBins][kEtaBins] = {
      // {0.7131, 0.7196, 0.7167, 0.7160, 0.6762, 0.6637, 0.6472, 0.6308, 0.6612, 0.6700}, //65% low pt
      // {0.7502, 0.7476, 0.7395, 0.7335, 0.6607, 0.6594, 0.6675, 0.6772, 0.7452, 0.7593}, //65% low pt, with DS
      {0.7887, 0.7849, 0.7783, 0.7744, 0.7276, 0.7098, 0.7216, 0.7306, 0.7877, 0.8067}, //50 (from with ds)
      // {0.2200, 0.2296, 0.2566, 0.2914, 0.3030, 0.3209, 0.4382, 0.4496, 0.3806, 0.3119}}; //65% high pt
      // {0.2654, 0.2804, 0.3037, 0.3155, 0.3337, 0.3374, 0.4277, 0.4803, 0.4997, 0.4662}}; //65% high pt, with DS
      {0.3112, 0.3231, 0.3405, 0.3679, 0.4102, 0.4031, 0.4809, 0.5259, 0.5396, 0.5135}}; //50% (from with ds)
    ALPAKA_STATIC_ACC_MEM_GLOBAL constexpr float kWp_displaced_tight[kPtBins][kEtaBins] = {
      // {0.3813, 0.4149, 0.4553, 0.4929, 0.4587, 0.4883, 0.5352, 0.6079, 0.7113, 0.7305}, //65% low pt
      {0.3876, 0.4230, 0.4793, 0.5187, 0.3934, 0.4039, 0.4541, 0.4945, 0.5991, 0.6419}, //65% low pt, with DS
      // {0.6772, 0.7315, 0.7790, 0.7970, 0.6142, 0.6286, 0.6750, 0.7262, 0.8572, 0.8780}, //50% low pt, with DS
      // {0.5771, 0.5913, 0.6359, 0.6237, 0.5685, 0.4611, 0.4348, 0.4382, 0.5647, 0.5393}}; //65% high pt
      {0.5391, 0.5472, 0.5730, 0.5749, 0.5112, 0.4106, 0.4130, 0.3961, 0.4434, 0.4561}}; //65% high pt, with DS
      // {0.6035, 0.6074, 0.6214, 0.6316, 0.6006, 0.4864, 0.4751, 0.4496, 0.4902, 0.5029}}; //50% high pt, with DS
  }  // namespace t4dnn

  namespace pt4dnn {
    ALPAKA_STATIC_ACC_MEM_GLOBAL constexpr float kEta_norm = 2.5f;
    // pt, eta binned
    // constexpr unsigned int kPtBins = 2;
    constexpr unsigned int kEtaBins = 10;
    // ALPAKA_STATIC_ACC_MEM_GLOBAL constexpr float kWp[kPtBins][kEtaBins] = {
    //   {0.7725, 0.8018, 0.7875, 0.8187, 0.7454, 0.6384, 0.641, 0.5701, 0.4969, 0.3537},
    //   {0.8957, 0.8968, 0.8978, 0.8915, 0.8668, 0.7666, 0.6267, 0.6929, 0.6253, 0.4507}}; //99% retention, no rphi cuts
        // {0.6517, 0.6897, 0.671, 0.688, 0.5998, 0.4248, 0.4812, 0.3878, 0.323, 0.2114},
        // {0.8789, 0.8886, 0.889, 0.8647, 0.7862, 0.5944, 0.6028, 0.6163, 0.7972, 0.7568}}; //99% retention, no rphi, w DS
    ALPAKA_STATIC_ACC_MEM_GLOBAL constexpr float kWp[kEtaBins] = 
        // {0.8093, 0.8475, 0.8591, 0.8498, 0.772, 0.6158, 0.7007, 0.7072, 0.6852, 0.5364}; //99% retention
        // {0.7749, 0.803, 0.7901, 0.8194, 0.7461, 0.6411, 0.6406, 0.5715, 0.4979, 0.3563}; //99% retention, no rphi cuts
        // {0.8575, 0.8707, 0.879, 0.8838, 0.8231, 0.7616, 0.77, 0.7232, 0.6716, 0.5538}; //98% retention no rphi
        {0.8497, 0.8674, 0.8501, 0.8416, 0.7457, 0.7593, 0.7884, 0.7622, 0.658, 0.5348}; //98% retention, updated for uncertainty in t4 dnn
        // {0.9445, 0.9551, 0.9505, 0.9476, 0.9074, 0.8888, 0.8925, 0.888, 0.8871, 0.8402}; //93% retention, no rphi
        // {0.6539, 0.6918, 0.6738, 0.6908, 0.6008, 0.4268, 0.4828, 0.3907, 0.3263, 0.2158}; //99% retention, no rphi, w downsample
  }  // namespace pt4dnn
//#endif
  namespace pt3dnn {
      ALPAKA_STATIC_ACC_MEM_GLOBAL constexpr float kEta_norm = 2.5f;
      ALPAKA_STATIC_ACC_MEM_GLOBAL constexpr float kEtaSize = 0.25f;
      constexpr unsigned int kEtaBins = 10;
      ALPAKA_STATIC_ACC_MEM_GLOBAL constexpr float kWp[kEtaBins] = {
          0.189f, 0.1805f, 0.2267f, 0.3104f, 0.4719f, 0.3159f, 0.1372f, 0.1571f, 0.3198f, 0.186f};
      ALPAKA_STATIC_ACC_MEM_GLOBAL constexpr float kWpHigh = 0.0473f;
    }  // namespace pt3dnn
}  //namespace lst
#endif
