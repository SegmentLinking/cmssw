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
      // {0.6967, 0.7051, 0.7020, 0.7009, 0.6498, 0.6385, 0.6241, 0.6086, 0.6416, 0.6512}, //70% low pt
      // {0.6514, 0.6671, 0.6632, 0.6620, 0.5720, 0.5687, 0.5615, 0.5473, 0.5857, 0.5879}, //80% low pt
      // {0.7131, 0.7196, 0.7167, 0.7160, 0.6762, 0.6637, 0.6472, 0.6308, 0.6612, 0.6700}, //65% low pt
      // {0.5531, 0.5826, 0.5817, 0.5835, 0.4285, 0.4377, 0.4481, 0.4317, 0.4665, 0.4508}, //90% low pt 
      // {0.7332, 0.7309, 0.7231, 0.7173, 0.6362, 0.6404, 0.6453, 0.6572, 0.7257, 0.7374}, //70% low pt, with DS
      {0.7117, 0.7102, 0.7025, 0.6972, 0.6083, 0.6180, 0.6192, 0.6338, 0.7017, 0.7123}, //75% low pt, with DS
      // {0.1895, 0.1950, 0.2269, 0.2614, 0.2581, 0.2676, 0.3672, 0.3995, 0.3469, 0.2702}}; //75% high pt
      // {0.2252, 0.2498, 0.2765, 0.2948, 0.2971, 0.2957, 0.3775, 0.4306, 0.4598, 0.3966}}; //75% hight pt, with DS
      {0.2001, 0.2230, 0.2600, 0.2829, 0.2794, 0.2715, 0.3439, 0.4018, 0.4332, 0.3548}}; //80% high pt, with DS
      // {0.2061, 0.2141, 0.2427, 0.2766, 0.2803, 0.2946, 0.4045, 0.4253, 0.3659, 0.2947}}; //70% high pt
      // {0.0854, 0.0919, 0.1614, 0.1891, 0.1712, 0.1551, 0.2330, 0.2811, 0.2525, 0.1491}}; //90% high pt
      // {0.1679, 0.1769, 0.2072, 0.2384, 0.2355, 0.2349, 0.3273, 0.3730, 0.3290, 0.2449}}; //80% high pt
      ALPAKA_STATIC_ACC_MEM_GLOBAL constexpr float kWp_displaced[kPtBins][kEtaBins] = {
      // {0.3226, 0.3448, 0.3746, 0.4018, 0.3868, 0.4155, 0.4587, 0.5205, 0.5905, 0.6168}, //70 low
      {0.3092, 0.3312, 0.3757, 0.4073, 0.3422, 0.3500, 0.3945, 0.4233, 0.4754, 0.5115}, //70% low pt, with DS
      // {0.2403, 0.2553, 0.2735, 0.2894, 0.2812, 0.2987, 0.3378, 0.3769, 0.3898, 0.3912}, //80% low pt
      // {0.3813, 0.4149, 0.4553, 0.4929, 0.4587, 0.4883, 0.5352, 0.6079, 0.7113, 0.7305}, //65% low pt
      // {0.0132, 0.0213, 0.0493, 0.0800, 0.0978, 0.1131, 0.1361, 0.1463, 0.1592, 0.1723}, //99% low pt
      // {0.4943, 0.5050, 0.5547, 0.5408, 0.4812, 0.4056, 0.3820, 0.3611, 0.5057, 0.4820}}; //75% high pt
      {0.4811, 0.4768, 0.5232, 0.5122, 0.4487, 0.3415, 0.3641, 0.3567, 0.4224, 0.4338}}; //75% hight pt, with DS
      // {0.5343, 0.5454, 0.5993, 0.5818, 0.5337, 0.4410, 0.4108, 0.4019, 0.5440, 0.5082}}; //70% high pt
      // {0.0009, 0.0022, 0.0354, 0.0343, 0.0587, 0.0476, 0.1060, 0.0456, 0.1918, 0.0122}}; //99% high pt
      // {0.4356, 0.4601, 0.5120, 0.4883, 0.4314, 0.3639, 0.3538, 0.3229, 0.4480, 0.4599}}; //80% high pt
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
        {0.8575, 0.8707, 0.879, 0.8838, 0.8231, 0.7616, 0.77, 0.7232, 0.6716, 0.5538}; //98% retention no rphi
        // {0.9445, 0.9551, 0.9505, 0.9476, 0.9074, 0.8888, 0.8925, 0.888, 0.8871, 0.8402}; //93% retention, no rphi
        // {0.6539, 0.6918, 0.6738, 0.6908, 0.6008, 0.4268, 0.4828, 0.3907, 0.3263, 0.2158}; //99% retention, no rphi, w downsample
  }  // namespace pt4dnn

}  //namespace lst
#endif
