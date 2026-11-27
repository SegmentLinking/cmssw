#ifndef RecoTracker_MkFitAlpaka_interface_matriplex_MatriplexBackend_h
#define RecoTracker_MkFitAlpaka_interface_matriplex_MatriplexBackend_h

// Per-backend lane width and namespace aliases for device code in ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev.
// Include only from Alpaka-compiled sources (*.dev.cc and the headers they include).
//
//   ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev::kNN  = 1 on GPU backends (CUDA, ROCm), 8 on CPU backends.
//   Inside ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev the MkFitCore spellings work unchanged:
//     Matriplex::..., vdt::fast_sincosf(...), Config::..., Const::..., MPlexLS<N>, hipo(...), sincos4(...)

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "RecoTracker/MkFitAlpaka/interface/matriplex/Matrix.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev {

#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED) || defined(ALPAKA_ACC_GPU_HIP_ENABLED)
  constexpr ::mkfitdev::Matriplex::idx_t kNN = ::mkfitdev::kNNGpu;
#else
  constexpr ::mkfitdev::Matriplex::idx_t kNN = ::mkfitdev::kNNCpu;
#endif

  namespace Matriplex = ::mkfitdev::Matriplex;
  namespace vdt = ::mkfitdev::vdt;
  namespace Config = ::mkfitdev::Config;
  namespace Const = ::mkfitdev::Const;

  using ::mkfitdev::HH;
  using ::mkfitdev::LL;

  using ::mkfitdev::MPlex;
  using ::mkfitdev::MPlex22;
  using ::mkfitdev::MPlex2H;
  using ::mkfitdev::MPlex2S;
  using ::mkfitdev::MPlex2V;
  using ::mkfitdev::MPlex52;
  using ::mkfitdev::MPlex55;
  using ::mkfitdev::MPlex56;
  using ::mkfitdev::MPlex5S;
  using ::mkfitdev::MPlex5V;
  using ::mkfitdev::MPlex65;
  using ::mkfitdev::MPlexH2;
  using ::mkfitdev::MPlexHH;
  using ::mkfitdev::MPlexHL;
  using ::mkfitdev::MPlexHS;
  using ::mkfitdev::MPlexHV;
  using ::mkfitdev::MPlexL2;
  using ::mkfitdev::MPlexLH;
  using ::mkfitdev::MPlexLL;
  using ::mkfitdev::MPlexLS;
  using ::mkfitdev::MPlexLV;
  using ::mkfitdev::MPlexQB;
  using ::mkfitdev::MPlexQF;
  using ::mkfitdev::MPlexQH;
  using ::mkfitdev::MPlexQI;
  using ::mkfitdev::MPlexQUH;
  using ::mkfitdev::MPlexQUI;
  using ::mkfitdev::MPlexSym;

  using ::mkfitdev::diagonalOnly;
  using ::mkfitdev::SMatrix;
  using ::mkfitdev::SMatrix22;
  using ::mkfitdev::SMatrix26;
  using ::mkfitdev::SMatrix33;
  using ::mkfitdev::SMatrix36;
  using ::mkfitdev::SMatrix62;
  using ::mkfitdev::SMatrix63;
  using ::mkfitdev::SMatrix66;
  using ::mkfitdev::SMatrixSym;
  using ::mkfitdev::SMatrixSym22;
  using ::mkfitdev::SMatrixSym33;
  using ::mkfitdev::SMatrixSym66;
  using ::mkfitdev::SVector;
  using ::mkfitdev::SVector2;
  using ::mkfitdev::SVector3;
  using ::mkfitdev::SVector6;

  using ::mkfitdev::cdist;
  using ::mkfitdev::cube;
  using ::mkfitdev::getEta;
  using ::mkfitdev::getHypot;
  using ::mkfitdev::getInvRad2;
  using ::mkfitdev::getInvRadErr2;
  using ::mkfitdev::getPhi;
  using ::mkfitdev::getPhiErr2;
  using ::mkfitdev::getRad2;
  using ::mkfitdev::getRadErr2;
  using ::mkfitdev::getTheta;
  using ::mkfitdev::hipo;
  using ::mkfitdev::hipo_sqr;
  using ::mkfitdev::sincos4;
  using ::mkfitdev::sqr;
  using ::mkfitdev::squashPhiGeneral;
  using ::mkfitdev::squashPhiMinimal;

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::mkfitdev

#endif
