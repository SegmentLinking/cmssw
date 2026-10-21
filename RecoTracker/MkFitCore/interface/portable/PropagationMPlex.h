#ifndef RecoTracker_MkFitCore_interface_portable_PropagationMPlex_h
#define RecoTracker_MkFitCore_interface_portable_PropagationMPlex_h

// MkFitCore's propagation, templated on the number of lanes N of the Matriplexes and callable from GPU code.
// MkFitCore's CPU code instantiates it with N = NN in src/PropagationMPlex*.cc (see src/PropagationMPlex.h).

#include <type_traits>

#include "RecoTracker/MkFitCore/interface/Config.h"
#include "RecoTracker/MkFitCore/interface/PropagationConfig.h"
#include "RecoTracker/MkFitCore/interface/portable/Macros.h"
#include "RecoTracker/MkFitCore/interface/portable/MPlexTypes.h"

namespace mkfit::portable::inline MKFIT_PORTABLE_NAMESPACE {

  template <int N>
  MKFIT_HOST_DEVICE inline void squashPhiMPlex(MPlexLV<N>& par, const int N_proc) {
#pragma omp simd
    for (int n = 0; n < N; ++n) {
      if (n < N_proc) {
        if (par(n, 4, 0) >= Const::PI)
          par(n, 4, 0) -= Const::TwoPI;
        if (par(n, 4, 0) < -Const::PI)
          par(n, 4, 0) += Const::TwoPI;
      }
    }
  }

  template <int N>
  MKFIT_HOST_DEVICE inline void squashPhiMPlexGeneral(MPlexLV<N>& par, const int N_proc) {
#pragma omp simd
    for (int n = 0; n < N; ++n) {
      par(n, 4, 0) -= std::floor(0.5f * Const::InvPI * (par(n, 4, 0) + Const::PI)) * Const::TwoPI;
    }
  }

  // Barrel / R: PropagationMPlexBarrel.h

  template <int N>
  MKFIT_HOST_DEVICE void propagateLineToRMPlex(const MPlexLS<N>& psErr,
                                               const MPlexLV<N>& psPar,
                                               const MPlexHS<N>& msErr,
                                               const MPlexHV<N>& msPar,
                                               MPlexLS<N>& outErr,
                                               MPlexLV<N>& outPar,
                                               const int N_proc);

  template <int N>
  MKFIT_HOST_DEVICE void propagateHelixToRMPlex(const MPlexLS<N>& inErr,
                                                const MPlexLV<N>& inPar,
                                                const MPlexQI<N>& inChg,
                                                const MPlexQF<N>& msRad,
                                                MPlexLS<N>& outErr,
                                                MPlexLV<N>& outPar,
                                                MPlexQI<N>& outFailFlag,
                                                const int N_proc,
                                                const PropagationFlags& pflags,
                                                const std::type_identity_t<MPlexQI<N>>* noMatEffPtr = nullptr);

  template <int N>
  MKFIT_HOST_DEVICE void helixAtRFromIterativeCCSFullJac(const MPlexLV<N>& inPar,
                                                         const MPlexQI<N>& inChg,
                                                         const MPlexQF<N>& msRad,
                                                         MPlexLV<N>& outPar,
                                                         MPlexLL<N>& errorProp,
                                                         const int N_proc);

  template <int N>
  MKFIT_HOST_DEVICE void helixAtRFromIterativeCCS(const MPlexLV<N>& inPar,
                                                  const MPlexQI<N>& inChg,
                                                  const MPlexQF<N>& msRad,
                                                  MPlexLV<N>& outPar,
                                                  MPlexLL<N>& errorProp,
                                                  MPlexQI<N>& outFailFlag,
                                                  const int N_proc,
                                                  const PropagationFlags& pflags);

  // Endcap / Z: PropagationMPlexEndcap.h

  template <int N>
  MKFIT_HOST_DEVICE void propagateHelixToZMPlex(const MPlexLS<N>& inErr,
                                                const MPlexLV<N>& inPar,
                                                const MPlexQI<N>& inChg,
                                                const MPlexQF<N>& msZ,
                                                MPlexLS<N>& outErr,
                                                MPlexLV<N>& outPar,
                                                MPlexQI<N>& outFailFlag,
                                                const int N_proc,
                                                const PropagationFlags& pflags,
                                                const std::type_identity_t<MPlexQI<N>>* noMatEffPtr = nullptr);

  template <int N>
  MKFIT_HOST_DEVICE void helixAtZ(const MPlexLV<N>& inPar,
                                  const MPlexQI<N>& inChg,
                                  const MPlexQF<N>& msZ,
                                  MPlexLV<N>& outPar,
                                  MPlexLL<N>& errorProp,
                                  MPlexQI<N>& outFailFlag,
                                  const int N_proc,
                                  const PropagationFlags& pflags);

  // Plane: PropagationMPlexPlane.h

  template <int N>
  MKFIT_HOST_DEVICE void helixAtPlane(const MPlexLV<N>& inPar,
                                      const MPlexQI<N>& inChg,
                                      const MPlexHV<N>& plPnt,
                                      const MPlexHV<N>& plNrm,
                                      MPlexQF<N>& pathL,
                                      MPlexLV<N>& outPar,
                                      MPlexLL<N>& errorProp,
                                      MPlexQI<N>& outFailFlag,
                                      const int N_proc,
                                      const PropagationFlags& pflags);

  template <int N>
  MKFIT_HOST_DEVICE void propagateHelixToPlaneMPlex(const MPlexLS<N>& inErr,
                                                    const MPlexLV<N>& inPar,
                                                    const MPlexQI<N>& inChg,
                                                    const MPlexHV<N>& plPnt,
                                                    const MPlexHV<N>& plNrm,
                                                    MPlexLS<N>& outErr,
                                                    MPlexLV<N>& outPar,
                                                    MPlexQI<N>& outFailFlag,
                                                    const int N_proc,
                                                    const PropagationFlags& pflags,
                                                    const std::type_identity_t<MPlexQI<N>>* noMatEffPtr = nullptr);

  // Common functions: PropagationMPlexCommon.h

  template <int N>
  MKFIT_HOST_DEVICE void applyMaterialEffects(const MPlexQF<N>& hitsRl,
                                              const MPlexQF<N>& hitsXi,
                                              const MPlexQF<N>& propSign,
                                              const MPlexHV<N>& plNrm,
                                              MPlexLS<N>& outErr,
                                              MPlexLV<N>& outPar,
                                              const int N_proc,
                                              const PropagationEnv& env);

  template <int N>
  MKFIT_HOST_DEVICE void MultHelixPropFull(const MPlexLL<N>& A, const MPlexLS<N>& B, MPlexLL<N>& C);
  template <int N>
  MKFIT_HOST_DEVICE void MultHelixPropTranspFull(const MPlexLL<N>& A, const MPlexLL<N>& B, MPlexLS<N>& C);

}  // namespace mkfit::portable::inline MKFIT_PORTABLE_NAMESPACE

// MkFitCore's debug printouts (src/Debug.h) are active if the including .cc defines DEBUG and includes Debug.h
// first; otherwise they compile to nothing, and the no-op macros are removed again at the end of this header.
#ifndef dprint
#define MKFIT_PORTABLE_PROPAGATION_NO_DEBUG
#define dprint(x) (void(0))
#define dprint_np(n, x) (void(0))
#define dcall(x) (void(0))
#define dprintf(...) (void(0))
#define dprintf_np(n, ...) (void(0))
#endif

#include "RecoTracker/MkFitCore/interface/portable/PropagationMPlexCommon.h"
#include "RecoTracker/MkFitCore/interface/portable/PropagationMPlexBarrel.h"
#include "RecoTracker/MkFitCore/interface/portable/PropagationMPlexEndcap.h"
#include "RecoTracker/MkFitCore/interface/portable/PropagationMPlexPlane.h"

#ifdef MKFIT_PORTABLE_PROPAGATION_NO_DEBUG
#undef MKFIT_PORTABLE_PROPAGATION_NO_DEBUG
#undef dprint
#undef dprint_np
#undef dcall
#undef dprintf
#undef dprintf_np
#endif

#endif
