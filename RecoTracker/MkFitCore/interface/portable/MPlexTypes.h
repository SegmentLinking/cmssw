#ifndef RecoTracker_MkFitCore_interface_portable_MPlexTypes_h
#define RecoTracker_MkFitCore_interface_portable_MPlexTypes_h

#include "RecoTracker/MkFitCore/interface/portable/Macros.h"
#include "RecoTracker/MkFitCore/interface/portable/MathUtils.h"

// Provide fast_xyzz() Matriplex methods and operators using VDT.
#define MPLEX_VDT
// Define the following to have fast_xyzz() functions actually call std:: stuff.
// #define MPLEX_VDT_USE_STD

#include "RecoTracker/MkFitCore/interface/portable/Matriplex/MatriplexSym.h"

namespace mkfit {

  constexpr Matriplex::idx_t LL = 6;  // Dimension of large/long  MPlex entities
  constexpr Matriplex::idx_t HH = 3;  // Dimension of small/short MPlex entities

  // The portable physics (propagation, Kalman update) is templated on the number of lanes N of its Matriplexes:
  // MkFitCore's CPU code instantiates it with N = NN (Matrix.h), GPU code with N = 1.
  namespace portable {

    namespace vdt = ::Matriplex::vdt;

    template <int N>
    using MPlexLL = Matriplex::Matriplex<float, LL, LL, N>;
    template <int N>
    using MPlexLV = Matriplex::Matriplex<float, LL, 1, N>;
    template <int N>
    using MPlexLS = Matriplex::MatriplexSym<float, LL, N>;

    template <int N>
    using MPlexHH = Matriplex::Matriplex<float, HH, HH, N>;
    template <int N>
    using MPlexHV = Matriplex::Matriplex<float, HH, 1, N>;
    template <int N>
    using MPlexHS = Matriplex::MatriplexSym<float, HH, N>;

    template <int N>
    using MPlex5V = Matriplex::Matriplex<float, 5, 1, N>;
    template <int N>
    using MPlex5S = Matriplex::MatriplexSym<float, 5, N>;

    template <int N>
    using MPlex55 = Matriplex::Matriplex<float, 5, 5, N>;
    template <int N>
    using MPlex56 = Matriplex::Matriplex<float, 5, 6, N>;
    template <int N>
    using MPlex65 = Matriplex::Matriplex<float, 6, 5, N>;

    template <int N>
    using MPlex22 = Matriplex::Matriplex<float, 2, 2, N>;
    template <int N>
    using MPlex2V = Matriplex::Matriplex<float, 2, 1, N>;
    template <int N>
    using MPlex2S = Matriplex::MatriplexSym<float, 2, N>;

    template <int N>
    using MPlexLH = Matriplex::Matriplex<float, LL, HH, N>;
    template <int N>
    using MPlexHL = Matriplex::Matriplex<float, HH, LL, N>;

    template <int N>
    using MPlex52 = Matriplex::Matriplex<float, 5, 2, N>;
    template <int N>
    using MPlexL2 = Matriplex::Matriplex<float, LL, 2, N>;
    template <int N>
    using MPlexH2 = Matriplex::Matriplex<float, HH, 2, N>;
    template <int N>
    using MPlex2H = Matriplex::Matriplex<float, 2, HH, N>;

    template <int N>
    using MPlexQF = Matriplex::Matriplex<float, 1, 1, N>;
    template <int N>
    using MPlexQI = Matriplex::Matriplex<int, 1, 1, N>;
    template <int N>
    using MPlexQUI = Matriplex::Matriplex<unsigned int, 1, 1, N>;
    template <int N>
    using MPlexQH = Matriplex::Matriplex<short, 1, 1, N>;
    template <int N>
    using MPlexQUH = Matriplex::Matriplex<unsigned short, 1, 1, N>;

    template <int N>
    using MPlexQB = Matriplex::Matriplex<bool, 1, 1, N>;

  }  // namespace portable

}  // end namespace mkfit

#endif
