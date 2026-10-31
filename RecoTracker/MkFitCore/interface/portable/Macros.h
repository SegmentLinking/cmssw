#ifndef RecoTracker_MkFitCore_interface_portable_Macros_h
#define RecoTracker_MkFitCore_interface_portable_Macros_h

// Host/device annotation of the portable mkFit physics (Matriplex, propagation, Kalman update), without a
// dependency on Alpaka or on GPU headers: empty for host compilers, so MkFitCore's own build is unchanged.
#if defined(__CUDACC__) || defined(__HIPCC__)
#define MKFIT_HOST_DEVICE __host__ __device__
#else
#define MKFIT_HOST_DEVICE
#endif

// Defined while a GPU compiler generates device code.
#if defined(__CUDA_ARCH__) || defined(__HIP_DEVICE_COMPILE__)
#define MKFIT_DEVICE_COMPILATION
#endif

// Inline namespace of the portable function templates: each Alpaka backend instantiates them under its own
// symbols, so they never interpose with MkFitCore's CPU instantiations, which are built with other flags.
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
#define MKFIT_PORTABLE_NAMESPACE cuda
#elif defined(ALPAKA_ACC_GPU_HIP_ENABLED)
#define MKFIT_PORTABLE_NAMESPACE rocm
#elif defined(ALPAKA_ACC_CPU_B_TBB_T_SEQ_ENABLED)
#define MKFIT_PORTABLE_NAMESPACE cpu_tbb
#elif defined(ALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED)
#define MKFIT_PORTABLE_NAMESPACE cpu_serial
#else
#define MKFIT_PORTABLE_NAMESPACE host
#endif

#endif
