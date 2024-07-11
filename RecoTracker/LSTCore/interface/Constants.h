#ifndef RecoTracker_LSTCore_interface_Constants_h
#define RecoTracker_LSTCore_interface_Constants_h


namespace SDL {

// If a compile time flag does not define PT_CUT, default to 0.8 (GeV)
#ifndef PT_CUT
  constexpr float PT_CUT = 0.8f;
#endif

  constexpr unsigned int MAX_BLOCKS = 80;
  constexpr unsigned int MAX_CONNECTED_MODULES = 40;

  constexpr unsigned int N_MAX_PIXEL_SEGMENTS_PER_MODULE = 50000;

  constexpr unsigned int N_MAX_PIXEL_MD_PER_MODULES = 2 * N_MAX_PIXEL_SEGMENTS_PER_MODULE;

  constexpr unsigned int N_MAX_PIXEL_TRIPLETS = 5000;
  constexpr unsigned int N_MAX_PIXEL_QUINTUPLETS = 15000;

  constexpr unsigned int N_MAX_PIXEL_TRACK_CANDIDATES = 30000;
  constexpr unsigned int N_MAX_NONPIXEL_TRACK_CANDIDATES = 1000;

  constexpr unsigned int size_superbins = 45000;

  //defining the constant host device variables right up here
  // Currently pixel tracks treated as LSs with 2 double layers (IT layers 1+2 and 3+4) and 4 hits. To be potentially handled better in the future.
  struct Params_pLS {
    static constexpr int kLayers = 2, kHits = 4;
  };
  struct Params_LS {
    static constexpr int kLayers = 2, kHits = 4;
  };
  struct Params_T3 {
    static constexpr int kLayers = 3, kHits = 6;
  };
  struct Params_pT3 {
    static constexpr int kLayers = 5, kHits = 10;
  };
  struct Params_T5 {
    static constexpr int kLayers = 5, kHits = 10;
  };
  struct Params_pT5 {
    static constexpr int kLayers = 7, kHits = 14;
  };

}  //namespace SDL

#endif
