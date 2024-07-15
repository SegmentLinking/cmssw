#ifndef Event_cuh
#define Event_cuh

#include "RecoTracker/LSTCore/interface/alpaka/Constants.h"
#include "RecoTracker/LSTCore/interface/alpaka/Module.h"
#include "RecoTracker/LSTCore/interface/alpaka/LST.h"

#include "Hit.h"
#include "ModuleMethods.h"
#include "Segment.h"
#include "Triplet.h"
#include "Kernels.h"
#include "Quintuplet.h"
#include "MiniDoublet.h"
#include "PixelTriplet.h"
#include "TrackCandidate.h"

#include "HeterogeneousCore/AlpakaInterface/interface/host.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::SDL {
  template <typename TAcc>
  class Event {};

  template <>
  class Event<Acc3D> {
  private:
    Queue queue;
    Device devAcc;
    DevHost devHost;
    bool addObjects;

    std::array<unsigned int, 6> n_hits_by_layer_barrel_;
    std::array<unsigned int, 5> n_hits_by_layer_endcap_;
    std::array<unsigned int, 6> n_minidoublets_by_layer_barrel_;
    std::array<unsigned int, 5> n_minidoublets_by_layer_endcap_;
    std::array<unsigned int, 6> n_segments_by_layer_barrel_;
    std::array<unsigned int, 5> n_segments_by_layer_endcap_;
    std::array<unsigned int, 6> n_triplets_by_layer_barrel_;
    std::array<unsigned int, 5> n_triplets_by_layer_endcap_;
    std::array<unsigned int, 6> n_trackCandidates_by_layer_barrel_;
    std::array<unsigned int, 5> n_trackCandidates_by_layer_endcap_;
    std::array<unsigned int, 6> n_quintuplets_by_layer_barrel_;
    std::array<unsigned int, 5> n_quintuplets_by_layer_endcap_;

    //Device stuff
    unsigned int nTotalSegments;
    struct objectRanges* rangesInGPU;
    struct objectRangesBuffer<Device>* rangesBuffers;
    struct hits* hitsInGPU;
    struct hitsBuffer<Device>* hitsBuffers;
    struct miniDoublets* mdsInGPU;
    struct miniDoubletsBuffer<Device>* miniDoubletsBuffers;
    struct segments* segmentsInGPU;
    struct segmentsBuffer<Device>* segmentsBuffers;
    struct triplets* tripletsInGPU;
    struct tripletsBuffer<Device>* tripletsBuffers;
    struct quintuplets* quintupletsInGPU;
    struct quintupletsBuffer<Device>* quintupletsBuffers;
    struct trackCandidates* trackCandidatesInGPU;
    struct trackCandidatesBuffer<Device>* trackCandidatesBuffers;
    struct pixelTriplets* pixelTripletsInGPU;
    struct pixelTripletsBuffer<Device>* pixelTripletsBuffers;
    struct pixelQuintuplets* pixelQuintupletsInGPU;
    struct pixelQuintupletsBuffer<Device>* pixelQuintupletsBuffers;

    //CPU interface stuff
    objectRangesBuffer<DevHost>* rangesInCPU;
    hitsBuffer<DevHost>* hitsInCPU;
    miniDoubletsBuffer<DevHost>* mdsInCPU;
    segmentsBuffer<DevHost>* segmentsInCPU;
    tripletsBuffer<DevHost>* tripletsInCPU;
    trackCandidatesBuffer<DevHost>* trackCandidatesInCPU;
    modulesBuffer<DevHost>* modulesInCPU;
    quintupletsBuffer<DevHost>* quintupletsInCPU;
    pixelTripletsBuffer<DevHost>* pixelTripletsInCPU;
    pixelQuintupletsBuffer<DevHost>* pixelQuintupletsInCPU;

    void init(bool verbose);

    int* superbinCPU;
    int8_t* pixelTypeCPU;

    // Stuff that used to be global
    const uint16_t nModules_;
    const uint16_t nLowerModules_;
    const unsigned int nPixels_;
    const unsigned int nEndCapMap_;
    const std::shared_ptr<const modulesBuffer<Device>> modulesBuffers_;
    const std::shared_ptr<const ::SDL::pixelMap> pixelMapping_;
    const std::shared_ptr<const EndcapGeometryBuffer<Device>> endcapGeometryBuffers_;

  public:
    // Constructor used for CMSSW integration. Uses an external queue.
    template <typename TQueue>
    Event(bool verbose, TQueue const& q, const LSTESData<Device>* deviceESData)
        : queue(q),
          devAcc(alpaka::getDev(q)),
          devHost(cms::alpakatools::host()),
          nModules_(deviceESData->nModules),
          nLowerModules_(deviceESData->nLowerModules),
          nPixels_(deviceESData->nPixels),
          nEndCapMap_(deviceESData->nEndCapMap),
          modulesBuffers_(deviceESData->modulesBuffers),
          pixelMapping_(deviceESData->pixelMapping),
          endcapGeometryBuffers_(deviceESData->endcapGeometryBuffers) {
      init(verbose);
    }
    void resetEvent();

    void addHitToEvent(
        std::vector<float> x,
        std::vector<float> y,
        std::vector<float> z,
        std::vector<unsigned int> detId,
        std::vector<unsigned int> idxInNtuple);  //call the appropriate hit function, then increment the counter here
    void addPixelSegmentToEvent(std::vector<unsigned int> hitIndices0,
                                std::vector<unsigned int> hitIndices1,
                                std::vector<unsigned int> hitIndices2,
                                std::vector<unsigned int> hitIndices3,
                                std::vector<float> dPhiChange,
                                std::vector<float> ptIn,
                                std::vector<float> ptErr,
                                std::vector<float> px,
                                std::vector<float> py,
                                std::vector<float> pz,
                                std::vector<float> eta,
                                std::vector<float> etaErr,
                                std::vector<float> phi,
                                std::vector<int> charge,
                                std::vector<unsigned int> seedIdx,
                                std::vector<int> superbin,
                                std::vector<int8_t> pixelType,
                                std::vector<char> isQuad);

    // functions that map the objects to the appropriate modules
    void addMiniDoubletsToEventExplicit();
    void addSegmentsToEventExplicit();
    void addTripletsToEventExplicit();
    void addQuintupletsToEventExplicit();
    void resetObjectsInModule();

    void createMiniDoublets();
    void createSegmentsWithModuleMap();
    void createTriplets();
    void createPixelTracklets();
    void createPixelTrackletsWithMap();
    void createTrackCandidates(bool no_pls_dupclean, bool tc_pls_triplets);
    void createExtendedTracks();
    void createQuintuplets();
    void createPixelTriplets();
    void createPixelQuintuplets();
    void pixelLineSegmentCleaning(bool no_pls_dupclean);

    unsigned int getNumberOfHits();
    unsigned int getNumberOfHitsByLayer(unsigned int layer);
    unsigned int getNumberOfHitsByLayerBarrel(unsigned int layer);
    unsigned int getNumberOfHitsByLayerEndcap(unsigned int layer);

    unsigned int getNumberOfMiniDoublets();
    unsigned int getNumberOfMiniDoubletsByLayer(unsigned int layer);
    unsigned int getNumberOfMiniDoubletsByLayerBarrel(unsigned int layer);
    unsigned int getNumberOfMiniDoubletsByLayerEndcap(unsigned int layer);

    unsigned int getNumberOfSegments();
    unsigned int getNumberOfSegmentsByLayer(unsigned int layer);
    unsigned int getNumberOfSegmentsByLayerBarrel(unsigned int layer);
    unsigned int getNumberOfSegmentsByLayerEndcap(unsigned int layer);

    unsigned int getNumberOfTriplets();
    unsigned int getNumberOfTripletsByLayer(unsigned int layer);
    unsigned int getNumberOfTripletsByLayerBarrel(unsigned int layer);
    unsigned int getNumberOfTripletsByLayerEndcap(unsigned int layer);

    int getNumberOfTrackCandidates();
    int getNumberOfPixelTrackCandidates();
    int getNumberOfPT5TrackCandidates();
    int getNumberOfPT3TrackCandidates();
    int getNumberOfT5TrackCandidates();
    int getNumberOfPLSTrackCandidates();

    unsigned int getNumberOfQuintuplets();
    unsigned int getNumberOfQuintupletsByLayer(unsigned int layer);
    unsigned int getNumberOfQuintupletsByLayerBarrel(unsigned int layer);
    unsigned int getNumberOfQuintupletsByLayerEndcap(unsigned int layer);

    int getNumberOfPixelTriplets();
    int getNumberOfPixelQuintuplets();

    objectRangesBuffer<DevHost>* getRanges();
    hitsBuffer<DevHost>* getHits();
    hitsBuffer<DevHost>* getHitsInCMSSW();
    miniDoubletsBuffer<DevHost>* getMiniDoublets();
    segmentsBuffer<DevHost>* getSegments();
    tripletsBuffer<DevHost>* getTriplets();
    quintupletsBuffer<DevHost>* getQuintuplets();
    trackCandidatesBuffer<DevHost>* getTrackCandidates();
    trackCandidatesBuffer<DevHost>* getTrackCandidatesInCMSSW();
    pixelTripletsBuffer<DevHost>* getPixelTriplets();
    pixelQuintupletsBuffer<DevHost>* getPixelQuintuplets();
    modulesBuffer<DevHost>* getModules(bool isFull = false);
  };

}  // namespace SDL
#endif
