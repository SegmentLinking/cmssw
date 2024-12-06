#ifndef AccessHelper_h
#define AccessHelper_h

#include <vector>
#include <tuple>
#include "Event.h"

using LSTEvent = lst::Event<ALPAKA_ACCELERATOR_NAMESPACE::Acc3D>;

enum { kpT5 = 7, kpT3 = 5, kT5 = 4, kpLS = 8, kpT2 = 10 };

// ----* Hit *----
std::tuple<std::vector<unsigned int>, std::vector<unsigned int>> convertHitsToHitIdxsAndHitTypes(
    LSTEvent* event, std::vector<unsigned int> hits);

// ----* pLS *----
std::vector<unsigned int> getPixelHitsFrompLS(LSTEvent* event, unsigned int pLS);
std::vector<unsigned int> getPixelHitIdxsFrompLS(LSTEvent* event, unsigned int pLS);
std::vector<unsigned int> getPixelHitTypesFrompLS(LSTEvent* event, unsigned int pLS);
std::tuple<std::vector<unsigned int>, std::vector<unsigned int>> getHitIdxsAndHitTypesFrompLS(LSTEvent* event,
                                                                                              unsigned pLS);

// ----* MD *----
std::vector<unsigned int> getHitsFromMD(LSTEvent* event, unsigned int MD);
std::tuple<std::vector<unsigned int>, std::vector<unsigned int>> getHitIdxsAndHitTypesFromMD(LSTEvent* event,
                                                                                             unsigned MD);

// ----* LS *----
std::vector<unsigned int> getMDsFromLS(LSTEvent* event, unsigned int LS);
std::vector<unsigned int> getHitsFromLS(LSTEvent* event, unsigned int LS);
std::tuple<std::vector<unsigned int>, std::vector<unsigned int>> getHitIdxsAndHitTypesFromLS(LSTEvent* event,
                                                                                             unsigned LS);

// ----* T3 *----
std::vector<unsigned int> getLSsFromT3(LSTEvent* event, unsigned int T3);
std::vector<unsigned int> getMDsFromT3(LSTEvent* event, unsigned int T3);
std::vector<unsigned int> getHitsFromT3(LSTEvent* event, unsigned int T3);
std::tuple<std::vector<unsigned int>, std::vector<unsigned int>> getHitIdxsAndHitTypesFromT3(LSTEvent* event,
                                                                                             unsigned T3);

// ----* T5 *----
std::vector<unsigned int> getT3sFromT5(LSTEvent* event, unsigned int T5);
std::vector<unsigned int> getLSsFromT5(LSTEvent* event, unsigned int T5);
std::vector<unsigned int> getMDsFromT5(LSTEvent* event, unsigned int T5);
std::vector<unsigned int> getHitsFromT5(LSTEvent* event, unsigned int T5);
std::vector<unsigned int> getHitIdxsFromT5(LSTEvent* event, unsigned int T5);
std::vector<unsigned int> getHitTypesFromT5(LSTEvent* event, unsigned int T5);
std::vector<unsigned int> getModuleIdxsFromT5(LSTEvent* event, unsigned int T5);
std::tuple<std::vector<unsigned int>, std::vector<unsigned int>> getHitIdxsAndHitTypesFromT5(LSTEvent* event,
                                                                                             unsigned T5);

// ----* pT3 *----
unsigned int getPixelLSFrompT3(LSTEvent* event, unsigned int pT3);
unsigned int getT3FrompT3(LSTEvent* event, unsigned int pT3);
std::vector<unsigned int> getLSsFrompT3(LSTEvent* event, unsigned int pT3);
std::vector<unsigned int> getMDsFrompT3(LSTEvent* event, unsigned int pT3);
std::vector<unsigned int> getOuterTrackerHitsFrompT3(LSTEvent* event, unsigned int pT3);
std::vector<unsigned int> getPixelHitsFrompT3(LSTEvent* event, unsigned int pT3);
std::vector<unsigned int> getHitsFrompT3(LSTEvent* event, unsigned int pT3);
std::vector<unsigned int> getHitIdxsFrompT3(LSTEvent* event, unsigned int pT3);
std::vector<unsigned int> getHitTypesFrompT3(LSTEvent* event, unsigned int pT3);
std::vector<unsigned int> getModuleIdxsFrompT3(LSTEvent* event, unsigned int pT3);
std::tuple<std::vector<unsigned int>, std::vector<unsigned int>> getHitIdxsAndHitTypesFrompT3(LSTEvent* event,
                                                                                              unsigned pT3);

// ----* pT5 *----
unsigned int getPixelLSFrompT5(LSTEvent* event, unsigned int pT5);
unsigned int getT5FrompT5(LSTEvent* event, unsigned int pT5);
std::vector<unsigned int> getT3sFrompT5(LSTEvent* event, unsigned int pT5);
std::vector<unsigned int> getLSsFrompT5(LSTEvent* event, unsigned int pT5);
std::vector<unsigned int> getMDsFrompT5(LSTEvent* event, unsigned int pT5);
std::vector<unsigned int> getOuterTrackerHitsFrompT5(LSTEvent* event, unsigned int pT5);
std::vector<unsigned int> getPixelHitsFrompT5(LSTEvent* event, unsigned int pT5);
std::vector<unsigned int> getHitsFrompT5(LSTEvent* event, unsigned int pT5);
std::vector<unsigned int> getHitIdxsFrompT5(LSTEvent* event, unsigned int pT5);
std::vector<unsigned int> getHitTypesFrompT5(LSTEvent* event, unsigned int pT5);
std::vector<unsigned int> getModuleIdxsFrompT5(LSTEvent* event, unsigned int pT5);
std::tuple<std::vector<unsigned int>, std::vector<unsigned int>> getHitIdxsAndHitTypesFrompT5(LSTEvent* event,
                                                                                              unsigned pT5);

// ----* pT2 *----
unsigned int getPixelLSFrompT2(LSTEvent* event, unsigned int pT2);
unsigned int getT2FrompT2(LSTEvent* event, unsigned int pT2);
unsigned int getLSsFrompT2(LSTEvent* event, unsigned int pT2);
std::vector<unsigned int> getMDsFrompT2(LSTEvent* event, unsigned int pT2);
std::vector<unsigned int> getPixelHitsFrompT2(LSTEvent* event, unsigned int pT2);
std::vector<unsigned int> getHitsFrompT2(LSTEvent* event, unsigned int pT2);
std::vector<unsigned int> getHitIdxsFrompT2(LSTEvent* event, unsigned int pT2);
std::vector<unsigned int> getHitTypesFrompT2(LSTEvent* event, unsigned int pT2);
std::vector<unsigned int> getModuleIdxsFrompT2(LSTEvent* event, unsigned int pT2);
std::tuple<std::vector<unsigned int>, std::vector<unsigned int>> getHitIdxsAndHitTypesFrompT2(LSTEvent* event,
                                                                                              unsigned pT2);
  
// ----* TC *----
std::vector<unsigned int> getLSsFromTC(LSTEvent* event, unsigned int TC);
std::vector<unsigned int> getHitsFromTC(LSTEvent* event, unsigned int TC);
std::tuple<std::vector<unsigned int>, std::vector<unsigned int>> getHitIdxsAndHitTypesFromTC(LSTEvent* event,
                                                                                             unsigned int TC);

#endif
