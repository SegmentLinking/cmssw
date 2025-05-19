#include "DataFormats/Portable/interface/PortableHostCollectionReadRules.h"
#include "RecoTracker/LSTCore/interface/LSTInputHostCollection.h"
#include "RecoTracker/LSTCore/interface/LSTOutputHostCollection.h"

#ifndef LST_STANDALONE
SET_PORTABLEHOSTMULTICOLLECTION_READ_RULES(lst::LSTInputHostCollection);
SET_PORTABLEHOSTCOLLECTION_READ_RULES(lst::LSTOutputHostCollection);
#endif
