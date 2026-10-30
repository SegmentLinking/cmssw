import FWCore.ParameterSet.Config as cms

import collections

# RecoTracker/MkFitAlpaka producers -> SoA layout of their device product; its last readers order the release of the
# block (GPU backends)
_mkFitAlpakaProducts = {
    "MkFitAlpakaOTRecHitsProducer@alpaka": "OTRecHitSoA",
    "MkFitAlpakaBuildProducer@alpaka": "Track",
}

def customiseEarlyDeleteForMkFit(process, products):
    
    references = collections.defaultdict(list)

    def _branchName(productType, moduleLabel, instanceLabel=""):
        return "%s_%s_%s_%s" % (productType, moduleLabel, instanceLabel, process.name_())

    for name, module in process.producers_().items():
        cppType = module.type_()
        if cppType == "MkFitSiPixelHitConverter":
            products[name].extend([
                _branchName("MkFitHitWrapper", name),
                _branchName("MkFitClusterIndexToHit", name),
            ])
        elif cppType == "MkFitSiStripHitConverter":
            products[name].extend([
                _branchName("MkFitHitWrapper", name),
                _branchName("MkFitClusterIndexToHit", name),
                _branchName("floats", name)
            ])
        elif cppType == "MkFitEventOfHitsProducer":
            products[name].append(_branchName("MkFitEventOfHits", name))
            references[_branchName("MkFitEventOfHits", name)] = [
                    _branchName("MkFitHitWrapper", module.pixelHits.moduleLabel),
                    _branchName("MkFitHitWrapper", module.stripHits.moduleLabel)
                    ]
        elif cppType == "MkFitSeedConverter":
            products[name].append(_branchName("MkFitSeedWrapper", name))
        elif cppType == "MkFitProducer":
            products[name].append(_branchName("MkFitOutputWrapper", name))
        elif cppType == "MkFitAlpakaEventOfHitsProducer@alpaka":
            # GPU backends: the two-allocation collection; CPU backends: returns its block to the pool
            for dev in ("Cuda", "Hip"):
                products[name].append(_branchName("alpakaDev%sRtmkfitdevEventOfHitsSplitDeviceCollection"
                                                  "edmDeviceProduct" % dev, name))
            products[name].append(_branchName("mkfitdevEventOfHitsPooledHostCollection", name))
        elif cppType in _mkFitAlpakaProducts:
            for dev in ("Cuda", "Hip"):
                products[name].append(_branchName("alpakaDev%sRt128falsemkfitdev%sLayoutvoid"
                                                  "PortableDeviceCollectionedmDeviceProduct"
                                                  % (dev, _mkFitAlpakaProducts[cppType]), name))

    return (products,references)
