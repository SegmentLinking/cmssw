# LSTCore proof of concept

**This is a proof of concept for how I think we could continue working towards the CMSSW integration while keeping the standalone version alive.**

This branch of CMSSW contains all of the relevant LST code and can be built entirely within CMSSW. The setup process is what you would expect.

```bash
export CMSSW_VERSION=CMSSW_14_1_0_pre0
export CMSSW_BRANCH=${CMSSW_VERSION}_LST_X_LSTCore
source /cvmfs/cms.cern.ch/cmsset_default.sh
cmsrel $CMSSW_VERSION
cd $CMSSW_VERSION/src
cmsenv
git cms-init
git remote add SegLink https://github.com/SegmentLinking/cmssw.git
git fetch SegLink ${CMSSW_BRANCH}:SegLink_cmssw
git checkout SegLink_cmssw
git cms-addpkg RecoTracker/LST RecoTracker/LSTCore Configuration/ProcessModifiers RecoTracker/ConversionSeedGenerators RecoTracker/FinalTrackSelectors RecoTracker/IterativeTracking
scram b -j 8
```

## How it works

The `SDL` and `data` directories of the [TrackLooper](https://github.com/SegmentLinking/TrackLooper) are copy-pasted into `RecoTracker/LSTCore` and the rest of the structure is set up using symlinks. Since all the warnings that were treated as errors are now resolved, it was just a matter of writing some simple `BuildFile.xml` file.

If we do decide to go with this option, I think we shouldn't actually include the `SDL` and `data` directories in this repo (and add a `.gitignore` to prevent them from being added), but instead the CI or the person doing the tests would copy them over before compiling.

## Benefits

- It would make it easier to work towards the full integration if we have a self-contained thing. It would probably be easier to slowly adapt more of the "proper" CMSSW conventions instead of having to switch them all at once.
- We can keep the standalone version alive for as long as needed.
- Our CI can start running the checks that are done by the `cms-bot` for CMSSW PRs.

## Disadvantages

- I might be better to work towards having a single CMSSW package instead of having them separated in `LST` and `LSTCore`. However, I think we could use a similar approach in that case.
- I couldn't think of anything else, but there's likely other disadvantages.

## Things to do

- There are a few minor changes that need to be made to the current LST package to get it to work with LSTCore.
- At some point we'll have to figure out how to properly integrate the `data` directory.
- We should try to separate the actual stuff that needs to be in the `interface` folder versus the rest. For now I just put everything in there since they are very interconnected. Ideally, it should be `LST.h` and likely `Module.h`.
