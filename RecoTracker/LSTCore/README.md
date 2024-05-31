# LSTCore proof of concept

**This is a proof of concept for how I think we could continue working towards the CMSSW integration while keeping the standalone version alive.**

This branch of CMSSW contains all of the relevant LST code and can be built entirely within CMSSW. The setup process is what you would expect.

```bash
export CMSSW_VERSION=CMSSW_14_1_0_pre3
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
git submodule update --init --recursive
scram b -j 8
```

## How it works

The [TrackLooper repository](https://github.com/SegmentLinking/TrackLooper) is included as a git submodule in `RecoTracker/LSTCore` and the rest of the structure is set up using symlinks. Since we have made a lot of progress getting the code ready for CMSSW, it was just a matter of writing a simple `BuildFile.xml` file.

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

