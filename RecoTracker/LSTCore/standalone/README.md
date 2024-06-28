# How to set up standalone LST

## Setting up LSTPerformanceWeb (only for lnx7188 and lnx4555)

For lnx7188 and lnx4555 this needs to be done once

    cd /cdat/tem/${USER}/
    git clone git@github.com:SegmentLinking/LSTPerformanceWeb.git

## Setting up container (only for lnx7188)

For lnx7188 this needs to be done before compiling or running the code:

    singularity shell --nv --bind /mnt/data1:/data --bind /data2/segmentlinking/ --bind /opt --bind /nfs --bind /mnt --bind /usr/local/cuda/bin/ --bind /cvmfs  /cvmfs/unpacked.cern.ch/registry.hub.docker.com/cmssw/el8:x86_64

## Setting up LST

There are two way to set up LST as a standalone, either by setting up a full CMSSW area, which provides a unified setup for standalone and CMSSW tests, or by `sparse-checkout` only the relevant package and using them independent of CMSSW. A CVMFS-less setup is also provided for the second option.

### Setting up LST within CMSSW (preferred option)

```bash
CMSSW_VERSION=CMSSW_14_1_0_pre3 # Change with latest/preferred CMSSW version
LST_BRANCH=CMSSW_14_1_0_pre3_LST_X_LSTCore_realfiles_batch1_devel # Change to the appropriate branch
cmsrel ${CMSSW_VERSION}
cd ${CMSSW_VERSION}/src/
cmsenv
git cms-init
git remote add SegLink git@github.com:SegmentLinking/cmssw.git
git fetch SegLink ${LST_BRANCH}
git checkout ${LST_BRANCH}
git cms-addpkg RecoTracker/LST RecoTracker/LSTCore
scram b -j 12
cd RecoTracker/LSTCore/standalone
```

Currently, the data files for LST need to be copied manually (under `$CMSSW_BASE/external/$SCRAM_ARCH/data/RecoTracker/LSTCore/data/`). This is done by running:

```bash
mkdir -p $CMSSW_BASE/external/$SCRAM_ARCH/data/RecoTracker/LSTCore/
cd $CMSSW_BASE/external/$SCRAM_ARCH/data/RecoTracker/LSTCore/
git clone git@github.com:cms-data/RecoTracker-LSTCore.git data
cd -
```

### Setting up LST outside of CMSSW

For this setup, dependencies are still provided from CMSSW through CVMFS but no CMSSW area is setup. This is done by running the following commands.

``` bash
LST_BRANCH=CMSSW_14_1_0_pre3_LST_X_LSTCore_realfiles # Change to the development branch
git clone --filter=blob:none --no-checkout --depth 1 --sparse --branch ${LST_BRANCH} https://github.com/SegmentLinking/cmssw.git TrackLooper
cd TrackLooper
git sparse-checkout add HeterogeneousCore/AlpakaInterface RecoTracker/LSTCore
git checkout
cd RecoTracker/LSTCore/standalone/
```

The data files for LST need to be copied manually (under `RecoTracker/LSTCore/data/`). This is done by running:

```bash
cd ..
git clone git@github.com:cms-data/RecoTracker-LSTCore.git data
cd -
```

<!---
#### Running LST in a CVMFS-less setup

The setup scripts included in this repository assume that the [CernVM File System (CVMFS)](https://cernvm.cern.ch/fs/) is installed. This provides a convenient way to fetch the required dependencies, but it is not necessary to run LST in standalone mode. Here, we briefly describe how to build and run it when CVMFS is not available.

The necessary dependencies are CUDA, ROOT, the Boost libraries, Alpaka, and some CMSSW headers. CUDA, ROOT, and Boost, are fairly standard libraries and are available from multiple package managers. For the remaining necessary headers you will need to clone the [Alpaka](https://github.com/alpaka-group/alpaka) and [CMSSW](https://github.com/cms-sw/cmssw) repositories (since the CMSSW repository is extremely large, the instructions just above need to be followed to checkout the strictly required part of it).

Then all that is left to do is set some environment variables. We give an example of how to do this in lnx7188/cgpu-1.

```bash
# These two lines are only needed to set the right version of gcc and nvcc. They are not needed for standard installations.
export PATH=/cvmfs/cms.cern.ch/el8_amd64_gcc12/external/gcc/12.3.1-40d504be6370b5a30e3947a6e575ca28/bin:/cvmfs/cms.cern.ch/el8_amd64_gcc12/cms/cmssw/CMSSW_14_1_0_pre3/external/el8_amd64_gcc12/bin:$PATH
export LD_LIBRARY_PATH=/cvmfs/cms.cern.ch/el8_amd64_gcc12/cms/cmssw/CMSSW_14_1_0_pre3/biglib/el8_amd64_gcc12:/cvmfs/cms.cern.ch/el8_amd64_gcc12/cms/cmssw/CMSSW_14_1_0_pre3/lib/el8_amd64_gcc12:/cvmfs/cms.cern.ch/el8_amd64_gcc12/cms/cmssw/CMSSW_14_1_0_pre3/external/el8_amd64_gcc12/lib:/cvmfs/cms.cern.ch/el8_amd64_gcc12/external/gcc/12.3.1-40d504be6370b5a30e3947a6e575ca28/lib64:/cvmfs/cms.cern.ch/el8_amd64_gcc12/external/gcc/12.3.1-40d504be6370b5a30e3947a6e575ca28/lib:$LD_LIBRARY_PATH

# These are the lines that you need to manually change for a CVMFS-less setup.
# In this example we use cvmfs paths since that is where the dependencies are in lnx7188/cgpu1, but they can point to local directories.
export BOOST_ROOT=/cvmfs/cms.cern.ch/el8_amd64_gcc12/external/boost/1.80.0-60a217837b5db1cff00c7d88ec42f53a
export ALPAKA_ROOT=/cvmfs/cms.cern.ch/el8_amd64_gcc12/external/alpaka/1.1.0-7d0324257db47fde2d27987e7ff98fb4
export CUDA_HOME=/cvmfs/cms.cern.ch/el8_amd64_gcc12/external/cuda/12.4.1-06cde0cd9f95a73a1ea05c8535f60bde
export ROOT_ROOT=/cvmfs/cms.cern.ch/el8_amd64_gcc12/lcg/root/6.30.07-21947a33e64ceb827a089697ad72e468
export CMSSW_BASE=/cvmfs/cms.cern.ch/el8_amd64_gcc12/cms/cmssw/CMSSW_14_1_0_pre3

# These lines are needed to account for some extra environment variables that are exported in the setup script.
export LD_LIBRARY_PATH=$PWD/SDL:$LD_LIBRARY_PATH
export PATH=$PWD/bin:$PATH
export PATH=$PWD/efficiency/bin:$PATH
export PATH=$PWD/efficiency/python:$PATH
export TRACKLOOPERDIR=$PWD
export TRACKINGNTUPLEDIR=/data2/segmentlinking/CMSSW_12_2_0_pre2/
export LSTOUTPUTDIR=.
source $PWD/code/rooutil/thisrooutil.sh

# After this, you can compile and run LST as usual, as described below.
```
-->

## Running the code

Each time the standalone version of LST is to be used, the following command should be run from the `RecoTracker/LSTCore/standalone` directory:
```bash
# Source one of the commands below, depending on the site
source setup.sh # if on UCSD or Cornell
source setup_hpg.sh # if on Florida
```

For running the code:

    sdl_make_tracklooper -mc
    sdl_<backend> -i PU200 -o LSTNtuple.root
    createPerfNumDenHists -i LSTNtuple.root -o LSTNumDen.root
    lst_plot_performance.py LSTNumDen.root -t "myTag" # or
    python3 efficiency/python/lst_plot_performance.py LSTNumDen.root -t "myTag" # if you are on cgpu-1 or Cornell

The above can be even simplified

    sdl_run -f -mc -s PU200 -n -1 -t myTag

The `-f` flag can be omitted when the code has already been compiled. If multiple backends were compiled, then the `-b` flag can be used to specify a backend. For example

    sdl_run -b cpu -s PU200 -n -1 -t myTag

### Command explanations

Compile the code with option flags. If none of `C,G,R,A` are used, then it defaults to compiling for CUDA and CPU.

    sdl_make_tracklooper -mc
    -m: make clean binaries
    -c: run with the cmssw caching allocator
    -C: compile CPU backend
    -G: compile CUDA backend
    -R: compile ROCm backend
    -A: compile all backends
    -h: show help screen with all options

Run the code
 
    sdl_<backend> -n <nevents> -v <verbose> -w <writeout> -s <streams> -i <dataset> -o <output>

    -i: PU200; muonGun, etc
    -n: number of events; default: all
    -v: 0-no printout; 1- timing printout only; 2- multiplicity printout; default: 0
    -s: number of streams/events in flight; default: 1
    -w: 0- no writeout; 1- minimum writeout; default: 1
    -o: provide an output root file name (e.g. LSTNtuple.root); default: debug.root
    -l: add lower level object (pT3, pT5, T5, etc.) branches to the output

Plotting numerators and denominators of performance plots

    createPerfNumDenHists -i <input> -o <output> [-g <pdgids> -n <nevents>]

    -i: Path to LSTNtuple.root
    -o: provide an output root file name (e.g. num_den_hist.root)
    -n: (optional) number of events
    -g: (optional) comma separated pdgids to add more efficiency plots with different sim particle slices
    
Plotting performance plots

    lst_plot_performance.py num_den_hist.root -t "mywork"

There are several options you can provide to restrict number of plots being produced.
And by default, it creates a certain set of objects.
One can specifcy the type, range, metric, etc.
To see the full information type

    lst_plot_performance.py --help

To give an example of plotting efficiency, object type of lower level T5, for |eta| < 2.5 only.

    lst_plot_performance.py num_den_hist.root -t "mywork" -m eff -o T5_lower -s loweta

NOTE: in order to plot lower level object, ```-l``` option must have been used during ```sdl``` step!

When running on ```cgpu-1``` remember to specify python3 as there is no python.
The shebang on the ```lst_plot_performance.py``` is not updated as ```lnx7188``` works with python2...

    python3 efficiency/python/lst_plot_performance.py num_den_hist.root -t "mywork" # If running on cgpu-1
                                                                                                                                                           
Comparing two different runs

    lst_plot_performance.py \
        num_den_hist_1.root \     # Reference
        num_den_hist_2.root \     # New work
        -L BaseLine,MyNewWork \   # Labeling
        -t "mywork" \
        --compare

# How to set up CMSSW with LST

## Setting up the area

To set up the are, the following commands need to be used:

```bash
CMSSW_VERSION=CMSSW_14_1_0_pre3
LST_BRANCH=CMSSW_14_1_0_pre3_LST_X_LSTCore_realfiles_batch1_devel
cmsrel ${CMSSW_VERSION}
cd ${CMSSW_VERSION}/src/
cmsenv
git cms-init
git remote add SegLink git@github.com:SegmentLinking/cmssw.git
git fetch SegLink ${LST_BRANCH}
git checkout ${LST_BRANCH}
git cms-addpkg Configuration RecoTracker/ConversionSeedGenerators RecoTracker/FinalTrackSelectors RecoTracker/IterativeTracking RecoTracker/LST RecoTracker/LSTCore
scram b -j 12
```

The `LST_BRANCH` and packages to be added depend on the specific use case.

## Run the LST reconstruction in CMSSW (read to the end, before running)

Two complete workflows have been implemented within CMSSW to run a two-iteration, tracking-only reconstruction with LST:
 - 24834.703 (CPU)
 - 24834.704 (GPU)

We will use the second one in the example below. To get the commands of this workflow, one can run:

    runTheMatrix.py -w upgrade -n -e -l 24834.704

For convenience, the workflow has been run for 100 events and the output is stored here:

    /data2/segmentlinking/CMSSW_14_1_0_pre0/step2_21034.1_100Events.root

The input files in each step may need to be properly adjusted to match the ones produced by the previous step/provided externally, hence it is better to run the commands with the `--no_exec` option included.

Running the configuration file with `cmsRun`, the output file will have a name starting with `DQM`. The name is the same every time this step runs,
so it is good practice to rename the file, e.g. to `step4_24834.704.root`.
The MTV plots can be produced with the command:

    makeTrackValidationPlots.py --extended step4_24834.704.root

Comparison plots can be made by including multiple ROOT files as arguments.

## Code formatting and checking

Using the first setup option above, it is prefered to run the checks provided by CMSSW using the following commands.

```
scram b -j 12 code-checks >& c.log && scram b -j 12 code-format >& f.log
```
