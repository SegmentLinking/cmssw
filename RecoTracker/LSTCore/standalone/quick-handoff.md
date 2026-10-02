
    bin/lst_cpu -i trackingNtuple-100.root -n -1 -v 1 -w 0
    output being captured to /tmp/lst_cpu_timing.log. This run was in progress / not yet confirmed complete.

    Remaining work:
    1. Confirm the CPU run finished and capture its timing output from /tmp/lst_cpu_timing.log.
    2. Run the equivalent command on the CUDA backend: bin/lst_cuda -i trackingNtuple-100.root -n -1 -v 1 -w 0, capture its timing output.
    3. Present/compare the CPU vs. CUDA timing printouts to the user.
    4. Restore the stashed change with git stash pop to bring back the performance.cc modifications once both timing runs are done.
    5. Environment setup reminder for any fresh shell: source /cvmfs/cms.cern.ch/cmsset_default.sh && cd /mnt/data1/kk829/CMSSW_16_1_1 && eval $(scram runtime -sh) && cd
    src/RecoTracker/LSTCore/standalone && source setup.sh — required before any build/run command in this repo.