#!/bin/bash
# The early-delete branch names of customiseEarlyDeleteForMkFit against the names the framework registers for the
# MkFitAlpaka products on every backend built: every name the customise writes must be registered by some backend, and
# every device product of an asynchronous backend (edm::DeviceProduct) must be in the customise's list.
set -e
workdir=$(mktemp -d "${TMPDIR:-/tmp}/earlydelete.XXXXXX")
trap 'rm -rf "$workdir"' EXIT
for backend in SerialSync CudaAsync ROCmAsync; do
  if command -v "testMkFitAlpakaEarlyDeleteNames$backend" > /dev/null; then
    "testMkFitAlpakaEarlyDeleteNames$backend" >> "$workdir/registered.txt"
  fi
done
python3 - "$workdir/registered.txt" <<'PYEOF'
import collections
import sys

import FWCore.ParameterSet.Config as cms
from RecoTracker.Configuration.customiseEarlyDeleteForMkFit import customiseEarlyDeleteForMkFit

registered = collections.defaultdict(set)  # producer type -> branch class names over the backends
for line in open(sys.argv[1]):
    moduleType, className = line.split()
    registered[moduleType].add(className)
process = cms.Process('TEST')
labels = {}
for i, moduleType in enumerate(sorted(registered)):
    labels[moduleType] = 'producer%d' % i
    setattr(process, labels[moduleType], cms.EDProducer(moduleType))
products, references = customiseEarlyDeleteForMkFit(process, collections.defaultdict(list))
failures = 0
for moduleType, label in labels.items():
    written = {name[:-len('_%s__TEST' % label)] for name in products[label]}
    unknown = written - registered[moduleType]
    missing = {name for name in registered[moduleType] if name.endswith('edmDeviceProduct')} - written
    good = not unknown and not missing
    failures += not good
    print('%s %s: customise %s, registered %s%s%s' % ('ok  ' if good else 'FAIL', moduleType, sorted(written),
                                                     sorted(registered[moduleType]),
                                                     ', not registered: %s' % sorted(unknown) if unknown else '',
                                                     ', not deleted early: %s' % sorted(missing) if missing else ''))
sys.exit(1 if failures else 0)
PYEOF
