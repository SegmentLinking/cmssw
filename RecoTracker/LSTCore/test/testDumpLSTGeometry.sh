#!/bin/sh

function die { echo $1: status $2; exit $2; }

if [ "${SCRAM_TEST_NAME}" != "" ] ; then
  mkdir ${SCRAM_TEST_NAME}
  cd ${SCRAM_TEST_NAME}
fi

(cmsRun ${SCRAM_TEST_PATH}/dumpLSTGeometry.py --conditions auto:phase2_realistic_T33 --geometry ExtendedRun4D110 --era Phase2C17I13M9) || die "failed to run dumpLSTGeometry.py" $?
