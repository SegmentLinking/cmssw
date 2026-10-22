#!/bin/bash
# Regenerate the MkFitCore propagation/Kalman .ah products (GenMPlexOps.pl, CMSSW_20_1_0_pre2) in portable form
# into directory $1.
# The expressions are identical to the standard branch of the MkFitCore .ah files (checked for all 31 products).
#   usage: interface/matriplex/gen/gen_portable_ops.sh <output dir>
set -e
GEN=$(cd "$(dirname "$0")" && pwd)
OUT=${1:?output directory}
mkdir -p "$OUT"
cd "$OUT"
sed -e "s#use lib \"../Matriplex\";#use lib \"$GEN\";#" -e 's/dump_multiply_std_and_intrinsic(/dump_multiply_portable(/g' \
  "$GEN/GenMPlexOps.pl" > .GenMPlexOpsPortable.pl
perl -I"$GEN" .GenMPlexOpsPortable.pl >/dev/null 2>&1
rm -f .GenMPlexOpsPortable.pl
ls *.ah | wc -l
