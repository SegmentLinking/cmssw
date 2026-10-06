#!/bin/bash
# replay_kernel_registers.sh [-l LOG] [-a ARCH] [-t] [-q] [-j N] [-W FILE] [-K FILE] [lib ...]
# Register / spill report of the MkFitAlpaka CUDA code.
#
# THE GATE = ptxas's "Registers are spilled to local memory" lines for our functions. cuobjdump's LOCAL field is
# always 0 under the CUDA ABI (spill slots and local arrays live in the STACK frame), so it can not see a spill.
#   default  re-run ptxas on the PTX embedded in the package's CUDA objects ($CMSSW_BASE/tmp/$SCRAM_ARCH/src/
#            RecoTracker/MkFitAlpaka/{src,plugins}, -t adds test/), one run per object and arch, with the ptxas
#            options the build recorded in the object (cuobjdump -xptx prints them). Same PTX + same ptxas + same
#            options = the build's own ptxas lines, but with the arch known. Takes a few minutes (-j parallel).
#   -l LOG   instead parse a build log's ptxas lines (no arch in them; fast).
# Waivers: -W FILE (default test/register_spill_waivers.txt): "<kernel regex> <arch regex> <reason>" per line, for
# instantiations the menu never launches on a GPU. A waived spill is listed, not counted.
# Known spills: -K FILE: "<kernel regex> <arch regex>
# <device cost> <reason>" for kernels the menu runs whose spill is kept by a decision. They are a separate, counted
# section with the measured device cost, and the verdict is KNOWN-SPILLS, never PASS. A known entry that no
# longer spills is reported (remove it). The known list wins over the waiver list.
# Information only: cuobjdump -res-usage REG / STACK / SHARED per kernel of the CudaAsync libraries for -a ARCH
# (default sm_89 = L40 / L4). STACK > 0 is NOT a spill by itself (div / sqrt slow paths, local arrays).
# Exit status: 0 no spill outside the waivers (PASS), 1 unwaived spills (FAIL), 2 nothing to check, 3 known spills
# only (KNOWN-SPILLS).
here=$(dirname $(readlink -f $0))
ARCH=sm_89; TESTS=0; QUIET=0; LOG=""; NJ=8; WAIVERS=$here/register_spill_waivers.txt; KNOWN=$here/register_spill_known.txt
while getopts "l:a:tqj:W:K:" o; do
  case $o in l) LOG=$OPTARG ;; a) ARCH=$OPTARG ;; t) TESTS=1 ;; q) QUIET=1 ;; j) NJ=$OPTARG ;; W) WAIVERS=$OPTARG ;; K) KNOWN=$OPTARG ;; *) exit 2 ;; esac
done
shift $((OPTIND - 1))
LIBS="$@"
[ -z "$LIBS" ] && LIBS=$(ls $CMSSW_BASE/lib/$SCRAM_ARCH/*MkFitAlpaka*CudaAsync*.so 2>/dev/null)
work=${TMPDIR:-$CMSSW_BASE/tmp}/spillgate.$$
mkdir -p $work
trap "rm -rf $work" EXIT

# 1. ptxas spill lines -> $work/spills.txt as "<arch> <stores> <loads> <mangled>"
if [ -n "$LOG" ]; then
  grep -o "Registers are spilled to local memory in function '[^']*', [0-9]* bytes spill stores, [0-9]* bytes spill loads" $LOG |
    sed -E "s/.*function '([^']*)', ([0-9]+) bytes spill stores, ([0-9]+) bytes spill loads/? \2 \3 \1/" > $work/spills.txt
  echo "[spills] source: build log $LOG ($(wc -l < $work/spills.txt) ptxas spill lines; arch not in the log)"
else
  T=$CMSSW_BASE/tmp/$SCRAM_ARCH/src/RecoTracker/MkFitAlpaka
  dirs="$T/src $T/plugins"; [ $TESTS = 1 ] && dirs="$dirs $T/test"
  objs=$(find $dirs -path '*CudaAsync*' -name '*.o' ! -name '*_nv.o' ! -name '*_cudadlink.o' 2>/dev/null)
  [ -z "$objs" ] && { echo "no MkFitAlpaka CUDA objects under $T: skipped"; exit 2; }
  n=0
  for o in $objs; do
    n=$((n + 1)); d=$work/o$n; mkdir -p $d
    (cd $d && cuobjdump -xptx all $o 2>/dev/null) | grep 'Extracting PTX file and ptxas options' |
      sed -E "s#.*: +(\S+\.ptx) +(.*)#$d \1 \2#" >> $work/jobs.txt
  done
  [ -s $work/jobs.txt ] || { echo "no embedded PTX found: skipped"; exit 2; }
  # one ptxas per (object, arch); output to /dev/null, warnings to <ptx>.log
  awk '{d=$1; f=$2; $1=""; $2=""; printf "cd %s && ptxas %s %s -o /dev/null > %s.log 2>&1\n", d, $0, f, f}' $work/jobs.txt |
    xargs -P $NJ -I{} bash -c '{}'
  for lg in $work/o*/*.ptx.log; do
    a=$(echo $lg | grep -o 'sm_[0-9]*' | tail -1)
    grep -o "Registers are spilled to local memory in function '[^']*', [0-9]* bytes spill stores, [0-9]* bytes spill loads" $lg |
      sed -E "s/.*function '([^']*)', ([0-9]+) bytes spill stores, ([0-9]+) bytes spill loads/$a \2 \3 \1/"
    grep -i "error" $lg | head -3 | sed "s#^#[spills] ptxas ERROR ($lg): #"
  done > $work/spills.txt
  echo "[spills] source: ptxas re-run on $(wc -l < $work/jobs.txt) embedded PTX files of $n objects ($(grep -c . $work/spills.txt) spill lines)"
fi

# 2. information table from cuobjdump
for f in $LIBS; do echo "#FILE $(basename $f)"; cuobjdump -res-usage $f 2>/dev/null; done > $work/resusage.txt

python3 - "$work" "$ARCH" "$QUIET" "$WAIVERS" "$KNOWN" <<'EOF'
import re, subprocess, sys, os
work, arch_show, quiet, wfile, kfile = sys.argv[1], sys.argv[2], sys.argv[3] == "1", sys.argv[4], sys.argv[5]

def demangle(names):
    names = sorted(set(names))
    if not names: return {}
    # nvcc prefixes file-static kernels with __nv_static_<n>__<hash>__; strip it for c++filt
    # ptxas names a cloned device function <mangled>$<n>: strip the suffix too (else the waiver regexes never match it)
    clean = [re.sub(r"\$\d+$", "", re.sub(r"^__nv_static_\w+?_(_Z)", r"\1", n)) for n in names]
    out = subprocess.run(["c++filt"], input="\n".join(clean), capture_output=True, text=True).stdout.split("\n")
    return dict(zip(names, out))

def short(d):
    m = re.search(r"gpuKernel<([^,<]+(?:<[^>]*>)?)", d)
    s = m.group(1) if m else d
    s = s.replace("alpaka_cuda_async::", "").replace("mkfitdev::", "").replace("(anonymous namespace)::", "")
    return re.sub(r"\(.*$", "", s)[:90]

# spills
rows = []
for line in open(os.path.join(work, "spills.txt")):
    p = line.split()
    if len(p) == 4: rows.append((p[0], int(p[1]), int(p[2]), p[3]))
dm = demangle([r[3] for r in rows])
ours = [r for r in rows if "mkfitdev" in dm[r[3]] or "mkfitdev" in r[3]]
waivers = []
if os.path.exists(wfile):
    for l in open(wfile):
        l = l.strip()
        if not l or l.startswith("#"): continue
        k, a, why = (l.split(None, 2) + ["", ""])[:3]
        waivers.append((re.compile(k), re.compile(a), why))
known = []
if os.path.exists(kfile):
    for l in open(kfile):
        l = l.strip()
        if not l or l.startswith("#"): continue
        k, a, cost, why = (l.split(None, 3) + ["", "", ""])[:4]
        known.append((re.compile(k), re.compile(a), cost, why, k))
agg = {}
for a, st, ld, m in ours:
    key = (short(dm[m]), a)
    s0, l0 = agg.get(key, (0, 0))
    agg[key] = (max(s0, st), max(l0, ld))
bad, waived, kept, hit = [], [], [], set()
for (k, a), (st, ld) in sorted(agg.items()):
    kn = next((x for x in known if x[0].search(k) and x[1].fullmatch(a)), None)
    if kn:
        hit.add(kn[4]); kept.append((k, a, st, ld, kn[2], kn[3])); continue
    w = next((w for w in waivers if w[0].search(k) and w[1].fullmatch(a)), None)
    (waived if w else bad).append((k, a, st, ld, w[2] if w else ""))
print("[spills] spilling functions of the package (kernel, arch, max spill stores / loads in bytes):")
for k, a, st, ld, why in bad:
    print("[spills] SPILL  %-90s %-7s %5d / %5d" % (k, a, st, ld))
for k, a, st, ld, why in waived:
    print("[spills] waived %-90s %-7s %5d / %5d  (%s)" % (k, a, st, ld, why))
if not agg: print("[spills] none")
print("[spills] KNOWN SPILLS (kept by decision; counted, never a PASS): %d (kernel, arch) pairs in %d kernels" % (
    len(kept), len({x[0] for x in kept})))
for k, a, st, ld, cost, why in kept:
    print("[spills] KNOWN  %-90s %-7s %5d / %5d  device %s  (%s)" % (k, a, st, ld, cost, why))
for x in known:
    if x[4] not in hit: print("[spills] KNOWN entry without a spill now (remove it from %s): %s" % (os.path.basename(kfile), x[4]))

# info table
res = []
f = arch = fn = None
for line in open(os.path.join(work, "resusage.txt")):
    if line.startswith("#FILE "): f = line.split()[1]; continue
    m = re.match(r"\s*arch = (sm_\d+)", line)
    if m: arch = m.group(1); continue
    m = re.match(r"\s*Function (\S+):", line)
    if m: fn = m.group(1); continue
    if "REG:" in line and fn:
        d = dict(re.findall(r"(\w+(?:\[\d+\])?):(\d+)", line))
        res.append((f, arch, fn, int(d.get("REG", 0)), int(d.get("STACK", 0)), int(d.get("SHARED", 0))))
        fn = None
rdm = demangle([r[2] for r in res])
shown = [r for r in res if r[1] == arch_show and not rdm[r[2]].startswith("__cuda")]
if not quiet:
    print("%-40s %-70s %s" % ("file", "kernel (" + arch_show + ")", "REG / STACK bytes / SHARED bytes (information)"))
    seen = set()
    for r in shown:
        key = (r[0], short(rdm[r[2]]), r[3], r[4], r[5])
        if key in seen: continue
        seen.add(key)
        print("%-40s %-70s REG:%d STACK:%d SHARED:%d" % (r[0][:40], short(rdm[r[2]])[:70], r[3], r[4], r[5]))
mx = max(shown, key=lambda r: r[3]) if shown else None
archs = sorted({r[1] for r in res if r[1]})
print("[registers] %d kernels x %d archs (%s); max REG on %s: %s; STACK > 0 on %s: %d (information)" % (
    len({(r[0], r[2]) for r in res}), len(archs), ",".join(archs), arch_show,
    ("%d (%s)" % (mx[3], short(rdm[mx[2]]))) if mx else "-", arch_show, len({(r[0], r[2]) for r in shown if r[4] > 0})))
print("[spills] %d spilling (kernel, arch) pairs: %d unwaived, %d known, %d waived -> %s" % (
    len(agg), len(bad), len(kept), len(waived), "FAIL" if bad else ("KNOWN-SPILLS" if kept else "PASS")))
sys.exit(1 if bad else (3 if kept else 0))
EOF
