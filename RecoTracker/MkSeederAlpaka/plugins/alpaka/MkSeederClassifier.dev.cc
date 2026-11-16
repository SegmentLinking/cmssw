#include <bit>
#include <cstdint>

#include <alpaka/alpaka.hpp>

#include "HeterogeneousCore/AlpakaInterface/interface/config.h"
#include "HeterogeneousCore/AlpakaInterface/interface/workdivision.h"

#include "MkSeederBackend.h"
#include "MkSeederClassifier.h"

namespace ALPAKA_ACCELERATOR_NAMESPACE::mkseeder {

  namespace {
    // a NaN by its bits (as xgboost's missing value; the host's isNaN)
    ALPAKA_FN_HOST_ACC inline bool isNaN(float v) { return (std::bit_cast<uint32_t>(v) & 0x7fffffffu) > 0x7f800000u; }

    // GPU: one thread per seed, the trees in order. CPU backends: the seeds of a thread in blocks of kBlock, the trees in
    // order and per tree the seeds of the block (the tree's nodes stay in L1, the seeds' independent paths overlap); the
    // same comparisons and the same double sums in the same order per seed, so the same scores
    class KernelClassifySeeds {
    public:
      static constexpr int kBlock = 16;
      template <typename TAcc>
      ALPAKA_FN_ACC void operator()(const TAcc &acc,
                                    ::mkseeder::MkSeederSeedFeaturesSoA::ConstView features,
                                    const uint32_t *count,
                                    ::mkseeder::MkSeederForestSoA::ConstView forest,
                                    SeedClassifierThresholds thresholds,
                                    ::mkseeder::MkSeederSeedScoresSoA::View scores) const {
        using namespace ::mkseeder::seedfeature;
        const uint32_t rows = scores.metadata().size();
        const uint32_t n = alpaka::math::min(acc, *count, rows);
        const auto nodes = forest.nodes();
        const auto leaves = forest.leaves();
        const uint32_t nTrees = nodes.nTrees();
        const int depth = nodes.depth();
        const uint32_t nSplits = (1u << depth) - 1, nLeaves = 1u << depth;
        const double baseMargin = nodes.baseMargin();
        // the model's input f of seed i (the z mirror image with zFold)
        auto zSign = [&](uint32_t i) {
          const auto inputs = features.inputs(i);
          return thresholds.zFold && inputs(kLastHitZ) < inputs(kFirstHitZ) ? -1.f : 1.f;
        };
        auto finish = [&](uint32_t i, double margin) {
          const auto inputs = features.inputs(i);
          const float score = 1. / (1. + alpaka::math::exp(acc, -margin));
          double cut = thresholds.threshold;
          if (thresholds.tripletThreshold >= 0 && inputs(kNHits) == 3.f)
            cut = thresholds.tripletThreshold;
          else if (thresholds.nEtaEdges > 0) {
            int k = 0;
            while (k < thresholds.nEtaEdges && inputs(kAbsEta) >= thresholds.etaEdges[k])
              ++k;
            cut = thresholds.etaThresholds[k];
          }
          scores.score(i) = score;
          scores.isSelected(i) = !features.isDropped(i) && score >= cut;
        };
        if constexpr (kCpuAcc<TAcc>) {
          uint32_t blk[kBlock];
          int nb = 0;
          float x[kBlock][kNFeatures];
          auto flush = [&]() {
            double margin[kBlock];
            for (int b = 0; b < nb; ++b) {
              const auto inputs = features.inputs(blk[b]);
              const float zs = zSign(blk[b]);
              for (int f = 0; f < kNFeatures; ++f)
                x[b][f] = (f == kDz || f == kFirstHitZ || f == kLastHitZ) ? zs * inputs(f) : inputs(f);
              margin[b] = baseMargin;
            }
            const float *thr = nodes.threshold().data();
            const uint32_t *feat = nodes.feature().data();
            const uint8_t *nanRight = nodes.nanGoesRight().data();
            const float *leaf = leaves.value().data();
            for (uint32_t t = 0; t < nTrees; ++t) {
              const uint32_t first = t * nSplits;
              uint32_t h[kBlock];
              for (int b = 0; b < nb; ++b)
                h[b] = 0;
              for (int level = 0; level < depth; ++level)
                for (int b = 0; b < nb; ++b) {
                  const uint32_t node = first + h[b];
                  const float v = x[b][feat[node]];
                  const uint32_t right = isNaN(v) ? nanRight[node] : uint32_t(!(v < thr[node]));
                  h[b] = 2 * h[b] + 1 + right;
                }
              for (int b = 0; b < nb; ++b)
                margin[b] += static_cast<double>(leaf[t * nLeaves + h[b] - nSplits]);
            }
            for (int b = 0; b < nb; ++b)
              finish(blk[b], margin[b]);
            nb = 0;
          };
          for (uint32_t i : cms::alpakatools::uniform_elements(acc, rows)) {
            if (i >= n) {
              scores.score(i) = 0.f, scores.isSelected(i) = 0;
              continue;
            }
            blk[nb++] = i;
            if (nb == kBlock)
              flush();
          }
          flush();
        } else {
          for (uint32_t i : cms::alpakatools::uniform_elements(acc, rows)) {
            if (i >= n) {
              scores.score(i) = 0.f, scores.isSelected(i) = 0;
              continue;
            }
            const auto inputs = features.inputs(i);
            const float zs = zSign(i);
            double margin = baseMargin;
            for (uint32_t t = 0; t < nTrees; ++t) {
              const uint32_t first = t * nSplits;
              uint32_t h = 0;
              for (int level = 0; level < depth; ++level) {
                const uint32_t node = first + h;
                const int f = nodes.feature(node);
                const float x = (f == kDz || f == kFirstHitZ || f == kLastHitZ) ? zs * inputs(f) : inputs(f);
                const uint32_t right = isNaN(x) ? nodes.nanGoesRight(node) : uint32_t(!(x < nodes.threshold(node)));
                h = 2 * h + 1 + right;
              }
              margin += static_cast<double>(leaves.value(t * nLeaves + h - nSplits));
            }
            finish(i, margin);
          }
        }
      }
    };
  }  // namespace

  void classifySeeds(Queue &queue,
                     ::mkseeder::MkSeederSeedFeaturesSoA::ConstView features,
                     const uint32_t *count,
                     ::mkseeder::MkSeederForestSoA::ConstView forest,
                     const SeedClassifierThresholds &thresholds,
                     ::mkseeder::MkSeederSeedScoresSoA::View scores) {
    const uint32_t rows = scores.metadata().size();
    if (rows == 0)
      return;
    constexpr uint32_t kThreads = 128;
    const auto workDiv =
        cms::alpakatools::make_workdiv<Acc1D>(cms::alpakatools::divide_up_by(rows, kThreads), kThreads);
    alpaka::exec<Acc1D>(queue, workDiv, KernelClassifySeeds{}, features, count, forest, thresholds, scores);
  }

}  // namespace ALPAKA_ACCELERATOR_NAMESPACE::mkseeder
