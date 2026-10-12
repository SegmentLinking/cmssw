#ifdef LST_TRANSFORMER

// libtorch before any ROOT header: ROOT's ClassDef macro breaks torch::jit::ClassDef
#include <ATen/Parallel.h>
#include <torch/cuda.h>
#include <torch/script.h>

#include "TransformerModel.h"

#include "AnalysisConfig.h"
#include "TStopwatch.h"

#include <algorithm>
#include <iostream>
#include <memory>
#include <numeric>
#include <stdexcept>

namespace {
#if defined(ALPAKA_ACC_GPU_CUDA_ENABLED)
  const torch::Device kTorchDevice(torch::kCUDA);
#else
  const torch::Device kTorchDevice(torch::kCPU);
#endif

  std::unique_ptr<torch::jit::Module> model;
}  // namespace

//___________________________________________________________________________________________________________________________________________________________________________________________
void loadTransformerModel(std::string const& path) {
  if (kTorchDevice.is_cuda() and not torch::cuda::is_available())
    throw std::runtime_error("loadTransformerModel: libtorch does not see a CUDA device");
  model = std::make_unique<torch::jit::Module>(torch::jit::load(path, kTorchDevice));
  model->eval();
  std::cout << "Loaded transformer model " << path << " on " << kTorchDevice << " (libtorch, " << at::get_num_threads()
            << " CPU threads)" << std::endl;
}

//___________________________________________________________________________________________________________________________________________________________________________________________
float runTransformer(LSTEvent* event, TransformerOutput& out) {
  if (not model)
    throw std::runtime_error("runTransformer: call loadTransformerModel first");
  TStopwatch my_timer;
  if (ana.verbose >= 2)
    std::cout << "Reco transformer start" << std::endl;
  my_timer.Start();

  // v1: features on host -> torch device -> forward(x_raw) -> host
  auto const features = event->getT3Features();
  unsigned int const nT3s = features.nT3s();
  out.nT3s = nT3s;
  out.tripletIndex.assign(features.tripletIndex().begin(), features.tripletIndex().begin() + nT3s);
  out.beta.clear();
  out.latent.clear();
  if (nT3s > 0) {
    torch::NoGradGuard noGrad;
    // rows are contiguous [nT3s, 49]; the model scales a clone, so the LST buffer is not modified
    auto x = torch::from_blob(const_cast<float*>(features.features()[0].data()),
                              {static_cast<int64_t>(nT3s), static_cast<int64_t>(lst::t3features::kN)},
                              torch::kFloat32)
                 .to(kTorchDevice);
    auto result = model->forward({x}).toTuple();
    auto beta = result->elements()[0].toTensor().reshape({-1}).to(torch::kCPU).contiguous();
    auto latent = result->elements()[1].toTensor().to(torch::kCPU).contiguous();
    out.latentDim = latent.size(1);
    out.beta.assign(beta.data_ptr<float>(), beta.data_ptr<float>() + nT3s);
    out.latent.assign(latent.data_ptr<float>(), latent.data_ptr<float>() + nT3s * out.latentDim);
  }
  float transformer_elapsed = my_timer.RealTime();

  if (ana.verbose >= 2) {
    std::cout << "Reco transformer processing time: " << transformer_elapsed << " secs" << std::endl;
    unsigned int nAbove = 0;
    for (float b : out.beta)
      nAbove += b > 0.005f;  // beta_threshold of the DBSCAN prefilter in transformer-oc
    std::cout << "# of T3s through the transformer: " << nT3s << " (beta > 0.005: " << nAbove
              << ", latent dim " << out.latentDim << ")" << std::endl;
  }

  return transformer_elapsed;
}

//___________________________________________________________________________________________________________________________________________________________________________________________
float runTransformerClustering(TransformerOutput& out, double eps, float betaThreshold) {
  TStopwatch my_timer;
  if (ana.verbose >= 2)
    std::cout << "Reco transformer clustering start" << std::endl;
  my_timer.Start();

  unsigned int const nT3s = out.nT3s;
  unsigned int const dim = out.latentDim;
  out.nClusters = 0;
  out.clusterLabel.assign(nT3s, -1);

  // beta prefilter: only these rows enter DBSCAN
  std::vector<int64_t> selected;
  for (unsigned int i = 0; i < nT3s; ++i)
    if (out.beta[i] > betaThreshold)
      selected.push_back(i);
  int64_t const nSelected = selected.size();

  // With min_samples = 2 (the point itself counts) every point with a neighbour within eps is a core point, so
  // there are no border points and the clusters are the connected components of the eps-graph with >= 2 points.
  // v1: candidate pairs from squared distances in float32 on the torch device, with a margin; the decision
  // is taken on the host in double precision; components with a union-find whose root is the smallest row.
  std::vector<int64_t> parent(nSelected);
  std::iota(parent.begin(), parent.end(), 0);
  auto find = [&parent](int64_t a) {
    while (parent[a] != a) {
      parent[a] = parent[parent[a]];
      a = parent[a];
    }
    return a;
  };

  if (nSelected > 1) {
    torch::NoGradGuard noGrad;
    double const eps2 = eps * eps;
    float const candidateMax = eps2 + 1e-4;  // float32 rounding of the squared distances is O(1e-6)
    int64_t const chunk = 1024;

    auto x = torch::from_blob(out.latent.data(), {static_cast<int64_t>(nT3s), static_cast<int64_t>(dim)}, torch::kFloat32)
                 .index_select(0, torch::from_blob(selected.data(), {nSelected}, torch::kInt64))
                 .to(kTorchDevice);
    auto norm2 = (x * x).sum(1);
    for (int64_t start = 0; start < nSelected; start += chunk) {
      int64_t const stop = std::min(start + chunk, nSelected);
      // rows [start, stop) against rows [start, nSelected): every pair is seen once
      auto d2 = norm2.slice(0, start, stop).unsqueeze(1) + norm2.slice(0, start).unsqueeze(0) -
                2 * torch::mm(x.slice(0, start, stop), x.slice(0, start).t());
      auto candidates = torch::nonzero(d2 <= candidateMax).to(torch::kCPU).contiguous();
      int64_t const* pairs = candidates.data_ptr<int64_t>();
      for (int64_t k = 0; k < candidates.size(0); ++k) {
        int64_t const a = start + pairs[2 * k];
        int64_t const b = start + pairs[2 * k + 1];
        if (b <= a)
          continue;
        int64_t const rootA = find(a);
        int64_t const rootB = find(b);
        if (rootA == rootB)
          continue;
        float const* va = &out.latent[selected[a] * dim];
        float const* vb = &out.latent[selected[b] * dim];
        double dist2 = 0;
        for (unsigned int c = 0; c < dim; ++c) {
          double const diff = static_cast<double>(va[c]) - static_cast<double>(vb[c]);
          dist2 += diff * diff;
        }
        if (dist2 <= eps2)
          parent[std::max(rootA, rootB)] = std::min(rootA, rootB);
      }
    }
  }

  // sklearn numbers the clusters in the order of their first point
  std::vector<int> size(nSelected, 0);
  for (int64_t a = 0; a < nSelected; ++a)
    size[find(a)]++;
  std::vector<int> label(nSelected, -1);
  unsigned int nClustered = 0;
  for (int64_t a = 0; a < nSelected; ++a) {
    int64_t const root = find(a);
    if (size[root] < 2)
      continue;
    if (label[root] < 0)
      label[root] = out.nClusters++;
    out.clusterLabel[selected[a]] = label[root];
    nClustered++;
  }
  float clustering_elapsed = my_timer.RealTime();

  if (ana.verbose >= 2) {
    std::cout << "Reco transformer clustering processing time: " << clustering_elapsed << " secs" << std::endl;
    std::cout << "# of transformer clusters: " << out.nClusters << " (" << nClustered << " of " << nSelected
              << " T3s with beta > " << betaThreshold << ", eps " << eps << ")" << std::endl;
  }

  return clustering_elapsed;
}

//___________________________________________________________________________________________________________________________________________________________________________________________
void dumpTransformerOutput(TransformerOutput const& out, int evt, std::ofstream& file) {
  file.write(reinterpret_cast<char const*>(&evt), sizeof(evt));
  file.write(reinterpret_cast<char const*>(&out.nT3s), sizeof(out.nT3s));
  file.write(reinterpret_cast<char const*>(&out.latentDim), sizeof(out.latentDim));
  file.write(reinterpret_cast<char const*>(out.beta.data()), sizeof(float) * out.beta.size());
  file.write(reinterpret_cast<char const*>(out.latent.data()), sizeof(float) * out.latent.size());
  file.write(reinterpret_cast<char const*>(out.clusterLabel.data()), sizeof(int) * out.clusterLabel.size());
}

#endif  // LST_TRANSFORMER
