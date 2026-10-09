#ifdef LST_TRANSFORMER

// libtorch before any ROOT header: ROOT's ClassDef macro breaks torch::jit::ClassDef
#include <ATen/Parallel.h>
#include <torch/cuda.h>
#include <torch/script.h>

#include "TransformerModel.h"

#include "AnalysisConfig.h"
#include "TStopwatch.h"

#include <iostream>
#include <memory>
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
void dumpTransformerOutput(TransformerOutput const& out, int evt, std::ofstream& file) {
  file.write(reinterpret_cast<char const*>(&evt), sizeof(evt));
  file.write(reinterpret_cast<char const*>(&out.nT3s), sizeof(out.nT3s));
  file.write(reinterpret_cast<char const*>(&out.latentDim), sizeof(out.latentDim));
  file.write(reinterpret_cast<char const*>(out.beta.data()), sizeof(float) * out.beta.size());
  file.write(reinterpret_cast<char const*>(out.latent.data()), sizeof(float) * out.latent.size());
}

#endif  // LST_TRANSFORMER
