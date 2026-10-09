#ifndef TransformerModel_h
#define TransformerModel_h

// Transformer (object condensation) inference on LST T3s with libtorch: only in builds with the transformer
// flag (lst_make_tracklooper -T). libtorch headers are kept out of this header.
#ifdef LST_TRANSFORMER

#include "LSTEvent.h"

#include <fstream>
#include <string>
#include <vector>

using LSTEvent = ALPAKA_ACCELERATOR_NAMESPACE::lst::LSTEvent;

// Model outputs for the T3 feature rows of one event (v1: host copies)
struct TransformerOutput {
  unsigned int nT3s = 0;
  unsigned int latentDim = 0;
  std::vector<unsigned int> tripletIndex;  // index into the Triplets collection, per row
  std::vector<float> beta;                 // [nT3s]
  std::vector<float> latent;               // [nT3s, latentDim], L2-normalised
};

// Load the exported TorchScript model (transformer-oc export/export_torchscript.py) once per job, on the device
// of the backend (CUDA for lst_cuda, CPU otherwise).
void loadTransformerModel(std::string const& path);

// Run the model on the event's T3 features (call after createT3Features). Returns the elapsed time.
float runTransformer(LSTEvent* event, TransformerOutput& out);

// Binary dump per event: int32 evt, uint32 nT3s, uint32 latentDim, float32 beta[nT3s], float32 latent[nT3s][latentDim]
void dumpTransformerOutput(TransformerOutput const& out, int evt, std::ofstream& file);

#endif  // LST_TRANSFORMER

#endif
