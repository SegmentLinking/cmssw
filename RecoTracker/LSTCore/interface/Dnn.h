#ifndef RecoTracker_LSTCore_interface_Dnn_h
#define RecoTracker_LSTCore_interface_Dnn_h

#include <tuple>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <type_traits>
#include <utility>

/**
 * A general Dnn class that holds a sequence (tuple) of DenseLayer<T> types,
 * each with compile-time fixed dimensions.
 *
 * Layers: A parameter pack of layer types (e.g. DenseLayer<23,32>, DenseLayer<32,1>, etc.)
 */
template <class... Layers>
class Dnn {
public:
  Dnn() = default;
  explicit Dnn(const std::string& filename) { load(filename); }

  /**
   * Loads biases and weights for each layer in the tuple from a binary file.
   */
  void load(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file) {
      throw std::runtime_error("Failed to open file: " + filename);
    }

    loadLayers<0>(file);

    if (!file.good()) {
      throw std::runtime_error("Error reading from file: " + filename);
    }
    file.close();
  }

  /**
   * Prints the biases and weights of each layer to stdout.
   */
  void print() const { printLayers<0>(); }

  /**
   * A const reference to the underlying tuple of layers.
   */
  const std::tuple<Layers...>& getLayers() const { return layers_; }

  /**
   * A reference to the underlying tuple of layers.
   */
  std::tuple<Layers...>& getLayers() { return layers_; }

private:
  // Store all layers in a compile-time tuple
  std::tuple<Layers...> layers_;

  /**
   * Internal compile-time recursion for loading each layer from file
   */
  template <std::size_t I>
  typename std::enable_if<I == sizeof...(Layers), void>::type loadLayers(std::ifstream&) {
    // Base case: no more layers to load
  }

  template <std::size_t I>
      typename std::enable_if < I<sizeof...(Layers), void>::type loadLayers(std::ifstream& file) {
    auto& layer = std::get<I>(layers_);

    // Read and verify header information
    uint32_t layer_id, num_inputs, num_outputs;
    file.read(reinterpret_cast<char*>(&layer_id), sizeof(layer_id));
    file.read(reinterpret_cast<char*>(&num_inputs), sizeof(num_inputs));
    file.read(reinterpret_cast<char*>(&num_outputs), sizeof(num_outputs));

    // Verify the dimensions match our template parameters
    if (num_inputs != layer.inputSize || num_outputs != layer.outputSize) {
      throw std::runtime_error("Layer " + std::to_string(I) +
                               " dimension mismatch: "
                               "expected " +
                               std::to_string(layer.inputSize) + "x" + std::to_string(layer.outputSize) + ", got " +
                               std::to_string(num_inputs) + "x" + std::to_string(num_outputs));
    }

    // Verify layer index matches
    if (layer_id != I + 1) {  // Assumes 1-based layer IDs
      throw std::runtime_error("Layer index mismatch: expected " + std::to_string(I + 1) + ", got " +
                               std::to_string(layer_id));
    }

    // Read biases
    file.read(reinterpret_cast<char*>(layer.biases.data()), layer.biases.size() * sizeof(float));

    // Read weights row by row
    for (auto& row : layer.weights) {
      file.read(reinterpret_cast<char*>(row.data()), row.size() * sizeof(float));
    }

    if (!file.good()) {
      throw std::runtime_error("Failed to read parameters for layer " + std::to_string(I));
    }

    // Recurse to next layer
    loadLayers<I + 1>(file);
  }

  /**
   * Internal compile-time recursion for printing each layer
   */
  template <std::size_t I>
  typename std::enable_if<I == sizeof...(Layers), void>::type printLayers() const {
    // Base case: no more layers to print
  }

  template <std::size_t I>
      typename std::enable_if < I<sizeof...(Layers), void>::type printLayers() const {
    const auto& layer = std::get<I>(layers_);
    std::cout << "\n=== Layer " << I + 1 << " ===\nInputs=" << layer.inputSize << ", Outputs=" << layer.outputSize
              << "\n\nBiases:\n";

    for (float b : layer.biases) {
      std::cout << b << " ";
    }
    std::cout << "\n\nWeights:\n";

    for (std::size_t in = 0; in < layer.inputSize; ++in) {
      std::cout << "  [ ";
      for (std::size_t out = 0; out < layer.outputSize; ++out) {
        std::cout << layer.getWeight(in, out) << " ";
      }
      std::cout << "]\n";
    }

    // Recurse to next layer
    printLayers<I + 1>();
  }
};

#endif