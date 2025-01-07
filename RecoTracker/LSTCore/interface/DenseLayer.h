#ifndef RecoTracker_LSTCore_interface_DenseLayer_h
#define RecoTracker_LSTCore_interface_DenseLayer_h

#include <array>
#include <cstddef>
#include <cstdint>

/**
 * Represents a dense (fully connected) layer with fixed input and output sizes.
 *
 * IN:  Number of input neurons
 * OUT: Number of output neurons
 */
template <std::size_t IN, std::size_t OUT>
struct DenseLayer {
  /**
   * Biases: one float per output neuron.
   */
  std::array<float, OUT> biases{};

  /**
   * Weights: stored as IN rows of OUT columns.
   */
  std::array<std::array<float, OUT>, IN> weights{};

  /**
   * Returns the weight from input neuron index `in` to output neuron index `out`.
   */
  float getWeight(std::size_t in, std::size_t out) const { return weights[in][out]; }

  static constexpr std::size_t inputSize = IN;
  static constexpr std::size_t outputSize = OUT;
};

#endif