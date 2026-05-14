#pragma once

#include <array>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace stage1 {

constexpr std::array<char, 8> kRankGatePayloadMagic = {'F', 'H', 'M', '3', 'R', 'G', 'A', 'T'};
constexpr std::uint32_t kRankGatePayloadFormatVersion = 5;

struct RankGatePayloadConfig {
  std::uint32_t d_model = 0;
  std::uint32_t d_model_pad = 0;
  std::uint32_t mimo_rank = 0;
  std::uint32_t rank_pad = 0;
  std::uint32_t d_state = 0;
  std::uint32_t model_baby_step = 0;
  std::uint32_t rank_baby_step = 0;
  std::uint32_t layer_index = 0;
  std::uint32_t prompt_token = 0;
  double norm_eps = 0.0;
};

struct RankGatePayloadArray {
  std::string name;
  std::vector<std::uint64_t> shape;
  std::vector<double> values;
};

struct RankGatePayload {
  RankGatePayloadConfig config;
  std::vector<RankGatePayloadArray> arrays;

  [[nodiscard]] const RankGatePayloadArray& array(const std::string& name) const {
    for (const auto& item : arrays) {
      if (item.name == name) {
        return item;
      }
    }
    throw std::runtime_error("rank/gate payload missing array: " + name);
  }
};

inline auto rank_gate_payload_array_order() -> const std::vector<std::string>& {
  static const std::vector<std::string> names = {
      "rms_input",
      "effective_rank_weight",
      "conv_bias",
      "gate_weight",
      "d_skip",
      "reference_conv_pre",
      "reference_rank_input",
      "reference_gate_pre",
      "reference_gate",
      "reference_skip_update",
      "rank_silu_coefficients",
      "gate_silu_coefficients",
      "reference_rank_input_poly",
      "reference_gate_poly",
      "reference_skip_update_poly",
      "b_weight",
      "c_weight",
      "reference_b_vec_poly",
      "reference_c_vec_poly",
      "reference_b_state_major_poly",
      "reference_c_state_major_poly",
      "dt_in_weight",
      "dt_proj_weight",
      "dt_proj_bias",
      "reference_dt_hidden_poly",
      "reference_dt_pre_poly",
      "reference_dt_state_major_poly",
      "decay_coefficients",
      "reference_decay_state_major_poly",
      "reference_decay_state_major_exact",
      "decay_metadata",
      "residual_input",
      "previous_state",
      "w_out",
      "reference_state_new_poly",
      "reference_readout_rank_poly",
      "reference_rank_output_poly",
      "reference_rank_payload_poly",
      "reference_output_model_poly",
      "reference_output_model_exact",
      "tail_metadata",
      "polynomial_metadata",
  };
  return names;
}

inline void require_rank_gate_divisible(
    std::uint64_t value,
    std::uint64_t divisor,
    const std::string& name) {
  if (divisor == 0 || value % divisor != 0) {
    throw std::runtime_error("rank/gate payload length is not divisible for " + name);
  }
}

inline auto rank_gate_payload_expected_shape(
    const RankGatePayloadConfig& config,
    const std::string& name,
    std::uint64_t encoded_length = 0) -> std::vector<std::uint64_t> {
  if (name == "rms_input") {
    return {config.d_model};
  }
  if (name == "effective_rank_weight" || name == "gate_weight") {
    return {config.mimo_rank, config.d_model};
  }
  if (name == "conv_bias" || name == "d_skip" || name == "reference_conv_pre" ||
      name == "reference_rank_input" || name == "reference_gate_pre" ||
      name == "reference_gate" || name == "reference_skip_update" ||
      name == "reference_rank_input_poly" || name == "reference_gate_poly" ||
      name == "reference_skip_update_poly") {
    return {config.mimo_rank};
  }
  if (name == "b_weight" || name == "c_weight") {
    return {config.d_state, config.mimo_rank};
  }
  if (name == "reference_b_vec_poly" || name == "reference_c_vec_poly") {
    return {config.d_state};
  }
  if (name == "reference_b_state_major_poly" || name == "reference_c_state_major_poly") {
    return {config.d_state, config.mimo_rank};
  }
  if (name == "dt_in_weight") {
    if (encoded_length == 0) {
      return {1, config.mimo_rank};
    }
    require_rank_gate_divisible(encoded_length, config.mimo_rank, name);
    return {encoded_length / config.mimo_rank, config.mimo_rank};
  }
  if (name == "dt_proj_weight") {
    if (encoded_length == 0) {
      return {config.mimo_rank, 1};
    }
    require_rank_gate_divisible(encoded_length, config.mimo_rank, name);
    return {config.mimo_rank, encoded_length / config.mimo_rank};
  }
  if (name == "dt_proj_bias" || name == "reference_dt_pre_poly") {
    return {config.mimo_rank};
  }
  if (name == "reference_dt_hidden_poly") {
    return {encoded_length == 0 ? 1 : encoded_length};
  }
  if (name == "reference_dt_state_major_poly" ||
      name == "reference_decay_state_major_poly" ||
      name == "reference_decay_state_major_exact") {
    return {config.d_state, config.mimo_rank};
  }
  if (name == "decay_coefficients") {
    const auto state_rank = static_cast<std::uint64_t>(config.d_state) * config.mimo_rank;
    if (encoded_length == 0) {
      return {1, config.d_state, config.mimo_rank};
    }
    require_rank_gate_divisible(encoded_length, state_rank, name);
    return {encoded_length / state_rank, config.d_state, config.mimo_rank};
  }
  if (name == "decay_metadata") {
    return {4};
  }
  if (name == "residual_input" || name == "reference_output_model_poly" ||
      name == "reference_output_model_exact") {
    return {config.d_model};
  }
  if (name == "previous_state" || name == "reference_state_new_poly") {
    return {config.d_state, config.mimo_rank};
  }
  if (name == "w_out") {
    return {config.d_model, config.mimo_rank};
  }
  if (name == "reference_readout_rank_poly" || name == "reference_rank_output_poly" ||
      name == "reference_rank_payload_poly") {
    return {config.mimo_rank};
  }
  if (name == "tail_metadata") {
    return {2};
  }
  if (name == "rank_silu_coefficients" || name == "gate_silu_coefficients") {
    return {encoded_length == 0 ? 1 : encoded_length};
  }
  if (name == "polynomial_metadata") {
    return {3};
  }
  throw std::runtime_error("unknown rank/gate payload array: " + name);
}

inline auto rank_gate_shape_size(const std::vector<std::uint64_t>& shape) -> std::uint64_t {
  std::uint64_t total = 1;
  for (const auto value : shape) {
    total *= value;
  }
  return total;
}

template <typename T>
inline auto read_rank_gate_scalar(std::istream& input, const std::string& name) -> T {
  T value{};
  input.read(reinterpret_cast<char*>(&value), sizeof(T));
  if (!input) {
    throw std::runtime_error("truncated rank/gate payload while reading " + name);
  }
  return value;
}

inline auto read_rank_gate_payload(const std::string& path) -> RankGatePayload {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw std::runtime_error("failed to open rank/gate payload: " + path);
  }

  std::array<char, 8> magic{};
  input.read(magic.data(), static_cast<std::streamsize>(magic.size()));
  if (!input || magic != kRankGatePayloadMagic) {
    throw std::runtime_error("invalid rank/gate payload magic");
  }

  const auto version = read_rank_gate_scalar<std::uint32_t>(input, "format_version");
  if (version != kRankGatePayloadFormatVersion) {
    throw std::runtime_error("unsupported rank/gate payload format version");
  }

  RankGatePayload payload;
  payload.config.d_model = read_rank_gate_scalar<std::uint32_t>(input, "d_model");
  payload.config.d_model_pad = read_rank_gate_scalar<std::uint32_t>(input, "d_model_pad");
  payload.config.mimo_rank = read_rank_gate_scalar<std::uint32_t>(input, "mimo_rank");
  payload.config.rank_pad = read_rank_gate_scalar<std::uint32_t>(input, "rank_pad");
  payload.config.d_state = read_rank_gate_scalar<std::uint32_t>(input, "d_state");
  payload.config.model_baby_step =
      read_rank_gate_scalar<std::uint32_t>(input, "model_baby_step");
  payload.config.rank_baby_step =
      read_rank_gate_scalar<std::uint32_t>(input, "rank_baby_step");
  payload.config.layer_index = read_rank_gate_scalar<std::uint32_t>(input, "layer_index");
  payload.config.prompt_token = read_rank_gate_scalar<std::uint32_t>(input, "prompt_token");
  payload.config.norm_eps = read_rank_gate_scalar<double>(input, "norm_eps");

  const auto array_count = read_rank_gate_scalar<std::uint32_t>(input, "array_count");
  if (array_count != rank_gate_payload_array_order().size()) {
    throw std::runtime_error("rank/gate payload array count mismatch");
  }
  payload.arrays.reserve(array_count);
  for (const auto& name : rank_gate_payload_array_order()) {
    const auto length = read_rank_gate_scalar<std::uint64_t>(input, name + "_length");
    auto shape = rank_gate_payload_expected_shape(payload.config, name, length);
    const auto expected_length = rank_gate_shape_size(shape);
    if (length != expected_length) {
      throw std::runtime_error("rank/gate payload length mismatch for " + name);
    }
    std::vector<double> values(length);
    input.read(
        reinterpret_cast<char*>(values.data()),
        static_cast<std::streamsize>(length * sizeof(double)));
    if (!input) {
      throw std::runtime_error("truncated rank/gate payload data for " + name);
    }
    payload.arrays.push_back(RankGatePayloadArray{
        .name = name,
        .shape = std::move(shape),
        .values = std::move(values)});
  }

  char trailing = 0;
  if (input.read(&trailing, 1)) {
    throw std::runtime_error("rank/gate payload has trailing bytes");
  }
  return payload;
}

}  // namespace stage1
