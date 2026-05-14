#pragma once

#include <array>
#include <cstdint>
#include <fstream>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace stage1 {

constexpr std::array<char, 8> kTailPayloadMagic = {'F', 'H', 'M', '3', 'T', 'A', 'I', 'L'};
constexpr std::uint32_t kTailPayloadFormatVersion = 1;

struct TailPayloadConfig {
  std::uint32_t d_model = 0;
  std::uint32_t d_model_pad = 0;
  std::uint32_t mimo_rank = 0;
  std::uint32_t rank_pad = 0;
  std::uint32_t d_state = 0;
  std::uint32_t model_baby_step = 0;
  std::uint32_t rank_baby_step = 0;
  std::uint32_t layer_index = 0;
  std::uint32_t prompt_token = 0;
  std::uint32_t dt_rank = 0;
  double norm_eps = 0.0;
  double previous_state_scale = 0.0;
  std::int64_t previous_state_seed = 0;
};

struct TailPayloadArray {
  std::string name;
  std::vector<std::uint64_t> shape;
  std::vector<double> values;
};

struct TailPayload {
  TailPayloadConfig config;
  std::vector<TailPayloadArray> arrays;

  [[nodiscard]] const TailPayloadArray& array(const std::string& name) const {
    for (const auto& item : arrays) {
      if (item.name == name) {
        return item;
      }
    }
    throw std::runtime_error("tail payload missing array: " + name);
  }
};

inline auto tail_payload_array_order() -> const std::vector<std::string>& {
  static const std::vector<std::string> names = {
      "residual_input",
      "rank_input",
      "gate",
      "b",
      "c",
      "decay",
      "previous_state",
      "skip_update",
      "w_out",
      "source_readout_rank",
      "source_final_output",
      "reference_state_new",
      "reference_readout_rank",
      "reference_rank_output",
      "reference_rank_payload",
      "reference_output_model",
  };
  return names;
}

inline auto tail_payload_expected_shape(
    const TailPayloadConfig& config,
    const std::string& name) -> std::vector<std::uint64_t> {
  if (name == "residual_input" || name == "source_final_output" ||
      name == "reference_output_model") {
    return {config.d_model};
  }
  if (name == "rank_input" || name == "gate" || name == "skip_update" ||
      name == "source_readout_rank" || name == "reference_readout_rank" ||
      name == "reference_rank_output" || name == "reference_rank_payload") {
    return {config.mimo_rank};
  }
  if (name == "b" || name == "c" || name == "decay" || name == "previous_state" ||
      name == "reference_state_new") {
    return {config.d_state, config.mimo_rank};
  }
  if (name == "w_out") {
    return {config.d_model, config.mimo_rank};
  }
  throw std::runtime_error("unknown tail payload array: " + name);
}

inline auto shape_size(const std::vector<std::uint64_t>& shape) -> std::uint64_t {
  std::uint64_t total = 1;
  for (const auto value : shape) {
    total *= value;
  }
  return total;
}

template <typename T>
inline auto read_scalar(std::istream& input, const std::string& name) -> T {
  T value{};
  input.read(reinterpret_cast<char*>(&value), sizeof(T));
  if (!input) {
    throw std::runtime_error("truncated tail payload while reading " + name);
  }
  return value;
}

inline auto read_tail_payload(const std::string& path) -> TailPayload {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw std::runtime_error("failed to open tail payload: " + path);
  }

  std::array<char, 8> magic{};
  input.read(magic.data(), static_cast<std::streamsize>(magic.size()));
  if (!input || magic != kTailPayloadMagic) {
    throw std::runtime_error("invalid tail payload magic");
  }

  const auto version = read_scalar<std::uint32_t>(input, "format_version");
  if (version != kTailPayloadFormatVersion) {
    throw std::runtime_error("unsupported tail payload format version");
  }

  TailPayload payload;
  payload.config.d_model = read_scalar<std::uint32_t>(input, "d_model");
  payload.config.d_model_pad = read_scalar<std::uint32_t>(input, "d_model_pad");
  payload.config.mimo_rank = read_scalar<std::uint32_t>(input, "mimo_rank");
  payload.config.rank_pad = read_scalar<std::uint32_t>(input, "rank_pad");
  payload.config.d_state = read_scalar<std::uint32_t>(input, "d_state");
  payload.config.model_baby_step = read_scalar<std::uint32_t>(input, "model_baby_step");
  payload.config.rank_baby_step = read_scalar<std::uint32_t>(input, "rank_baby_step");
  payload.config.layer_index = read_scalar<std::uint32_t>(input, "layer_index");
  payload.config.prompt_token = read_scalar<std::uint32_t>(input, "prompt_token");
  payload.config.dt_rank = read_scalar<std::uint32_t>(input, "dt_rank");
  payload.config.norm_eps = read_scalar<double>(input, "norm_eps");
  payload.config.previous_state_scale = read_scalar<double>(input, "previous_state_scale");
  payload.config.previous_state_seed =
      read_scalar<std::int64_t>(input, "previous_state_seed");

  const auto array_count = read_scalar<std::uint32_t>(input, "array_count");
  if (array_count != tail_payload_array_order().size()) {
    throw std::runtime_error("tail payload array count mismatch");
  }
  payload.arrays.reserve(array_count);
  for (const auto& name : tail_payload_array_order()) {
    const auto length = read_scalar<std::uint64_t>(input, name + "_length");
    auto shape = tail_payload_expected_shape(payload.config, name);
    const auto expected_length = shape_size(shape);
    if (length != expected_length) {
      throw std::runtime_error("tail payload length mismatch for " + name);
    }
    std::vector<double> values(length);
    input.read(
        reinterpret_cast<char*>(values.data()),
        static_cast<std::streamsize>(length * sizeof(double)));
    if (!input) {
      throw std::runtime_error("truncated tail payload data for " + name);
    }
    payload.arrays.push_back(
        TailPayloadArray{.name = name, .shape = std::move(shape), .values = std::move(values)});
  }

  char trailing = 0;
  if (input.read(&trailing, 1)) {
    throw std::runtime_error("tail payload has trailing bytes");
  }
  return payload;
}

}  // namespace stage1
