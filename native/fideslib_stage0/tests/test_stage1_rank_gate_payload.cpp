#include "stage1_rank_gate_eval.hpp"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

namespace {

template <typename T>
void write_scalar(std::ostream& output, const T& value) {
  output.write(reinterpret_cast<const char*>(&value), sizeof(T));
}

void require_true(bool condition, const char* name) {
  if (condition) {
    return;
  }
  std::cerr << name << " failed\n";
  std::exit(EXIT_FAILURE);
}

void require_equal(std::uint64_t actual, std::uint64_t expected, const char* name) {
  if (actual == expected) {
    return;
  }
  std::cerr << name << " mismatch: " << actual << " != " << expected << "\n";
  std::exit(EXIT_FAILURE);
}

void require_close(double actual, double expected, double atol, const char* name) {
  if (std::abs(actual - expected) <= atol) {
    return;
  }
  std::cerr << name << " mismatch: " << actual << " != " << expected << "\n";
  std::exit(EXIT_FAILURE);
}

void require_string_equal(const std::string& actual, const std::string& expected, const char* name) {
  if (actual == expected) {
    return;
  }
  std::cerr << name << " mismatch: " << actual << " != " << expected << "\n";
  std::exit(EXIT_FAILURE);
}

auto write_demo_payload() -> std::filesystem::path {
  const auto path = std::filesystem::current_path() / "stage1_rank_gate_payload_test.bin";
  std::ofstream output(path, std::ios::binary);
  require_true(static_cast<bool>(output), "open demo rank/gate payload");

  output.write(stage1::kRankGatePayloadMagic.data(), stage1::kRankGatePayloadMagic.size());
  write_scalar<std::uint32_t>(output, stage1::kRankGatePayloadFormatVersion);
  write_scalar<std::uint32_t>(output, 8);   // d_model
  write_scalar<std::uint32_t>(output, 8);   // d_model_pad
  write_scalar<std::uint32_t>(output, 6);   // mimo_rank
  write_scalar<std::uint32_t>(output, 8);   // rank_pad
  write_scalar<std::uint32_t>(output, 2);   // d_state
  write_scalar<std::uint32_t>(output, 64);  // model_baby_step
  write_scalar<std::uint32_t>(output, 4);   // rank_baby_step
  write_scalar<std::uint32_t>(output, 1);   // layer_index
  write_scalar<std::uint32_t>(output, 3);   // prompt_token
  write_scalar<double>(output, 1e-5);

  write_scalar<std::uint32_t>(
      output,
      static_cast<std::uint32_t>(stage1::rank_gate_payload_array_order().size()));
  double value = 0.0;
  stage1::RankGatePayloadConfig config;
  config.d_model = 8;
  config.mimo_rank = 6;
  config.d_state = 2;
  for (const auto& name : stage1::rank_gate_payload_array_order()) {
    const auto shape = stage1::rank_gate_payload_expected_shape(config, name);
    const auto length = stage1::rank_gate_shape_size(shape);
    write_scalar<std::uint64_t>(output, length);
    for (std::uint64_t index = 0; index < length; ++index) {
      value += 0.125;
      write_scalar<double>(output, value);
    }
  }
  require_true(static_cast<bool>(output), "write demo rank/gate payload");
  return path;
}

void test_read_rank_gate_payload() {
  const auto path = write_demo_payload();
  const auto payload = stage1::read_rank_gate_payload(path.string());

  require_equal(payload.config.d_model, 8, "d_model");
  require_equal(payload.config.mimo_rank, 6, "mimo_rank");
  require_equal(payload.config.d_state, 2, "d_state");
  require_equal(payload.config.layer_index, 1, "layer_index");
  require_equal(payload.config.prompt_token, 3, "prompt_token");
  require_equal(
      payload.arrays.size(),
      stage1::rank_gate_payload_array_order().size(),
      "array count");
  require_string_equal(payload.arrays.front().name, "rms_input", "first array name");
  require_equal(payload.array("effective_rank_weight").values.size(), 48, "rank weight length");
  require_equal(payload.array("gate_weight").values.size(), 48, "gate weight length");

  std::filesystem::remove(path);
}

auto make_consistent_payload() -> stage1::RankGatePayload {
  stage1::RankGatePayload payload;
  payload.config.d_model = 3;
  payload.config.d_model_pad = 4;
  payload.config.mimo_rank = 4;
  payload.config.rank_pad = 4;
  payload.config.d_state = 2;
  payload.config.model_baby_step = 2;
  payload.config.rank_baby_step = 2;

  const auto d_model = payload.config.d_model;
  const auto rank = payload.config.mimo_rank;
  std::vector<double> rms_input = {1.0, -0.5, 0.25};
  std::vector<double> effective_rank_weight = {
      0.1, -0.2, 0.3, 0.2, 0.05, -0.1, -0.3, 0.4, 0.2, 0.01, -0.02, 0.03};
  std::vector<double> conv_bias = {0.01, -0.02, 0.03, -0.04};
  std::vector<double> gate_weight = {
      -0.1, 0.3, 0.05, 0.2, -0.4, 0.1, 0.01, 0.02, -0.03, 0.3, 0.2, -0.1};
  std::vector<double> d_skip = {0.7, 0.8, -0.2, 0.5};
  std::vector<double> conv_pre(rank, 0.0);
  std::vector<double> rank_input(rank, 0.0);
  std::vector<double> gate_pre(rank, 0.0);
  std::vector<double> gate(rank, 0.0);
  std::vector<double> skip_update(rank, 0.0);
  std::vector<double> rank_coefficients = {0.0, 0.5, 0.1};
  std::vector<double> gate_coefficients = {0.0, 0.5, 0.1};
  std::vector<double> rank_input_poly(rank, 0.0);
  std::vector<double> gate_poly(rank, 0.0);
  std::vector<double> skip_update_poly(rank, 0.0);
  for (std::uint32_t rank_index = 0; rank_index < rank; ++rank_index) {
    const auto base = rank_index * d_model;
    conv_pre[rank_index] = conv_bias[rank_index];
    for (std::uint32_t model_index = 0; model_index < d_model; ++model_index) {
      conv_pre[rank_index] += effective_rank_weight[base + model_index] * rms_input[model_index];
      gate_pre[rank_index] += gate_weight[base + model_index] * rms_input[model_index];
    }
    rank_input[rank_index] = stage1::silu(conv_pre[rank_index]);
    gate[rank_index] = stage1::silu(gate_pre[rank_index]);
    skip_update[rank_index] = rank_input[rank_index] * d_skip[rank_index];
    rank_input_poly[rank_index] =
        rank_coefficients[0] + rank_coefficients[1] * conv_pre[rank_index] +
        rank_coefficients[2] * conv_pre[rank_index] * conv_pre[rank_index];
    gate_poly[rank_index] = gate_coefficients[0] + gate_coefficients[1] * gate_pre[rank_index] +
                            gate_coefficients[2] * gate_pre[rank_index] * gate_pre[rank_index];
    skip_update_poly[rank_index] = rank_input_poly[rank_index] * d_skip[rank_index];
  }

  auto push = [&](const std::string& name, std::vector<double> values) {
    std::vector<std::uint64_t> shape;
    if (name == "rank_silu_coefficients" || name == "gate_silu_coefficients") {
      shape = {values.size()};
    } else {
      shape = stage1::rank_gate_payload_expected_shape(payload.config, name);
    }
    payload.arrays.push_back(stage1::RankGatePayloadArray{
        .name = name,
        .shape = std::move(shape),
        .values = std::move(values)});
  };
  push("rms_input", rms_input);
  push("effective_rank_weight", effective_rank_weight);
  push("conv_bias", conv_bias);
  push("gate_weight", gate_weight);
  push("d_skip", d_skip);
  push("reference_conv_pre", conv_pre);
  push("reference_rank_input", rank_input);
  push("reference_gate_pre", gate_pre);
  push("reference_gate", gate);
  push("reference_skip_update", skip_update);
  push("rank_silu_coefficients", rank_coefficients);
  push("gate_silu_coefficients", gate_coefficients);
  push("reference_rank_input_poly", rank_input_poly);
  push("reference_gate_poly", gate_poly);
  push("reference_skip_update_poly", skip_update_poly);
  push("polynomial_metadata", {2.0, 2.0, 8.0});
  return payload;
}

void test_evaluate_rank_gate_payload_matches_reference() {
  const auto result = stage1::evaluate_rank_gate_payload(make_consistent_payload());
  require_close(result.max_abs_error, 0.0, 1e-15, "rank/gate eval max error");
  require_equal(result.skip_update.size(), 4, "skip update length");
}

}  // namespace

auto main() -> int {
  test_read_rank_gate_payload();
  test_evaluate_rank_gate_payload_matches_reference();
  return EXIT_SUCCESS;
}
