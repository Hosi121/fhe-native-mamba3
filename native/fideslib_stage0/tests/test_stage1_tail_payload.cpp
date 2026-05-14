#include "stage1_tail_eval.hpp"

#include <cstdint>
#include <cstdlib>
#include <cmath>
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
  const auto path = std::filesystem::current_path() / "stage1_tail_payload_test.bin";
  std::ofstream output(path, std::ios::binary);
  require_true(static_cast<bool>(output), "open demo payload");

  output.write(stage1::kTailPayloadMagic.data(), stage1::kTailPayloadMagic.size());
  write_scalar<std::uint32_t>(output, stage1::kTailPayloadFormatVersion);
  write_scalar<std::uint32_t>(output, 8);   // d_model
  write_scalar<std::uint32_t>(output, 8);   // d_model_pad
  write_scalar<std::uint32_t>(output, 6);   // mimo_rank
  write_scalar<std::uint32_t>(output, 8);   // rank_pad
  write_scalar<std::uint32_t>(output, 2);   // d_state
  write_scalar<std::uint32_t>(output, 64);  // model_baby_step
  write_scalar<std::uint32_t>(output, 4);   // rank_baby_step
  write_scalar<std::uint32_t>(output, 1);   // layer_index
  write_scalar<std::uint32_t>(output, 3);   // prompt_token
  write_scalar<std::uint32_t>(output, 4);   // dt_rank
  write_scalar<double>(output, 1e-5);
  write_scalar<double>(output, 0.05);
  write_scalar<std::int64_t>(output, 7);

  write_scalar<std::uint32_t>(
      output,
      static_cast<std::uint32_t>(stage1::tail_payload_array_order().size()));
  double value = 0.0;
  stage1::TailPayloadConfig config;
  config.d_model = 8;
  config.mimo_rank = 6;
  config.d_state = 2;
  for (const auto& name : stage1::tail_payload_array_order()) {
    const auto shape = stage1::tail_payload_expected_shape(config, name);
    const auto length = stage1::shape_size(shape);
    write_scalar<std::uint64_t>(output, length);
    for (std::uint64_t index = 0; index < length; ++index) {
      value += 0.25;
      write_scalar<double>(output, value);
    }
  }
  require_true(static_cast<bool>(output), "write demo payload");
  return path;
}

void test_read_tail_payload() {
  const auto path = write_demo_payload();
  const auto payload = stage1::read_tail_payload(path.string());

  require_equal(payload.config.d_model, 8, "d_model");
  require_equal(payload.config.mimo_rank, 6, "mimo_rank");
  require_equal(payload.config.d_state, 2, "d_state");
  require_equal(payload.config.layer_index, 1, "layer_index");
  require_equal(payload.config.prompt_token, 3, "prompt_token");
  require_equal(payload.arrays.size(), stage1::tail_payload_array_order().size(), "array count");
  require_string_equal(payload.arrays.front().name, "residual_input", "first array name");
  require_equal(payload.array("w_out").values.size(), 48, "w_out length");
  require_equal(payload.array("b").shape.size(), 2, "b rank");
  require_equal(payload.array("b").shape[0], 2, "b dim0");
  require_equal(payload.array("b").shape[1], 6, "b dim1");

  std::filesystem::remove(path);
}

auto demo_values(std::uint64_t count, double scale) -> std::vector<double> {
  std::vector<double> values(count);
  for (std::uint64_t index = 0; index < count; ++index) {
    values[index] = static_cast<double>(index + 1) * scale;
  }
  return values;
}

auto make_consistent_payload() -> stage1::TailPayload {
  stage1::TailPayload payload;
  payload.config.d_model = 3;
  payload.config.d_model_pad = 4;
  payload.config.mimo_rank = 4;
  payload.config.rank_pad = 4;
  payload.config.d_state = 2;
  payload.config.model_baby_step = 2;
  payload.config.rank_baby_step = 2;

  const auto d_model = payload.config.d_model;
  const auto rank = payload.config.mimo_rank;
  const auto d_state = payload.config.d_state;
  std::vector<double> residual_input = {1.0, -0.5, 0.25};
  std::vector<double> rank_input = {0.2, -0.1, 0.05, 0.3};
  std::vector<double> gate = {0.7, 0.8, -0.2, 0.5};
  std::vector<double> b = demo_values(d_state * rank, 0.01);
  std::vector<double> c = demo_values(d_state * rank, -0.02);
  std::vector<double> decay(d_state * rank, 0.9);
  std::vector<double> previous_state = demo_values(d_state * rank, 0.03);
  std::vector<double> skip_update = {0.01, -0.02, 0.03, -0.04};
  std::vector<double> w_out = demo_values(d_model * rank, 0.005);

  std::vector<double> state_new(d_state * rank, 0.0);
  std::vector<double> readout_rank(rank, 0.0);
  std::vector<double> rank_output(rank, 0.0);
  std::vector<double> rank_payload(rank, 0.0);
  auto output_model = residual_input;
  for (std::uint32_t state_index = 0; state_index < d_state; ++state_index) {
    const auto base = state_index * rank;
    for (std::uint32_t rank_index = 0; rank_index < rank; ++rank_index) {
      const auto index = base + rank_index;
      state_new[index] = decay[index] * previous_state[index] + b[index] * rank_input[rank_index];
      readout_rank[rank_index] += c[index] * state_new[index];
    }
  }
  for (std::uint32_t rank_index = 0; rank_index < rank; ++rank_index) {
    rank_output[rank_index] = readout_rank[rank_index] + skip_update[rank_index];
    rank_payload[rank_index] = gate[rank_index] * rank_output[rank_index];
  }
  for (std::uint32_t model_index = 0; model_index < d_model; ++model_index) {
    const auto base = model_index * rank;
    for (std::uint32_t rank_index = 0; rank_index < rank; ++rank_index) {
      output_model[model_index] += w_out[base + rank_index] * rank_payload[rank_index];
    }
  }

  auto push = [&](const std::string& name, std::vector<double> values) {
    payload.arrays.push_back(stage1::TailPayloadArray{
        .name = name,
        .shape = stage1::tail_payload_expected_shape(payload.config, name),
        .values = std::move(values)});
  };
  push("residual_input", residual_input);
  push("rank_input", rank_input);
  push("gate", gate);
  push("b", b);
  push("c", c);
  push("decay", decay);
  push("previous_state", previous_state);
  push("skip_update", skip_update);
  push("w_out", w_out);
  push("source_readout_rank", readout_rank);
  push("source_final_output", output_model);
  push("reference_state_new", state_new);
  push("reference_readout_rank", readout_rank);
  push("reference_rank_output", rank_output);
  push("reference_rank_payload", rank_payload);
  push("reference_output_model", output_model);
  return payload;
}

void test_evaluate_tail_payload_matches_reference() {
  const auto result = stage1::evaluate_tail_payload(make_consistent_payload());
  require_close(result.max_abs_error, 0.0, 1e-15, "tail eval max error");
  require_equal(result.output_model.size(), 3, "output model length");
}

}  // namespace

auto main() -> int {
  test_read_tail_payload();
  test_evaluate_tail_payload_matches_reference();
  return EXIT_SUCCESS;
}
