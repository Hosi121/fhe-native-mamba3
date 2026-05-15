#include "stage1_rank_gate_eval.hpp"

#include <fideslib.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

using namespace fideslib;

namespace {

constexpr double kPlaintextCoefficientFloor = 1e-8;

struct Config {
  std::string input;
  std::string input_chain;
  std::string output_json;
  std::string artifact_version = "0.0.0+unknown";
  std::string repo_commit = "unknown";
  int ring_dim = 131072;
  int multiplicative_depth = 48;
  int scaling_mod_size = 40;
  int first_mod_size = 60;
  double atol = 1e-5;
  double rank_projection_scale = 1.0;
  double dt_projection_scale = 1.0;
  int chain_steps = 1;
  std::set<int> bootstrap_before_chain_steps;
  int bootstrap_level_budget_cts = 5;
  int bootstrap_level_budget_stc = 4;
  int bootstrap_bsgs_dim_cts = 0;
  int bootstrap_bsgs_dim_stc = 0;
  std::string security = "128-classic";
  std::string secret_key_dist = "sparse-ternary";
};

struct TailCiphertexts {
  Ciphertext<DCRTPoly> readout;
  Ciphertext<DCRTPoly> rank_output;
  Ciphertext<DCRTPoly> rank_payload;
  Ciphertext<DCRTPoly> output_delta;
  Ciphertext<DCRTPoly> output_model;
};

struct ChainReference {
  std::vector<double> state_new;
  std::vector<double> readout_rank;
  std::vector<double> rank_output;
  std::vector<double> rank_payload;
  std::vector<double> output_model;
};

struct LayerCiphertexts {
  Ciphertext<DCRTPoly> rms;
  Ciphertext<DCRTPoly> conv_pre;
  Ciphertext<DCRTPoly> conv_pre_for_silu;
  Ciphertext<DCRTPoly> gate_pre;
  Ciphertext<DCRTPoly> rank_input_poly;
  Ciphertext<DCRTPoly> gate_poly;
  Ciphertext<DCRTPoly> skip_update_poly;
  Ciphertext<DCRTPoly> dt_hidden_poly;
  Ciphertext<DCRTPoly> dt_pre_poly;
  Ciphertext<DCRTPoly> dt_state_major_poly;
  Ciphertext<DCRTPoly> decay_state_major_poly;
  Ciphertext<DCRTPoly> b_vec_poly;
  Ciphertext<DCRTPoly> c_vec_poly;
  Ciphertext<DCRTPoly> b_state_major_poly;
  Ciphertext<DCRTPoly> c_state_major_poly;
  Ciphertext<DCRTPoly> x_state_major_poly;
  Ciphertext<DCRTPoly> input_state_term;
  Ciphertext<DCRTPoly> state_new_poly;
  Ciphertext<DCRTPoly> readout_poly;
  Ciphertext<DCRTPoly> rank_output_poly;
  Ciphertext<DCRTPoly> rank_payload_poly;
  Ciphertext<DCRTPoly> output_delta_poly;
  Ciphertext<DCRTPoly> output_model_poly;
  std::vector<double> gate_poly_slots_for_error;
  int dt_rank = 0;
  bool previous_state_is_zero = true;
};

struct OperationCounts {
  int rotations = 0;
  int ct_pt_mul = 0;
  int ct_ct_mul = 0;
  int adds = 0;
  int unity_level_align_muls = 0;
  int bootstraps = 0;
};

using BabyRotationCache = std::map<int, Ciphertext<DCRTPoly>>;

auto now() -> std::chrono::steady_clock::time_point { return std::chrono::steady_clock::now(); }

auto seconds_since(std::chrono::steady_clock::time_point start) -> double {
  return std::chrono::duration<double>(now() - start).count();
}

auto add_counts(OperationCounts lhs, const OperationCounts& rhs) -> OperationCounts {
  lhs.rotations += rhs.rotations;
  lhs.ct_pt_mul += rhs.ct_pt_mul;
  lhs.ct_ct_mul += rhs.ct_ct_mul;
  lhs.adds += rhs.adds;
  lhs.unity_level_align_muls += rhs.unity_level_align_muls;
  lhs.bootstraps += rhs.bootstraps;
  return lhs;
}

auto subtract_counts(const OperationCounts& after, const OperationCounts& before)
    -> OperationCounts {
  return OperationCounts{
      .rotations = after.rotations - before.rotations,
      .ct_pt_mul = after.ct_pt_mul - before.ct_pt_mul,
      .ct_ct_mul = after.ct_ct_mul - before.ct_ct_mul,
      .adds = after.adds - before.adds,
      .unity_level_align_muls =
          after.unity_level_align_muls - before.unity_level_align_muls,
      .bootstraps = after.bootstraps - before.bootstraps,
  };
}

auto parse_int(std::string_view name, const char* value) -> int {
  try {
    return std::stoi(value);
  } catch (const std::exception& exc) {
    throw std::invalid_argument(std::string("invalid integer for ") + std::string(name) + ": " +
                                exc.what());
  }
}

auto parse_double(std::string_view name, const char* value) -> double {
  try {
    return std::stod(value);
  } catch (const std::exception& exc) {
    throw std::invalid_argument(std::string("invalid float for ") + std::string(name) + ": " +
                                exc.what());
  }
}

auto parse_int_set(std::string_view name, std::string_view value) -> std::set<int> {
  std::set<int> output;
  std::string text(value);
  std::stringstream stream(text);
  std::string token;
  while (std::getline(stream, token, ',')) {
    if (token.empty()) {
      continue;
    }
    output.insert(parse_int(name, token.c_str()));
  }
  return output;
}

auto split_paths(std::string_view value) -> std::vector<std::string> {
  std::vector<std::string> paths;
  std::string current;
  for (const char character : value) {
    if (character == ',') {
      if (!current.empty()) {
        paths.push_back(current);
        current.clear();
      }
      continue;
    }
    current.push_back(character);
  }
  if (!current.empty()) {
    paths.push_back(current);
  }
  return paths;
}

auto parse_args(int argc, char* argv[]) -> Config {
  Config config;
  for (int i = 1; i < argc; ++i) {
    const std::string_view arg(argv[i]);
    if (i + 1 >= argc) {
      throw std::invalid_argument(std::string("missing value for ") + std::string(arg));
    }
    const char* value = argv[++i];
    if (arg == "--input") {
      config.input = value;
    } else if (arg == "--input-chain") {
      config.input_chain = value;
    } else if (arg == "--output-json") {
      config.output_json = value;
    } else if (arg == "--artifact-version") {
      config.artifact_version = value;
    } else if (arg == "--repo-commit") {
      config.repo_commit = value;
    } else if (arg == "--ring-dim") {
      config.ring_dim = parse_int(arg, value);
    } else if (arg == "--multiplicative-depth") {
      config.multiplicative_depth = parse_int(arg, value);
    } else if (arg == "--scaling-mod-size") {
      config.scaling_mod_size = parse_int(arg, value);
    } else if (arg == "--first-mod-size") {
      config.first_mod_size = parse_int(arg, value);
    } else if (arg == "--atol") {
      config.atol = parse_double(arg, value);
    } else if (arg == "--rank-projection-scale") {
      config.rank_projection_scale = parse_double(arg, value);
    } else if (arg == "--dt-projection-scale") {
      config.dt_projection_scale = parse_double(arg, value);
    } else if (arg == "--chain-steps") {
      config.chain_steps = parse_int(arg, value);
    } else if (arg == "--bootstrap-before-chain-steps") {
      config.bootstrap_before_chain_steps = parse_int_set(arg, value);
    } else if (arg == "--bootstrap-level-budget-cts") {
      config.bootstrap_level_budget_cts = parse_int(arg, value);
    } else if (arg == "--bootstrap-level-budget-stc") {
      config.bootstrap_level_budget_stc = parse_int(arg, value);
    } else if (arg == "--bootstrap-bsgs-dim-cts") {
      config.bootstrap_bsgs_dim_cts = parse_int(arg, value);
    } else if (arg == "--bootstrap-bsgs-dim-stc") {
      config.bootstrap_bsgs_dim_stc = parse_int(arg, value);
    } else if (arg == "--security") {
      config.security = value;
    } else if (arg == "--secret-key-dist") {
      config.secret_key_dist = value;
    } else {
      throw std::invalid_argument(std::string("unknown argument: ") + std::string(arg));
    }
  }
  if (config.input.empty() == config.input_chain.empty()) {
    throw std::invalid_argument("exactly one of --input or --input-chain is required");
  }
  if (config.ring_dim <= 0 || (config.ring_dim & (config.ring_dim - 1)) != 0) {
    throw std::invalid_argument("ring-dim must be a positive power of two");
  }
  if (config.multiplicative_depth <= 0 || config.scaling_mod_size <= 0 ||
      config.first_mod_size <= 0 || config.atol < 0.0 || config.rank_projection_scale <= 0.0 ||
      config.dt_projection_scale <= 0.0 || config.chain_steps <= 0 ||
      config.bootstrap_level_budget_cts <= 0 || config.bootstrap_level_budget_stc <= 0 ||
      config.bootstrap_bsgs_dim_cts < 0 || config.bootstrap_bsgs_dim_stc < 0) {
    throw std::invalid_argument("invalid CKKS parameters");
  }
  for (const int step : config.bootstrap_before_chain_steps) {
    if (step <= 1 || step > config.chain_steps) {
      throw std::invalid_argument(
          "bootstrap-before-chain-steps must be in [2, chain_steps]");
    }
  }
  if (!config.input_chain.empty() && config.chain_steps != 1) {
    throw std::invalid_argument("--input-chain currently requires --chain-steps 1");
  }
  if (config.security != "not-set" && config.security != "128-classic") {
    throw std::invalid_argument("security must be not-set or 128-classic");
  }
  if (config.secret_key_dist != "sparse-ternary" &&
      config.secret_key_dist != "uniform-ternary" &&
      config.secret_key_dist != "sparse-encapsulated") {
    throw std::invalid_argument(
        "secret-key-dist must be sparse-ternary, uniform-ternary, or sparse-encapsulated");
  }
  return config;
}

auto resolve_security(const std::string& value) -> SecurityLevel {
  if (value == "128-classic") {
    return HEStd_128_classic;
  }
  return HEStd_NotSet;
}

auto resolve_secret_key_dist(const std::string& value) -> SecretKeyDist {
  if (value == "uniform-ternary") {
    return UNIFORM_TERNARY;
  }
  if (value == "sparse-encapsulated") {
    return fideslib::SPARSE_ENCAPSULATED;
  }
  return SPARSE_TERNARY;
}

auto python_mod(int value, int modulus) -> int {
  const int result = value % modulus;
  return result < 0 ? result + modulus : result;
}

auto slot_bsgs_giant_with_zero(int input_dim, int output_dim, int baby_step) -> std::vector<int> {
  std::set<int> values;
  const int min_offset = -(output_dim - 1);
  const int max_offset = input_dim - 1;
  for (int offset = min_offset; offset <= max_offset; ++offset) {
    values.insert(offset - python_mod(offset, baby_step));
  }
  return {values.begin(), values.end()};
}

auto slot_bsgs_rotations(int input_dim, int output_dim, int baby_step) -> std::vector<int32_t> {
  std::set<int32_t> rotations;
  for (int baby = 1; baby < baby_step; ++baby) {
    rotations.insert(static_cast<int32_t>(baby));
  }
  for (const int giant : slot_bsgs_giant_with_zero(input_dim, output_dim, baby_step)) {
    if (giant != 0) {
      rotations.insert(static_cast<int32_t>(giant));
    }
  }
  return {rotations.begin(), rotations.end()};
}

auto rank_to_vector_reduce_rotations(
    const stage1::RankGatePayloadConfig& config,
    std::uint32_t output_dim) -> std::vector<int32_t> {
  std::set<int32_t> rotations;
  for (std::uint32_t step = 1; step < config.rank_pad; step *= 2) {
    rotations.insert(static_cast<int32_t>(step));
  }
  for (std::uint32_t output_index = 1; output_index < output_dim; ++output_index) {
    rotations.insert(-static_cast<int32_t>(output_index));
  }
  return {rotations.begin(), rotations.end()};
}

auto rank_to_state_vector_reduce_rotations(const stage1::RankGatePayloadConfig& config)
    -> std::vector<int32_t> {
  return rank_to_vector_reduce_rotations(config, config.d_state);
}

auto vector_to_rank_block_expand_rotations(
    int input_dim,
    const stage1::RankGatePayloadConfig& config) -> std::vector<int32_t> {
  std::set<int32_t> rotations;
  for (int input_index = 1; input_index < input_dim; ++input_index) {
    rotations.insert(static_cast<int32_t>(input_index));
  }
  for (std::uint32_t step = 1; step < config.rank_pad; step *= 2) {
    rotations.insert(-static_cast<int32_t>(step));
  }
  return {rotations.begin(), rotations.end()};
}

auto bounded_baby_step(int requested, int input_dim) -> int {
  return std::max(1, std::min(requested, input_dim));
}

auto state_vector_to_state_major_rotations(const stage1::RankGatePayloadConfig& config)
    -> std::vector<int32_t> {
  std::set<int32_t> rotations;
  for (std::uint32_t state_index = 0; state_index < config.d_state; ++state_index) {
    const auto target_slot = static_cast<int>(state_index * config.rank_pad);
    const auto shift = static_cast<int>(state_index) - target_slot;
    if (shift != 0) {
      rotations.insert(static_cast<int32_t>(shift));
    }
    for (std::uint32_t step = 1; step < config.rank_pad; step *= 2) {
      rotations.insert(-static_cast<int32_t>(step));
    }
  }
  return {rotations.begin(), rotations.end()};
}

auto rank_block0_to_state_major_rotations(const stage1::RankGatePayloadConfig& config)
    -> std::vector<int32_t> {
  std::set<int32_t> rotations;
  for (std::uint32_t step = 1; step < config.d_state; step *= 2) {
    rotations.insert(-static_cast<int32_t>(step * config.rank_pad));
  }
  return {rotations.begin(), rotations.end()};
}

auto state_major_to_rank_block0_rotations(const stage1::RankGatePayloadConfig& config)
    -> std::vector<int32_t> {
  std::set<int32_t> rotations;
  for (std::uint32_t step = 1; step < config.d_state; step *= 2) {
    rotations.insert(static_cast<int32_t>(step * config.rank_pad));
  }
  return {rotations.begin(), rotations.end()};
}

auto required_rank_gate_rotations(const stage1::RankGatePayload& payload)
    -> std::vector<int32_t> {
  const auto& config = payload.config;
  std::set<int32_t> rotations;
  const auto model_rotations =
      slot_bsgs_rotations(config.d_model, config.mimo_rank, config.model_baby_step);
  rotations.insert(model_rotations.begin(), model_rotations.end());
  const auto bc_rotations = rank_to_state_vector_reduce_rotations(config);
  rotations.insert(bc_rotations.begin(), bc_rotations.end());
  const int dt_rank = static_cast<int>(payload.array("dt_in_weight").shape.at(0));
  const auto dt_hidden_rotations =
      rank_to_vector_reduce_rotations(config, static_cast<std::uint32_t>(dt_rank));
  rotations.insert(dt_hidden_rotations.begin(), dt_hidden_rotations.end());
  const auto dt_project_expand_rotations = vector_to_rank_block_expand_rotations(dt_rank, config);
  rotations.insert(dt_project_expand_rotations.begin(), dt_project_expand_rotations.end());
  const auto state_rotations = state_vector_to_state_major_rotations(config);
  rotations.insert(state_rotations.begin(), state_rotations.end());
  const auto rank_broadcast_rotations = rank_block0_to_state_major_rotations(config);
  rotations.insert(rank_broadcast_rotations.begin(), rank_broadcast_rotations.end());
  const auto rank_reduce_rotations = state_major_to_rank_block0_rotations(config);
  rotations.insert(rank_reduce_rotations.begin(), rank_reduce_rotations.end());
  const auto out_rotations = slot_bsgs_rotations(
      config.mimo_rank,
      config.d_model,
      bounded_baby_step(
          static_cast<int>(config.rank_baby_step),
          static_cast<int>(config.mimo_rank)));
  rotations.insert(out_rotations.begin(), out_rotations.end());
  return {rotations.begin(), rotations.end()};
}

auto configured_input_paths(const Config& args) -> std::vector<std::string> {
  if (!args.input_chain.empty()) {
    auto paths = split_paths(args.input_chain);
    if (paths.empty()) {
      throw std::runtime_error("--input-chain must contain at least one path");
    }
    return paths;
  }
  return {args.input};
}

void require_same_payload_config(
    const stage1::RankGatePayloadConfig& expected,
    const stage1::RankGatePayloadConfig& actual,
    std::size_t index) {
  if (expected.d_model == actual.d_model &&
      expected.d_model_pad == actual.d_model_pad &&
      expected.mimo_rank == actual.mimo_rank &&
      expected.rank_pad == actual.rank_pad &&
      expected.d_state == actual.d_state &&
      expected.model_baby_step == actual.model_baby_step &&
      expected.rank_baby_step == actual.rank_baby_step) {
    return;
  }
  throw std::runtime_error("rank/gate input-chain payload config mismatch at index " +
                           std::to_string(index));
}

auto read_rank_gate_payloads(const std::vector<std::string>& paths)
    -> std::vector<stage1::RankGatePayload> {
  std::vector<stage1::RankGatePayload> payloads;
  payloads.reserve(paths.size());
  for (const auto& path : paths) {
    payloads.push_back(stage1::read_rank_gate_payload(path));
  }
  if (payloads.empty()) {
    throw std::runtime_error("rank/gate payload list must not be empty");
  }
  for (std::size_t index = 1; index < payloads.size(); ++index) {
    require_same_payload_config(payloads.front().config, payloads[index].config, index);
  }
  return payloads;
}

auto required_rank_gate_rotations_union(const std::vector<stage1::RankGatePayload>& payloads)
    -> std::vector<int32_t> {
  std::set<int32_t> rotations;
  for (const auto& payload : payloads) {
    const auto payload_rotations = required_rank_gate_rotations(payload);
    rotations.insert(payload_rotations.begin(), payload_rotations.end());
  }
  return {rotations.begin(), rotations.end()};
}

auto pack_block0(
    const std::vector<double>& values,
    const stage1::RankGatePayloadConfig& config) -> std::vector<double> {
  std::vector<double> slots(static_cast<size_t>(config.rank_pad) * config.d_state, 0.0);
  std::copy(values.begin(), values.end(), slots.begin());
  return slots;
}

auto scaled_values(const std::vector<double>& values, double scale) -> std::vector<double> {
  std::vector<double> output(values.size(), 0.0);
  for (std::size_t index = 0; index < values.size(); ++index) {
    output[index] = values[index] * scale;
  }
  return output;
}

auto unscaled_values(const std::vector<double>& values, double scale) -> std::vector<double> {
  std::vector<double> output(values.size(), 0.0);
  for (std::size_t index = 0; index < values.size(); ++index) {
    output[index] = values[index] / scale;
  }
  return output;
}

auto scaled_decay_coefficients(
    const std::vector<double>& coefficients,
    const stage1::RankGatePayloadConfig& config,
    double input_scale) -> std::vector<double> {
  const auto state_rank = static_cast<std::size_t>(config.d_state) * config.mimo_rank;
  if (state_rank == 0 || coefficients.empty() || coefficients.size() % state_rank != 0) {
    throw std::runtime_error("decay coefficient vector has invalid size");
  }
  std::vector<double> output(coefficients.size(), 0.0);
  double denominator = 1.0;
  const auto degree_count = coefficients.size() / state_rank;
  for (std::size_t degree = 0; degree < degree_count; ++degree) {
    for (std::size_t index = 0; index < state_rank; ++index) {
      const double value = coefficients[degree * state_rank + index] / denominator;
      output[degree * state_rank + index] =
          std::abs(value) < kPlaintextCoefficientFloor ? 0.0 : value;
    }
    denominator *= input_scale;
  }
  return output;
}

auto repeated_rank_values(double value, const stage1::RankGatePayloadConfig& config)
    -> std::vector<double> {
  return std::vector<double>(static_cast<std::size_t>(config.mimo_rank), value);
}

auto pack_state_major_values(
    const std::vector<double>& values,
    const stage1::RankGatePayloadConfig& config) -> std::vector<double> {
  const auto expected = static_cast<std::size_t>(config.d_state) * config.mimo_rank;
  if (values.size() != expected) {
    throw std::runtime_error("state-major values length mismatch");
  }
  std::vector<double> slots(static_cast<size_t>(config.rank_pad) * config.d_state, 0.0);
  std::size_t index = 0;
  for (std::uint32_t state_index = 0; state_index < config.d_state; ++state_index) {
    const auto base = static_cast<std::size_t>(state_index * config.rank_pad);
    for (std::uint32_t rank_index = 0; rank_index < config.mimo_rank; ++rank_index) {
      slots[base + rank_index] = values[index++];
    }
  }
  return slots;
}

auto rank_valid_mask(const stage1::RankGatePayloadConfig& config) -> std::vector<double> {
  std::vector<double> mask(static_cast<size_t>(config.rank_pad) * config.d_state, 0.0);
  std::fill(mask.begin(), mask.begin() + static_cast<std::ptrdiff_t>(config.mimo_rank), 1.0);
  return mask;
}

auto is_near_zero_vector(const std::vector<double>& values) -> bool {
  return std::all_of(values.begin(), values.end(), [](double value) {
    return std::abs(value) < 1e-15;
  });
}

auto slot_bsgs_pre_mask(
    const std::vector<double>& weights,
    int input_dim,
    int output_dim,
    int batch_size,
    int giant,
    int offset) -> std::vector<double> {
  std::vector<double> mask(static_cast<size_t>(batch_size), 0.0);
  for (int output = 0; output < output_dim; ++output) {
    const int input = output + offset;
    if (input < 0 || input >= input_dim) {
      continue;
    }
    const int source_slot = python_mod(output + giant, batch_size);
    const double value = weights[static_cast<size_t>(output) * input_dim + input];
    mask[static_cast<size_t>(source_slot)] =
        std::abs(value) < kPlaintextCoefficientFloor ? 0.0 : value;
  }
  return mask;
}

auto slot_bsgs_precompute_baby_rotations(
    const CryptoContext<DCRTPoly>& cc,
    const Ciphertext<DCRTPoly>& input_ct,
    int baby_step,
    int& rotations) -> BabyRotationCache {
  BabyRotationCache baby_ct;
  baby_ct[0] = input_ct;
  for (int baby = 1; baby < baby_step; ++baby) {
    baby_ct[baby] = cc->EvalRotate(input_ct, baby);
    ++rotations;
  }
  return baby_ct;
}

auto slot_bsgs_linear_block0_from_babies(
    const CryptoContext<DCRTPoly>& cc,
    const BabyRotationCache& baby_ct,
    const std::vector<double>& weights,
    int input_dim,
    int output_dim,
    int baby_step,
    int batch_size,
    int& rotations,
    int& ct_pt_muls,
    int& adds) -> Ciphertext<DCRTPoly> {
  Ciphertext<DCRTPoly> accumulator;
  bool has_accumulator = false;
  for (const int giant : slot_bsgs_giant_with_zero(input_dim, output_dim, baby_step)) {
    Ciphertext<DCRTPoly> inner;
    bool has_inner = false;
    for (int baby = 0; baby < baby_step; ++baby) {
      const int offset = giant + baby;
      auto mask = slot_bsgs_pre_mask(weights, input_dim, output_dim, batch_size, giant, offset);
      if (std::all_of(mask.begin(), mask.end(), [](double value) { return value == 0.0; })) {
        continue;
      }
      auto plain = cc->MakeCKKSPackedPlaintext(mask);
      plain->SetLength(static_cast<size_t>(batch_size));
      auto term = cc->EvalMult(baby_ct.at(baby), plain);
      ++ct_pt_muls;
      if (!has_inner) {
        inner = term;
        has_inner = true;
      } else {
        inner = cc->EvalAdd(inner, term);
        ++adds;
      }
    }
    if (!has_inner) {
      continue;
    }
    if (giant != 0) {
      inner = cc->EvalRotate(inner, giant);
      ++rotations;
    }
    if (!has_accumulator) {
      accumulator = inner;
      has_accumulator = true;
    } else {
      accumulator = cc->EvalAdd(accumulator, inner);
      ++adds;
    }
  }
  if (!has_accumulator) {
    throw std::runtime_error("slot BSGS produced no terms");
  }
  return accumulator;
}

auto slot_bsgs_linear_block0(
    const CryptoContext<DCRTPoly>& cc,
    const Ciphertext<DCRTPoly>& input_ct,
    const std::vector<double>& weights,
    int input_dim,
    int output_dim,
    int baby_step,
    int batch_size,
    int& rotations,
    int& ct_pt_muls,
    int& adds) -> Ciphertext<DCRTPoly> {
  auto baby_ct = slot_bsgs_precompute_baby_rotations(cc, input_ct, baby_step, rotations);
  return slot_bsgs_linear_block0_from_babies(
      cc,
      baby_ct,
      weights,
      input_dim,
      output_dim,
      baby_step,
      batch_size,
      rotations,
      ct_pt_muls,
      adds);
}

auto rank_to_vector_linear_reduce(
    const CryptoContext<DCRTPoly>& cc,
    const Ciphertext<DCRTPoly>& rank_ct,
    const std::vector<double>& weights,
    const stage1::RankGatePayloadConfig& config,
    std::uint32_t output_dim,
    int batch_size,
    int& rotations,
    int& ct_pt_muls,
    int& adds) -> Ciphertext<DCRTPoly> {
  if (weights.size() != static_cast<std::size_t>(output_dim) * config.mimo_rank) {
    throw std::runtime_error("rank-to-vector reduction weights have invalid size");
  }
  if (output_dim == 0 || output_dim > static_cast<std::uint32_t>(batch_size)) {
    throw std::runtime_error("rank-to-vector reduction output_dim exceeds batch size");
  }
  std::vector<double> slot0_mask(static_cast<std::size_t>(batch_size), 0.0);
  slot0_mask[0] = 1.0;
  auto slot0_plain = cc->MakeCKKSPackedPlaintext(slot0_mask);
  slot0_plain->SetLength(static_cast<std::size_t>(batch_size));

  Ciphertext<DCRTPoly> accumulator;
  bool has_accumulator = false;
  for (std::uint32_t output_index = 0; output_index < output_dim; ++output_index) {
    std::vector<double> mask(static_cast<std::size_t>(batch_size), 0.0);
    const auto weight_base = static_cast<std::size_t>(output_index) * config.mimo_rank;
    for (std::uint32_t rank_index = 0; rank_index < config.mimo_rank; ++rank_index) {
      const double value = weights[weight_base + rank_index];
      mask[rank_index] = std::abs(value) < kPlaintextCoefficientFloor ? 0.0 : value;
    }
    auto weight_plain = cc->MakeCKKSPackedPlaintext(mask);
    weight_plain->SetLength(static_cast<std::size_t>(batch_size));
    auto state_sum_ct = cc->EvalMult(rank_ct, weight_plain);
    ++ct_pt_muls;
    for (std::uint32_t step = 1; step < config.rank_pad; step *= 2) {
      state_sum_ct = cc->EvalAdd(
          state_sum_ct,
          cc->EvalRotate(state_sum_ct, static_cast<int32_t>(step)));
      ++rotations;
      ++adds;
    }
    state_sum_ct = cc->EvalMult(state_sum_ct, slot0_plain);
    ++ct_pt_muls;
    if (output_index != 0) {
      state_sum_ct = cc->EvalRotate(state_sum_ct, -static_cast<int32_t>(output_index));
      ++rotations;
    }
    if (!has_accumulator) {
      accumulator = state_sum_ct;
      has_accumulator = true;
    } else {
      accumulator = cc->EvalAdd(accumulator, state_sum_ct);
      ++adds;
    }
  }
  if (!has_accumulator) {
    throw std::runtime_error("rank-to-vector reduction produced no terms");
  }
  return accumulator;
}

auto rank_to_state_vector_linear_reduce(
    const CryptoContext<DCRTPoly>& cc,
    const Ciphertext<DCRTPoly>& rank_ct,
    const std::vector<double>& weights,
    const stage1::RankGatePayloadConfig& config,
    int batch_size,
    int& rotations,
    int& ct_pt_muls,
    int& adds) -> Ciphertext<DCRTPoly> {
  return rank_to_vector_linear_reduce(
      cc,
      rank_ct,
      weights,
      config,
      config.d_state,
      batch_size,
      rotations,
      ct_pt_muls,
      adds);
}

auto vector_to_rank_block_linear_expand(
    const CryptoContext<DCRTPoly>& cc,
    const Ciphertext<DCRTPoly>& input_ct,
    const std::vector<double>& weights,
    int input_dim,
    const stage1::RankGatePayloadConfig& config,
    int batch_size,
    int& rotations,
    int& ct_pt_muls,
    int& adds) -> Ciphertext<DCRTPoly> {
  if (input_dim <= 0 || static_cast<std::uint32_t>(input_dim) > config.rank_pad) {
    throw std::runtime_error("vector-to-rank expansion input_dim exceeds rank pad");
  }
  if (weights.size() != static_cast<std::size_t>(config.mimo_rank) * input_dim) {
    throw std::runtime_error("vector-to-rank expansion weights have invalid size");
  }
  Ciphertext<DCRTPoly> accumulator;
  bool has_accumulator = false;
  for (int input_index = 0; input_index < input_dim; ++input_index) {
    std::vector<double> select_mask(static_cast<std::size_t>(batch_size), 0.0);
    select_mask[static_cast<std::size_t>(input_index)] = 1.0;
    auto select_plain = cc->MakeCKKSPackedPlaintext(select_mask);
    select_plain->SetLength(static_cast<std::size_t>(batch_size));
    auto column_ct = cc->EvalMult(input_ct, select_plain);
    ++ct_pt_muls;
    if (input_index != 0) {
      column_ct = cc->EvalRotate(column_ct, static_cast<int32_t>(input_index));
      ++rotations;
    }
    for (std::uint32_t step = 1; step < config.rank_pad; step *= 2) {
      column_ct = cc->EvalAdd(column_ct, cc->EvalRotate(column_ct, -static_cast<int32_t>(step)));
      ++rotations;
      ++adds;
    }

    std::vector<double> weight_mask(static_cast<std::size_t>(batch_size), 0.0);
    bool has_column = false;
    for (std::uint32_t rank_index = 0; rank_index < config.mimo_rank; ++rank_index) {
      const auto weight_index =
          static_cast<std::size_t>(rank_index) * input_dim + static_cast<std::size_t>(input_index);
      const double value = weights[weight_index];
      if (std::abs(value) < kPlaintextCoefficientFloor) {
        continue;
      }
      weight_mask[rank_index] = value;
      has_column = true;
    }
    if (!has_column) {
      continue;
    }
    auto weight_plain = cc->MakeCKKSPackedPlaintext(weight_mask);
    weight_plain->SetLength(static_cast<std::size_t>(batch_size));
    auto term = cc->EvalMult(column_ct, weight_plain);
    ++ct_pt_muls;
    if (!has_accumulator) {
      accumulator = term;
      has_accumulator = true;
    } else {
      accumulator = cc->EvalAdd(accumulator, term);
      ++adds;
    }
  }
  if (!has_accumulator) {
    throw std::runtime_error("vector-to-rank expansion produced no terms");
  }
  return accumulator;
}

void align_levels(
    const CryptoContext<DCRTPoly>& cc,
    Ciphertext<DCRTPoly>& lhs,
    Ciphertext<DCRTPoly>& rhs,
    int& unity_multiplies);

auto evaluate_power_polynomial_block0(
    const CryptoContext<DCRTPoly>& cc,
    const PublicKey<DCRTPoly>& public_key,
    Ciphertext<DCRTPoly> input_ct,
    const std::vector<double>& coefficients,
    const stage1::RankGatePayloadConfig& config,
    int batch_size,
    int& ct_ct_muls,
    int& adds,
    int& unity_multiplies) -> Ciphertext<DCRTPoly> {
  if (coefficients.empty()) {
    throw std::runtime_error("polynomial coefficient vector must not be empty");
  }
  auto make_constant_ct = [&](double coefficient) {
    auto plain = cc->MakeCKKSPackedPlaintext(
        pack_block0(repeated_rank_values(coefficient, config), config));
    plain->SetLength(static_cast<std::size_t>(batch_size));
    return cc->Encrypt(public_key, plain);
  };

  auto result = make_constant_ct(coefficients.back());
  if (coefficients.size() == 1) {
    return result;
  }
  for (auto it = coefficients.rbegin() + 1; it != coefficients.rend(); ++it) {
    result = cc->EvalMult(result, input_ct);
    ++ct_ct_muls;
    const double coefficient = *it;
    if (coefficient != 0.0) {
      auto constant_ct = make_constant_ct(coefficient);
      align_levels(cc, result, constant_ct, unity_multiplies);
      result = cc->EvalAdd(result, constant_ct);
      ++adds;
    }
  }
  return result;
}

auto evaluate_state_major_vector_power_polynomial(
    const CryptoContext<DCRTPoly>& cc,
    const PublicKey<DCRTPoly>& public_key,
    Ciphertext<DCRTPoly> input_ct,
    const std::vector<double>& coefficients,
    const stage1::RankGatePayloadConfig& config,
    int batch_size,
    int& ct_ct_muls,
    int& adds,
    int& unity_multiplies) -> Ciphertext<DCRTPoly> {
  const auto state_rank = static_cast<std::size_t>(config.d_state) * config.mimo_rank;
  if (state_rank == 0 || coefficients.empty() || coefficients.size() % state_rank != 0) {
    throw std::runtime_error("decay coefficient vector has invalid size");
  }
  const auto degree_count = coefficients.size() / state_rank;
  auto make_constant_ct = [&](std::size_t coefficient_index) {
    const auto begin = coefficients.begin() +
                       static_cast<std::ptrdiff_t>(coefficient_index * state_rank);
    const auto end = begin + static_cast<std::ptrdiff_t>(state_rank);
    auto plain = cc->MakeCKKSPackedPlaintext(
        pack_state_major_values(std::vector<double>(begin, end), config));
    plain->SetLength(static_cast<std::size_t>(batch_size));
    return cc->Encrypt(public_key, plain);
  };

  auto result = make_constant_ct(degree_count - 1);
  if (degree_count == 1) {
    return result;
  }
  for (std::size_t index = degree_count - 1; index-- > 0;) {
    result = cc->EvalMult(result, input_ct);
    ++ct_ct_muls;
    auto constant_ct = make_constant_ct(index);
    align_levels(cc, result, constant_ct, unity_multiplies);
    result = cc->EvalAdd(result, constant_ct);
    ++adds;
  }
  return result;
}

void align_levels(
    const CryptoContext<DCRTPoly>& cc,
    Ciphertext<DCRTPoly>& lhs,
    Ciphertext<DCRTPoly>& rhs,
    int& unity_multiplies) {
  for (int guard = 0; guard < 128 && lhs->GetLevel() < rhs->GetLevel(); ++guard) {
    const auto before = lhs->GetLevel();
    cc->EvalMultInPlace(lhs, 1.0);
    ++unity_multiplies;
    if (lhs->GetLevel() == before) {
      break;
    }
  }
  for (int guard = 0; guard < 128 && rhs->GetLevel() < lhs->GetLevel(); ++guard) {
    const auto before = rhs->GetLevel();
    cc->EvalMultInPlace(rhs, 1.0);
    ++unity_multiplies;
    if (rhs->GetLevel() == before) {
      break;
    }
  }
}

auto decrypt_slots(
    const CryptoContext<DCRTPoly>& cc,
    const PrivateKey<DCRTPoly>& secret_key,
    Ciphertext<DCRTPoly> ciphertext,
    size_t length) -> std::vector<double> {
  Plaintext plaintext;
  cc->Decrypt(secret_key, ciphertext, &plaintext);
  plaintext->SetLength(length);
  auto values = plaintext->GetRealPackedValue();
  values.resize(length);
  return values;
}

auto first_values(const std::vector<double>& values, size_t count) -> std::vector<double> {
  return {values.begin(), values.begin() + static_cast<std::ptrdiff_t>(count)};
}

auto state_major_valid_values(
    const std::vector<double>& values,
    const stage1::RankGatePayloadConfig& config) -> std::vector<double> {
  std::vector<double> output;
  output.reserve(static_cast<std::size_t>(config.d_state) * config.mimo_rank);
  for (std::uint32_t state_index = 0; state_index < config.d_state; ++state_index) {
    const auto base = static_cast<std::size_t>(state_index * config.rank_pad);
    for (std::uint32_t rank_index = 0; rank_index < config.mimo_rank; ++rank_index) {
      output.push_back(values[base + rank_index]);
    }
  }
  return output;
}

auto state_vector_to_state_major_ciphertext(
    const CryptoContext<DCRTPoly>& cc,
    const Ciphertext<DCRTPoly>& state_vector_ct,
    const stage1::RankGatePayloadConfig& config,
    int batch_size,
    int& rotations,
    int& ct_pt_muls,
    int& adds) -> Ciphertext<DCRTPoly> {
  Ciphertext<DCRTPoly> output;
  bool has_output = false;
  for (std::uint32_t state_index = 0; state_index < config.d_state; ++state_index) {
    std::vector<double> mask(static_cast<std::size_t>(batch_size), 0.0);
    mask[static_cast<std::size_t>(state_index)] = 1.0;
    auto plain = cc->MakeCKKSPackedPlaintext(mask);
    plain->SetLength(static_cast<std::size_t>(batch_size));
    auto block = cc->EvalMult(state_vector_ct, plain);
    ++ct_pt_muls;
    const auto target_slot = static_cast<int>(state_index * config.rank_pad);
    const auto shift = static_cast<int>(state_index) - target_slot;
    if (shift != 0) {
      block = cc->EvalRotate(block, shift);
      ++rotations;
    }
    for (std::uint32_t step = 1; step < config.rank_pad; step *= 2) {
      block = cc->EvalAdd(block, cc->EvalRotate(block, -static_cast<int32_t>(step)));
      ++rotations;
      ++adds;
    }
    if (!has_output) {
      output = block;
      has_output = true;
    } else {
      output = cc->EvalAdd(output, block);
      ++adds;
    }
  }
  if (!has_output) {
    throw std::runtime_error("state-vector broadcast produced no terms");
  }
  return output;
}

auto rank_block0_to_state_major_ciphertext(
    const CryptoContext<DCRTPoly>& cc,
    const Ciphertext<DCRTPoly>& rank_ct,
    const stage1::RankGatePayloadConfig& config,
    int& rotations,
    int& adds) -> Ciphertext<DCRTPoly> {
  auto output = rank_ct;
  for (std::uint32_t step = 1; step < config.d_state; step *= 2) {
    output = cc->EvalAdd(
        output,
        cc->EvalRotate(output, -static_cast<int32_t>(step * config.rank_pad)));
    ++rotations;
    ++adds;
  }
  return output;
}

auto read_status_kib(const std::string& key) -> long long {
  std::ifstream status("/proc/self/status");
  std::string line;
  while (std::getline(status, line)) {
    if (line.rfind(key + ":", 0) != 0) {
      continue;
    }
    std::stringstream stream(line.substr(key.size() + 1));
    long long value = 0;
    std::string unit;
    stream >> value >> unit;
    return value;
  }
  return 0;
}

auto rss_gib() -> double { return static_cast<double>(read_status_kib("VmRSS")) / (1024.0 * 1024.0); }

auto peak_rss_gib() -> double {
  return static_cast<double>(read_status_kib("VmHWM")) / (1024.0 * 1024.0);
}

void log_phase(const std::string& message) {
  std::cerr << "[stage1_rank_gate_fideslib] " << message << " rss_gib=" << rss_gib()
            << " peak_rss_gib=" << peak_rss_gib() << std::endl;
}

void write_payload(const std::string& output_json, const std::string& payload) {
  if (output_json.empty()) {
    std::cout << payload << std::endl;
    return;
  }
  std::ofstream output(output_json);
  if (!output) {
    throw std::runtime_error("failed to open output-json path");
  }
  output << payload << std::endl;
}

auto payload_file_exists(const std::string& output_json) -> bool {
  if (output_json.empty()) {
    return false;
  }
  std::ifstream input(output_json, std::ios::ate);
  return input && input.tellg() > 0;
}

void write_level_field(
    std::ostringstream& out,
    std::string_view name,
    const Ciphertext<DCRTPoly>& ciphertext,
    bool& first) {
  if (!first) {
    out << ",";
  }
  first = false;
  out << "\"" << name << "\":" << ciphertext->GetLevel();
}

void write_int_set_json(std::ostringstream& out, const std::set<int>& values) {
  out << "[";
  bool first = true;
  for (const int value : values) {
    if (!first) {
      out << ",";
    }
    first = false;
    out << value;
  }
  out << "]";
}

auto json_escape(std::string_view value) -> std::string {
  std::string output;
  output.reserve(value.size());
  for (const char character : value) {
    switch (character) {
      case '\\':
        output += "\\\\";
        break;
      case '"':
        output += "\\\"";
        break;
      case '\n':
        output += "\\n";
        break;
      case '\r':
        output += "\\r";
        break;
      case '\t':
        output += "\\t";
        break;
      default:
        output.push_back(character);
        break;
    }
  }
  return output;
}

void write_double_map_json(
    std::ostringstream& out,
    const std::map<std::string, double>& values) {
  out << "{";
  bool first = true;
  for (const auto& [name, value] : values) {
    if (!first) {
      out << ",";
    }
    first = false;
    out << "\"" << json_escape(name) << "\":" << value;
  }
  out << "}";
}

void write_operation_counts_json(std::ostringstream& out, const OperationCounts& counts) {
  out << "{";
  out << "\"rotations\":" << counts.rotations << ",";
  out << "\"ct_pt_mul\":" << counts.ct_pt_mul << ",";
  out << "\"ct_ct_mul\":" << counts.ct_ct_mul << ",";
  out << "\"adds\":" << counts.adds << ",";
  out << "\"unity_level_align_muls\":" << counts.unity_level_align_muls << ",";
  out << "\"bootstraps\":" << counts.bootstraps;
  out << "}";
}

void write_phase_operation_counts_json(
    std::ostringstream& out,
    const std::map<std::string, OperationCounts>& values) {
  out << "{";
  bool first = true;
  for (const auto& [name, counts] : values) {
    if (!first) {
      out << ",";
    }
    first = false;
    out << "\"" << json_escape(name) << "\":";
    write_operation_counts_json(out, counts);
  }
  out << "}";
}

void write_artifact_prefix(std::ostringstream& out, const Config& args) {
  out << "\"version\":\"" << json_escape(args.artifact_version) << "\",";
  out << "\"repo_commit\":\"" << json_escape(args.repo_commit) << "\",";
  out << "\"stage\":\"stage1-rank-gate-fideslib-projection\",";
  out << "\"backend\":\"fideslib-gpu\",";
  out << "\"encrypted\":true,";
  out << "\"config\":{\"input_mode\":\"stage1-rank-gate-fideslib-projection\"},";
}

void write_runtime_failure_payload(
    const Config& args,
    std::string_view phase,
    std::string_view message) {
  if (args.output_json.empty()) {
    return;
  }
  std::ostringstream out;
  out << "{";
  write_artifact_prefix(out, args);
  out << "\"status\":\"failed\",";
  out << "\"passed\":false,";
  out << "\"failure_phase\":\"" << json_escape(phase) << "\",";
  out << "\"error_message\":\"" << json_escape(message) << "\",";
  out << "\"parameters\":{";
  out << "\"ring_dimension\":" << args.ring_dim << ",";
  out << "\"multiplicative_depth\":" << args.multiplicative_depth << ",";
  out << "\"scaling_mod_size\":" << args.scaling_mod_size << ",";
  out << "\"chain_steps\":" << args.chain_steps << ",";
  out << "\"bootstrap_before_chain_steps\":";
  write_int_set_json(out, args.bootstrap_before_chain_steps);
  out << "},";
  out << "\"operation_counts\":{";
  out << "\"bootstraps\":0,";
  out << "\"rotations\":0,";
  out << "\"ct_ct_mul\":0,";
  out << "\"ct_pt_mul\":0";
  out << "},";
  out << "\"measurement_scope\":{";
  out << "\"rank_gate_payload\":true,";
  out << "\"state_major_layout\":true,";
  out << "\"scheduled_bootstrap_chain\":"
      << (args.bootstrap_before_chain_steps.empty() ? "false" : "true") << ",";
  out << "\"bootstrap_before_chain_steps\":";
  write_int_set_json(out, args.bootstrap_before_chain_steps);
  out << ",";
  out << "\"non_success_probe\":true,";
  out << "\"success_not_expected\":true,";
  out << "\"full_model_correctness_claimed\":false,";
  out << "\"claim\":\"Native FIDESlib rank/gate chain failed before final decrypt; "
         "this artifact preserves the failure phase for collection.\"";
  out << "}";
  out << "}";
  write_payload(args.output_json, out.str());
}

auto build_repeated_chain_reference(
    const stage1::RankGatePayload& payload,
    int chain_steps) -> ChainReference {
  const auto& config = payload.config;
  const auto state_rank = static_cast<std::size_t>(config.d_state) * config.mimo_rank;
  const auto& decay = payload.array("reference_decay_state_major_poly").values;
  const auto& previous = payload.array("previous_state").values;
  const auto& first_state = payload.array("reference_state_new_poly").values;
  if (decay.size() != state_rank || previous.size() != state_rank ||
      first_state.size() != state_rank) {
    throw std::runtime_error("invalid state-major reference vector sizes");
  }
  std::vector<double> input_term(state_rank, 0.0);
  std::vector<double> state = previous;
  for (std::size_t index = 0; index < state_rank; ++index) {
    input_term[index] = first_state[index] - decay[index] * previous[index];
  }
  for (int step = 0; step < chain_steps; ++step) {
    for (std::size_t index = 0; index < state_rank; ++index) {
      state[index] = decay[index] * state[index] + input_term[index];
    }
  }

  const auto& c_state = payload.array("reference_c_state_major_poly").values;
  const auto& skip = payload.array("reference_skip_update_poly").values;
  const auto& gate = payload.array("reference_gate_poly").values;
  const auto& w_out = payload.array("w_out").values;
  const auto& residual = payload.array("residual_input").values;
  if (c_state.size() != state_rank || skip.size() != config.mimo_rank ||
      gate.size() != config.mimo_rank ||
      w_out.size() != static_cast<std::size_t>(config.d_model) * config.mimo_rank ||
      residual.size() != config.d_model) {
    throw std::runtime_error("invalid chain readout reference vector sizes");
  }

  std::vector<double> readout(config.mimo_rank, 0.0);
  for (std::uint32_t state_index = 0; state_index < config.d_state; ++state_index) {
    const auto base = static_cast<std::size_t>(state_index) * config.mimo_rank;
    for (std::uint32_t rank_index = 0; rank_index < config.mimo_rank; ++rank_index) {
      const auto index = base + rank_index;
      readout[rank_index] += c_state[index] * state[index];
    }
  }

  std::vector<double> rank_output(config.mimo_rank, 0.0);
  std::vector<double> rank_payload(config.mimo_rank, 0.0);
  for (std::uint32_t rank_index = 0; rank_index < config.mimo_rank; ++rank_index) {
    rank_output[rank_index] = readout[rank_index] + skip[rank_index];
    rank_payload[rank_index] = rank_output[rank_index] * gate[rank_index];
  }

  std::vector<double> output_model(config.d_model, 0.0);
  for (std::uint32_t output = 0; output < config.d_model; ++output) {
    double value = residual[output];
    const auto base = static_cast<std::size_t>(output) * config.mimo_rank;
    for (std::uint32_t rank_index = 0; rank_index < config.mimo_rank; ++rank_index) {
      value += w_out[base + rank_index] * rank_payload[rank_index];
    }
    output_model[output] = value;
  }
  return ChainReference{
      .state_new = state,
      .readout_rank = readout,
      .rank_output = rank_output,
      .rank_payload = rank_payload,
      .output_model = output_model,
  };
}

}  // namespace

auto main(int argc, char* argv[]) -> int {
  Config args;
  bool args_available = false;
  try {
    args = parse_args(argc, argv);
    args_available = true;
    const auto input_paths = configured_input_paths(args);
    const auto payloads = read_rank_gate_payloads(input_paths);
    const auto& first_payload = payloads.front();
    const auto& payload = payloads.back();
    const bool model_layout_ciphertext_handoff = payloads.size() > 1;
    const auto handoff_reference = stage1::evaluate_rank_gate_payload_chain_handoff(payloads);
    const auto reference = stage1::evaluate_rank_gate_payload(payload);
    const auto required_rotations = required_rank_gate_rotations_union(payloads);
    const auto batch_size =
        static_cast<int>(first_payload.config.rank_pad * first_payload.config.d_state);
    if (batch_size <= 0 || batch_size > args.ring_dim / 2) {
      throw std::runtime_error("payload batch size does not fit ring dimension");
    }
    log_phase(
        "payload loaded count=" + std::to_string(payloads.size()) +
        " d_model=" + std::to_string(first_payload.config.d_model) +
        " rank=" + std::to_string(first_payload.config.mimo_rank) +
        " d_state=" + std::to_string(first_payload.config.d_state) +
        " rotations=" + std::to_string(required_rotations.size()));

    const auto setup_start = now();
    log_phase("context setup begin");
    CCParams<CryptoContextCKKSRNS> parameters;
    parameters.SetSecretKeyDist(resolve_secret_key_dist(args.secret_key_dist));
    parameters.SetSecurityLevel(resolve_security(args.security));
    parameters.SetRingDim(static_cast<uint32_t>(args.ring_dim));
    parameters.SetScalingTechnique(FLEXIBLEAUTO);
    parameters.SetFirstModSize(static_cast<uint32_t>(args.first_mod_size));
    parameters.SetKeySwitchTechnique(HYBRID);
    parameters.SetMultiplicativeDepth(static_cast<uint32_t>(args.multiplicative_depth));
    parameters.SetScalingModSize(static_cast<uint32_t>(args.scaling_mod_size));
    parameters.SetBatchSize(static_cast<uint32_t>(batch_size));
    parameters.SetDevices({0});
    parameters.SetPlaintextAutoload(false);
    parameters.SetCiphertextAutoload(true);
    if (args.secret_key_dist == "sparse-ternary" ||
        args.secret_key_dist == "sparse-encapsulated") {
      parameters.SetNumLargeDigits(3);
    }

    CryptoContext<DCRTPoly> cc = GenCryptoContext(parameters);
    cc->Enable(PKE);
    cc->Enable(KEYSWITCH);
    cc->Enable(LEVELEDSHE);
    if (!args.bootstrap_before_chain_steps.empty()) {
      cc->Enable(ADVANCEDSHE);
      cc->Enable(FHE);
    }
    auto keys = cc->KeyGen();
    cc->EvalMultKeyGen(keys.secretKey);
    log_phase("context setup done");
    const auto keygen_start = now();
    log_phase("rotation keygen begin");
    cc->EvalRotateKeyGen(keys.secretKey, required_rotations);
    const double rotate_keygen_seconds = seconds_since(keygen_start);
    log_phase("rotation keygen done");
    double bootstrap_precompute_seconds = 0.0;
    if (!args.bootstrap_before_chain_steps.empty()) {
      const auto bootstrap_precompute_start = now();
      log_phase("bootstrap setup/keygen begin");
      const std::vector<uint32_t> level_budget = {
          static_cast<uint32_t>(args.bootstrap_level_budget_cts),
          static_cast<uint32_t>(args.bootstrap_level_budget_stc),
      };
      const std::vector<uint32_t> bsgs_dim = {
          static_cast<uint32_t>(args.bootstrap_bsgs_dim_cts),
          static_cast<uint32_t>(args.bootstrap_bsgs_dim_stc),
      };
      cc->EvalBootstrapSetup(level_budget, bsgs_dim, static_cast<uint32_t>(batch_size), 0);
      cc->EvalBootstrapKeyGen(keys.secretKey, static_cast<uint32_t>(batch_size));
      bootstrap_precompute_seconds = seconds_since(bootstrap_precompute_start);
      log_phase("bootstrap setup/keygen done");
    }
    const auto load_start = now();
    log_phase("load context begin");
    cc->LoadContext(keys.publicKey);
    const double load_context_seconds = seconds_since(load_start);
    log_phase("load context done");
    const double setup_seconds = seconds_since(setup_start);

    auto make_plain = [&](const std::vector<double>& values) {
      auto plain = cc->MakeCKKSPackedPlaintext(values);
      plain->SetLength(static_cast<size_t>(batch_size));
      return plain;
    };
    auto encrypt_values = [&](const std::vector<double>& values) {
      auto plain = make_plain(values);
      return cc->Encrypt(keys.publicKey, plain);
    };

    const auto eval_start = now();
    log_phase("encrypt/projection eval begin");
    int rotations = 0;
    int ct_pt_muls = 0;
    int ct_ct_muls = 0;
    int unity_multiplies = 0;
    int adds = 0;
    int bootstraps = 0;
    double bootstrap_eval_seconds = 0.0;
    std::map<std::string, double> phase_timings;
    std::map<std::string, OperationCounts> phase_operation_counts;

    auto current_operation_counts = [&]() {
      return OperationCounts{
          .rotations = rotations,
          .ct_pt_mul = ct_pt_muls,
          .ct_ct_mul = ct_ct_muls,
          .adds = adds,
          .unity_level_align_muls = unity_multiplies,
          .bootstraps = bootstraps,
      };
    };
    auto record_phase = [&](
                            const std::string& name,
                            std::chrono::steady_clock::time_point phase_start,
                            const OperationCounts& before) {
      phase_timings[name] += seconds_since(phase_start);
      phase_operation_counts[name] = add_counts(
          phase_operation_counts[name],
          subtract_counts(current_operation_counts(), before));
    };
    auto time_phase = [&](const std::string& name, auto&& work)
        -> std::invoke_result_t<decltype(work)&> {
      using Result = std::invoke_result_t<decltype(work)&>;
      const auto before = current_operation_counts();
      const auto phase_start = now();
      if constexpr (std::is_void_v<Result>) {
        work();
        record_phase(name, phase_start, before);
      } else {
        Result result = work();
        record_phase(name, phase_start, before);
        return result;
      }
    };

    auto evaluate_payload_layer = [&](
                                      const stage1::RankGatePayload& layer_payload,
                                      const Ciphertext<DCRTPoly>* residual_override_ct,
                                      std::size_t payload_index) -> LayerCiphertexts {
      LayerCiphertexts result;
      log_phase("payload layer " + std::to_string(payload_index) + " eval begin");
      const std::string phase_prefix = "layer_" + std::to_string(payload_index) + ".";
      time_phase(phase_prefix + "encrypt_rms_input", [&]() {
        result.rms = encrypt_values(
            pack_block0(layer_payload.array("rms_input").values, layer_payload.config));
      });
      const int model_baby_step = static_cast<int>(layer_payload.config.model_baby_step);
      auto rms_model_babies = time_phase(phase_prefix + "rms_model_baby_rotations", [&]() {
        return slot_bsgs_precompute_baby_rotations(
            cc,
            result.rms,
            model_baby_step,
            rotations);
      });
      time_phase(phase_prefix + "conv_projection", [&]() {
        result.conv_pre = slot_bsgs_linear_block0_from_babies(
            cc,
            rms_model_babies,
            scaled_values(
                layer_payload.array("effective_rank_weight").values,
                args.rank_projection_scale),
            static_cast<int>(layer_payload.config.d_model),
            static_cast<int>(layer_payload.config.mimo_rank),
            model_baby_step,
            batch_size,
            rotations,
            ct_pt_muls,
            adds);
        auto conv_bias_ct =
            encrypt_values(pack_block0(
                scaled_values(layer_payload.array("conv_bias").values, args.rank_projection_scale),
                layer_payload.config));
        align_levels(cc, result.conv_pre, conv_bias_ct, unity_multiplies);
        result.conv_pre = cc->EvalAdd(result.conv_pre, conv_bias_ct);
        ++adds;
      });
      log_phase("payload layer " + std::to_string(payload_index) + " conv projection done");
      result.conv_pre_for_silu = result.conv_pre;
      if (args.rank_projection_scale != 1.0) {
        time_phase(phase_prefix + "conv_unscale", [&]() {
          auto unscale_plain = make_plain(
              pack_block0(
                  repeated_rank_values(
                      1.0 / args.rank_projection_scale,
                      layer_payload.config),
                  layer_payload.config));
          result.conv_pre_for_silu = cc->EvalMult(result.conv_pre, unscale_plain);
          ++ct_pt_muls;
        });
      }

      result.gate_pre = time_phase(phase_prefix + "gate_projection", [&]() {
        return slot_bsgs_linear_block0_from_babies(
            cc,
            rms_model_babies,
            layer_payload.array("gate_weight").values,
            static_cast<int>(layer_payload.config.d_model),
            static_cast<int>(layer_payload.config.mimo_rank),
            model_baby_step,
            batch_size,
            rotations,
            ct_pt_muls,
            adds);
      });
      log_phase("payload layer " + std::to_string(payload_index) + " gate projection done");
      result.rank_input_poly = time_phase(phase_prefix + "rank_silu_polynomial", [&]() {
        return evaluate_power_polynomial_block0(
            cc,
            keys.publicKey,
            result.conv_pre_for_silu,
            layer_payload.array("rank_silu_coefficients").values,
            layer_payload.config,
            batch_size,
            ct_ct_muls,
            adds,
            unity_multiplies);
      });
      log_phase("payload layer " + std::to_string(payload_index) + " rank SiLU polynomial done");
      result.gate_poly = time_phase(phase_prefix + "gate_silu_polynomial", [&]() {
        return evaluate_power_polynomial_block0(
            cc,
            keys.publicKey,
            result.gate_pre,
            layer_payload.array("gate_silu_coefficients").values,
            layer_payload.config,
            batch_size,
            ct_ct_muls,
            adds,
            unity_multiplies);
      });
      log_phase("payload layer " + std::to_string(payload_index) + " gate SiLU polynomial done");
      time_phase(phase_prefix + "skip_update", [&]() {
        auto d_skip_plain =
            make_plain(pack_block0(layer_payload.array("d_skip").values, layer_payload.config));
        result.skip_update_poly = cc->EvalMult(result.rank_input_poly, d_skip_plain);
        ++ct_pt_muls;
      });
      log_phase("payload layer " + std::to_string(payload_index) + " skip update done");
      result.dt_rank = static_cast<int>(layer_payload.array("dt_in_weight").shape.at(0));
      result.dt_hidden_poly = time_phase(phase_prefix + "dt_hidden_projection", [&]() {
        return rank_to_vector_linear_reduce(
            cc,
            result.rank_input_poly,
            scaled_values(layer_payload.array("dt_in_weight").values, args.dt_projection_scale),
            layer_payload.config,
            static_cast<std::uint32_t>(result.dt_rank),
            batch_size,
            rotations,
            ct_pt_muls,
            adds);
      });
      log_phase("payload layer " + std::to_string(payload_index) + " dt hidden projection done");
      time_phase(phase_prefix + "dt_rank_projection", [&]() {
        result.dt_pre_poly = vector_to_rank_block_linear_expand(
            cc,
            result.dt_hidden_poly,
            layer_payload.array("dt_proj_weight").values,
            result.dt_rank,
            layer_payload.config,
            batch_size,
            rotations,
            ct_pt_muls,
            adds);
        auto dt_bias_ct = encrypt_values(
            pack_block0(
                scaled_values(layer_payload.array("dt_proj_bias").values, args.dt_projection_scale),
                layer_payload.config));
        align_levels(cc, result.dt_pre_poly, dt_bias_ct, unity_multiplies);
        result.dt_pre_poly = cc->EvalAdd(result.dt_pre_poly, dt_bias_ct);
        ++adds;
      });
      log_phase("payload layer " + std::to_string(payload_index) + " dt rank projection done");
      log_phase("payload layer " + std::to_string(payload_index) + " dt bias add done");
      result.dt_state_major_poly = time_phase(phase_prefix + "dt_state_major_broadcast", [&]() {
        return rank_block0_to_state_major_ciphertext(
            cc,
            result.dt_pre_poly,
            layer_payload.config,
            rotations,
            adds);
      });
      log_phase("payload layer " + std::to_string(payload_index) + " dt state-major broadcast done");
      result.decay_state_major_poly =
          time_phase(phase_prefix + "decay_state_major_polynomial", [&]() {
            return evaluate_state_major_vector_power_polynomial(
                cc,
                keys.publicKey,
                result.dt_state_major_poly,
                scaled_decay_coefficients(
                    layer_payload.array("decay_coefficients").values,
                    layer_payload.config,
                    args.dt_projection_scale),
                layer_payload.config,
                batch_size,
                ct_ct_muls,
                adds,
                unity_multiplies);
          });
      log_phase(
          "payload layer " + std::to_string(payload_index) +
          " dynamic decay projection/polynomial done");
      result.b_vec_poly = time_phase(phase_prefix + "dynamic_b_projection", [&]() {
        return rank_to_state_vector_linear_reduce(
            cc,
            result.rank_input_poly,
            layer_payload.array("b_weight").values,
            layer_payload.config,
            batch_size,
            rotations,
            ct_pt_muls,
            adds);
      });
      result.c_vec_poly = time_phase(phase_prefix + "dynamic_c_projection", [&]() {
        return rank_to_state_vector_linear_reduce(
            cc,
            result.rank_input_poly,
            layer_payload.array("c_weight").values,
            layer_payload.config,
            batch_size,
            rotations,
            ct_pt_muls,
            adds);
      });
      log_phase("payload layer " + std::to_string(payload_index) + " dynamic B/C projections done");
      result.b_state_major_poly = time_phase(phase_prefix + "dynamic_b_broadcast", [&]() {
        return state_vector_to_state_major_ciphertext(
            cc,
            result.b_vec_poly,
            layer_payload.config,
            batch_size,
            rotations,
            ct_pt_muls,
            adds);
      });
      result.c_state_major_poly = time_phase(phase_prefix + "dynamic_c_broadcast", [&]() {
        return state_vector_to_state_major_ciphertext(
            cc,
            result.c_vec_poly,
            layer_payload.config,
            batch_size,
            rotations,
            ct_pt_muls,
            adds);
      });
      log_phase(
          "payload layer " + std::to_string(payload_index) + " dynamic B/C broadcasts done");
      result.gate_poly_slots_for_error =
          time_phase(phase_prefix + "gate_diagnostic_decrypt", [&]() {
            return decrypt_slots(
                cc,
                keys.secretKey,
                result.gate_poly,
                static_cast<size_t>(batch_size));
          });
      log_phase("payload layer " + std::to_string(payload_index) + " gate diagnostic decrypt done");
      result.x_state_major_poly = time_phase(phase_prefix + "rank_input_broadcast", [&]() {
        return rank_block0_to_state_major_ciphertext(
            cc,
            result.rank_input_poly,
            layer_payload.config,
            rotations,
            adds);
      });
      log_phase(
          "payload layer " + std::to_string(payload_index) + " rank input broadcast done");
      time_phase(phase_prefix + "input_state_term", [&]() {
        Ciphertext<DCRTPoly> b_state_major_aligned_ct = result.b_state_major_poly->Clone();
        align_levels(cc, b_state_major_aligned_ct, result.x_state_major_poly, unity_multiplies);
        result.input_state_term =
            cc->EvalMult(b_state_major_aligned_ct, result.x_state_major_poly);
        ++ct_ct_muls;
      });
      log_phase("payload layer " + std::to_string(payload_index) + " input state term done");

      result.previous_state_is_zero =
          is_near_zero_vector(layer_payload.array("previous_state").values);
      if (result.previous_state_is_zero) {
        time_phase(phase_prefix + "zero_state_select", [&]() {
          result.state_new_poly = result.input_state_term;
        });
        log_phase("payload layer " + std::to_string(payload_index) + " zero state skipped");
      } else {
        time_phase(phase_prefix + "state_update", [&]() {
          auto previous_state_ct = encrypt_values(
              pack_state_major_values(
                  layer_payload.array("previous_state").values,
                  layer_payload.config));
          Ciphertext<DCRTPoly> decay_state_major_aligned_ct =
              result.decay_state_major_poly->Clone();
          align_levels(cc, decay_state_major_aligned_ct, previous_state_ct, unity_multiplies);
          auto decay_state_term_ct =
              cc->EvalMult(decay_state_major_aligned_ct, previous_state_ct);
          ++ct_ct_muls;
          auto input_state_term_for_update_ct = result.input_state_term->Clone();
          align_levels(cc, decay_state_term_ct, input_state_term_for_update_ct, unity_multiplies);
          result.state_new_poly = cc->EvalAdd(decay_state_term_ct, input_state_term_for_update_ct);
          ++adds;
        });
        log_phase("payload layer " + std::to_string(payload_index) + " state update done");
      }

      auto rank_mask_plain = make_plain(rank_valid_mask(layer_payload.config));
      const int out_baby_step = bounded_baby_step(
          static_cast<int>(layer_payload.config.rank_baby_step),
          static_cast<int>(layer_payload.config.mimo_rank));
      auto residual_base_ct = residual_override_ct == nullptr
                                  ? encrypt_values(pack_block0(
                                        layer_payload.array("residual_input").values,
                                        layer_payload.config))
                                  : (*residual_override_ct)->Clone();
      auto evaluate_tail_from_state =
          [&](const Ciphertext<DCRTPoly>& state_ct) -> TailCiphertexts {
        auto readout_ct = time_phase(phase_prefix + "readout_reduce", [&]() {
          Ciphertext<DCRTPoly> c_state_major_aligned_ct = result.c_state_major_poly->Clone();
          auto state_new_for_readout_ct = state_ct->Clone();
          align_levels(cc, c_state_major_aligned_ct, state_new_for_readout_ct, unity_multiplies);
          auto local_readout_ct = cc->EvalMult(c_state_major_aligned_ct, state_new_for_readout_ct);
          ++ct_ct_muls;
          for (const auto step : state_major_to_rank_block0_rotations(layer_payload.config)) {
            local_readout_ct =
                cc->EvalAdd(local_readout_ct, cc->EvalRotate(local_readout_ct, step));
            ++rotations;
            ++adds;
          }
          local_readout_ct = cc->EvalMult(local_readout_ct, rank_mask_plain);
          ++ct_pt_muls;
          return local_readout_ct;
        });
        auto rank_output_ct = time_phase(phase_prefix + "rank_skip_add", [&]() {
          auto readout_for_rank_output_ct = readout_ct->Clone();
          auto skip_for_rank_output_ct = result.skip_update_poly->Clone();
          align_levels(cc, readout_for_rank_output_ct, skip_for_rank_output_ct, unity_multiplies);
          auto local_rank_output_ct =
              cc->EvalAdd(readout_for_rank_output_ct, skip_for_rank_output_ct);
          ++adds;
          return local_rank_output_ct;
        });
        auto rank_payload_ct = time_phase(phase_prefix + "rank_gate_product", [&]() {
          auto rank_output_for_gate_ct = rank_output_ct->Clone();
          auto gate_for_tail_ct = result.gate_poly->Clone();
          align_levels(cc, rank_output_for_gate_ct, gate_for_tail_ct, unity_multiplies);
          auto local_rank_payload_ct = cc->EvalMult(rank_output_for_gate_ct, gate_for_tail_ct);
          ++ct_ct_muls;
          return local_rank_payload_ct;
        });
        auto output_delta_ct = time_phase(phase_prefix + "output_projection", [&]() {
          return slot_bsgs_linear_block0(
              cc,
              rank_payload_ct,
              layer_payload.array("w_out").values,
              static_cast<int>(layer_payload.config.mimo_rank),
              static_cast<int>(layer_payload.config.d_model),
              out_baby_step,
              batch_size,
              rotations,
              ct_pt_muls,
              adds);
        });
        auto output_model_ct = time_phase(phase_prefix + "residual_add", [&]() {
          auto residual_ct = residual_base_ct->Clone();
          align_levels(cc, output_delta_ct, residual_ct, unity_multiplies);
          auto local_output_model_ct = cc->EvalAdd(output_delta_ct, residual_ct);
          ++adds;
          return local_output_model_ct;
        });
        return TailCiphertexts{
            .readout = readout_ct,
            .rank_output = rank_output_ct,
            .rank_payload = rank_payload_ct,
            .output_delta = output_delta_ct,
            .output_model = output_model_ct,
        };
      };

      auto tail_ciphertexts = evaluate_tail_from_state(result.state_new_poly);
      result.readout_poly = tail_ciphertexts.readout;
      result.rank_output_poly = tail_ciphertexts.rank_output;
      result.rank_payload_poly = tail_ciphertexts.rank_payload;
      result.output_delta_poly = tail_ciphertexts.output_delta;
      result.output_model_poly = tail_ciphertexts.output_model;
      for (int chain_step = 2; chain_step <= args.chain_steps; ++chain_step) {
        if (args.bootstrap_before_chain_steps.count(chain_step) > 0) {
          const auto bootstrap_start = now();
          time_phase(
              phase_prefix + "chain_step_" + std::to_string(chain_step) + ".bootstrap",
              [&]() {
                result.state_new_poly = cc->EvalBootstrap(result.state_new_poly);
                ++bootstraps;
              });
          bootstrap_eval_seconds += seconds_since(bootstrap_start);
          log_phase("bootstrap before ciphertext chain step " + std::to_string(chain_step) + " done");
        }
        time_phase(
            phase_prefix + "chain_step_" + std::to_string(chain_step) + ".state_update",
            [&]() {
              Ciphertext<DCRTPoly> decay_state_major_aligned_ct =
                  result.decay_state_major_poly->Clone();
              auto previous_state_for_chain_ct = result.state_new_poly->Clone();
              align_levels(
                  cc,
                  decay_state_major_aligned_ct,
                  previous_state_for_chain_ct,
                  unity_multiplies);
              auto decay_state_term_ct =
                  cc->EvalMult(decay_state_major_aligned_ct, previous_state_for_chain_ct);
              ++ct_ct_muls;
              auto input_state_term_for_chain_ct = result.input_state_term->Clone();
              align_levels(cc, decay_state_term_ct, input_state_term_for_chain_ct, unity_multiplies);
              result.state_new_poly = cc->EvalAdd(decay_state_term_ct, input_state_term_for_chain_ct);
              ++adds;
            });
        tail_ciphertexts = evaluate_tail_from_state(result.state_new_poly);
        result.readout_poly = tail_ciphertexts.readout;
        result.rank_output_poly = tail_ciphertexts.rank_output;
        result.rank_payload_poly = tail_ciphertexts.rank_payload;
        result.output_delta_poly = tail_ciphertexts.output_delta;
        result.output_model_poly = tail_ciphertexts.output_model;
        log_phase("ciphertext chain step " + std::to_string(chain_step) + " done");
      }
      log_phase("payload layer " + std::to_string(payload_index) + " eval done");
      return result;
    };

    std::vector<LayerCiphertexts> layer_results;
    layer_results.reserve(payloads.size());
    Ciphertext<DCRTPoly> residual_override_ct;
    bool has_residual_override = false;
    for (std::size_t payload_index = 0; payload_index < payloads.size(); ++payload_index) {
      auto layer_result = evaluate_payload_layer(
          payloads[payload_index],
          has_residual_override ? &residual_override_ct : nullptr,
          payload_index);
      residual_override_ct = layer_result.output_model_poly;
      has_residual_override = true;
      layer_results.push_back(layer_result);
    }
    const auto& final_layer = layer_results.back();
    auto rms_ct = final_layer.rms;
    auto conv_pre_ct = final_layer.conv_pre;
    auto conv_pre_for_silu_ct = final_layer.conv_pre_for_silu;
    auto gate_pre_ct = final_layer.gate_pre;
    auto rank_input_poly_ct = final_layer.rank_input_poly;
    auto gate_poly_ct = final_layer.gate_poly;
    auto skip_update_poly_ct = final_layer.skip_update_poly;
    auto dt_hidden_poly_ct = final_layer.dt_hidden_poly;
    auto dt_pre_poly_ct = final_layer.dt_pre_poly;
    auto dt_state_major_poly_ct = final_layer.dt_state_major_poly;
    auto decay_state_major_poly_ct = final_layer.decay_state_major_poly;
    auto b_vec_poly_ct = final_layer.b_vec_poly;
    auto c_vec_poly_ct = final_layer.c_vec_poly;
    auto b_state_major_poly_ct = final_layer.b_state_major_poly;
    auto c_state_major_poly_ct = final_layer.c_state_major_poly;
    auto x_state_major_poly_ct = final_layer.x_state_major_poly;
    auto input_state_term_ct = final_layer.input_state_term;
    auto state_new_poly_ct = final_layer.state_new_poly;
    auto readout_poly_ct = final_layer.readout_poly;
    auto rank_output_poly_ct = final_layer.rank_output_poly;
    auto rank_payload_poly_ct = final_layer.rank_payload_poly;
    auto output_delta_poly_ct = final_layer.output_delta_poly;
    auto output_model_poly_ct = final_layer.output_model_poly;
    const auto gate_poly_slots_for_error = final_layer.gate_poly_slots_for_error;
    const int dt_rank = final_layer.dt_rank;
    const bool previous_state_is_zero = final_layer.previous_state_is_zero;
    log_phase("polynomial recurrence/readout/output tail done");
    const double eval_seconds = seconds_since(eval_start);
    log_phase(
        "projection eval done rotations=" + std::to_string(rotations) +
        " ct_pt=" + std::to_string(ct_pt_muls) +
        " ct_ct=" + std::to_string(ct_ct_muls));

    auto build_ckks_levels_json = [&]() -> std::string {
      std::ostringstream levels;
      levels << "{";
      bool first_ckks_level = true;
      write_level_field(levels, "rms_input", rms_ct, first_ckks_level);
      write_level_field(levels, "conv_pre", conv_pre_ct, first_ckks_level);
      write_level_field(levels, "conv_pre_for_silu", conv_pre_for_silu_ct, first_ckks_level);
      write_level_field(levels, "gate_pre", gate_pre_ct, first_ckks_level);
      write_level_field(levels, "rank_input_poly", rank_input_poly_ct, first_ckks_level);
      write_level_field(levels, "gate_poly", gate_poly_ct, first_ckks_level);
      write_level_field(levels, "skip_update_poly", skip_update_poly_ct, first_ckks_level);
      write_level_field(levels, "dt_hidden_poly", dt_hidden_poly_ct, first_ckks_level);
      write_level_field(levels, "dt_pre_poly", dt_pre_poly_ct, first_ckks_level);
      write_level_field(levels, "dt_state_major_poly", dt_state_major_poly_ct, first_ckks_level);
      write_level_field(levels, "decay_state_major_poly", decay_state_major_poly_ct, first_ckks_level);
      write_level_field(levels, "b_vec_poly", b_vec_poly_ct, first_ckks_level);
      write_level_field(levels, "c_vec_poly", c_vec_poly_ct, first_ckks_level);
      write_level_field(levels, "b_state_major_poly", b_state_major_poly_ct, first_ckks_level);
      write_level_field(levels, "c_state_major_poly", c_state_major_poly_ct, first_ckks_level);
      write_level_field(levels, "x_state_major_poly", x_state_major_poly_ct, first_ckks_level);
      write_level_field(levels, "input_state_term", input_state_term_ct, first_ckks_level);
      write_level_field(levels, "state_new_poly", state_new_poly_ct, first_ckks_level);
      write_level_field(levels, "readout_poly", readout_poly_ct, first_ckks_level);
      write_level_field(levels, "rank_output_poly", rank_output_poly_ct, first_ckks_level);
      write_level_field(levels, "rank_payload_poly", rank_payload_poly_ct, first_ckks_level);
      write_level_field(levels, "output_delta_poly", output_delta_poly_ct, first_ckks_level);
      write_level_field(levels, "output_model_poly", output_model_poly_ct, first_ckks_level);
      levels << "}";
      return levels.str();
    };
    const auto ckks_levels_json = build_ckks_levels_json();
    auto write_decrypt_failure_payload = [&](const std::string& message) {
      std::ostringstream failure;
      failure << "{";
      write_artifact_prefix(failure, args);
      failure << "\"status\":\"failed\",";
      failure << "\"passed\":false,";
      failure << "\"failure_phase\":\"decrypt\",";
      failure << "\"error_message\":\"" << json_escape(message) << "\",";
      failure << "\"parameters\":{";
      failure << "\"d_model\":" << payload.config.d_model << ",";
      failure << "\"d_state\":" << payload.config.d_state << ",";
      failure << "\"mimo_rank\":" << payload.config.mimo_rank << ",";
      failure << "\"rank_pad\":" << payload.config.rank_pad << ",";
      failure << "\"batch_size\":" << batch_size << ",";
      failure << "\"ring_dimension\":" << args.ring_dim << ",";
      failure << "\"multiplicative_depth\":" << args.multiplicative_depth << ",";
      failure << "\"scaling_mod_size\":" << args.scaling_mod_size << ",";
      failure << "\"payload_count\":" << payloads.size() << ",";
      failure << "\"chain_steps\":" << args.chain_steps << ",";
      failure << "\"bootstrap_before_chain_steps\":";
      write_int_set_json(failure, args.bootstrap_before_chain_steps);
      failure << "},";
      failure << "\"measurements\":{";
      failure << "\"required_application_rotation_key_count\":" << required_rotations.size()
              << ",";
      failure << "\"previous_state_nonzero\":"
              << (previous_state_is_zero ? "false" : "true") << ",";
      failure << "\"model_layout_ciphertext_handoff\":"
              << (model_layout_ciphertext_handoff ? "true" : "false") << ",";
      failure << "\"peak_rss_gib\":" << peak_rss_gib() << ",";
      failure << "\"rss_gib\":" << rss_gib();
      failure << "},";
      failure << "\"ckks_levels\":" << ckks_levels_json << ",";
      failure << "\"timing\":{";
      failure << "\"setup_seconds\":" << setup_seconds << ",";
      failure << "\"rotate_keygen_seconds\":" << rotate_keygen_seconds << ",";
      failure << "\"bootstrap_precompute_seconds\":" << bootstrap_precompute_seconds << ",";
      failure << "\"bootstrap_eval_seconds\":" << bootstrap_eval_seconds << ",";
      failure << "\"load_context_seconds\":" << load_context_seconds << ",";
      failure << "\"eval_seconds\":" << eval_seconds;
      failure << "},";
      failure << "\"phase_timings\":";
      write_double_map_json(failure, phase_timings);
      failure << ",";
      failure << "\"phase_operation_counts\":";
      write_phase_operation_counts_json(failure, phase_operation_counts);
      failure << ",";
      failure << "\"operation_counts\":{";
      failure << "\"rotations\":" << rotations << ",";
      failure << "\"ct_pt_mul\":" << ct_pt_muls << ",";
      failure << "\"ct_ct_mul\":" << ct_ct_muls << ",";
      failure << "\"adds\":" << adds << ",";
      failure << "\"unity_level_align_muls\":" << unity_multiplies << ",";
      failure << "\"bootstraps\":" << bootstraps;
      failure << "},";
      failure << "\"measurement_scope\":{";
      failure << "\"rank_gate_payload\":true,";
      failure << "\"state_major_layout\":true,";
      failure << "\"recurrence_tail_executed\":true,";
      failure << "\"ciphertext_recurrent_state_chain\":"
              << (args.chain_steps > 1 ? "true" : "false") << ",";
      failure << "\"model_layout_ciphertext_handoff\":"
              << (model_layout_ciphertext_handoff ? "true" : "false") << ",";
      failure << "\"pre_recurrence_payload_reference_per_layer\":"
              << (model_layout_ciphertext_handoff ? "true" : "false") << ",";
      failure << "\"payload_count\":" << payloads.size() << ",";
      failure << "\"chain_steps\":" << args.chain_steps << ",";
      failure << "\"scheduled_bootstrap_chain\":"
              << (args.bootstrap_before_chain_steps.empty() ? "false" : "true") << ",";
      failure << "\"bootstrap_before_chain_steps\":";
      write_int_set_json(failure, args.bootstrap_before_chain_steps);
      failure << ",";
      failure << "\"executed_bootstrap_count\":" << bootstraps << ",";
      failure << "\"chain_exact_reference_available\":"
              << (args.chain_steps == 1 ? "true" : "false") << ",";
      failure << "\"previous_state_nonzero\":"
              << (previous_state_is_zero ? "false" : "true") << ",";
      failure << "\"ckks_level_telemetry\":true,";
      failure << "\"decrypt_failure_artifact\":true,";
      failure << "\"full_model_correctness_claimed\":false,";
      failure << "\"claim\":\"Native FIDESlib rank/gate chain reached decrypt but failed "
                 "before final correctness comparison; this artifact preserves telemetry "
                 "and does not claim full Mamba model correctness.\"";
      failure << "}";
      failure << "}";
      write_payload(args.output_json, failure.str());
    };

    std::vector<double> conv_pre_slots;
    std::vector<double> gate_pre_slots;
    std::vector<double> rank_input_poly_slots;
    std::vector<double> skip_update_poly_slots;
    std::vector<double> dt_hidden_poly_slots;
    std::vector<double> dt_pre_poly_slots;
    std::vector<double> dt_state_major_poly_slots;
    std::vector<double> decay_state_major_poly_slots;
    std::vector<double> b_vec_poly_slots;
    std::vector<double> c_vec_poly_slots;
    std::vector<double> b_state_major_poly_slots;
    std::vector<double> c_state_major_poly_slots;
    std::vector<double> state_new_poly_slots;
    std::vector<double> readout_poly_slots;
    std::vector<double> rank_output_poly_slots;
    std::vector<double> rank_payload_poly_slots;
    std::vector<double> output_model_poly_slots;
    log_phase("decrypt begin");
    try {
      conv_pre_slots = decrypt_slots(
          cc,
          keys.secretKey,
          conv_pre_ct,
          static_cast<size_t>(batch_size));
      gate_pre_slots = decrypt_slots(
          cc,
          keys.secretKey,
          gate_pre_ct,
          static_cast<size_t>(batch_size));
      rank_input_poly_slots = decrypt_slots(
          cc,
          keys.secretKey,
          rank_input_poly_ct,
          static_cast<size_t>(batch_size));
      skip_update_poly_slots = decrypt_slots(
          cc,
          keys.secretKey,
          skip_update_poly_ct,
          static_cast<size_t>(batch_size));
      dt_hidden_poly_slots = decrypt_slots(
          cc,
          keys.secretKey,
          dt_hidden_poly_ct,
          static_cast<size_t>(batch_size));
      dt_pre_poly_slots = decrypt_slots(
          cc,
          keys.secretKey,
          dt_pre_poly_ct,
          static_cast<size_t>(batch_size));
      dt_state_major_poly_slots = decrypt_slots(
          cc,
          keys.secretKey,
          dt_state_major_poly_ct,
          static_cast<size_t>(batch_size));
      decay_state_major_poly_slots = decrypt_slots(
          cc,
          keys.secretKey,
          decay_state_major_poly_ct,
          static_cast<size_t>(batch_size));
      b_vec_poly_slots = decrypt_slots(
          cc,
          keys.secretKey,
          b_vec_poly_ct,
          static_cast<size_t>(batch_size));
      c_vec_poly_slots = decrypt_slots(
          cc,
          keys.secretKey,
          c_vec_poly_ct,
          static_cast<size_t>(batch_size));
      b_state_major_poly_slots = decrypt_slots(
          cc,
          keys.secretKey,
          b_state_major_poly_ct,
          static_cast<size_t>(batch_size));
      c_state_major_poly_slots = decrypt_slots(
          cc,
          keys.secretKey,
          c_state_major_poly_ct,
          static_cast<size_t>(batch_size));
      state_new_poly_slots = decrypt_slots(
          cc,
          keys.secretKey,
          state_new_poly_ct,
          static_cast<size_t>(batch_size));
      readout_poly_slots = decrypt_slots(
          cc,
          keys.secretKey,
          readout_poly_ct,
          static_cast<size_t>(batch_size));
      rank_output_poly_slots = decrypt_slots(
          cc,
          keys.secretKey,
          rank_output_poly_ct,
          static_cast<size_t>(batch_size));
      rank_payload_poly_slots = decrypt_slots(
          cc,
          keys.secretKey,
          rank_payload_poly_ct,
          static_cast<size_t>(batch_size));
      output_model_poly_slots = decrypt_slots(
          cc,
          keys.secretKey,
          output_model_poly_ct,
          static_cast<size_t>(batch_size));
    } catch (const std::exception& exc) {
      write_decrypt_failure_payload(exc.what());
      throw;
    }
    const auto gate_poly_slots = gate_poly_slots_for_error;
    log_phase("decrypt done");

    const auto conv_pre = unscaled_values(
        first_values(conv_pre_slots, static_cast<size_t>(payload.config.mimo_rank)),
        args.rank_projection_scale);
    const auto gate_pre =
        first_values(gate_pre_slots, static_cast<size_t>(payload.config.mimo_rank));
    const auto rank_input_poly =
        first_values(rank_input_poly_slots, static_cast<size_t>(payload.config.mimo_rank));
    const auto gate_poly =
        first_values(gate_poly_slots, static_cast<size_t>(payload.config.mimo_rank));
    const auto skip_update_poly =
        first_values(skip_update_poly_slots, static_cast<size_t>(payload.config.mimo_rank));
    const auto dt_hidden_poly =
        unscaled_values(first_values(dt_hidden_poly_slots, static_cast<size_t>(dt_rank)),
                        args.dt_projection_scale);
    const auto dt_pre_poly =
        unscaled_values(
            first_values(dt_pre_poly_slots, static_cast<size_t>(payload.config.mimo_rank)),
            args.dt_projection_scale);
    const auto dt_state_major_poly = unscaled_values(
        state_major_valid_values(dt_state_major_poly_slots, payload.config),
        args.dt_projection_scale);
    const auto decay_state_major_poly =
        state_major_valid_values(decay_state_major_poly_slots, payload.config);
    const auto b_vec_poly =
        first_values(b_vec_poly_slots, static_cast<size_t>(payload.config.d_state));
    const auto c_vec_poly =
        first_values(c_vec_poly_slots, static_cast<size_t>(payload.config.d_state));
    const auto b_state_major_poly =
        state_major_valid_values(b_state_major_poly_slots, payload.config);
    const auto c_state_major_poly =
        state_major_valid_values(c_state_major_poly_slots, payload.config);
    const auto state_new_poly = state_major_valid_values(state_new_poly_slots, payload.config);
    const auto readout_poly =
        first_values(readout_poly_slots, static_cast<size_t>(payload.config.mimo_rank));
    const auto rank_output_poly =
        first_values(rank_output_poly_slots, static_cast<size_t>(payload.config.mimo_rank));
    const auto rank_payload_poly =
        first_values(rank_payload_poly_slots, static_cast<size_t>(payload.config.mimo_rank));
    const auto output_model_poly =
        first_values(output_model_poly_slots, static_cast<size_t>(payload.config.d_model));
    const auto chain_reference = build_repeated_chain_reference(payload, args.chain_steps);
    const auto conv_error =
        stage1::max_abs_delta(conv_pre, payload.array("reference_conv_pre").values);
    const auto gate_error =
        stage1::max_abs_delta(gate_pre, payload.array("reference_gate_pre").values);
    const auto rank_poly_error =
        stage1::max_abs_delta(rank_input_poly, payload.array("reference_rank_input_poly").values);
    const auto gate_poly_error =
        stage1::max_abs_delta(gate_poly, payload.array("reference_gate_poly").values);
    const auto skip_poly_error =
        stage1::max_abs_delta(skip_update_poly, payload.array("reference_skip_update_poly").values);
    const auto dt_hidden_poly_error =
        stage1::max_abs_delta(dt_hidden_poly, payload.array("reference_dt_hidden_poly").values);
    const auto dt_pre_poly_error =
        stage1::max_abs_delta(dt_pre_poly, payload.array("reference_dt_pre_poly").values);
    const auto dt_state_major_poly_error = stage1::max_abs_delta(
        dt_state_major_poly,
        payload.array("reference_dt_state_major_poly").values);
    const auto decay_state_major_poly_error = stage1::max_abs_delta(
        decay_state_major_poly,
        payload.array("reference_decay_state_major_poly").values);
    const auto b_vec_poly_error =
        stage1::max_abs_delta(b_vec_poly, payload.array("reference_b_vec_poly").values);
    const auto c_vec_poly_error =
        stage1::max_abs_delta(c_vec_poly, payload.array("reference_c_vec_poly").values);
    const auto b_state_major_poly_error = stage1::max_abs_delta(
        b_state_major_poly,
        payload.array("reference_b_state_major_poly").values);
    const auto c_state_major_poly_error = stage1::max_abs_delta(
        c_state_major_poly,
        payload.array("reference_c_state_major_poly").values);
    const auto state_new_poly_error =
        stage1::max_abs_delta(state_new_poly, chain_reference.state_new);
    const auto readout_poly_error =
        stage1::max_abs_delta(readout_poly, chain_reference.readout_rank);
    const auto rank_output_poly_error =
        stage1::max_abs_delta(rank_output_poly, chain_reference.rank_output);
    const auto rank_payload_poly_error = stage1::max_abs_delta(
        rank_payload_poly,
        chain_reference.rank_payload);
    const auto output_model_poly_error = stage1::max_abs_delta(
        output_model_poly,
        chain_reference.output_model);
    const auto rank_poly_approx_error = stage1::max_abs_delta(
        payload.array("reference_rank_input_poly").values,
        payload.array("reference_rank_input").values);
    const auto gate_poly_approx_error =
        stage1::max_abs_delta(payload.array("reference_gate_poly").values,
                              payload.array("reference_gate").values);
    const auto skip_poly_approx_error = stage1::max_abs_delta(
        payload.array("reference_skip_update_poly").values,
        payload.array("reference_skip_update").values);
    const auto decay_poly_approx_error = stage1::max_abs_delta(
        payload.array("reference_decay_state_major_poly").values,
        payload.array("reference_decay_state_major_exact").values);
    const auto output_model_poly_vs_exact_error = stage1::max_abs_delta(
        payload.array("reference_output_model_poly").values,
        payload.array("reference_output_model_exact").values);
    const auto diagnostic_max_error = std::max(
        {conv_error,
         gate_error,
         rank_poly_error,
         gate_poly_error,
         skip_poly_error,
         dt_hidden_poly_error,
         dt_pre_poly_error,
         dt_state_major_poly_error,
         decay_state_major_poly_error,
         b_vec_poly_error,
         c_vec_poly_error,
         b_state_major_poly_error,
         c_state_major_poly_error,
         state_new_poly_error,
         readout_poly_error,
         rank_output_poly_error,
         rank_payload_poly_error,
         output_model_poly_error});
    const auto max_error = output_model_poly_error;
    const bool passed = max_error <= args.atol;
    const auto& polynomial_metadata = payload.array("polynomial_metadata").values;

    std::ostringstream out;
    out << "{";
    write_artifact_prefix(out, args);
    out << "\"status\":\"" << (passed ? "passed" : "failed") << "\",";
    out << "\"passed\":" << (passed ? "true" : "false") << ",";
    out << "\"parameters\":{";
    out << "\"d_model\":" << payload.config.d_model << ",";
    out << "\"d_state\":" << payload.config.d_state << ",";
    out << "\"mimo_rank\":" << payload.config.mimo_rank << ",";
    out << "\"rank_pad\":" << payload.config.rank_pad << ",";
    out << "\"batch_size\":" << batch_size << ",";
    out << "\"ring_dimension\":" << args.ring_dim << ",";
    out << "\"multiplicative_depth\":" << args.multiplicative_depth << ",";
    out << "\"scaling_mod_size\":" << args.scaling_mod_size << ",";
    out << "\"payload_count\":" << payloads.size() << ",";
    out << "\"chain_steps\":" << args.chain_steps << ",";
    out << "\"bootstrap_before_chain_steps\":";
    write_int_set_json(out, args.bootstrap_before_chain_steps);
    out << ",";
    out << "\"model_baby_step\":" << payload.config.model_baby_step;
    out << ",\"rank_baby_step\":" << payload.config.rank_baby_step;
    out << ",\"rank_projection_scale\":" << args.rank_projection_scale;
    out << ",\"dt_projection_scale\":" << args.dt_projection_scale;
    out << ",\"polynomial_degree\":" << polynomial_metadata[0];
    out << ",\"gate_polynomial_degree\":" << polynomial_metadata[1];
    out << ",\"polynomial_range\":" << polynomial_metadata[2];
    const auto& decay_metadata = payload.array("decay_metadata").values;
    out << ",\"dt_rank\":" << decay_metadata[0];
    out << ",\"decay_polynomial_degree\":" << decay_metadata[1];
    out << ",\"decay_polynomial_range_lower\":" << decay_metadata[2];
    out << ",\"decay_polynomial_range_upper\":" << decay_metadata[3];
    out << "},";
    out << "\"measurements\":{";
    out << "\"max_abs_error\":" << max_error << ",";
    out << "\"diagnostic_max_abs_error\":" << diagnostic_max_error << ",";
    out << "\"conv_pre_max_abs_error\":" << conv_error << ",";
    out << "\"gate_pre_max_abs_error\":" << gate_error << ",";
    out << "\"rank_input_poly_max_abs_error\":" << rank_poly_error << ",";
    out << "\"gate_poly_max_abs_error\":" << gate_poly_error << ",";
    out << "\"skip_update_poly_max_abs_error\":" << skip_poly_error << ",";
    out << "\"dt_hidden_poly_max_abs_error\":" << dt_hidden_poly_error << ",";
    out << "\"dt_pre_poly_max_abs_error\":" << dt_pre_poly_error << ",";
    out << "\"dt_state_major_poly_max_abs_error\":" << dt_state_major_poly_error << ",";
    out << "\"decay_state_major_poly_max_abs_error\":" << decay_state_major_poly_error << ",";
    out << "\"b_vec_poly_max_abs_error\":" << b_vec_poly_error << ",";
    out << "\"c_vec_poly_max_abs_error\":" << c_vec_poly_error << ",";
    out << "\"b_state_major_poly_max_abs_error\":" << b_state_major_poly_error << ",";
    out << "\"c_state_major_poly_max_abs_error\":" << c_state_major_poly_error << ",";
    out << "\"state_new_poly_max_abs_error\":" << state_new_poly_error << ",";
    out << "\"readout_rank_poly_max_abs_error\":" << readout_poly_error << ",";
    out << "\"rank_output_poly_max_abs_error\":" << rank_output_poly_error << ",";
    out << "\"rank_payload_poly_max_abs_error\":" << rank_payload_poly_error << ",";
    out << "\"output_model_poly_max_abs_error\":" << output_model_poly_error << ",";
    out << "\"rank_input_poly_vs_exact_max_abs_error\":" << rank_poly_approx_error << ",";
    out << "\"gate_poly_vs_exact_max_abs_error\":" << gate_poly_approx_error << ",";
    out << "\"skip_update_poly_vs_exact_max_abs_error\":" << skip_poly_approx_error << ",";
    out << "\"decay_poly_vs_exact_max_abs_error\":" << decay_poly_approx_error << ",";
    out << "\"output_model_poly_vs_exact_max_abs_error\":" << output_model_poly_vs_exact_error
        << ",";
    out << "\"output_model_poly_vs_exact_reference_steps\":" << payloads.size() << ",";
    out << "\"native_plaintext_reference_max_abs_error\":" << reference.max_abs_error << ",";
    out << "\"payload_chain_reference_max_abs_error\":" << handoff_reference.max_abs_error << ",";
    out << "\"model_layout_handoff_max_abs_error\":"
        << handoff_reference.model_layout_handoff_max_abs_error << ",";
    out << "\"required_application_rotation_key_count\":" << required_rotations.size() << ",";
    out << "\"plaintext_coefficient_floor\":" << kPlaintextCoefficientFloor << ",";
    out << "\"rank_projection_scaled\":" << (args.rank_projection_scale == 1.0 ? "false" : "true")
        << ",";
    out << "\"previous_state_nonzero\":" << (previous_state_is_zero ? "false" : "true") << ",";
    out << "\"payload_count\":" << payloads.size() << ",";
    out << "\"model_layout_ciphertext_handoff\":"
        << (model_layout_ciphertext_handoff ? "true" : "false") << ",";
    out << "\"chain_steps\":" << args.chain_steps << ",";
    out << "\"scheduled_bootstrap_chain\":"
        << (args.bootstrap_before_chain_steps.empty() ? "false" : "true") << ",";
    out << "\"executed_bootstrap_count\":" << bootstraps << ",";
    out << "\"peak_rss_gib\":" << peak_rss_gib() << ",";
    out << "\"rss_gib\":" << rss_gib();
    out << "},";
    out << "\"ckks_levels\":" << ckks_levels_json << ",";
    out << "\"timing\":{";
    out << "\"setup_seconds\":" << setup_seconds << ",";
    out << "\"rotate_keygen_seconds\":" << rotate_keygen_seconds << ",";
    out << "\"bootstrap_precompute_seconds\":" << bootstrap_precompute_seconds << ",";
    out << "\"bootstrap_eval_seconds\":" << bootstrap_eval_seconds << ",";
    out << "\"load_context_seconds\":" << load_context_seconds << ",";
    out << "\"eval_seconds\":" << eval_seconds;
    out << "},";
    out << "\"phase_timings\":";
    write_double_map_json(out, phase_timings);
    out << ",";
    out << "\"phase_operation_counts\":";
    write_phase_operation_counts_json(out, phase_operation_counts);
    out << ",";
    out << "\"operation_counts\":{";
    out << "\"rotations\":" << rotations << ",";
    out << "\"ct_pt_mul\":" << ct_pt_muls << ",";
    out << "\"ct_ct_mul\":" << ct_ct_muls << ",";
    out << "\"adds\":" << adds << ",";
    out << "\"unity_level_align_muls\":" << unity_multiplies << ",";
    out << "\"bootstraps\":" << bootstraps;
    out << "},";
    out << "\"measurement_scope\":{";
    out << "\"rank_gate_payload\":true,";
    out << "\"state_major_layout\":true,";
    out << "\"pre_recurrence_rank_gate_projection\":true,";
    out << "\"rank_projection_scaled\":"
        << (args.rank_projection_scale == 1.0 ? "false" : "true") << ",";
    out << "\"pre_recurrence_silu_activation\":true,";
    out << "\"pre_recurrence_skip_update\":true,";
    out << "\"pre_recurrence_dynamic_bc\":true,";
    out << "\"pre_recurrence_decay\":true,";
    out << "\"recurrence_tail_executed\":true,";
    out << "\"ciphertext_recurrent_state_chain\":"
        << (args.chain_steps > 1 ? "true" : "false") << ",";
    out << "\"model_layout_ciphertext_handoff\":"
        << (model_layout_ciphertext_handoff ? "true" : "false") << ",";
    out << "\"pre_recurrence_payload_reference_per_layer\":"
        << (model_layout_ciphertext_handoff ? "true" : "false") << ",";
    out << "\"payload_count\":" << payloads.size() << ",";
    out << "\"chain_steps\":" << args.chain_steps << ",";
    out << "\"scheduled_bootstrap_chain\":"
        << (args.bootstrap_before_chain_steps.empty() ? "false" : "true") << ",";
    out << "\"bootstrap_before_chain_steps\":";
    write_int_set_json(out, args.bootstrap_before_chain_steps);
    out << ",";
    out << "\"executed_bootstrap_count\":" << bootstraps << ",";
    out << "\"chain_exact_reference_available\":"
        << (args.chain_steps == 1 ? "true" : "false") << ",";
    out << "\"previous_state_nonzero\":"
        << (previous_state_is_zero ? "false" : "true") << ",";
    out << "\"ckks_level_telemetry\":true,";
    out << "\"full_one_layer_polynomial_output_checked\":true,";
    out << "\"fideslib_encrypted_execution\":true,";
    out << "\"full_layer_pre_recurrence_computed_in_kernel\":true,";
    out << "\"full_model_correctness_claimed\":false,";
    out << "\"claim\":\"Native FIDESlib encrypted rank/gate recurrence-tail projection "
           "artifact for one checkpoint layer, recurrent chain, or model-layout "
           "ciphertext handoff chain; it does not claim lm_head decoding, full "
           "24-layer model correctness, or complete Mamba inference.\"";
    out << "}";
    out << "}";
    write_payload(args.output_json, out.str());
    return passed ? EXIT_SUCCESS : EXIT_FAILURE;
  } catch (const std::exception& exc) {
    if (args_available && !payload_file_exists(args.output_json)) {
      try {
        write_runtime_failure_payload(args, "runtime", exc.what());
      } catch (const std::exception& write_exc) {
        std::cerr << "failed to write runtime failure artifact: " << write_exc.what()
                  << std::endl;
      }
    }
    std::cerr << "stage1_rank_gate_fideslib failed: " << exc.what() << std::endl;
    return EXIT_FAILURE;
  }
}
