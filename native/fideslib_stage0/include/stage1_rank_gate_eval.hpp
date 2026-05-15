#pragma once

#include "stage1_rank_gate_payload.hpp"
#include "stage1_tail_eval.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace stage1 {

struct RankGateEvalResult {
  std::vector<double> conv_pre;
  std::vector<double> rank_input;
  std::vector<double> gate_pre;
  std::vector<double> gate;
  std::vector<double> skip_update;
  double max_abs_error = 0.0;
  double conv_pre_max_abs_error = 0.0;
  double rank_input_max_abs_error = 0.0;
  double gate_pre_max_abs_error = 0.0;
  double gate_max_abs_error = 0.0;
  double skip_update_max_abs_error = 0.0;
};

struct RankGateChainHandoffEvalResult {
  std::size_t payload_count = 0;
  double max_abs_error = 0.0;
  double rank_gate_max_abs_error = 0.0;
  double model_layout_handoff_max_abs_error = 0.0;
};

inline auto rank_gate_array_values(const RankGatePayload& payload, const std::string& name)
    -> const std::vector<double>& {
  return payload.array(name).values;
}

inline auto silu(double value) -> double {
  return value / (1.0 + std::exp(-value));
}

inline auto evaluate_rank_gate_payload(const RankGatePayload& payload) -> RankGateEvalResult {
  const auto d_model = static_cast<std::size_t>(payload.config.d_model);
  const auto rank = static_cast<std::size_t>(payload.config.mimo_rank);

  const auto& rms_input = rank_gate_array_values(payload, "rms_input");
  const auto& effective_rank_weight = rank_gate_array_values(payload, "effective_rank_weight");
  const auto& conv_bias = rank_gate_array_values(payload, "conv_bias");
  const auto& gate_weight = rank_gate_array_values(payload, "gate_weight");
  const auto& d_skip = rank_gate_array_values(payload, "d_skip");

  RankGateEvalResult result;
  result.conv_pre.assign(rank, 0.0);
  result.rank_input.assign(rank, 0.0);
  result.gate_pre.assign(rank, 0.0);
  result.gate.assign(rank, 0.0);
  result.skip_update.assign(rank, 0.0);

  for (std::size_t rank_index = 0; rank_index < rank; ++rank_index) {
    const auto base = rank_index * d_model;
    double conv = conv_bias[rank_index];
    double gate_value = 0.0;
    for (std::size_t model_index = 0; model_index < d_model; ++model_index) {
      conv += effective_rank_weight[base + model_index] * rms_input[model_index];
      gate_value += gate_weight[base + model_index] * rms_input[model_index];
    }
    result.conv_pre[rank_index] = conv;
    result.rank_input[rank_index] = silu(conv);
    result.gate_pre[rank_index] = gate_value;
    result.gate[rank_index] = silu(gate_value);
    result.skip_update[rank_index] = result.rank_input[rank_index] * d_skip[rank_index];
  }

  result.conv_pre_max_abs_error =
      max_abs_delta(result.conv_pre, rank_gate_array_values(payload, "reference_conv_pre"));
  result.rank_input_max_abs_error =
      max_abs_delta(result.rank_input, rank_gate_array_values(payload, "reference_rank_input"));
  result.gate_pre_max_abs_error =
      max_abs_delta(result.gate_pre, rank_gate_array_values(payload, "reference_gate_pre"));
  result.gate_max_abs_error =
      max_abs_delta(result.gate, rank_gate_array_values(payload, "reference_gate"));
  result.skip_update_max_abs_error =
      max_abs_delta(result.skip_update, rank_gate_array_values(payload, "reference_skip_update"));
  result.max_abs_error = std::max(
      {result.conv_pre_max_abs_error,
       result.rank_input_max_abs_error,
       result.gate_pre_max_abs_error,
       result.gate_max_abs_error,
       result.skip_update_max_abs_error});
  return result;
}

inline auto evaluate_rank_gate_payload_chain_handoff(
    const std::vector<RankGatePayload>& payloads) -> RankGateChainHandoffEvalResult {
  if (payloads.empty()) {
    throw std::runtime_error("rank/gate payload chain must not be empty");
  }
  RankGateChainHandoffEvalResult result;
  result.payload_count = payloads.size();
  for (const auto& payload : payloads) {
    const auto rank_gate = evaluate_rank_gate_payload(payload);
    result.rank_gate_max_abs_error =
        std::max(result.rank_gate_max_abs_error, rank_gate.max_abs_error);
  }
  for (std::size_t index = 1; index < payloads.size(); ++index) {
    const auto& previous_output =
        rank_gate_array_values(payloads[index - 1], "reference_output_model_poly");
    const auto& current_residual = rank_gate_array_values(payloads[index], "residual_input");
    result.model_layout_handoff_max_abs_error =
        std::max(
            result.model_layout_handoff_max_abs_error,
            max_abs_delta(current_residual, previous_output));
  }
  result.max_abs_error =
      std::max(result.rank_gate_max_abs_error, result.model_layout_handoff_max_abs_error);
  return result;
}

}  // namespace stage1
