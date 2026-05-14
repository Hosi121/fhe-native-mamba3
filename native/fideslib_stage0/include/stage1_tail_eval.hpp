#pragma once

#include "stage1_tail_payload.hpp"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <stdexcept>
#include <string>
#include <vector>

namespace stage1 {

struct TailEvalResult {
  std::vector<double> state_new;
  std::vector<double> readout_rank;
  std::vector<double> rank_output;
  std::vector<double> rank_payload;
  std::vector<double> output_model;
  double max_abs_error = 0.0;
  double state_new_max_abs_error = 0.0;
  double readout_rank_max_abs_error = 0.0;
  double rank_output_max_abs_error = 0.0;
  double rank_payload_max_abs_error = 0.0;
  double output_model_max_abs_error = 0.0;
};

inline auto max_abs_delta(const std::vector<double>& lhs, const std::vector<double>& rhs) -> double {
  if (lhs.size() != rhs.size()) {
    throw std::runtime_error("max_abs_delta size mismatch");
  }
  double output = 0.0;
  for (std::size_t index = 0; index < lhs.size(); ++index) {
    output = std::max(output, std::abs(lhs[index] - rhs[index]));
  }
  return output;
}

inline auto tail_array_values(const TailPayload& payload, const std::string& name)
    -> const std::vector<double>& {
  return payload.array(name).values;
}

inline auto evaluate_tail_payload(const TailPayload& payload) -> TailEvalResult {
  const auto d_model = static_cast<std::size_t>(payload.config.d_model);
  const auto rank = static_cast<std::size_t>(payload.config.mimo_rank);
  const auto d_state = static_cast<std::size_t>(payload.config.d_state);

  const auto& residual_input = tail_array_values(payload, "residual_input");
  const auto& rank_input = tail_array_values(payload, "rank_input");
  const auto& gate = tail_array_values(payload, "gate");
  const auto& b = tail_array_values(payload, "b");
  const auto& c = tail_array_values(payload, "c");
  const auto& decay = tail_array_values(payload, "decay");
  const auto& previous_state = tail_array_values(payload, "previous_state");
  const auto& skip_update = tail_array_values(payload, "skip_update");
  const auto& w_out = tail_array_values(payload, "w_out");

  TailEvalResult result;
  result.state_new.assign(d_state * rank, 0.0);
  result.readout_rank.assign(rank, 0.0);
  result.rank_output.assign(rank, 0.0);
  result.rank_payload.assign(rank, 0.0);
  result.output_model = residual_input;

  for (std::size_t state_index = 0; state_index < d_state; ++state_index) {
    const auto base = state_index * rank;
    for (std::size_t rank_index = 0; rank_index < rank; ++rank_index) {
      const auto index = base + rank_index;
      result.state_new[index] =
          decay[index] * previous_state[index] + b[index] * rank_input[rank_index];
      result.readout_rank[rank_index] += c[index] * result.state_new[index];
    }
  }
  for (std::size_t rank_index = 0; rank_index < rank; ++rank_index) {
    result.rank_output[rank_index] = result.readout_rank[rank_index] + skip_update[rank_index];
    result.rank_payload[rank_index] = gate[rank_index] * result.rank_output[rank_index];
  }
  for (std::size_t model_index = 0; model_index < d_model; ++model_index) {
    double update = 0.0;
    const auto base = model_index * rank;
    for (std::size_t rank_index = 0; rank_index < rank; ++rank_index) {
      update += w_out[base + rank_index] * result.rank_payload[rank_index];
    }
    result.output_model[model_index] += update;
  }

  result.state_new_max_abs_error =
      max_abs_delta(result.state_new, tail_array_values(payload, "reference_state_new"));
  result.readout_rank_max_abs_error =
      max_abs_delta(result.readout_rank, tail_array_values(payload, "reference_readout_rank"));
  result.rank_output_max_abs_error =
      max_abs_delta(result.rank_output, tail_array_values(payload, "reference_rank_output"));
  result.rank_payload_max_abs_error =
      max_abs_delta(result.rank_payload, tail_array_values(payload, "reference_rank_payload"));
  result.output_model_max_abs_error =
      max_abs_delta(result.output_model, tail_array_values(payload, "reference_output_model"));
  result.max_abs_error = std::max(
      {result.state_new_max_abs_error,
       result.readout_rank_max_abs_error,
       result.rank_output_max_abs_error,
       result.rank_payload_max_abs_error,
       result.output_model_max_abs_error});
  return result;
}

}  // namespace stage1
