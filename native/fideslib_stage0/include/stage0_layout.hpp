#pragma once

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <ostream>
#include <vector>

namespace stage0 {

inline auto state_slots(int d_state, int mimo_rank) -> int {
  return d_state * mimo_rank;
}

inline auto make_readout_rotations(int d_state, int mimo_rank) -> std::vector<int32_t> {
  std::vector<int32_t> rotations;
  for (int step = 1; step < d_state; step *= 2) {
    rotations.push_back(static_cast<int32_t>(step));
  }
  for (int rank = 1; rank < mimo_rank; ++rank) {
    const int shift = rank * d_state - rank;
    if (shift != 0) {
      rotations.push_back(static_cast<int32_t>(shift));
    }
  }
  std::sort(rotations.begin(), rotations.end());
  rotations.erase(std::unique(rotations.begin(), rotations.end()), rotations.end());
  return rotations;
}

inline auto make_reduce_mask(int d_state, int mimo_rank, int step) -> std::vector<double> {
  const int slots = state_slots(d_state, mimo_rank);
  std::vector<double> mask(static_cast<size_t>(slots), 0.0);
  for (int rank = 0; rank < mimo_rank; ++rank) {
    for (int n = 0; n < d_state; ++n) {
      if (n + step < d_state && n % (2 * step) == 0) {
        mask[static_cast<size_t>(rank * d_state + n)] = 1.0;
      }
    }
  }
  return mask;
}

inline auto make_scatter_mask(int d_state, int mimo_rank, int rank) -> std::vector<double> {
  const int slots = state_slots(d_state, mimo_rank);
  std::vector<double> mask(static_cast<size_t>(slots), 0.0);
  mask[static_cast<size_t>(rank * d_state)] = 1.0;
  return mask;
}

inline auto make_reduce_steps(int d_state) -> std::vector<int> {
  std::vector<int> steps;
  for (int step = 1; step < d_state; step *= 2) {
    steps.push_back(step);
  }
  return steps;
}

inline auto make_scatter_shifts(int d_state, int mimo_rank) -> std::vector<int> {
  std::vector<int> shifts;
  for (int rank = 0; rank < mimo_rank; ++rank) {
    shifts.push_back(rank * d_state - rank);
  }
  return shifts;
}

inline void print_json_vector(std::ostream& out, const std::vector<double>& values, int length) {
  out << "[";
  for (int i = 0; i < length; ++i) {
    if (i > 0) {
      out << ",";
    }
    const auto value = values.at(static_cast<size_t>(i));
    if (std::isfinite(value)) {
      out << value;
    } else {
      out << "null";
    }
  }
  out << "]";
}

}  // namespace stage0
