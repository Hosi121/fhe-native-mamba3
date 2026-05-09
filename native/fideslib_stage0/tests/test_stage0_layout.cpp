#include "stage0_layout.hpp"

#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace {

template <typename T>
void require_equal(const std::vector<T>& actual, const std::vector<T>& expected, const char* name) {
  if (actual == expected) {
    return;
  }
  std::cerr << name << " mismatch\nactual:   ";
  for (const auto& value : actual) {
    std::cerr << value << " ";
  }
  std::cerr << "\nexpected: ";
  for (const auto& value : expected) {
    std::cerr << value << " ";
  }
  std::cerr << "\n";
  std::exit(EXIT_FAILURE);
}

void require_string_equal(const std::string& actual, const std::string& expected, const char* name) {
  if (actual == expected) {
    return;
  }
  std::cerr << name << " mismatch\nactual: " << actual << "\nexpected: " << expected << "\n";
  std::exit(EXIT_FAILURE);
}

void test_readout_rotation_inventory() {
  require_equal<int32_t>(
      stage0::make_readout_rotations(4, 4),
      {1, 2, 3, 6, 9},
      "readout rotations");
}

void test_reduce_masks_follow_rank_major_layout() {
  require_equal<double>(
      stage0::make_reduce_mask(4, 2, 1),
      {1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0},
      "step=1 reduce mask");
  require_equal<double>(
      stage0::make_reduce_mask(4, 2, 2),
      {1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0},
      "step=2 reduce mask");
}

void test_scatter_masks_and_shifts() {
  require_equal<double>(
      stage0::make_scatter_mask(4, 2, 1),
      {0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0},
      "scatter mask");
  require_equal<int>(stage0::make_scatter_shifts(4, 4), {0, 3, 6, 9}, "scatter shifts");
}

void test_nonfinite_values_are_valid_json_nulls() {
  std::ostringstream out;
  stage0::print_json_vector(out, {1.0, NAN, INFINITY, -2.5}, 4);
  require_string_equal(out.str(), "[1,null,null,-2.5]", "json vector");
}

}  // namespace

auto main() -> int {
  test_readout_rotation_inventory();
  test_reduce_masks_follow_rank_major_layout();
  test_scatter_masks_and_shifts();
  test_nonfinite_values_are_valid_json_nulls();
  return EXIT_SUCCESS;
}
