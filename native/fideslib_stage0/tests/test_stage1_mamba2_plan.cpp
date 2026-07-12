#include "stage1_mamba2_plan.hpp"

#include <algorithm>
#include <cstdint>
#include <functional>
#include <numeric>
#include <set>
#include <stdexcept>
#include <vector>

namespace {

void require(bool condition, const char* message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

void require_invalid(const std::function<void()>& operation) {
  try {
    operation();
  } catch (const std::invalid_argument&) {
    return;
  }
  throw std::runtime_error("expected invalid_argument");
}

auto model_dims() -> fhemamba::stage1::M1Payload {
  fhemamba::stage1::M1Payload payload;
  payload.d_model = 768;
  payload.d_inner = 1536;
  payload.num_heads = 24;
  payload.head_dim = 64;
  payload.state_size = 64;
  payload.n_groups = 1;
  payload.conv_dim = 1664;
  payload.proj_dim = 3352;
  return payload;
}

}  // namespace

auto main() -> int {
  using namespace fhemamba::stage1;

  const auto payload = model_dims();
  const auto packing = derive_packing(payload, 32768);
  require(packing.group_heads == 8, "unexpected heads per state group");
  require(packing.group_count == 3, "unexpected state group count");
  require(packing.group_block == 512, "unexpected state group width");

  const auto rep_in =
      resolve_replicated_shape(payload.proj_dim, payload.d_model, 32768, 0);
  const auto rep_out =
      resolve_replicated_shape(payload.d_model, payload.d_inner, 32768, 0);
  require(rep_in.replicas == 7 && rep_in.window == 4608,
          "unexpected in-projection replicated shape");
  require(rep_out.replicas == 10 && rep_out.window == 3072,
          "unexpected out-projection replicated shape");

  const auto rotations = required_rotations(payload, packing, rep_in, rep_out);
  require(!rotations.empty(), "rotation plan is empty");
  require(std::is_sorted(rotations.begin(), rotations.end()),
          "rotation plan is not deterministic");
  verify_naf(rotations);
  const auto positive_steps = naf_steps(28);
  require(std::accumulate(positive_steps.begin(), positive_steps.end(), 0) == 28,
          "positive NAF decomposition is incorrect");
  const auto negative_steps = naf_steps(-13);
  require(std::accumulate(negative_steps.begin(), negative_steps.end(), 0) == -13,
          "negative NAF decomposition is incorrect");

  const auto frequencies = rotation_frequencies(
      payload, packing, 24, 1, 32768, rep_in, rep_out);
  const std::set<int32_t> required(rotations.begin(), rotations.end());
  for (const auto& [index, frequency] : frequencies) {
    require(required.count(index) == 1, "frequency contains an unplanned rotation");
    require(frequency > 0.0, "rotation frequency is not positive");
  }
  require(rotation_key_gib_estimate(65536, 44) > 0.0,
          "rotation key estimate is not positive");

  auto bsgs_in = rep_in;
  auto bsgs_out = rep_out;
  bsgs_in.baby_step = 10;
  bsgs_out.baby_step = 12;
  const auto bsgs_rotations = required_rotations(payload, packing, bsgs_in, bsgs_out);
  require(bsgs_rotations.size() < rotations.size(),
          "true replicated BSGS did not reduce the rotation-key inventory");
  const auto bsgs_frequencies = rotation_frequencies(
      payload, packing, 24, 1, 32768, bsgs_in, bsgs_out);
  const std::set<int32_t> bsgs_required(bsgs_rotations.begin(), bsgs_rotations.end());
  for (const auto& [index, frequency] : bsgs_frequencies) {
    require(bsgs_required.count(index) == 1,
            "BSGS frequency contains an unplanned rotation");
    require(frequency > 0.0, "BSGS rotation frequency is not positive");
  }
  const auto state_rotations =
      required_rotations(payload, packing, bsgs_in, bsgs_out, true);
  const std::set<int32_t> state_required(state_rotations.begin(),
                                         state_rotations.end());
  require(state_required.count(-(packing.group_block - 1)) == 1,
          "replicated state stride rotation is missing");
  require(state_required.count(-2 * (packing.group_block - 1)) == 1,
          "replicated state doubling rotation is missing");
  const auto state_frequencies = rotation_frequencies(
      payload, packing, 24, 1, 32768, bsgs_in, bsgs_out, true);
  for (const auto& [index, frequency] : state_frequencies) {
    require(state_required.count(index) == 1,
            "replicated-state frequency contains an unplanned rotation");
    require(frequency > 0.0,
            "replicated-state rotation frequency is not positive");
  }

  const auto small_shape = resolve_replicated_shape(4, 4, 32, 0);
  std::vector<double> weights(16);
  std::iota(weights.begin(), weights.end(), 1.0);
  const auto mask = replicated_bsgs_mask(weights, 4, 4, 0, small_shape, 32);
  require(mask.size() == 32, "replicated mask has the wrong size");
  const auto small_bsgs_base = resolve_replicated_shape(4, 16, 128, 0);
  std::vector<double> bsgs_weights(64);
  std::iota(bsgs_weights.begin(), bsgs_weights.end(), 1.0);
  auto small_bsgs_shape = small_bsgs_base;
  small_bsgs_shape.baby_step = 2;
  const auto first_pre_mask =
      replicated_bsgs_pre_mask(bsgs_weights, 4, 16, 0, small_bsgs_shape, 128);
  const auto second_pre_mask =
      replicated_bsgs_pre_mask(bsgs_weights, 4, 16, 2, small_bsgs_shape, 128);
  require(first_pre_mask == replicated_bsgs_mask(bsgs_weights, 4, 16, 0,
                                                  small_bsgs_shape, 128),
          "zero-giant BSGS mask was unexpectedly shifted");
  require(second_pre_mask != replicated_bsgs_mask(bsgs_weights, 4, 16, 2,
                                                   small_bsgs_shape, 128),
          "nonzero-giant BSGS mask was not shifted");

  require_invalid([] { resolve_replicated_shape(4, 0, 32, 0); });
  require_invalid([] { python_mod(1, 0); });
  require_invalid([] { int_log2(0); });
  require_invalid([&] {
    replicated_bsgs_mask({}, 4, 4, 0, small_shape, 32);
  });
  require_invalid([&] {
    rotation_frequencies(payload, packing, 0, 1, 32768, rep_in, rep_out);
  });
  return 0;
}
