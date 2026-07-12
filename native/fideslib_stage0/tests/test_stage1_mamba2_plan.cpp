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

  const auto small_shape = resolve_replicated_shape(4, 4, 32, 0);
  std::vector<double> weights(16);
  std::iota(weights.begin(), weights.end(), 1.0);
  const auto mask = replicated_bsgs_mask(weights, 4, 4, 0, small_shape, 32);
  require(mask.size() == 32, "replicated mask has the wrong size");

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
