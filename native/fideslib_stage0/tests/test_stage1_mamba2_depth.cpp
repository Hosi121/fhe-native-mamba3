#include "stage1_mamba2_depth.hpp"

#include <cmath>
#include <functional>
#include <stdexcept>
#include <string>
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

auto depth_payload() -> fhemamba::stage1::M1Payload {
  using fhemamba::stage1::PolySpec;
  fhemamba::stage1::M1Payload payload;
  auto polynomial = [](int degree) {
    PolySpec spec;
    spec.coeffs.assign(static_cast<std::size_t>(degree + 1), 0.01);
    return spec;
  };
  payload.polys["conv_silu"] = polynomial(8);
  payload.polys["gate_silu"] = polynomial(8);
  payload.polys["dt_softplus"] = polynomial(8);
  payload.polys["decay_exp"] = polynomial(8);
  payload.polys["decay_exp"].squarings = 4;
  payload.polys["rms_invsqrt"] = polynomial(8);
  payload.polys["rms_invsqrt"].iterations = 4;
  payload.polys["gated_rms_invsqrt"] = polynomial(8);
  payload.polys["gated_rms_invsqrt"].iterations = 4;
  return payload;
}

}  // namespace

auto main() -> int {
  using namespace fhemamba::stage1;

  const std::vector<double> coeffs = {0.5, 0.25, -0.1, 0.05, -0.025};
  const int baby_size = cheb_baby_size(static_cast<int>(coeffs.size()) - 1);
  for (int sample = 0; sample <= 20; ++sample) {
    const double value = -1.0 + sample / 10.0;
    require(std::abs(cheb_clenshaw_host(coeffs, value) -
                     cheb_ps_host(coeffs, value, baby_size)) < 1e-12,
            "Paterson-Stockmeyer evaluation differs from Clenshaw");
  }
  verify_cheb_ps_host("test", coeffs);
  require(cheb_ps_depth(0) == 0, "constant polynomial consumes a level");
  require(cheb_ps_depth(8) > 0, "non-constant polynomial has zero depth");

  const auto estimate =
      estimate_levels(depth_payload(), 3, {}, {}, false, 0, false, 1);
  require(estimate.token_output_levels.size() == 3,
          "depth estimate omitted token outputs");
  require(estimate.required_depth > 0 && estimate.max_segment > 0,
          "depth estimate is not positive");

  require_invalid([] { ceil_log2(0); });
  require_invalid([] { cheb_baby_size(-1); });
  require_invalid([] { cheb_clenshaw_host({}, 0.0); });
  require_invalid([] { cheb_ps_host({1.0}, 0.0, 0); });
  require_invalid([] { cheb_ps_depth(-1); });
  const auto periodic =
      estimate_levels(depth_payload(), 4, {}, {}, false, 2, false, 1);
  require(periodic.token_output_levels[2] < estimate_levels(
      depth_payload(), 4, {}, {}, false, 0, false, 1).token_output_levels[2],
          "periodic state refresh was omitted from the depth estimate");
  const auto replicated =
      estimate_levels(depth_payload(), 1, {}, {}, false, 0, true, 1);
  require(replicated.req_conv == estimate.req_conv + 1,
          "replicated state layout segment depth was not modeled");
  require(replicated.update_level >= estimate.update_level,
          "replicated state layout reduced the modeled update level");
  require_invalid([] {
    estimate_levels(depth_payload(), 0, {}, {}, false, 0, false, 1);
  });
  require_invalid([] {
    estimate_levels(depth_payload(), 1, {}, {}, false, -1, false, 1);
  });
  auto invalid_newton = depth_payload();
  invalid_newton.polys["rms_invsqrt"].iterations = 0;
  require_invalid([&] {
    estimate_levels(invalid_newton, 1, {}, {}, false, 0, false, 1);
  });
  return 0;
}
