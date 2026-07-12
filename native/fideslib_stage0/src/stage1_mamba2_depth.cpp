#include "stage1_mamba2_depth.hpp"

#include <algorithm>
#include <cmath>
#include <functional>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace fhemamba::stage1 {

auto ceil_log2(int value) -> int {
  if (value <= 0) {
    throw std::invalid_argument("log2 input must be positive");
  }
  int log = 0;
  while ((1 << log) < value) {
    ++log;
  }
  return log;
}

auto cheb_baby_size(int degree) -> int {
  if (degree < 0) {
    throw std::invalid_argument("Chebyshev degree must be non-negative");
  }
  const int levels = std::max(1, ceil_log2(degree + 1));
  return 1 << ((levels + 1) / 2);
}

auto cheb_clenshaw_host(const std::vector<double>& coeffs, double t) -> double {
  if (coeffs.empty()) {
    throw std::invalid_argument("Chebyshev coefficients must not be empty");
  }
  double b1 = 0.0;
  double b2 = 0.0;
  for (std::size_t index = coeffs.size(); index-- > 1;) {
    const double next = 2.0 * t * b1 - b2 + coeffs[index];
    b2 = b1;
    b1 = next;
  }
  return t * b1 - b2 + coeffs[0];
}

auto cheb_ps_host(const std::vector<double>& coeffs, double u, int m) -> double {
  if (coeffs.empty() || m <= 0) {
    throw std::invalid_argument("Chebyshev coefficients and baby size are invalid");
  }
  std::vector<double> t_values(std::max<std::size_t>(coeffs.size() + 1, 2), 0.0);
  t_values[0] = 1.0;
  t_values[1] = u;
  for (std::size_t i = 2; i < t_values.size(); ++i) {
    t_values[i] = 2.0 * u * t_values[i - 1] - t_values[i - 2];
  }
  std::function<double(std::vector<double>)> rec = [&](std::vector<double> c) -> double {
    const int n = static_cast<int>(c.size()) - 1;
    if (n < m) {
      double sum = 0.0;
      for (int i = 0; i <= n; ++i) {
        sum += c[static_cast<std::size_t>(i)] * t_values[static_cast<std::size_t>(i)];
      }
      return sum;
    }
    int k = m;
    while (2 * k - 1 < n) {
      k *= 2;
    }
    std::vector<double> btil(static_cast<std::size_t>(n - k + 1), 0.0);
    for (int j = 0; j <= n - k; ++j) {
      btil[static_cast<std::size_t>(j)] = 2.0 * c[static_cast<std::size_t>(k + j)];
    }
    btil[0] = c[static_cast<std::size_t>(k)];
    std::vector<double> aprime(c.begin(), c.begin() + k);
    for (int i = k + 1; i <= n; ++i) {
      aprime[static_cast<std::size_t>(2 * k - i)] -= c[static_cast<std::size_t>(i)];
    }
    return rec(aprime) + t_values[static_cast<std::size_t>(k)] * rec(btil);
  };
  return rec(coeffs);
}

void verify_cheb_ps_host(const std::string& name, const std::vector<double>& coeffs) {
  const int m = cheb_baby_size(static_cast<int>(coeffs.size()) - 1);
  double max_error = 0.0;
  for (int sample = 0; sample <= 400; ++sample) {
    const double u = -1.0 + 2.0 * sample / 400.0;
    const double reference = cheb_clenshaw_host(coeffs, u);
    const double value = cheb_ps_host(coeffs, u, m);
    max_error = std::max(max_error, std::abs(reference - value));
  }
  if (max_error > 1e-6) {
    throw std::runtime_error("Chebyshev PS self-check failed for " + name + ": max error " +
                             std::to_string(max_error));
  }
}

// Level ledger of the PS recursion (levels above the level of u), used only
// for the pre-run depth estimate.
auto cheb_ps_depth(int degree) -> int {
  if (degree < 0) {
    throw std::invalid_argument("Chebyshev degree must be non-negative");
  }
  const int m = cheb_baby_size(degree);
  std::map<int, int> t_level;
  std::function<int(int)> level_of = [&](int i) -> int {
    if (i <= 1) {
      return 0;
    }
    auto found = t_level.find(i);
    if (found != t_level.end()) {
      return found->second;
    }
    int level = 0;
    if (i % 2 == 0) {
      level = level_of(i / 2) + 1;
    } else {
      level = std::max(level_of((i + 1) / 2), level_of(i / 2)) + 1;
    }
    t_level[i] = level;
    return level;
  };
  std::function<int(int)> rec = [&](int n) -> int {
    if (n == 0) {
      return 0;  // constant term: fresh low-level ciphertext, aligned upward
    }
    if (n < m) {
      int deepest = 0;
      for (int i = 1; i <= n; ++i) {
        deepest = std::max(deepest, level_of(i));
      }
      return deepest + 1;  // scalar coefficient multiply
    }
    int k = m;
    while (2 * k - 1 < n) {
      k *= 2;
    }
    const int giant_term = std::max(level_of(k), rec(n - k)) + 1;
    return std::max(rec(k - 1), giant_term);
  };
  return rec(degree);
}

auto estimate_levels(
    const M1Payload& payload,
    int tokens,
    const std::set<int>& bootstrap_before_token,
    const std::set<int>& debug_client_reencrypt_before_token,
    bool refresh_recurrent_state_post, int state_refresh_interval,
    bool replicated_state_blocks,
    int streams) -> DepthEstimate {
  if (tokens <= 0 || state_refresh_interval < 0 || streams <= 0) {
    throw std::invalid_argument("depth estimate tokens and streams must be positive");
  }
  // streams > 1 costs one extra level in each RMS variance reduction (the
  // stream-base mask before the in-stride broadcast).
  const int norm_extra = streams > 1 ? 1 : 0;
  const auto& rms = payload.polys.at("rms_invsqrt");
  const auto& gated = payload.polys.at("gated_rms_invsqrt");
  if (rms.iterations < 1 || gated.iterations < 1) {
    throw std::invalid_argument("Newton polynomial iterations must be positive");
  }
  const int rms_depth = cheb_ps_depth(static_cast<int>(rms.coeffs.size()) - 1);
  const int conv_depth = cheb_ps_depth(
      static_cast<int>(payload.polys.at("conv_silu").coeffs.size()) - 1);
  const int gate_depth = cheb_ps_depth(
      static_cast<int>(payload.polys.at("gate_silu").coeffs.size()) - 1);
  const int dt_depth = cheb_ps_depth(
      static_cast<int>(payload.polys.at("dt_softplus").coeffs.size()) - 1);
  const auto& exp_spec = payload.polys.at("decay_exp");
  if (exp_spec.squarings < 0) {
    throw std::invalid_argument("exponential squarings must be non-negative");
  }
  const int exp_depth = cheb_ps_depth(static_cast<int>(exp_spec.coeffs.size()) - 1);

  const int inv1 = 2 + norm_extra + rms_depth + 2 * rms.iterations;
  const int proj = std::max(inv1, 1) + 1;
  const int xconv = proj + 1 + conv_depth;
  const int gate_lvl = proj + 1 + gate_depth;
  const int dt_lvl = proj + 1 + dt_depth + 1;
  const int decay_lvl = dt_lvl + 1 + exp_depth + exp_spec.squarings;
  const int x_exp = xconv + 1;
  const int bc_exp = xconv + (replicated_state_blocks ? 2 : 1);
  const int dt_exp = dt_lvl + 1;
  const int decay_exp_lvl = decay_lvl + 1;
  const int dtx = std::max(x_exp, dt_exp) + 1;
  const int update = std::max(dtx, bc_exp) + 1;

  DepthEstimate estimate;
  estimate.proj_level = proj;
  estimate.update_level = update;
  estimate.req_residual = proj + 1;
  estimate.req_proj =
      1 + std::max(conv_depth, std::max(gate_depth + 2, dt_depth + 1));
  estimate.req_fifo = 2 + conv_depth;
  estimate.req_conv = replicated_state_blocks ? 7 : 6;
  estimate.req_dt = 2 + exp_depth + exp_spec.squarings;
  estimate.req_decay = 3;
  estimate.req_state_pre = 5;
  estimate.req_state_tail = 4;
  estimate.req_y = 4 + norm_extra;
  estimate.req_out = 2;
  estimate.max_segment = std::max(
      {estimate.req_residual, estimate.req_proj, estimate.req_fifo, estimate.req_conv,
       estimate.req_dt, estimate.req_decay, estimate.req_state_pre,
       estimate.req_state_tail, estimate.req_y, kNewtonSegmentEstimate});
  int state = 0;
  bool has_state = false;
  for (int token = 0; token < tokens; ++token) {
    if (has_state && debug_client_reencrypt_before_token.count(token) > 0) {
      state = 0;
    } else if (has_state && bootstrap_before_token.count(token) > 0) {
      state = std::min(state, kAssumedBootstrapOutputLevel);
    }
    const bool recurrent_update = has_state;
    state = recurrent_update ? std::max(decay_exp_lvl, state) + 1 : update;
    const bool periodic_refresh =
        state_refresh_interval > 0 && token % state_refresh_interval == 0;
    if (recurrent_update &&
        (refresh_recurrent_state_post || periodic_refresh)) {
      state = kAssumedBootstrapOutputLevel;
    }
    has_state = true;
    const int readout = std::max(state, bc_exp) + 2;  // *C then packed mask
    const int y = std::max(readout, gate_lvl) + 1;
    const int variance = y + 1 + norm_extra;
    const int inv2 = variance + 1 + 2 * (gated.iterations - 1);
    const int out = std::max(y + 1, inv2) + 1;
    estimate.token_output_levels.push_back(out);
    estimate.required_depth = std::max(estimate.required_depth, out);
  }
  return estimate;
}

}  // namespace fhemamba::stage1
