#pragma once

#include "stage1_mamba2_payload.hpp"

#include <set>
#include <string>
#include <vector>

namespace fhemamba::stage1 {

inline constexpr double kChebCoefficientFloor = 1e-12;
inline constexpr int kAssumedBootstrapOutputLevel = 18;
inline constexpr int kNewtonSegmentEstimate = 14;

struct DepthEstimate {
  std::vector<int> token_output_levels;
  int required_depth = 0;
  int proj_level = 0;    // fresh-input level of the in_proj output
  int update_level = 0;  // fresh-input level of the token-0 state update
  // Segment requirements for the mid-circuit bootstrap checkpoints (same
  // formulas as LayerPlan; used for the pre-run geometry warning).
  int req_residual = 0;
  int req_proj = 0;
  int req_fifo = 0;
  int req_conv = 0;
  int req_dt = 0;
  int req_decay = 0;
  int req_state_pre = 0;
  int req_state_tail = 0;
  int req_y = 0;
  int req_out = 0;
  int max_segment = 0;
};

auto ceil_log2(int value) -> int;
auto cheb_baby_size(int degree) -> int;
auto cheb_clenshaw_host(const std::vector<double>& coeffs, double t) -> double;
auto cheb_ps_host(const std::vector<double>& coeffs, double u, int m) -> double;
void verify_cheb_ps_host(const std::string& name,
                         const std::vector<double>& coeffs);
auto cheb_ps_depth(int degree) -> int;
auto estimate_levels(
    const M1Payload& payload, int tokens,
    const std::set<int>& bootstrap_before_token,
    const std::set<int>& debug_client_reencrypt_before_token,
    bool refresh_recurrent_state_post, int streams) -> DepthEstimate;

}  // namespace fhemamba::stage1
