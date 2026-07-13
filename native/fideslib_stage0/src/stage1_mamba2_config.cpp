#include "stage1_mamba2_config.hpp"

#include <algorithm>
#include <cmath>
#include <cctype>
#include <exception>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>

namespace fhemamba::stage1 {

auto parse_int(std::string_view name, const char* value) -> int {
  try {
    const std::string text(value);
    std::size_t consumed = 0;
    const int parsed = std::stoi(text, &consumed);
    if (consumed != text.size()) {
      throw std::invalid_argument("trailing characters");
    }
    return parsed;
  } catch (const std::exception& exc) {
    throw std::invalid_argument(std::string("invalid integer for ") + std::string(name) + ": " +
                                exc.what());
  }
}

auto parse_double_arg(std::string_view name, const char* value) -> double {
  try {
    const std::string text(value);
    std::size_t consumed = 0;
    const double parsed = std::stod(text, &consumed);
    if (consumed != text.size() || !std::isfinite(parsed)) {
      throw std::invalid_argument("must be a finite number without trailing characters");
    }
    return parsed;
  } catch (const std::exception& exc) {
    throw std::invalid_argument(std::string("invalid float for ") + std::string(name) + ": " +
                                exc.what());
  }
}

auto parse_bool_arg(std::string_view name, std::string_view value) -> bool {
  if (value == "1" || value == "true") {
    return true;
  }
  if (value == "0" || value == "false") {
    return false;
  }
  throw std::invalid_argument(std::string("invalid boolean for ") + std::string(name) +
                              ": expected 0, 1, false, or true");
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

auto should_use_meta_bts(const Config& config, int active_layer,
                         bool carried, bool normalized_state,
                         std::string_view checkpoint) -> bool {
  if (!config.meta_bts ||
      (normalized_state && !config.normalized_state_meta_bts)) {
    return false;
  }
  const bool selected_residual =
      checkpoint.find("residual") != std::string_view::npos &&
      config.meta_bts_residual_layers.count(active_layer) > 0;
  return carried ||
         checkpoint.find("gated_poly_input") != std::string_view::npos ||
         selected_residual;
}

auto parse_args(int argc, char* argv[]) -> Config {
  Config config;
  for (int i = 1; i < argc; ++i) {
    const std::string_view arg(argv[i]);
    if (i + 1 >= argc) {
      throw std::invalid_argument(std::string("missing value for ") + std::string(arg));
    }
    const char* value = argv[++i];
    if (arg == "--debug-decrypt") {
      config.debug_decrypt = parse_bool_arg(arg, value);
    } else if (arg == "--debug-normalized-state-bootstrap-range") {
      config.debug_normalized_state_bootstrap_range = parse_bool_arg(arg, value);
    } else if (arg == "--debug-refresh-probes") {
      config.debug_refresh_probes = parse_bool_arg(arg, value);
    } else if (arg == "--debug-layer-errors") {
      config.debug_layer_errors = parse_bool_arg(arg, value);
    } else if (arg == "--debug-recurrence-token") {
      config.debug_recurrence_token = parse_int(arg, value);
    } else if (arg == "--debug-recurrence-layer") {
      config.debug_recurrence_layer = parse_int(arg, value);
    } else if (arg == "--bootstrap-norm-margin") {
      config.bootstrap_norm_margin = parse_double_arg(arg, value);
    } else if (arg == "--state-bootstrap-margin") {
      config.state_bootstrap_margin = parse_double_arg(arg, value);
    } else if (arg == "--meta-bts") {
      config.meta_bts = parse_bool_arg(arg, value);
    } else if (arg == "--meta-bts-alpha") {
      config.meta_bts_alpha = parse_int(arg, value);
    } else if (arg == "--state-meta-bts-alpha") {
      config.state_meta_bts_alpha = parse_int(arg, value);
    } else if (arg == "--meta-bts-residual-align-mode") {
      config.meta_bts_residual_align_mode = value;
    } else if (arg == "--meta-bts-residual-layers") {
      config.meta_bts_residual_layers = parse_int_set(arg, value);
    } else if (arg == "--input") {
      config.input = value;
    } else if (arg == "--input-chain") {
      config.input_chain = value;
    } else if (arg == "--max-layers") {
      config.max_layers = parse_int(arg, value);
    } else if (arg == "--auto-bootstrap-headroom") {
      config.auto_bootstrap_headroom = parse_int(arg, value);
    } else if (arg == "--residual-bootstrap-headroom") {
      config.residual_bootstrap_headroom = parse_int(arg, value);
    } else if (arg == "--carried-bootstrap-headroom") {
      config.carried_bootstrap_headroom = parse_int(arg, value);
    } else if (arg == "--pt-cache") {
      config.pt_cache = value;
    } else if (arg == "--pt-cache-gib") {
      config.pt_cache_gib = parse_double_arg(arg, value);
    } else if (arg == "--pt-cache-level") {
      config.pt_cache_level = parse_int(arg, value);
    } else if (arg == "--pt-cache-weight-level") {
      config.pt_cache_weight_level = parse_int(arg, value);
    } else if (arg == "--pt-miss-consumption-level") {
      config.pt_miss_consumption_level = parse_bool_arg(arg, value);
    } else if (arg == "--encode-threads") {
      config.encode_threads = parse_int(arg, value);
    } else if (arg == "--streams") {
      config.streams = parse_int(arg, value);
    } else if (arg == "--rotation-keys") {
      config.rotation_keys = value;
    } else if (arg == "--rotation-key-gib") {
      config.rotation_key_gib = parse_double_arg(arg, value);
    } else if (arg == "--level-align-mode") {
      config.level_align_mode = value;
    } else if (arg == "--bsgs-replicas") {
      config.bsgs_replicas = value;
    } else if (arg == "--replicated-true-bsgs") {
      config.replicated_true_bsgs = parse_bool_arg(arg, value);
    } else if (arg == "--fused-replicated-linear-transform") {
      config.fused_replicated_linear_transform = parse_bool_arg(arg, value);
    } else if (arg == "--fused-replicated-linear-transform-scope") {
      config.fused_replicated_linear_transform_scope = value;
    } else if (arg == "--interleaved-replicated-projection") {
      config.interleaved_replicated_projection = parse_bool_arg(arg, value);
    } else if (arg == "--replicated-state-blocks") {
      config.replicated_state_blocks = parse_bool_arg(arg, value);
    } else if (arg == "--projection-late-level") {
      config.projection_late_level = parse_bool_arg(arg, value);
    } else if (arg == "--output-json") {
      config.output_json = value;
    } else if (arg == "--artifact-version") {
      config.artifact_version = value;
    } else if (arg == "--repo-commit") {
      config.repo_commit = value;
    } else if (arg == "--binary-sha256") {
      config.binary_sha256 = value;
    } else if (arg == "--process-role") {
      config.process_role = value;
    } else if (arg == "--handoff-dir") {
      config.handoff_dir = value;
    } else if (arg == "--ring-dim") {
      config.ring_dim = parse_int(arg, value);
    } else if (arg == "--depth" || arg == "--multiplicative-depth") {
      config.multiplicative_depth = parse_int(arg, value);
    } else if (arg == "--scaling-mod-size") {
      config.scaling_mod_size = parse_int(arg, value);
    } else if (arg == "--first-mod-size") {
      config.first_mod_size = parse_int(arg, value);
    } else if (arg == "--tokens") {
      config.tokens = parse_int(arg, value);
    } else if (arg == "--tolerance") {
      config.tolerance = parse_double_arg(arg, value);
    } else if (arg == "--bootstrap-before-token") {
      config.bootstrap_before_token = parse_int_set(arg, value);
    } else if (arg == "--debug-client-reencrypt-before-token") {
      config.debug_client_reencrypt_before_token = parse_int_set(arg, value);
    } else if (arg == "--autoregressive-client-loop") {
      config.autoregressive_client_loop = parse_bool_arg(arg, value);
    } else if (arg == "--refresh-recurrent-state-post") {
      config.refresh_recurrent_state_post = parse_bool_arg(arg, value);
    } else if (arg == "--refresh-recurrent-state-post-layers") {
      config.refresh_recurrent_state_post_layers = parse_int_set(arg, value);
    } else if (arg == "--state-refresh-interval") {
      config.state_refresh_interval = parse_int(arg, value);
    } else if (arg == "--normalized-recurrent-state") {
      config.normalized_recurrent_state = parse_bool_arg(arg, value);
    } else if (arg == "--complex-state-pairing") {
      config.complex_state_pairing = parse_bool_arg(arg, value);
    } else if (arg == "--normalized-state-meta-bts") {
      config.normalized_state_meta_bts = parse_bool_arg(arg, value);
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
  if (config.process_role != "inline" && config.process_role != "client-init" &&
      config.process_role != "server-eval" &&
      config.process_role != "client-decrypt") {
    throw std::invalid_argument(
        "process-role must be inline, client-init, server-eval, or client-decrypt");
  }
  if (config.process_role != "inline" && config.handoff_dir.empty()) {
    throw std::invalid_argument("non-inline process-role requires --handoff-dir");
  }
  if (config.process_role != "inline" && config.output_json.empty()) {
    throw std::invalid_argument("non-inline process-role requires --output-json");
  }
  if (config.process_role != "inline" && config.autoregressive_client_loop) {
    throw std::invalid_argument(
        "process-separated autoregressive loop is not implemented yet");
  }
  if (config.process_role == "server-eval" &&
      (config.debug_decrypt || config.debug_refresh_probes ||
       config.debug_layer_errors ||
       config.debug_recurrence_token >= 0 || config.debug_recurrence_layer >= 0 ||
       config.debug_normalized_state_bootstrap_range ||
       !config.debug_client_reencrypt_before_token.empty())) {
    throw std::invalid_argument(
        "server-eval forbids every secret-key diagnostic option");
  }
  if (config.ring_dim <= 0 || (config.ring_dim & (config.ring_dim - 1)) != 0) {
    throw std::invalid_argument("ring-dim must be a positive power of two");
  }
  if ((config.debug_recurrence_token < 0) !=
      (config.debug_recurrence_layer < 0)) {
    throw std::invalid_argument(
        "debug-recurrence-token and debug-recurrence-layer must be set together");
  }
  if (config.debug_recurrence_token >= 0) {
    if (!config.autoregressive_client_loop) {
      throw std::invalid_argument(
          "recurrence debug requires autoregressive-client-loop");
    }
    if (config.debug_recurrence_token < 1 ||
        config.debug_recurrence_token >= config.tokens) {
      throw std::invalid_argument("debug-recurrence-token must be in [1, tokens-1]");
    }
  }
  if (config.multiplicative_depth <= 0 || config.scaling_mod_size <= 0 ||
      config.first_mod_size <= 0 || config.tokens <= 0 || config.tolerance <= 0.0 ||
      config.max_layers < 0 || config.auto_bootstrap_headroom < 0 ||
      config.residual_bootstrap_headroom < 0 ||
      config.carried_bootstrap_headroom < 0 ||
      config.bootstrap_level_budget_cts <= 0 || config.bootstrap_level_budget_stc <= 0 ||
      config.bootstrap_bsgs_dim_cts < 0 || config.bootstrap_bsgs_dim_stc < 0) {
    throw std::invalid_argument("invalid CKKS parameters");
  }
  for (const int token : config.bootstrap_before_token) {
    if (token < 1 || token >= config.tokens) {
      throw std::invalid_argument("bootstrap-before-token must be in [1, tokens-1]");
    }
  }
  for (const int token : config.debug_client_reencrypt_before_token) {
    if (token < 1 || token >= config.tokens) {
      throw std::invalid_argument(
          "debug-client-reencrypt-before-token must be in [1, tokens-1]");
    }
    if (config.bootstrap_before_token.count(token) > 0) {
      throw std::invalid_argument(
          "bootstrap and debug client re-encrypt schedules must be disjoint");
    }
  }
  for (const int layer : config.refresh_recurrent_state_post_layers) {
    if (layer < 0) {
      throw std::invalid_argument("refresh-recurrent-state-post-layers must be nonnegative");
    }
  }
  for (const int layer : config.meta_bts_residual_layers) {
    if (layer < 0) {
      throw std::invalid_argument("meta-bts-residual-layers must be nonnegative");
    }
  }
  if (config.state_refresh_interval < 0) {
    throw std::invalid_argument("state-refresh-interval must be nonnegative");
  }
  if (config.bootstrap_norm_margin <= 0.0) {
    throw std::invalid_argument("bootstrap-norm-margin must be positive");
  }
  if (config.state_bootstrap_margin <= 0.0) {
    throw std::invalid_argument("state-bootstrap-margin must be positive");
  }
  if (config.meta_bts_alpha < 0 || config.meta_bts_alpha > 40) {
    throw std::invalid_argument("meta-bts-alpha must be in [0, 40]");
  }
  if (config.state_meta_bts_alpha < -1 || config.state_meta_bts_alpha > 40) {
    throw std::invalid_argument("state-meta-bts-alpha must be in [-1, 40]");
  }
  if (config.meta_bts_residual_align_mode != "unity" &&
      config.meta_bts_residual_align_mode != "drop" &&
      config.meta_bts_residual_align_mode != "native") {
    throw std::invalid_argument(
        "meta-bts-residual-align-mode must be unity, drop, or native");
  }
  if (config.pt_cache != "auto" && config.pt_cache != "full" && config.pt_cache != "masks" &&
      config.pt_cache != "off") {
    throw std::invalid_argument("pt-cache must be auto, full, masks, or off");
  }
  if (config.pt_cache_gib < 0.0) {
    throw std::invalid_argument("pt-cache-gib must be non-negative");
  }
  if (config.pt_cache_level < 0 || config.pt_cache_level >= config.multiplicative_depth) {
    throw std::invalid_argument("pt-cache-level must be in [0, multiplicative-depth)");
  }
  if (config.pt_cache_weight_level < 0 ||
      config.pt_cache_weight_level >= config.multiplicative_depth) {
    throw std::invalid_argument(
        "pt-cache-weight-level must be in [0, multiplicative-depth)");
  }
  if (config.encode_threads < 1 || config.encode_threads > 64) {
    throw std::invalid_argument("encode-threads must be in [1, 64]");
  }
  if (config.streams < 1 || (config.streams & (config.streams - 1)) != 0) {
    throw std::invalid_argument("streams must be a positive power of two");
  }
  if (config.streams > 1 && !config.input_chain.empty()) {
    throw std::invalid_argument("streams > 1 requires --input (single-layer mode)");
  }
  if (config.autoregressive_client_loop && config.streams != 1) {
    throw std::invalid_argument("autoregressive-client-loop requires streams == 1");
  }
  if (config.autoregressive_client_loop && config.input_chain.empty()) {
    throw std::invalid_argument("autoregressive-client-loop requires --input-chain");
  }
  if (config.rotation_keys != "full" && config.rotation_keys != "balanced" &&
      config.rotation_keys != "compact") {
    throw std::invalid_argument("rotation-keys must be full, balanced, or compact");
  }
  if (config.rotation_key_gib <= 0.0) {
    throw std::invalid_argument("rotation-key-gib must be positive");
  }
  if (config.level_align_mode != "unity" && config.level_align_mode != "drop" &&
      config.level_align_mode != "native") {
    throw std::invalid_argument("level-align-mode must be unity, drop, or native");
  }
  if (config.bsgs_replicas != "auto" && config.bsgs_replicas != "1") {
    try {
      if (parse_int("--bsgs-replicas", config.bsgs_replicas.c_str()) < 1) {
        throw std::invalid_argument("");
      }
    } catch (const std::exception&) {
      throw std::invalid_argument("bsgs-replicas must be auto, 1, or a positive integer");
    }
  }
  if (config.bsgs_replicas != "1" && config.streams > 1) {
    throw std::invalid_argument("bsgs-replicas requires --streams 1");
  }
  if (config.replicated_true_bsgs && config.bsgs_replicas == "1") {
    throw std::invalid_argument("replicated-true-bsgs requires replicated layout");
  }
  if (config.fused_replicated_linear_transform &&
      !config.replicated_true_bsgs) {
    throw std::invalid_argument(
        "fused-replicated-linear-transform requires replicated-true-bsgs");
  }
  if (config.fused_replicated_linear_transform_scope != "all" &&
      config.fused_replicated_linear_transform_scope != "out-proj") {
    throw std::invalid_argument(
        "fused-replicated-linear-transform-scope must be all or out-proj");
  }
  if (config.interleaved_replicated_projection &&
      config.bsgs_replicas == "1") {
    throw std::invalid_argument(
        "interleaved-replicated-projection requires replicated layout");
  }
  if (config.normalized_state_meta_bts &&
      !config.normalized_recurrent_state) {
    throw std::invalid_argument(
        "normalized-state-meta-bts requires normalized-recurrent-state");
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
  if (config.binary_sha256 != "unknown" &&
      (config.binary_sha256.size() != 64 ||
       !std::all_of(config.binary_sha256.begin(), config.binary_sha256.end(),
                    [](unsigned char character) {
                      return std::isxdigit(character) != 0;
                    }))) {
    throw std::invalid_argument("binary-sha256 must be unknown or 64 hexadecimal characters");
  }
  return config;
}

}  // namespace fhemamba::stage1
