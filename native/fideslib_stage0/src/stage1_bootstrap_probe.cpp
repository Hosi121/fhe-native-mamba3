#include <fideslib.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <complex>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

using namespace fideslib;

namespace {

struct Config {
  int ring_dim = 65536;
  int num_slots = 32768;
  int multiplicative_depth = 28;
  int scaling_mod_size = 40;
  int first_mod_size = 60;
  int level_budget_cts = 5;
  int level_budget_stc = 4;
  int bsgs_dim_cts = 0;
  int bsgs_dim_stc = 0;
  int iterations = 3;
  int bootstrap_iterations = 1;
  int bootstrap_precision = 0;
  int seed = 0;
  double input_magnitude = 0.5;
  int encrypt_level = -1;  // -1: depth-1 (exhausted); else encrypt at this level
  // Square the fresh ciphertext before bootstrapping so the bootstrap input is
  // a real noiseScaleDeg-2 product (magnitude becomes input_magnitude^2).
  bool square_input = false;
  bool complex_pair = false;
  bool skip_decrypt = false;
  std::string security = "128-classic";
  std::string secret_key_dist = "sparse-ternary";
  std::string artifact_version = "0.0.0+bootstrap-probe";
  std::string repo_commit = "working-tree";
  std::string output_json;
};

auto now() -> std::chrono::steady_clock::time_point { return std::chrono::steady_clock::now(); }

auto seconds_since(std::chrono::steady_clock::time_point start) -> double {
  return std::chrono::duration<double>(now() - start).count();
}

auto parse_int(std::string_view name, const char* value) -> int {
  try {
    return std::stoi(value);
  } catch (const std::exception& exc) {
    throw std::invalid_argument(std::string("invalid integer for ") + std::string(name) + ": " +
                                exc.what());
  }
}

auto parse_args(int argc, char* argv[]) -> Config {
  Config config;
  for (int i = 1; i < argc; ++i) {
    const std::string_view arg(argv[i]);
    if (arg == "--skip-decrypt") {
      config.skip_decrypt = true;
      continue;
    }
    if (arg == "--complex-pair") {
      config.complex_pair = true;
      continue;
    }
    if (i + 1 >= argc) {
      throw std::invalid_argument(std::string("missing value for ") + std::string(arg));
    }
    const char* value = argv[++i];
    if (arg == "--ring-dim") {
      config.ring_dim = parse_int(arg, value);
    } else if (arg == "--num-slots") {
      config.num_slots = parse_int(arg, value);
    } else if (arg == "--multiplicative-depth") {
      config.multiplicative_depth = parse_int(arg, value);
    } else if (arg == "--scaling-mod-size") {
      config.scaling_mod_size = parse_int(arg, value);
    } else if (arg == "--first-mod-size") {
      config.first_mod_size = parse_int(arg, value);
    } else if (arg == "--level-budget-cts") {
      config.level_budget_cts = parse_int(arg, value);
    } else if (arg == "--level-budget-stc") {
      config.level_budget_stc = parse_int(arg, value);
    } else if (arg == "--bsgs-dim-cts") {
      config.bsgs_dim_cts = parse_int(arg, value);
    } else if (arg == "--bsgs-dim-stc") {
      config.bsgs_dim_stc = parse_int(arg, value);
    } else if (arg == "--iterations") {
      config.iterations = parse_int(arg, value);
    } else if (arg == "--bootstrap-iterations") {
      config.bootstrap_iterations = parse_int(arg, value);
    } else if (arg == "--bootstrap-precision") {
      config.bootstrap_precision = parse_int(arg, value);
    } else if (arg == "--input-magnitude") {
      config.input_magnitude = std::stod(value);
    } else if (arg == "--encrypt-level") {
      config.encrypt_level = parse_int(arg, value);
    } else if (arg == "--square-input") {
      config.square_input = (std::string_view(value) != "0");
    } else if (arg == "--seed") {
      config.seed = parse_int(arg, value);
    } else if (arg == "--security") {
      config.security = value;
    } else if (arg == "--secret-key-dist") {
      config.secret_key_dist = value;
    } else if (arg == "--artifact-version") {
      config.artifact_version = value;
    } else if (arg == "--repo-commit") {
      config.repo_commit = value;
    } else if (arg == "--output-json") {
      config.output_json = value;
    } else {
      throw std::invalid_argument(std::string("unknown argument: ") + std::string(arg));
    }
  }
  if (config.ring_dim <= 0 || (config.ring_dim & (config.ring_dim - 1)) != 0) {
    throw std::invalid_argument("ring-dim must be a positive power of two");
  }
  if (config.num_slots <= 0 || config.num_slots > config.ring_dim / 2) {
    throw std::invalid_argument("num-slots must be in [1, ring_dim / 2]");
  }
  if (config.multiplicative_depth <= 0 || config.scaling_mod_size <= 0 ||
      config.first_mod_size <= 0) {
    throw std::invalid_argument(
        "multiplicative-depth, scaling-mod-size, and first-mod-size must be positive");
  }
  if (config.level_budget_cts <= 0 || config.level_budget_stc <= 0 ||
      config.bsgs_dim_cts < 0 || config.bsgs_dim_stc < 0 || config.iterations <= 0) {
    throw std::invalid_argument("bootstrap budgets/dimensions and iterations are invalid");
  }
  if ((config.bootstrap_iterations != 1 && config.bootstrap_iterations != 2) ||
      config.bootstrap_precision < 0 || config.bootstrap_precision >= 31) {
    throw std::invalid_argument(
        "bootstrap-iterations must be 1 or 2 and bootstrap-precision must be in [0, 31)");
  }
  if (config.complex_pair && config.square_input) {
    throw std::invalid_argument("complex-pair and square-input cannot be combined");
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
  return config;
}

auto resolve_security(const std::string& value) -> SecurityLevel {
  if (value == "128-classic") {
    return HEStd_128_classic;
  }
  return HEStd_NotSet;
}

auto resolve_secret_key_dist(const std::string& value) -> SecretKeyDist {
  if (value == "uniform-ternary") {
    return UNIFORM_TERNARY;
  }
  if (value == "sparse-encapsulated") {
    return fideslib::SPARSE_ENCAPSULATED;
  }
  return SPARSE_TERNARY;
}

auto make_input(int slots, int seed, double magnitude) -> std::vector<double> {
  std::mt19937 rng(static_cast<uint32_t>(seed));
  std::uniform_real_distribution<double> distribution(-magnitude, magnitude);
  std::vector<double> values(static_cast<size_t>(slots));
  for (double& value : values) {
    value = distribution(rng);
  }
  return values;
}

auto mean(const std::vector<double>& values) -> double {
  double total = 0.0;
  for (double value : values) {
    total += value;
  }
  return total / static_cast<double>(values.size());
}

auto max_abs_error(const std::vector<double>& expected, const std::vector<double>& actual) -> double {
  double error = 0.0;
  for (size_t i = 0; i < expected.size(); ++i) {
    const double diff = std::abs(expected[i] - actual[i]);
    if (!std::isfinite(diff)) {
      // NaN must read as failure, not as zero: std::max(0.0, NaN) returns 0.0.
      return std::numeric_limits<double>::infinity();
    }
    error = std::max(error, diff);
  }
  return error;
}

struct ComplexErrors {
  double real = 0.0;
  double imag = 0.0;
};

auto complex_component_errors(
    const std::vector<double>& expected_real,
    const std::vector<double>& expected_imag,
    const std::vector<std::complex<double>>& actual) -> ComplexErrors {
  if (actual.size() < expected_real.size() || expected_real.size() != expected_imag.size()) {
    return {
        .real = std::numeric_limits<double>::infinity(),
        .imag = std::numeric_limits<double>::infinity(),
    };
  }
  ComplexErrors errors;
  for (size_t slot = 0; slot < expected_real.size(); ++slot) {
    const double real_error = std::abs(expected_real[slot] - actual[slot].real());
    const double imag_error = std::abs(expected_imag[slot] - actual[slot].imag());
    if (!std::isfinite(real_error) || !std::isfinite(imag_error)) {
      return {
          .real = std::numeric_limits<double>::infinity(),
          .imag = std::numeric_limits<double>::infinity(),
      };
    }
    errors.real = std::max(errors.real, real_error);
    errors.imag = std::max(errors.imag, imag_error);
  }
  return errors;
}

void write_json_vector(std::ostream& out, const std::vector<double>& values) {
  out << "[";
  for (size_t i = 0; i < values.size(); ++i) {
    if (i > 0) {
      out << ",";
    }
    out << values[i];
  }
  out << "]";
}

auto build_payload(
    const Config& config,
    int actual_ring_dim,
    double setup_seconds,
    double precompute_seconds,
    const std::vector<double>& latencies,
    const std::vector<double>& conjugate_latencies,
    const std::vector<double>& split_latencies,
    int initial_levels_remaining,
    const std::vector<int>& levels_after,
    double probe_max_abs_error,
    double pair_input_real_max_abs_error,
    double pair_input_imag_max_abs_error,
    double pair_gpu_load_real_max_abs_error,
    double pair_gpu_load_imag_max_abs_error,
    double pair_bootstrap_real_max_abs_error,
    double pair_bootstrap_imag_max_abs_error,
    double pair_real_max_abs_error,
    double pair_imag_max_abs_error,
    bool decrypt_checked,
    int decrypt_failure_count) -> std::string {
  const bool target_shape_compatible = config.ring_dim >= 65536 && config.num_slots >= 32768 &&
                                       config.multiplicative_depth >= 44 &&
                                       config.scaling_mod_size >= 54;
  const bool security_claimed = config.security == "128-classic";
  const bool pair_accuracy_passed =
      !config.complex_pair || !decrypt_checked || probe_max_abs_error <= 0.05;
  const bool passed = decrypt_failure_count == 0 && pair_accuracy_passed;
  std::ostringstream out;
  out << "{";
  out << "\"version\":\"" << config.artifact_version << "\",";
  out << "\"stage\":\""
      << (config.complex_pair ? "fideslib-gpu-stage1-complex-pair-bootstrap"
                              : "fideslib-gpu-stage1-bootstrap-latency")
      << "\",";
  out << "\"repo_commit\":\"" << config.repo_commit << "\",";
  out << "\"backend\":\"fideslib-gpu\",";
  out << "\"available\":true,";
  out << "\"encrypted\":true,";
  out << "\"status\":\"" << (passed ? "passed" : "failed") << "\",";
  out << "\"passed\":" << (passed ? "true" : "false") << ",";
  out << "\"ring_dimension\":" << actual_ring_dim << ",";
  out << "\"batch_size\":" << config.num_slots << ",";
  out << "\"config\":{";
  out << "\"input_mode\":\""
      << (config.complex_pair ? "complex-pair-bootstrap-probe" : "bootstrap-probe")
      << "\",";
  out << "\"ring_dimension\":" << actual_ring_dim << ",";
  out << "\"num_slots\":" << config.num_slots << ",";
  out << "\"multiplicative_depth\":" << config.multiplicative_depth << ",";
  out << "\"scaling_mod_size\":" << config.scaling_mod_size << ",";
  out << "\"first_mod_size\":" << config.first_mod_size << ",";
  out << "\"level_budget\":[" << config.level_budget_cts << "," << config.level_budget_stc
      << "],";
  out << "\"bsgs_dim\":[" << config.bsgs_dim_cts << "," << config.bsgs_dim_stc << "],";
  out << "\"security\":\"" << config.security << "\",";
  out << "\"secret_key_dist\":\"" << config.secret_key_dist << "\",";
  out << "\"iterations\":" << config.iterations << ",";
  out << "\"bootstrap_iterations\":" << config.bootstrap_iterations << ",";
  out << "\"bootstrap_precision\":" << config.bootstrap_precision << ",";
  out << "\"input_magnitude\":" << config.input_magnitude << ",";
  out << "\"encrypt_level\":" << config.encrypt_level << ",";
  out << "\"complex_pair\":" << (config.complex_pair ? "true" : "false") << ",";
  out << "\"skip_decrypt\":" << (config.skip_decrypt ? "true" : "false");
  out << "},";
  out << "\"latencies_sec\":";
  write_json_vector(out, latencies);
  out << ",";
  out << "\"conjugate_latencies_sec\":";
  write_json_vector(out, conjugate_latencies);
  out << ",";
  out << "\"split_latencies_sec\":";
  write_json_vector(out, split_latencies);
  out << ",";
  out << "\"mean_latency_sec\":" << mean(latencies) << ",";
  out << "\"min_latency_sec\":" << *std::min_element(latencies.begin(), latencies.end()) << ",";
  out << "\"max_latency_sec\":" << *std::max_element(latencies.begin(), latencies.end()) << ",";
  if (config.complex_pair) {
    out << "\"mean_conjugate_latency_sec\":" << mean(conjugate_latencies) << ",";
    out << "\"mean_split_latency_sec\":" << mean(split_latencies) << ",";
  }
  out << "\"measurements\":{";
  out << "\"probe_repetitions\":" << config.iterations << ",";
  out << "\"bootstrap_iterations_per_probe\":" << config.bootstrap_iterations << ",";
  out << "\"mean_latency_sec\":" << mean(latencies) << ",";
  out << "\"min_latency_sec\":" << *std::min_element(latencies.begin(), latencies.end()) << ",";
  out << "\"max_latency_sec\":" << *std::max_element(latencies.begin(), latencies.end()) << ",";
  out << "\"stage1_target_shape_compatible\":" << (target_shape_compatible ? "true" : "false")
      << ",";
  out << "\"stage1_target_compatible\":"
      << (target_shape_compatible && security_claimed ? "true" : "false");
  out << "},";
  out << "\"timing\":{";
  out << "\"setup_seconds\":" << setup_seconds << ",";
  out << "\"bootstrap_precompute_seconds\":" << precompute_seconds;
  out << "},";
  out << "\"levels_before\":" << initial_levels_remaining << ",";
  out << "\"levels_after\":[";
  for (size_t i = 0; i < levels_after.size(); ++i) {
    if (i > 0) {
      out << ",";
    }
    out << levels_after[i];
  }
  out << "],";
  out << "\"decrypt_checked\":" << (decrypt_checked ? "true" : "false") << ",";
  out << "\"decrypt_failure_count\":" << decrypt_failure_count << ",";
  out << "\"max_abs_error\":";
  if (decrypt_checked && decrypt_failure_count == 0 && std::isfinite(probe_max_abs_error)) {
    out << probe_max_abs_error;
  } else {
    out << "null";
  }
  out << ",";
  out << "\"pair_real_max_abs_error\":";
  if (config.complex_pair && decrypt_checked && decrypt_failure_count == 0 &&
      std::isfinite(pair_real_max_abs_error)) {
    out << pair_real_max_abs_error;
  } else {
    out << "null";
  }
  out << ",";
  out << "\"pair_imag_max_abs_error\":";
  if (config.complex_pair && decrypt_checked && decrypt_failure_count == 0 &&
      std::isfinite(pair_imag_max_abs_error)) {
    out << pair_imag_max_abs_error;
  } else {
    out << "null";
  }
  out << ",";
  out << "\"pair_input_real_max_abs_error\":";
  if (config.complex_pair && decrypt_checked && decrypt_failure_count == 0) {
    out << pair_input_real_max_abs_error;
  } else {
    out << "null";
  }
  out << ",";
  out << "\"pair_input_imag_max_abs_error\":";
  if (config.complex_pair && decrypt_checked && decrypt_failure_count == 0) {
    out << pair_input_imag_max_abs_error;
  } else {
    out << "null";
  }
  out << ",";
  out << "\"pair_gpu_load_real_max_abs_error\":";
  if (config.complex_pair && decrypt_checked && decrypt_failure_count == 0) {
    out << pair_gpu_load_real_max_abs_error;
  } else {
    out << "null";
  }
  out << ",";
  out << "\"pair_gpu_load_imag_max_abs_error\":";
  if (config.complex_pair && decrypt_checked && decrypt_failure_count == 0) {
    out << pair_gpu_load_imag_max_abs_error;
  } else {
    out << "null";
  }
  out << ",";
  out << "\"pair_bootstrap_real_max_abs_error\":";
  if (config.complex_pair && decrypt_checked && decrypt_failure_count == 0) {
    out << pair_bootstrap_real_max_abs_error;
  } else {
    out << "null";
  }
  out << ",";
  out << "\"pair_bootstrap_imag_max_abs_error\":";
  if (config.complex_pair && decrypt_checked && decrypt_failure_count == 0) {
    out << pair_bootstrap_imag_max_abs_error;
  } else {
    out << "null";
  }
  out << ",";
  out << "\"operation_counts\":{";
  out << "\"bootstraps\":"
      << config.iterations * config.bootstrap_iterations << ",";
  out << "\"conjugations\":" << (config.complex_pair ? config.iterations : 0) << ",";
  out << "\"rotations\":0,";
  out << "\"ct_ct_mul\":0,";
  out << "\"ct_pt_mul\":" << (config.complex_pair ? config.iterations : 0) << ",";
  out << "\"encrypt\":" << config.iterations << ",";
  out << "\"decrypt\":"
      << (decrypt_checked ? (config.complex_pair ? 5 : 1) * config.iterations : 0);
  out << "},";
  out << "\"measurement_scope\":{";
  out << "\"bootstrap_latency_probe\":true,";
  out << "\"complex_pair_extraction_probe\":"
      << (config.complex_pair ? "true" : "false") << ",";
  out << "\"gpu_bootstrap\":true,";
  out << "\"stage1_target_shape_compatible\":" << (target_shape_compatible ? "true" : "false")
      << ",";
  out << "\"stage1_target_compatible\":"
      << (target_shape_compatible && security_claimed ? "true" : "false") << ",";
  out << "\"he_security_claimed\":" << (security_claimed ? "true" : "false") << ",";
  out << "\"correctness_checked\":" << (decrypt_checked ? "true" : "false") << ",";
  out << "\"full_model_correctness_claimed\":false,";
  out << "\"claim\":\"FIDESlib GPU bootstrap latency probe at Stage 1-sized CKKS shape. "
         "Complex-pair mode additionally checks homomorphic real/imaginary extraction after "
         "one shared bootstrap. This does not claim full Mamba checkpoint execution.\"";
  out << "}";
  out << "}";
  return out.str();
}

void write_payload(const std::string& output_json, const std::string& payload) {
  if (output_json.empty()) {
    std::cout << payload << std::endl;
    return;
  }
  std::ofstream output(output_json);
  if (!output) {
    throw std::runtime_error("failed to open output-json path");
  }
  output << payload << std::endl;
}

}  // namespace

auto main(int argc, char* argv[]) -> int {
  try {
    const Config config = parse_args(argc, argv);
    const auto setup_start = now();

    CCParams<CryptoContextCKKSRNS> parameters;
    parameters.SetSecretKeyDist(resolve_secret_key_dist(config.secret_key_dist));
    parameters.SetSecurityLevel(resolve_security(config.security));
    parameters.SetRingDim(static_cast<uint32_t>(config.ring_dim));
    parameters.SetScalingTechnique(FLEXIBLEAUTO);
    parameters.SetFirstModSize(static_cast<uint32_t>(config.first_mod_size));
    parameters.SetKeySwitchTechnique(HYBRID);
    parameters.SetMultiplicativeDepth(static_cast<uint32_t>(config.multiplicative_depth));
    parameters.SetScalingModSize(static_cast<uint32_t>(config.scaling_mod_size));
    parameters.SetBatchSize(static_cast<uint32_t>(config.num_slots));
    parameters.SetCKKSDataType(config.complex_pair ? COMPLEX : REAL);
    parameters.SetDevices({0});
    parameters.SetPlaintextAutoload(false);
    parameters.SetCiphertextAutoload(!config.complex_pair);
    if (config.secret_key_dist == "sparse-ternary" ||
        config.secret_key_dist == "sparse-encapsulated") {
      parameters.SetNumLargeDigits(3);
    }

    CryptoContext<DCRTPoly> cc = GenCryptoContext(parameters);
    cc->Enable(PKE);
    cc->Enable(KEYSWITCH);
    cc->Enable(LEVELEDSHE);
    cc->Enable(ADVANCEDSHE);
    cc->Enable(FHE);

    auto keys = cc->KeyGen();
    cc->EvalMultKeyGen(keys.secretKey);

    const double setup_seconds = seconds_since(setup_start);
    const auto precompute_start = now();
    const std::vector<uint32_t> level_budget = {
        static_cast<uint32_t>(config.level_budget_cts),
        static_cast<uint32_t>(config.level_budget_stc),
    };
    const std::vector<uint32_t> bsgs_dim = {
        static_cast<uint32_t>(config.bsgs_dim_cts),
        static_cast<uint32_t>(config.bsgs_dim_stc),
    };
    cc->EvalBootstrapSetup(level_budget, bsgs_dim, static_cast<uint32_t>(config.num_slots), 0);
    cc->EvalBootstrapKeyGen(keys.secretKey, static_cast<uint32_t>(config.num_slots));
    cc->LoadContext(keys.publicKey);
    const double precompute_seconds = seconds_since(precompute_start);

    const auto input = make_input(config.num_slots, config.seed, config.input_magnitude);
    const auto pair_imag_input =
        make_input(config.num_slots, config.seed + 1, config.input_magnitude);
    std::vector<double> latencies;
    std::vector<double> conjugate_latencies;
    std::vector<double> split_latencies;
    std::vector<int> levels_after;
    latencies.reserve(static_cast<size_t>(config.iterations));
    conjugate_latencies.reserve(static_cast<size_t>(config.iterations));
    split_latencies.reserve(static_cast<size_t>(config.iterations));
    levels_after.reserve(static_cast<size_t>(config.iterations));
    double probe_max_abs_error = 0.0;
    double pair_input_real_max_abs_error = 0.0;
    double pair_input_imag_max_abs_error = 0.0;
    double pair_gpu_load_real_max_abs_error = 0.0;
    double pair_gpu_load_imag_max_abs_error = 0.0;
    double pair_bootstrap_real_max_abs_error = 0.0;
    double pair_bootstrap_imag_max_abs_error = 0.0;
    double pair_real_max_abs_error = 0.0;
    double pair_imag_max_abs_error = 0.0;
    int initial_levels_remaining = 0;
    int decrypt_failure_count = 0;

    auto decrypt_complex_errors = [&](Ciphertext<DCRTPoly>& value) -> ComplexErrors {
      Plaintext result;
      cc->Decrypt(keys.secretKey, value, &result);
      result->SetLength(static_cast<size_t>(config.num_slots));
      const auto errors =
          complex_component_errors(input, pair_imag_input, result->GetCKKSPackedValue());
      if (!std::isfinite(errors.real) || !std::isfinite(errors.imag)) {
        throw std::runtime_error("non-finite complex decode");
      }
      return errors;
    };

    for (int iteration = 0; iteration < config.iterations; ++iteration) {
      const uint32_t encode_level =
          config.encrypt_level >= 0
              ? static_cast<uint32_t>(config.encrypt_level)
              : static_cast<uint32_t>(config.multiplicative_depth - 1);
      Plaintext plaintext;
      if (config.complex_pair) {
        std::vector<std::complex<double>> packed_input(static_cast<size_t>(config.num_slots));
        for (size_t slot = 0; slot < packed_input.size(); ++slot) {
          packed_input[slot] = {input[slot], pair_imag_input[slot]};
        }
        plaintext = cc->MakeCKKSPackedPlaintext(
            packed_input,
            1,
            encode_level,
            nullptr,
            static_cast<uint32_t>(config.num_slots));
      } else {
        plaintext = cc->MakeCKKSPackedPlaintext(
            input,
            1,
            encode_level,
            nullptr,
            static_cast<uint32_t>(config.num_slots));
      }
      plaintext->SetLength(static_cast<size_t>(config.num_slots));
      Ciphertext<DCRTPoly> ciphertext = cc->Encrypt(keys.publicKey, plaintext);
      if (config.complex_pair && !config.skip_decrypt) {
        try {
          const auto errors = decrypt_complex_errors(ciphertext);
          pair_input_real_max_abs_error =
              std::max(pair_input_real_max_abs_error, errors.real);
          pair_input_imag_max_abs_error =
              std::max(pair_input_imag_max_abs_error, errors.imag);
        } catch (const std::exception& exc) {
          ++decrypt_failure_count;
          std::cerr << "complex-pair input decrypt failed at iteration " << iteration << ": "
                    << exc.what() << std::endl;
        }
      }
      if (config.complex_pair) {
        cc->LoadCiphertext(ciphertext);
        cc->Synchronize();
        if (!config.skip_decrypt) {
          try {
            const auto errors = decrypt_complex_errors(ciphertext);
            pair_gpu_load_real_max_abs_error =
                std::max(pair_gpu_load_real_max_abs_error, errors.real);
            pair_gpu_load_imag_max_abs_error =
                std::max(pair_gpu_load_imag_max_abs_error, errors.imag);
          } catch (const std::exception& exc) {
            ++decrypt_failure_count;
            std::cerr << "complex-pair GPU-load decrypt failed at iteration " << iteration
                      << ": " << exc.what() << std::endl;
          }
        }
      }
      std::vector<double> expected = input;
      if (config.square_input) {
        // Real product: bootstrap input becomes noiseScaleDeg 2 with
        // magnitude input_magnitude^2.
        ciphertext = cc->EvalMult(ciphertext, ciphertext);
        for (double& value : expected) {
          value *= value;
        }
        std::cerr << "square-input: level=" << ciphertext->GetLevel()
                  << " deg=" << ciphertext->GetNoiseScaleDeg() << std::endl;
      }
      if (iteration == 0) {
        initial_levels_remaining = config.multiplicative_depth - static_cast<int>(ciphertext->GetLevel());
      }
      const auto bootstrap_start = now();
      Ciphertext<DCRTPoly> bootstrapped = cc->EvalBootstrap(
          ciphertext, static_cast<uint32_t>(config.bootstrap_iterations),
          static_cast<uint32_t>(config.bootstrap_precision));
      latencies.push_back(seconds_since(bootstrap_start));
      levels_after.push_back(
          config.multiplicative_depth - static_cast<int>(bootstrapped->GetLevel()));

      if (config.skip_decrypt && !config.complex_pair) {
        continue;
      }
      if (config.complex_pair) {
        if (!config.skip_decrypt) {
          try {
            const auto errors = decrypt_complex_errors(bootstrapped);
            pair_bootstrap_real_max_abs_error =
                std::max(pair_bootstrap_real_max_abs_error, errors.real);
            pair_bootstrap_imag_max_abs_error =
                std::max(pair_bootstrap_imag_max_abs_error, errors.imag);
          } catch (const std::exception& exc) {
            ++decrypt_failure_count;
            std::cerr << "complex-pair post-bootstrap decrypt failed at iteration "
                      << iteration << ": " << exc.what() << std::endl;
          }
        }
        cc->Synchronize();
        const auto conjugate_start = now();
        auto conjugated = cc->EvalConjugate(bootstrapped);
        cc->Synchronize();
        conjugate_latencies.push_back(seconds_since(conjugate_start));

        std::vector<std::complex<double>> minus_half_i(
            static_cast<size_t>(config.num_slots), {0.0, -0.5});
        auto minus_half_i_plain = cc->MakeCKKSPackedPlaintext(
            minus_half_i,
            1,
            static_cast<uint32_t>(bootstrapped->GetLevel()),
            nullptr,
            static_cast<uint32_t>(config.num_slots));
        cc->LoadPlaintext(minus_half_i_plain);
        cc->Synchronize();

        const auto split_start = now();
        auto real_part = cc->EvalAdd(bootstrapped, conjugated);
        cc->EvalMultInPlace(real_part, 0.5);
        auto imag_part = cc->EvalSub(bootstrapped, conjugated);
        cc->EvalMultInPlace(imag_part, minus_half_i_plain);
        cc->Synchronize();
        split_latencies.push_back(seconds_since(split_start));

        if (config.skip_decrypt) {
          continue;
        }
        try {
          Plaintext real_result;
          cc->Decrypt(keys.secretKey, real_part, &real_result);
          real_result->SetLength(static_cast<size_t>(config.num_slots));
          auto real_decoded = real_result->GetRealPackedValue();
          real_decoded.resize(static_cast<size_t>(config.num_slots));

          Plaintext imag_result;
          cc->Decrypt(keys.secretKey, imag_part, &imag_result);
          imag_result->SetLength(static_cast<size_t>(config.num_slots));
          auto imag_decoded = imag_result->GetRealPackedValue();
          imag_decoded.resize(static_cast<size_t>(config.num_slots));

          const double real_error = max_abs_error(input, real_decoded);
          const double imag_error = max_abs_error(pair_imag_input, imag_decoded);
          if (!std::isfinite(real_error) || !std::isfinite(imag_error)) {
            ++decrypt_failure_count;
            std::cerr << "non-finite complex-pair decode after bootstrap iteration "
                      << iteration << std::endl;
            continue;
          }
          pair_real_max_abs_error = std::max(pair_real_max_abs_error, real_error);
          pair_imag_max_abs_error = std::max(pair_imag_max_abs_error, imag_error);
          probe_max_abs_error = std::max(probe_max_abs_error, std::max(real_error, imag_error));
        } catch (const std::exception& exc) {
          ++decrypt_failure_count;
          std::cerr << "complex-pair decrypt check failed after bootstrap iteration "
                    << iteration << ": " << exc.what() << std::endl;
        }
        continue;
      }
      Plaintext result;
      try {
        cc->Decrypt(keys.secretKey, bootstrapped, &result);
        result->SetLength(static_cast<size_t>(config.num_slots));
        auto decoded = result->GetRealPackedValue();
        decoded.resize(static_cast<size_t>(config.num_slots));
        const double iteration_error = max_abs_error(expected, decoded);
        if (!std::isfinite(iteration_error)) {
          // NaN/Inf in the decode is a failed refresh, not a zero-error one.
          ++decrypt_failure_count;
          std::cerr << "non-finite decode after bootstrap iteration " << iteration << std::endl;
          continue;
        }
        probe_max_abs_error = std::max(probe_max_abs_error, iteration_error);
      } catch (const std::exception& exc) {
        ++decrypt_failure_count;
        std::cerr << "decrypt check failed after bootstrap iteration " << iteration << ": "
                  << exc.what() << std::endl;
      }
    }

    write_payload(
        config.output_json,
        build_payload(
            config,
            static_cast<int>(cc->GetRingDimension()),
            setup_seconds,
            precompute_seconds,
            latencies,
            conjugate_latencies,
            split_latencies,
            initial_levels_remaining,
            levels_after,
            probe_max_abs_error,
            pair_input_real_max_abs_error,
            pair_input_imag_max_abs_error,
            pair_gpu_load_real_max_abs_error,
            pair_gpu_load_imag_max_abs_error,
            pair_bootstrap_real_max_abs_error,
            pair_bootstrap_imag_max_abs_error,
            pair_real_max_abs_error,
            pair_imag_max_abs_error,
            !config.skip_decrypt,
            decrypt_failure_count));
    if (decrypt_failure_count != 0 ||
        (config.complex_pair && !config.skip_decrypt && probe_max_abs_error > 0.05)) {
      return EXIT_FAILURE;
    }
  } catch (const std::exception& exc) {
    std::cerr << "stage1_bootstrap_probe failed: " << exc.what() << std::endl;
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
