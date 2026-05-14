#include <fideslib.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <random>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

using namespace fideslib;

namespace {

struct Config {
  int ring_dim = 131072;
  int num_slots = 32768;
  int multiplicative_depth = 48;
  int scaling_mod_size = 40;
  int first_mod_size = 60;
  int iterations = 1;
  int seed = 0;
  int rotation_limit = 0;
  bool skip_decrypt = true;
  std::string security = "128-classic";
  std::string secret_key_dist = "sparse-ternary";
  std::string rotations_csv;
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

auto parse_rotations_csv(const std::string& value) -> std::vector<int32_t> {
  std::set<int32_t> rotations;
  std::stringstream stream(value);
  std::string token;
  while (std::getline(stream, token, ',')) {
    if (token.empty()) {
      continue;
    }
    const int rotation = parse_int("--rotations-csv", token.c_str());
    if (rotation != 0) {
      rotations.insert(static_cast<int32_t>(rotation));
    }
  }
  if (rotations.empty()) {
    throw std::invalid_argument("rotations-csv must include at least one nonzero rotation");
  }
  return {rotations.begin(), rotations.end()};
}

auto parse_args(int argc, char* argv[]) -> Config {
  Config config;
  for (int i = 1; i < argc; ++i) {
    const std::string_view arg(argv[i]);
    if (arg == "--skip-decrypt") {
      config.skip_decrypt = true;
      continue;
    }
    if (arg == "--check-decrypt") {
      config.skip_decrypt = false;
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
    } else if (arg == "--iterations") {
      config.iterations = parse_int(arg, value);
    } else if (arg == "--seed") {
      config.seed = parse_int(arg, value);
    } else if (arg == "--rotation-limit") {
      config.rotation_limit = parse_int(arg, value);
    } else if (arg == "--security") {
      config.security = value;
    } else if (arg == "--secret-key-dist") {
      config.secret_key_dist = value;
    } else if (arg == "--rotations-csv") {
      config.rotations_csv = value;
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
      config.first_mod_size <= 0 || config.iterations <= 0 || config.rotation_limit < 0) {
    throw std::invalid_argument(
        "depth, scale, first modulus, iterations, and rotation limit are invalid");
  }
  if (config.rotations_csv.empty()) {
    throw std::invalid_argument("rotations-csv is required");
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

auto make_input(int slots, int seed) -> std::vector<double> {
  std::mt19937 rng(static_cast<uint32_t>(seed));
  std::uniform_real_distribution<double> distribution(-0.05, 0.05);
  std::vector<double> values(static_cast<size_t>(slots));
  for (double& value : values) {
    value = distribution(rng);
  }
  return values;
}

auto make_weights(int slots) -> std::vector<double> {
  std::vector<double> values(static_cast<size_t>(slots));
  for (int i = 0; i < slots; ++i) {
    values[static_cast<size_t>(i)] = 0.001 * static_cast<double>((i % 17) + 1);
  }
  return values;
}

auto read_status_kib(const std::string& key) -> long long {
  std::ifstream status("/proc/self/status");
  std::string line;
  while (std::getline(status, line)) {
    if (line.rfind(key + ":", 0) != 0) {
      continue;
    }
    std::stringstream stream(line.substr(key.size() + 1));
    long long value = 0;
    std::string unit;
    stream >> value >> unit;
    return value;
  }
  return 0;
}

auto rss_gib() -> double { return static_cast<double>(read_status_kib("VmRSS")) / (1024.0 * 1024.0); }

auto peak_rss_gib() -> double {
  return static_cast<double>(read_status_kib("VmHWM")) / (1024.0 * 1024.0);
}

void log_phase(const std::string& message) {
  std::cerr << "[stage1_rotation_probe] " << message << " rss_gib=" << rss_gib()
            << " peak_rss_gib=" << peak_rss_gib() << std::endl;
}

void write_json_int_vector(std::ostream& out, const std::vector<int32_t>& values) {
  out << "[";
  for (size_t i = 0; i < values.size(); ++i) {
    if (i > 0) {
      out << ",";
    }
    out << values[i];
  }
  out << "]";
}

void write_json_double_vector(std::ostream& out, const std::vector<double>& values) {
  out << "[";
  for (size_t i = 0; i < values.size(); ++i) {
    if (i > 0) {
      out << ",";
    }
    out << values[i];
  }
  out << "]";
}

auto limited_rotations(const std::vector<int32_t>& rotations, int limit) -> std::vector<int32_t> {
  if (limit <= 0 || static_cast<size_t>(limit) >= rotations.size()) {
    return rotations;
  }
  return {rotations.begin(), rotations.begin() + limit};
}

auto build_payload(
    const Config& config,
    const std::vector<int32_t>& requested_rotations,
    const std::vector<int32_t>& executed_rotations,
    int actual_ring_dim,
    double context_setup_seconds,
    double rotate_keygen_seconds,
    double load_context_seconds,
    double encrypt_seconds,
    const std::vector<double>& iteration_latencies,
    double memory_after_context_gib,
    double memory_after_rotate_keygen_gib,
    double memory_after_load_context_gib,
    double memory_after_eval_gib,
    int decrypt_failure_count,
    int final_level) -> std::string {
  const bool target_shape_compatible =
      config.ring_dim >= 131072 && config.num_slots >= 32768 &&
      config.multiplicative_depth >= 48 && config.scaling_mod_size <= 40 &&
      requested_rotations.size() <= 200;
  const bool security_claimed = config.security == "128-classic";
  const bool passed = decrypt_failure_count == 0 && !executed_rotations.empty();
  double mean_latency = 0.0;
  for (const double latency : iteration_latencies) {
    mean_latency += latency;
  }
  mean_latency /= static_cast<double>(iteration_latencies.size());

  std::ostringstream out;
  out << "{";
  out << "\"stage\":\"fideslib-gpu-stage1-state-major-rotation-probe\",";
  out << "\"backend\":\"fideslib-gpu\",";
  out << "\"available\":true,";
  out << "\"encrypted\":true,";
  out << "\"passed\":" << (passed ? "true" : "false") << ",";
  out << "\"ring_dimension\":" << actual_ring_dim << ",";
  out << "\"batch_size\":" << config.num_slots << ",";
  out << "\"config\":{";
  out << "\"input_mode\":\"state-major-rotation-probe\",";
  out << "\"ring_dimension\":" << actual_ring_dim << ",";
  out << "\"num_slots\":" << config.num_slots << ",";
  out << "\"multiplicative_depth\":" << config.multiplicative_depth << ",";
  out << "\"scaling_mod_size\":" << config.scaling_mod_size << ",";
  out << "\"first_mod_size\":" << config.first_mod_size << ",";
  out << "\"security\":\"" << config.security << "\",";
  out << "\"secret_key_dist\":\"" << config.secret_key_dist << "\",";
  out << "\"iterations\":" << config.iterations << ",";
  out << "\"rotation_limit\":" << config.rotation_limit << ",";
  out << "\"skip_decrypt\":" << (config.skip_decrypt ? "true" : "false");
  out << "},";
  out << "\"required_application_rotations\":";
  write_json_int_vector(out, requested_rotations);
  out << ",\"executed_rotations\":";
  write_json_int_vector(out, executed_rotations);
  out << ",\"latencies_sec\":";
  write_json_double_vector(out, iteration_latencies);
  out << ",";
  out << "\"mean_latency_sec\":" << mean_latency << ",";
  out << "\"min_latency_sec\":" << *std::min_element(iteration_latencies.begin(), iteration_latencies.end()) << ",";
  out << "\"max_latency_sec\":" << *std::max_element(iteration_latencies.begin(), iteration_latencies.end()) << ",";
  out << "\"measurements\":{";
  out << "\"requested_rotation_key_count\":" << requested_rotations.size() << ",";
  out << "\"executed_rotation_count\":" << executed_rotations.size() << ",";
  out << "\"mean_latency_sec\":" << mean_latency << ",";
  out << "\"memory_after_context_gib\":" << memory_after_context_gib << ",";
  out << "\"memory_after_rotate_keygen_gib\":" << memory_after_rotate_keygen_gib << ",";
  out << "\"memory_after_load_context_gib\":" << memory_after_load_context_gib << ",";
  out << "\"memory_after_eval_gib\":" << memory_after_eval_gib << ",";
  out << "\"peak_rss_gib\":" << peak_rss_gib() << ",";
  out << "\"stage1_state_major_target_compatible\":"
      << (target_shape_compatible && security_claimed ? "true" : "false");
  out << "},";
  out << "\"timing\":{";
  out << "\"context_setup_seconds\":" << context_setup_seconds << ",";
  out << "\"rotate_keygen_seconds\":" << rotate_keygen_seconds << ",";
  out << "\"load_context_seconds\":" << load_context_seconds << ",";
  out << "\"encrypt_seconds\":" << encrypt_seconds << ",";
  out << "\"eval_seconds\":" << mean_latency;
  out << "},";
  out << "\"operation_counts\":{";
  out << "\"bootstraps\":0,";
  out << "\"rotations\":" << executed_rotations.size() * static_cast<size_t>(config.iterations) << ",";
  out << "\"ct_ct_mul\":0,";
  out << "\"ct_pt_mul\":" << executed_rotations.size() * static_cast<size_t>(config.iterations) << ",";
  out << "\"encrypt\":" << config.iterations << ",";
  out << "\"decrypt\":" << (config.skip_decrypt ? 0 : config.iterations - decrypt_failure_count);
  out << "},";
  out << "\"final_level\":" << final_level << ",";
  out << "\"decrypt_checked\":" << (config.skip_decrypt ? "false" : "true") << ",";
  out << "\"decrypt_failure_count\":" << decrypt_failure_count << ",";
  out << "\"measurement_scope\":{";
  out << "\"stage1_fideslib_rotation_probe\":true,";
  out << "\"state_major_layout\":true,";
  out << "\"rank_pack_first\":true,";
  out << "\"gpu_rotations\":true,";
  out << "\"key_memory_probe\":true,";
  out << "\"representative_bsgs_rotation_group\":true,";
  out << "\"stage1_state_major_target_compatible\":"
      << (target_shape_compatible && security_claimed ? "true" : "false") << ",";
  out << "\"he_security_claimed\":" << (security_claimed ? "true" : "false") << ",";
  out << "\"full_model_correctness_claimed\":false,";
  out << "\"claim\":\"FIDESlib GPU probe for the bounded Stage 1 state-major "
         "rotation inventory. This measures key generation, key loading, and a "
         "representative rotation plus plaintext-multiply group; it does not claim "
         "full Mamba checkpoint execution.\"";
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
    const std::vector<int32_t> requested_rotations = parse_rotations_csv(config.rotations_csv);
    const std::vector<int32_t> executed_rotations =
        limited_rotations(requested_rotations, config.rotation_limit);
    log_phase(
        "parsed rotations requested=" + std::to_string(requested_rotations.size()) +
        " executed=" + std::to_string(executed_rotations.size()));
    const auto context_start = now();

    log_phase("context setup begin");
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
    parameters.SetDevices({0});
    parameters.SetPlaintextAutoload(false);
    parameters.SetCiphertextAutoload(true);
    if (config.secret_key_dist == "sparse-ternary" ||
        config.secret_key_dist == "sparse-encapsulated") {
      parameters.SetNumLargeDigits(3);
    }

    CryptoContext<DCRTPoly> cc = GenCryptoContext(parameters);
    cc->Enable(PKE);
    cc->Enable(KEYSWITCH);
    cc->Enable(LEVELEDSHE);

    auto keys = cc->KeyGen();
    cc->EvalMultKeyGen(keys.secretKey);
    const double context_setup_seconds = seconds_since(context_start);
    const double memory_after_context_gib = rss_gib();
    log_phase("context setup done");

    const auto rotate_keygen_start = now();
    log_phase("rotate keygen begin");
    cc->EvalRotateKeyGen(keys.secretKey, requested_rotations);
    const double rotate_keygen_seconds = seconds_since(rotate_keygen_start);
    const double memory_after_rotate_keygen_gib = rss_gib();
    log_phase("rotate keygen done");

    const auto load_context_start = now();
    log_phase("load context begin");
    cc->LoadContext(keys.publicKey);
    const double load_context_seconds = seconds_since(load_context_start);
    const double memory_after_load_context_gib = rss_gib();
    log_phase("load context done");

    const auto input = make_input(config.num_slots, config.seed);
    const auto weights = make_weights(config.num_slots);
    auto weight_plain = cc->MakeCKKSPackedPlaintext(weights);
    weight_plain->SetLength(static_cast<size_t>(config.num_slots));

    std::vector<double> iteration_latencies;
    iteration_latencies.reserve(static_cast<size_t>(config.iterations));
    int decrypt_failure_count = 0;
    int final_level = 0;
    double encrypt_seconds = 0.0;

    for (int iteration = 0; iteration < config.iterations; ++iteration) {
      log_phase("iteration " + std::to_string(iteration) + " encrypt begin");
      const auto encrypt_start = now();
      auto input_plain = cc->MakeCKKSPackedPlaintext(input);
      input_plain->SetLength(static_cast<size_t>(config.num_slots));
      auto ciphertext = cc->Encrypt(keys.publicKey, input_plain);
      encrypt_seconds += seconds_since(encrypt_start);

      const auto eval_start = now();
      log_phase("iteration " + std::to_string(iteration) + " eval begin");
      Ciphertext<DCRTPoly> accumulator;
      bool has_accumulator = false;
      for (const int32_t rotation : executed_rotations) {
        auto rotated = cc->EvalRotate(ciphertext, rotation);
        auto term = cc->EvalMult(rotated, weight_plain);
        if (!has_accumulator) {
          accumulator = term;
          has_accumulator = true;
        } else {
          accumulator = cc->EvalAdd(accumulator, term);
        }
      }
      iteration_latencies.push_back(seconds_since(eval_start));
      log_phase("iteration " + std::to_string(iteration) + " eval done");
      if (has_accumulator) {
        final_level = static_cast<int>(accumulator->GetLevel());
      }

      if (config.skip_decrypt) {
        continue;
      }
      try {
        Plaintext result;
        cc->Decrypt(keys.secretKey, accumulator, &result);
        result->SetLength(static_cast<size_t>(config.num_slots));
        auto decoded = result->GetRealPackedValue();
        decoded.resize(static_cast<size_t>(config.num_slots));
        for (const double value : decoded) {
          if (!std::isfinite(value)) {
            throw std::runtime_error("decrypted probe output contains nonfinite values");
          }
        }
      } catch (const std::exception& exc) {
        ++decrypt_failure_count;
        std::cerr << "decrypt check failed after rotation iteration " << iteration << ": "
                  << exc.what() << std::endl;
      }
    }
    const double memory_after_eval_gib = rss_gib();

    write_payload(
        config.output_json,
        build_payload(
            config,
            requested_rotations,
            executed_rotations,
            static_cast<int>(cc->GetRingDimension()),
            context_setup_seconds,
            rotate_keygen_seconds,
            load_context_seconds,
            encrypt_seconds,
            iteration_latencies,
            memory_after_context_gib,
            memory_after_rotate_keygen_gib,
            memory_after_load_context_gib,
            memory_after_eval_gib,
            decrypt_failure_count,
            final_level));
  } catch (const std::exception& exc) {
    std::cerr << "stage1_rotation_probe failed: " << exc.what() << std::endl;
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
