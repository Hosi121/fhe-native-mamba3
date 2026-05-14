#include "stage1_rank_gate_eval.hpp"

#include <fideslib.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

using namespace fideslib;

namespace {

constexpr double kPlaintextCoefficientFloor = 1e-8;

struct Config {
  std::string input;
  std::string output_json;
  int ring_dim = 131072;
  int multiplicative_depth = 48;
  int scaling_mod_size = 40;
  int first_mod_size = 60;
  double atol = 1e-5;
  double rank_projection_scale = 1.0;
  std::string security = "128-classic";
  std::string secret_key_dist = "sparse-ternary";
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

auto parse_double(std::string_view name, const char* value) -> double {
  try {
    return std::stod(value);
  } catch (const std::exception& exc) {
    throw std::invalid_argument(std::string("invalid float for ") + std::string(name) + ": " +
                                exc.what());
  }
}

auto parse_args(int argc, char* argv[]) -> Config {
  Config config;
  for (int i = 1; i < argc; ++i) {
    const std::string_view arg(argv[i]);
    if (i + 1 >= argc) {
      throw std::invalid_argument(std::string("missing value for ") + std::string(arg));
    }
    const char* value = argv[++i];
    if (arg == "--input") {
      config.input = value;
    } else if (arg == "--output-json") {
      config.output_json = value;
    } else if (arg == "--ring-dim") {
      config.ring_dim = parse_int(arg, value);
    } else if (arg == "--multiplicative-depth") {
      config.multiplicative_depth = parse_int(arg, value);
    } else if (arg == "--scaling-mod-size") {
      config.scaling_mod_size = parse_int(arg, value);
    } else if (arg == "--first-mod-size") {
      config.first_mod_size = parse_int(arg, value);
    } else if (arg == "--atol") {
      config.atol = parse_double(arg, value);
    } else if (arg == "--rank-projection-scale") {
      config.rank_projection_scale = parse_double(arg, value);
    } else if (arg == "--security") {
      config.security = value;
    } else if (arg == "--secret-key-dist") {
      config.secret_key_dist = value;
    } else {
      throw std::invalid_argument(std::string("unknown argument: ") + std::string(arg));
    }
  }
  if (config.input.empty()) {
    throw std::invalid_argument("--input is required");
  }
  if (config.ring_dim <= 0 || (config.ring_dim & (config.ring_dim - 1)) != 0) {
    throw std::invalid_argument("ring-dim must be a positive power of two");
  }
  if (config.multiplicative_depth <= 0 || config.scaling_mod_size <= 0 ||
      config.first_mod_size <= 0 || config.atol < 0.0 || config.rank_projection_scale <= 0.0) {
    throw std::invalid_argument("invalid CKKS parameters");
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

auto python_mod(int value, int modulus) -> int {
  const int result = value % modulus;
  return result < 0 ? result + modulus : result;
}

auto slot_bsgs_giant_with_zero(int input_dim, int output_dim, int baby_step) -> std::vector<int> {
  std::set<int> values;
  const int min_offset = -(output_dim - 1);
  const int max_offset = input_dim - 1;
  for (int offset = min_offset; offset <= max_offset; ++offset) {
    values.insert(offset - python_mod(offset, baby_step));
  }
  return {values.begin(), values.end()};
}

auto slot_bsgs_rotations(int input_dim, int output_dim, int baby_step) -> std::vector<int32_t> {
  std::set<int32_t> rotations;
  for (int baby = 1; baby < baby_step; ++baby) {
    rotations.insert(static_cast<int32_t>(baby));
  }
  for (const int giant : slot_bsgs_giant_with_zero(input_dim, output_dim, baby_step)) {
    if (giant != 0) {
      rotations.insert(static_cast<int32_t>(giant));
    }
  }
  return {rotations.begin(), rotations.end()};
}

auto required_rank_gate_rotations(const stage1::RankGatePayloadConfig& config)
    -> std::vector<int32_t> {
  return slot_bsgs_rotations(config.d_model, config.mimo_rank, config.model_baby_step);
}

auto pack_block0(
    const std::vector<double>& values,
    const stage1::RankGatePayloadConfig& config) -> std::vector<double> {
  std::vector<double> slots(static_cast<size_t>(config.rank_pad) * config.d_state, 0.0);
  std::copy(values.begin(), values.end(), slots.begin());
  return slots;
}

auto scaled_values(const std::vector<double>& values, double scale) -> std::vector<double> {
  std::vector<double> output(values.size(), 0.0);
  for (std::size_t index = 0; index < values.size(); ++index) {
    output[index] = values[index] * scale;
  }
  return output;
}

auto unscaled_values(const std::vector<double>& values, double scale) -> std::vector<double> {
  std::vector<double> output(values.size(), 0.0);
  for (std::size_t index = 0; index < values.size(); ++index) {
    output[index] = values[index] / scale;
  }
  return output;
}

auto repeated_rank_values(double value, const stage1::RankGatePayloadConfig& config)
    -> std::vector<double> {
  return std::vector<double>(static_cast<std::size_t>(config.mimo_rank), value);
}

auto slot_bsgs_pre_mask(
    const std::vector<double>& weights,
    int input_dim,
    int output_dim,
    int batch_size,
    int giant,
    int offset) -> std::vector<double> {
  std::vector<double> mask(static_cast<size_t>(batch_size), 0.0);
  for (int output = 0; output < output_dim; ++output) {
    const int input = output + offset;
    if (input < 0 || input >= input_dim) {
      continue;
    }
    const int source_slot = python_mod(output + giant, batch_size);
    const double value = weights[static_cast<size_t>(output) * input_dim + input];
    mask[static_cast<size_t>(source_slot)] =
        std::abs(value) < kPlaintextCoefficientFloor ? 0.0 : value;
  }
  return mask;
}

auto slot_bsgs_linear_block0(
    const CryptoContext<DCRTPoly>& cc,
    const Ciphertext<DCRTPoly>& input_ct,
    const std::vector<double>& weights,
    int input_dim,
    int output_dim,
    int baby_step,
    int batch_size,
    int& rotations,
    int& ct_pt_muls,
    int& adds) -> Ciphertext<DCRTPoly> {
  std::map<int, Ciphertext<DCRTPoly>> baby_ct;
  baby_ct[0] = input_ct;
  for (int baby = 1; baby < baby_step; ++baby) {
    baby_ct[baby] = cc->EvalRotate(input_ct, baby);
    ++rotations;
  }

  Ciphertext<DCRTPoly> accumulator;
  bool has_accumulator = false;
  for (const int giant : slot_bsgs_giant_with_zero(input_dim, output_dim, baby_step)) {
    Ciphertext<DCRTPoly> inner;
    bool has_inner = false;
    for (int baby = 0; baby < baby_step; ++baby) {
      const int offset = giant + baby;
      auto mask = slot_bsgs_pre_mask(weights, input_dim, output_dim, batch_size, giant, offset);
      if (std::all_of(mask.begin(), mask.end(), [](double value) { return value == 0.0; })) {
        continue;
      }
      auto plain = cc->MakeCKKSPackedPlaintext(mask);
      plain->SetLength(static_cast<size_t>(batch_size));
      auto term = cc->EvalMult(baby_ct[baby], plain);
      ++ct_pt_muls;
      if (!has_inner) {
        inner = term;
        has_inner = true;
      } else {
        inner = cc->EvalAdd(inner, term);
        ++adds;
      }
    }
    if (!has_inner) {
      continue;
    }
    if (giant != 0) {
      inner = cc->EvalRotate(inner, giant);
      ++rotations;
    }
    if (!has_accumulator) {
      accumulator = inner;
      has_accumulator = true;
    } else {
      accumulator = cc->EvalAdd(accumulator, inner);
      ++adds;
    }
  }
  if (!has_accumulator) {
    throw std::runtime_error("slot BSGS produced no terms");
  }
  return accumulator;
}

void align_levels(
    const CryptoContext<DCRTPoly>& cc,
    Ciphertext<DCRTPoly>& lhs,
    Ciphertext<DCRTPoly>& rhs,
    int& unity_multiplies);

auto evaluate_power_polynomial_block0(
    const CryptoContext<DCRTPoly>& cc,
    const PublicKey<DCRTPoly>& public_key,
    Ciphertext<DCRTPoly> input_ct,
    const std::vector<double>& coefficients,
    const stage1::RankGatePayloadConfig& config,
    int batch_size,
    int& ct_ct_muls,
    int& adds,
    int& unity_multiplies) -> Ciphertext<DCRTPoly> {
  if (coefficients.empty()) {
    throw std::runtime_error("polynomial coefficient vector must not be empty");
  }
  auto make_constant_ct = [&](double coefficient) {
    auto plain = cc->MakeCKKSPackedPlaintext(
        pack_block0(repeated_rank_values(coefficient, config), config));
    plain->SetLength(static_cast<std::size_t>(batch_size));
    return cc->Encrypt(public_key, plain);
  };

  auto result = make_constant_ct(coefficients.back());
  if (coefficients.size() == 1) {
    return result;
  }
  for (auto it = coefficients.rbegin() + 1; it != coefficients.rend(); ++it) {
    result = cc->EvalMult(result, input_ct);
    ++ct_ct_muls;
    const double coefficient = *it;
    if (coefficient != 0.0) {
      auto constant_ct = make_constant_ct(coefficient);
      align_levels(cc, result, constant_ct, unity_multiplies);
      result = cc->EvalAdd(result, constant_ct);
      ++adds;
    }
  }
  return result;
}

void align_levels(
    const CryptoContext<DCRTPoly>& cc,
    Ciphertext<DCRTPoly>& lhs,
    Ciphertext<DCRTPoly>& rhs,
    int& unity_multiplies) {
  for (int guard = 0; guard < 128 && lhs->GetLevel() < rhs->GetLevel(); ++guard) {
    const auto before = lhs->GetLevel();
    cc->EvalMultInPlace(lhs, 1.0);
    ++unity_multiplies;
    if (lhs->GetLevel() == before) {
      break;
    }
  }
  for (int guard = 0; guard < 128 && rhs->GetLevel() < lhs->GetLevel(); ++guard) {
    const auto before = rhs->GetLevel();
    cc->EvalMultInPlace(rhs, 1.0);
    ++unity_multiplies;
    if (rhs->GetLevel() == before) {
      break;
    }
  }
}

auto decrypt_slots(
    const CryptoContext<DCRTPoly>& cc,
    const PrivateKey<DCRTPoly>& secret_key,
    Ciphertext<DCRTPoly> ciphertext,
    size_t length) -> std::vector<double> {
  Plaintext plaintext;
  cc->Decrypt(secret_key, ciphertext, &plaintext);
  plaintext->SetLength(length);
  auto values = plaintext->GetRealPackedValue();
  values.resize(length);
  return values;
}

auto first_values(const std::vector<double>& values, size_t count) -> std::vector<double> {
  return {values.begin(), values.begin() + static_cast<std::ptrdiff_t>(count)};
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
  std::cerr << "[stage1_rank_gate_fideslib] " << message << " rss_gib=" << rss_gib()
            << " peak_rss_gib=" << peak_rss_gib() << std::endl;
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
    const auto args = parse_args(argc, argv);
    const auto payload = stage1::read_rank_gate_payload(args.input);
    const auto reference = stage1::evaluate_rank_gate_payload(payload);
    const auto required_rotations = required_rank_gate_rotations(payload.config);
    const auto batch_size = static_cast<int>(payload.config.rank_pad * payload.config.d_state);
    if (batch_size <= 0 || batch_size > args.ring_dim / 2) {
      throw std::runtime_error("payload batch size does not fit ring dimension");
    }
    log_phase(
        "payload loaded d_model=" + std::to_string(payload.config.d_model) +
        " rank=" + std::to_string(payload.config.mimo_rank) +
        " d_state=" + std::to_string(payload.config.d_state) +
        " rotations=" + std::to_string(required_rotations.size()));

    const auto setup_start = now();
    log_phase("context setup begin");
    CCParams<CryptoContextCKKSRNS> parameters;
    parameters.SetSecretKeyDist(resolve_secret_key_dist(args.secret_key_dist));
    parameters.SetSecurityLevel(resolve_security(args.security));
    parameters.SetRingDim(static_cast<uint32_t>(args.ring_dim));
    parameters.SetScalingTechnique(FLEXIBLEAUTO);
    parameters.SetFirstModSize(static_cast<uint32_t>(args.first_mod_size));
    parameters.SetKeySwitchTechnique(HYBRID);
    parameters.SetMultiplicativeDepth(static_cast<uint32_t>(args.multiplicative_depth));
    parameters.SetScalingModSize(static_cast<uint32_t>(args.scaling_mod_size));
    parameters.SetBatchSize(static_cast<uint32_t>(batch_size));
    parameters.SetDevices({0});
    parameters.SetPlaintextAutoload(false);
    parameters.SetCiphertextAutoload(true);
    if (args.secret_key_dist == "sparse-ternary" ||
        args.secret_key_dist == "sparse-encapsulated") {
      parameters.SetNumLargeDigits(3);
    }

    CryptoContext<DCRTPoly> cc = GenCryptoContext(parameters);
    cc->Enable(PKE);
    cc->Enable(KEYSWITCH);
    cc->Enable(LEVELEDSHE);
    auto keys = cc->KeyGen();
    cc->EvalMultKeyGen(keys.secretKey);
    log_phase("context setup done");
    const auto keygen_start = now();
    log_phase("rotation keygen begin");
    cc->EvalRotateKeyGen(keys.secretKey, required_rotations);
    const double rotate_keygen_seconds = seconds_since(keygen_start);
    log_phase("rotation keygen done");
    const auto load_start = now();
    log_phase("load context begin");
    cc->LoadContext(keys.publicKey);
    const double load_context_seconds = seconds_since(load_start);
    log_phase("load context done");
    const double setup_seconds = seconds_since(setup_start);

    auto make_plain = [&](const std::vector<double>& values) {
      auto plain = cc->MakeCKKSPackedPlaintext(values);
      plain->SetLength(static_cast<size_t>(batch_size));
      return plain;
    };
    auto encrypt_values = [&](const std::vector<double>& values) {
      auto plain = make_plain(values);
      return cc->Encrypt(keys.publicKey, plain);
    };

    const auto eval_start = now();
    log_phase("encrypt/projection eval begin");
    int rotations = 0;
    int ct_pt_muls = 0;
    int ct_ct_muls = 0;
    int unity_multiplies = 0;
    int adds = 0;

    auto rms_ct = encrypt_values(pack_block0(payload.array("rms_input").values, payload.config));
    auto conv_pre_ct = slot_bsgs_linear_block0(
        cc,
        rms_ct,
        scaled_values(payload.array("effective_rank_weight").values, args.rank_projection_scale),
        static_cast<int>(payload.config.d_model),
        static_cast<int>(payload.config.mimo_rank),
        static_cast<int>(payload.config.model_baby_step),
        batch_size,
        rotations,
        ct_pt_muls,
        adds);
    auto conv_bias_ct =
        encrypt_values(pack_block0(
            scaled_values(payload.array("conv_bias").values, args.rank_projection_scale),
            payload.config));
    align_levels(cc, conv_pre_ct, conv_bias_ct, unity_multiplies);
    conv_pre_ct = cc->EvalAdd(conv_pre_ct, conv_bias_ct);
    ++adds;
    log_phase("conv projection done");
    auto conv_pre_for_silu_ct = conv_pre_ct;
    if (args.rank_projection_scale != 1.0) {
      auto unscale_plain = make_plain(
          pack_block0(
              repeated_rank_values(1.0 / args.rank_projection_scale, payload.config),
              payload.config));
      conv_pre_for_silu_ct = cc->EvalMult(conv_pre_ct, unscale_plain);
      ++ct_pt_muls;
    }

    auto gate_pre_ct = slot_bsgs_linear_block0(
        cc,
        rms_ct,
        payload.array("gate_weight").values,
        static_cast<int>(payload.config.d_model),
        static_cast<int>(payload.config.mimo_rank),
        static_cast<int>(payload.config.model_baby_step),
        batch_size,
        rotations,
        ct_pt_muls,
        adds);
    log_phase("gate projection done");
    auto rank_input_poly_ct = evaluate_power_polynomial_block0(
        cc,
        keys.publicKey,
        conv_pre_for_silu_ct,
        payload.array("rank_silu_coefficients").values,
        payload.config,
        batch_size,
        ct_ct_muls,
        adds,
        unity_multiplies);
    log_phase("rank SiLU polynomial done");
    auto gate_poly_ct = evaluate_power_polynomial_block0(
        cc,
        keys.publicKey,
        gate_pre_ct,
        payload.array("gate_silu_coefficients").values,
        payload.config,
        batch_size,
        ct_ct_muls,
        adds,
        unity_multiplies);
    log_phase("gate SiLU polynomial done");
    auto d_skip_plain = make_plain(pack_block0(payload.array("d_skip").values, payload.config));
    auto skip_update_poly_ct = cc->EvalMult(rank_input_poly_ct, d_skip_plain);
    ++ct_pt_muls;
    log_phase("skip update done");
    const double eval_seconds = seconds_since(eval_start);
    log_phase(
        "projection eval done rotations=" + std::to_string(rotations) +
        " ct_pt=" + std::to_string(ct_pt_muls) +
        " ct_ct=" + std::to_string(ct_ct_muls));

    log_phase("decrypt begin");
    const auto conv_pre_slots = decrypt_slots(
        cc,
        keys.secretKey,
        conv_pre_ct,
        static_cast<size_t>(batch_size));
    const auto gate_pre_slots = decrypt_slots(
        cc,
        keys.secretKey,
        gate_pre_ct,
        static_cast<size_t>(batch_size));
    const auto rank_input_poly_slots = decrypt_slots(
        cc,
        keys.secretKey,
        rank_input_poly_ct,
        static_cast<size_t>(batch_size));
    const auto gate_poly_slots = decrypt_slots(
        cc,
        keys.secretKey,
        gate_poly_ct,
        static_cast<size_t>(batch_size));
    const auto skip_update_poly_slots = decrypt_slots(
        cc,
        keys.secretKey,
        skip_update_poly_ct,
        static_cast<size_t>(batch_size));
    log_phase("decrypt done");

    const auto conv_pre = unscaled_values(
        first_values(conv_pre_slots, static_cast<size_t>(payload.config.mimo_rank)),
        args.rank_projection_scale);
    const auto gate_pre =
        first_values(gate_pre_slots, static_cast<size_t>(payload.config.mimo_rank));
    const auto rank_input_poly =
        first_values(rank_input_poly_slots, static_cast<size_t>(payload.config.mimo_rank));
    const auto gate_poly =
        first_values(gate_poly_slots, static_cast<size_t>(payload.config.mimo_rank));
    const auto skip_update_poly =
        first_values(skip_update_poly_slots, static_cast<size_t>(payload.config.mimo_rank));
    const auto conv_error =
        stage1::max_abs_delta(conv_pre, payload.array("reference_conv_pre").values);
    const auto gate_error =
        stage1::max_abs_delta(gate_pre, payload.array("reference_gate_pre").values);
    const auto rank_poly_error =
        stage1::max_abs_delta(rank_input_poly, payload.array("reference_rank_input_poly").values);
    const auto gate_poly_error =
        stage1::max_abs_delta(gate_poly, payload.array("reference_gate_poly").values);
    const auto skip_poly_error =
        stage1::max_abs_delta(skip_update_poly, payload.array("reference_skip_update_poly").values);
    const auto rank_poly_approx_error = stage1::max_abs_delta(
        payload.array("reference_rank_input_poly").values,
        payload.array("reference_rank_input").values);
    const auto gate_poly_approx_error =
        stage1::max_abs_delta(payload.array("reference_gate_poly").values,
                              payload.array("reference_gate").values);
    const auto skip_poly_approx_error = stage1::max_abs_delta(
        payload.array("reference_skip_update_poly").values,
        payload.array("reference_skip_update").values);
    const auto max_error =
        std::max({conv_error, gate_error, rank_poly_error, gate_poly_error, skip_poly_error});
    const bool passed = max_error <= args.atol;
    const auto& polynomial_metadata = payload.array("polynomial_metadata").values;

    std::ostringstream out;
    out << "{";
    out << "\"stage\":\"stage1-rank-gate-fideslib-projection\",";
    out << "\"backend\":\"fideslib-gpu\",";
    out << "\"encrypted\":true,";
    out << "\"status\":\"" << (passed ? "passed" : "failed") << "\",";
    out << "\"passed\":" << (passed ? "true" : "false") << ",";
    out << "\"parameters\":{";
    out << "\"d_model\":" << payload.config.d_model << ",";
    out << "\"d_state\":" << payload.config.d_state << ",";
    out << "\"mimo_rank\":" << payload.config.mimo_rank << ",";
    out << "\"rank_pad\":" << payload.config.rank_pad << ",";
    out << "\"batch_size\":" << batch_size << ",";
    out << "\"ring_dimension\":" << args.ring_dim << ",";
    out << "\"multiplicative_depth\":" << args.multiplicative_depth << ",";
    out << "\"scaling_mod_size\":" << args.scaling_mod_size << ",";
    out << "\"model_baby_step\":" << payload.config.model_baby_step;
    out << ",\"rank_projection_scale\":" << args.rank_projection_scale;
    out << ",\"polynomial_degree\":" << polynomial_metadata[0];
    out << ",\"gate_polynomial_degree\":" << polynomial_metadata[1];
    out << ",\"polynomial_range\":" << polynomial_metadata[2];
    out << "},";
    out << "\"measurements\":{";
    out << "\"max_abs_error\":" << max_error << ",";
    out << "\"conv_pre_max_abs_error\":" << conv_error << ",";
    out << "\"gate_pre_max_abs_error\":" << gate_error << ",";
    out << "\"rank_input_poly_max_abs_error\":" << rank_poly_error << ",";
    out << "\"gate_poly_max_abs_error\":" << gate_poly_error << ",";
    out << "\"skip_update_poly_max_abs_error\":" << skip_poly_error << ",";
    out << "\"rank_input_poly_vs_exact_max_abs_error\":" << rank_poly_approx_error << ",";
    out << "\"gate_poly_vs_exact_max_abs_error\":" << gate_poly_approx_error << ",";
    out << "\"skip_update_poly_vs_exact_max_abs_error\":" << skip_poly_approx_error << ",";
    out << "\"native_plaintext_reference_max_abs_error\":" << reference.max_abs_error << ",";
    out << "\"required_application_rotation_key_count\":" << required_rotations.size() << ",";
    out << "\"plaintext_coefficient_floor\":" << kPlaintextCoefficientFloor << ",";
    out << "\"rank_projection_scaled\":" << (args.rank_projection_scale == 1.0 ? "false" : "true")
        << ",";
    out << "\"peak_rss_gib\":" << peak_rss_gib() << ",";
    out << "\"rss_gib\":" << rss_gib();
    out << "},";
    out << "\"timing\":{";
    out << "\"setup_seconds\":" << setup_seconds << ",";
    out << "\"rotate_keygen_seconds\":" << rotate_keygen_seconds << ",";
    out << "\"load_context_seconds\":" << load_context_seconds << ",";
    out << "\"eval_seconds\":" << eval_seconds;
    out << "},";
    out << "\"operation_counts\":{";
    out << "\"rotations\":" << rotations << ",";
    out << "\"ct_pt_mul\":" << ct_pt_muls << ",";
    out << "\"ct_ct_mul\":" << ct_ct_muls << ",";
    out << "\"adds\":" << adds << ",";
    out << "\"unity_level_align_muls\":" << unity_multiplies << ",";
    out << "\"bootstraps\":0";
    out << "},";
    out << "\"measurement_scope\":{";
    out << "\"rank_gate_payload\":true,";
    out << "\"state_major_layout\":true,";
    out << "\"pre_recurrence_rank_gate_projection\":true,";
    out << "\"rank_projection_scaled\":"
        << (args.rank_projection_scale == 1.0 ? "false" : "true") << ",";
    out << "\"pre_recurrence_silu_activation\":true,";
    out << "\"pre_recurrence_skip_update\":true,";
    out << "\"fideslib_encrypted_execution\":true,";
    out << "\"full_layer_pre_recurrence_computed_in_kernel\":false,";
    out << "\"full_model_correctness_claimed\":false";
    out << "}";
    out << "}";
    write_payload(args.output_json, out.str());
    return passed ? EXIT_SUCCESS : EXIT_FAILURE;
  } catch (const std::exception& exc) {
    std::cerr << "stage1_rank_gate_fideslib failed: " << exc.what() << std::endl;
    return EXIT_FAILURE;
  }
}
