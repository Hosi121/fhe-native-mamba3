#include "stage1_tail_eval.hpp"

#include <fideslib.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
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

struct Config {
  std::string input;
  std::string output_json;
  int ring_dim = 131072;
  int multiplicative_depth = 48;
  int scaling_mod_size = 40;
  int first_mod_size = 60;
  double atol = 1e-6;
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
      config.first_mod_size <= 0 || config.atol < 0.0) {
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

auto state_axis_steps(const stage1::TailPayloadConfig& config, int sign) -> std::vector<int32_t> {
  std::vector<int32_t> rotations;
  for (std::uint32_t step = 1; step < config.d_state; step *= 2) {
    rotations.push_back(static_cast<int32_t>(sign * step * config.rank_pad));
  }
  return rotations;
}

auto required_tail_rotations(const stage1::TailPayloadConfig& config) -> std::vector<int32_t> {
  std::set<int32_t> rotations;
  for (const auto value :
       slot_bsgs_rotations(config.mimo_rank, config.d_model, config.rank_baby_step)) {
    rotations.insert(value);
  }
  for (const auto value : state_axis_steps(config, -1)) {
    rotations.insert(value);
  }
  for (const auto value : state_axis_steps(config, 1)) {
    rotations.insert(value);
  }
  return {rotations.begin(), rotations.end()};
}

auto pack_rank_block0(
    const std::vector<double>& values,
    const stage1::TailPayloadConfig& config) -> std::vector<double> {
  std::vector<double> slots(static_cast<size_t>(config.rank_pad) * config.d_state, 0.0);
  std::copy(values.begin(), values.end(), slots.begin());
  return slots;
}

auto pack_model_block0(
    const std::vector<double>& values,
    const stage1::TailPayloadConfig& config) -> std::vector<double> {
  std::vector<double> slots(static_cast<size_t>(config.rank_pad) * config.d_state, 0.0);
  std::copy(values.begin(), values.end(), slots.begin());
  return slots;
}

auto pack_state_major(
    const std::vector<double>& values,
    const stage1::TailPayloadConfig& config) -> std::vector<double> {
  std::vector<double> slots(static_cast<size_t>(config.rank_pad) * config.d_state, 0.0);
  for (std::uint32_t state_index = 0; state_index < config.d_state; ++state_index) {
    const auto src_base = static_cast<size_t>(state_index) * config.mimo_rank;
    const auto dst_base = static_cast<size_t>(state_index) * config.rank_pad;
    std::copy(
        values.begin() + static_cast<std::ptrdiff_t>(src_base),
        values.begin() + static_cast<std::ptrdiff_t>(src_base + config.mimo_rank),
        slots.begin() + static_cast<std::ptrdiff_t>(dst_base));
  }
  return slots;
}

auto rank_valid_mask(const stage1::TailPayloadConfig& config) -> std::vector<double> {
  std::vector<double> mask(static_cast<size_t>(config.rank_pad) * config.d_state, 0.0);
  std::fill(mask.begin(), mask.begin() + config.mimo_rank, 1.0);
  return mask;
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
    mask[static_cast<size_t>(source_slot)] =
        weights[static_cast<size_t>(output) * input_dim + input];
  }
  return mask;
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

auto slot_bsgs_linear_block0(
    const CryptoContext<DCRTPoly>& cc,
    const Ciphertext<DCRTPoly>& input_ct,
    const std::vector<double>& weights,
    const stage1::TailPayloadConfig& config,
    int& rotations,
    int& ct_pt_muls,
    int& adds) -> Ciphertext<DCRTPoly> {
  std::map<int, Ciphertext<DCRTPoly>> baby_ct;
  baby_ct[0] = input_ct;
  for (int baby = 1; baby < static_cast<int>(config.rank_baby_step); ++baby) {
    baby_ct[baby] = cc->EvalRotate(input_ct, baby);
    ++rotations;
  }

  Ciphertext<DCRTPoly> accumulator;
  bool has_accumulator = false;
  const int batch_size = static_cast<int>(config.rank_pad * config.d_state);
  for (const int giant : slot_bsgs_giant_with_zero(
           static_cast<int>(config.mimo_rank),
           static_cast<int>(config.d_model),
           static_cast<int>(config.rank_baby_step))) {
    Ciphertext<DCRTPoly> inner;
    bool has_inner = false;
    for (int baby = 0; baby < static_cast<int>(config.rank_baby_step); ++baby) {
      const int offset = giant + baby;
      auto mask = slot_bsgs_pre_mask(
          weights,
          static_cast<int>(config.mimo_rank),
          static_cast<int>(config.d_model),
          batch_size,
          giant,
          offset);
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
    const auto payload = stage1::read_tail_payload(args.input);
    const auto reference = stage1::evaluate_tail_payload(payload);
    const auto required_rotations = required_tail_rotations(payload.config);
    const auto batch_size = static_cast<int>(payload.config.rank_pad * payload.config.d_state);
    if (batch_size <= 0 || batch_size > args.ring_dim / 2) {
      throw std::runtime_error("payload batch size does not fit ring dimension");
    }

    const auto setup_start = now();
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
    const auto keygen_start = now();
    cc->EvalRotateKeyGen(keys.secretKey, required_rotations);
    const double rotate_keygen_seconds = seconds_since(keygen_start);
    const auto load_start = now();
    cc->LoadContext(keys.publicKey);
    const double load_context_seconds = seconds_since(load_start);
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
    int rotations = 0;
    int ct_pt_muls = 0;
    int ct_ct_muls = 0;
    int adds = 0;
    int unity_multiplies = 0;

    const auto& arrays = payload.arrays;
    (void)arrays;
    auto rank_ct = encrypt_values(pack_rank_block0(payload.array("rank_input").values, payload.config));
    auto gate_ct = encrypt_values(pack_rank_block0(payload.array("gate").values, payload.config));
    auto skip_ct = encrypt_values(pack_rank_block0(payload.array("skip_update").values, payload.config));
    auto previous_ct =
        encrypt_values(pack_state_major(payload.array("previous_state").values, payload.config));
    auto residual_ct =
        encrypt_values(pack_model_block0(payload.array("residual_input").values, payload.config));

    auto x_state_ct = rank_ct;
    for (const auto step : state_axis_steps(payload.config, -1)) {
      x_state_ct = cc->EvalAdd(x_state_ct, cc->EvalRotate(x_state_ct, step));
      ++rotations;
      ++adds;
    }

    auto decay_plain = make_plain(pack_state_major(payload.array("decay").values, payload.config));
    auto decay_term = cc->EvalMult(previous_ct, decay_plain);
    ++ct_pt_muls;
    auto b_plain = make_plain(pack_state_major(payload.array("b").values, payload.config));
    auto input_term = cc->EvalMult(x_state_ct, b_plain);
    ++ct_pt_muls;
    align_levels(cc, decay_term, input_term, unity_multiplies);
    auto state_new_ct = cc->EvalAdd(decay_term, input_term);
    ++adds;

    auto c_plain = make_plain(pack_state_major(payload.array("c").values, payload.config));
    auto readout_ct = cc->EvalMult(state_new_ct, c_plain);
    ++ct_pt_muls;
    for (const auto step : state_axis_steps(payload.config, 1)) {
      readout_ct = cc->EvalAdd(readout_ct, cc->EvalRotate(readout_ct, step));
      ++rotations;
      ++adds;
    }
    auto rank_mask_plain = make_plain(rank_valid_mask(payload.config));
    readout_ct = cc->EvalMult(readout_ct, rank_mask_plain);
    ++ct_pt_muls;

    align_levels(cc, readout_ct, skip_ct, unity_multiplies);
    auto rank_output_ct = cc->EvalAdd(readout_ct, skip_ct);
    ++adds;
    align_levels(cc, rank_output_ct, gate_ct, unity_multiplies);
    auto rank_payload_ct = cc->EvalMult(rank_output_ct, gate_ct);
    ++ct_ct_muls;

    auto output_delta_ct = slot_bsgs_linear_block0(
        cc,
        rank_payload_ct,
        payload.array("w_out").values,
        payload.config,
        rotations,
        ct_pt_muls,
        adds);
    align_levels(cc, output_delta_ct, residual_ct, unity_multiplies);
    auto output_ct = cc->EvalAdd(output_delta_ct, residual_ct);
    ++adds;

    const double eval_seconds = seconds_since(eval_start);
    const auto output_slots = decrypt_slots(
        cc,
        keys.secretKey,
        output_ct,
        static_cast<size_t>(batch_size));
    const auto output_model =
        first_values(output_slots, static_cast<size_t>(payload.config.d_model));
    const auto output_error =
        stage1::max_abs_delta(output_model, payload.array("reference_output_model").values);
    const bool passed = output_error <= args.atol;

    std::ostringstream out;
    out << "{";
    out << "\"stage\":\"stage1-tail-fideslib-ciphertext\",";
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
    out << "\"rank_baby_step\":" << payload.config.rank_baby_step;
    out << "},";
    out << "\"measurements\":{";
    out << "\"max_abs_error\":" << output_error << ",";
    out << "\"native_plaintext_reference_max_abs_error\":" << reference.max_abs_error << ",";
    out << "\"required_application_rotation_key_count\":" << required_rotations.size() << ",";
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
    out << "\"checkpoint_tail_payload\":true,";
    out << "\"state_major_layout\":true,";
    out << "\"source_boundary_pre_recurrence\":true,";
    out << "\"fideslib_encrypted_execution\":true,";
    out << "\"full_layer_pre_recurrence_computed_in_kernel\":false,";
    out << "\"full_model_correctness_claimed\":false";
    out << "}";
    out << "}";
    write_payload(args.output_json, out.str());
    return passed ? EXIT_SUCCESS : EXIT_FAILURE;
  } catch (const std::exception& exc) {
    std::cerr << "stage1_tail_fideslib failed: " << exc.what() << std::endl;
    return EXIT_FAILURE;
  }
}
