#include <fideslib.hpp>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

using namespace fideslib;

namespace {

struct Config {
  int seq_len = 8;
  int d_state = 4;
  int mimo_rank = 2;
  int multiplicative_depth = 24;
  int scaling_mod_size = 59;
  int first_mod_size = 60;
  int ring_dim = 4096;
  std::string input_mode = "client-update";
  int update_level_offset = 0;
};

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
    if (i + 1 >= argc) {
      throw std::invalid_argument(std::string("missing value for ") + std::string(arg));
    }
    const char* value = argv[++i];
    if (arg == "--seq-len") {
      config.seq_len = parse_int(arg, value);
    } else if (arg == "--d-state") {
      config.d_state = parse_int(arg, value);
    } else if (arg == "--mimo-rank") {
      config.mimo_rank = parse_int(arg, value);
    } else if (arg == "--multiplicative-depth") {
      config.multiplicative_depth = parse_int(arg, value);
    } else if (arg == "--scaling-mod-size") {
      config.scaling_mod_size = parse_int(arg, value);
    } else if (arg == "--first-mod-size") {
      config.first_mod_size = parse_int(arg, value);
    } else if (arg == "--ring-dim") {
      config.ring_dim = parse_int(arg, value);
    } else if (arg == "--input-mode") {
      config.input_mode = value;
    } else if (arg == "--update-level-offset") {
      config.update_level_offset = parse_int(arg, value);
    } else {
      throw std::invalid_argument(std::string("unknown argument: ") + std::string(arg));
    }
  }
  if (config.seq_len <= 0 || config.d_state <= 0 || config.mimo_rank <= 0) {
    throw std::invalid_argument("seq-len, d-state, and mimo-rank must be positive");
  }
  if (config.input_mode != "server-bx" && config.input_mode != "client-update") {
    throw std::invalid_argument("input-mode must be server-bx or client-update");
  }
  return config;
}

auto now() -> std::chrono::steady_clock::time_point { return std::chrono::steady_clock::now(); }

auto seconds_since(std::chrono::steady_clock::time_point start) -> double {
  return std::chrono::duration<double>(now() - start).count();
}

auto make_initial_state(int slots) -> std::vector<double> {
  std::vector<double> values(slots);
  for (int i = 0; i < slots; ++i) {
    values[i] = 0.05 * static_cast<double>(i + 1);
  }
  return values;
}

auto make_input(int t, int slots) -> std::vector<double> {
  std::vector<double> values(slots);
  for (int i = 0; i < slots; ++i) {
    values[i] = 0.01 * static_cast<double>(t + 1) * static_cast<double>((i % 7) + 1);
  }
  return values;
}

auto make_b_vector(int t, int slots) -> std::vector<double> {
  std::vector<double> values(slots);
  for (int i = 0; i < slots; ++i) {
    values[i] = 0.20 + 0.01 * static_cast<double>((t + i) % 11);
  }
  return values;
}

auto make_decay(int t) -> double {
  return 0.91 - 0.01 * static_cast<double>(t % 3);
}

void print_vector(std::ostream& out, const std::vector<double>& values, int length) {
  out << "[";
  for (int i = 0; i < length; ++i) {
    if (i > 0) {
      out << ",";
    }
    out << values.at(static_cast<size_t>(i));
  }
  out << "]";
}

void align_levels(
    const CryptoContext<DCRTPoly>& cc,
    Ciphertext<DCRTPoly>& lhs,
    Ciphertext<DCRTPoly>& rhs,
    int& unity_multiplies) {
  for (int guard = 0; guard < 64 && lhs->GetLevel() < rhs->GetLevel(); ++guard) {
    const auto before = lhs->GetLevel();
    cc->EvalMultInPlace(lhs, 1.0);
    ++unity_multiplies;
    if (lhs->GetLevel() == before) {
      break;
    }
  }
  for (int guard = 0; guard < 64 && rhs->GetLevel() < lhs->GetLevel(); ++guard) {
    const auto before = rhs->GetLevel();
    cc->EvalMultInPlace(rhs, 1.0);
    ++unity_multiplies;
    if (rhs->GetLevel() == before) {
      break;
    }
  }
}

}  // namespace

auto main(int argc, char* argv[]) -> int {
  try {
    const auto config = parse_args(argc, argv);
    const int state_slots = config.d_state * config.mimo_rank;
    const auto setup_start = now();

    CCParams<CryptoContextCKKSRNS> parameters;
    parameters.SetSecretKeyDist(UNIFORM_TERNARY);
    parameters.SetSecurityLevel(HEStd_NotSet);
    parameters.SetRingDim(static_cast<uint32_t>(config.ring_dim));
    parameters.SetScalingTechnique(FLEXIBLEAUTO);
    parameters.SetFirstModSize(static_cast<uint32_t>(config.first_mod_size));
    parameters.SetKeySwitchTechnique(HYBRID);
    parameters.SetMultiplicativeDepth(static_cast<uint32_t>(config.multiplicative_depth));
    parameters.SetScalingModSize(static_cast<uint32_t>(config.scaling_mod_size));
    parameters.SetBatchSize(static_cast<uint32_t>(state_slots));
    parameters.SetDevices({0});
    parameters.SetPlaintextAutoload(false);
    parameters.SetCiphertextAutoload(true);

    CryptoContext<DCRTPoly> cc = GenCryptoContext(parameters);
    cc->Enable(PKE);
    cc->Enable(KEYSWITCH);
    cc->Enable(LEVELEDSHE);

    auto keys = cc->KeyGen();
    cc->EvalMultKeyGen(keys.secretKey);
    cc->LoadContext(keys.publicKey);

    auto h_expected = make_initial_state(state_slots);
    auto h_plain = cc->MakeCKKSPackedPlaintext(h_expected);
    h_plain->SetLength(static_cast<size_t>(state_slots));
    auto h_cipher = cc->Encrypt(keys.publicKey, h_plain);

    std::vector<Ciphertext<DCRTPoly>> encrypted_inputs;
    std::vector<Ciphertext<DCRTPoly>> encrypted_updates;
    std::vector<Plaintext> b_plaintexts;
    std::vector<double> decays;
    encrypted_inputs.reserve(static_cast<size_t>(config.seq_len));
    encrypted_updates.reserve(static_cast<size_t>(config.seq_len));
    b_plaintexts.reserve(static_cast<size_t>(config.seq_len));
    decays.reserve(static_cast<size_t>(config.seq_len));
    int client_plaintext_public_weight_multiplies = 0;

    for (int t = 0; t < config.seq_len; ++t) {
      auto x = make_input(t, state_slots);
      auto b = make_b_vector(t, state_slots);
      std::vector<double> update_values(static_cast<size_t>(state_slots));
      decays.push_back(make_decay(t));

      for (int i = 0; i < state_slots; ++i) {
        update_values[static_cast<size_t>(i)] =
            b[static_cast<size_t>(i)] * x[static_cast<size_t>(i)];
        h_expected[static_cast<size_t>(i)] =
            decays.back() * h_expected[static_cast<size_t>(i)] + update_values[static_cast<size_t>(i)];
      }

      if (config.input_mode == "server-bx") {
        auto x_plain = cc->MakeCKKSPackedPlaintext(x);
        x_plain->SetLength(static_cast<size_t>(state_slots));
        auto b_plain = cc->MakeCKKSPackedPlaintext(b);
        b_plain->SetLength(static_cast<size_t>(state_slots));
        encrypted_inputs.push_back(cc->Encrypt(keys.publicKey, x_plain));
        b_plaintexts.push_back(b_plain);
      } else {
        const auto target_level_int = t + config.update_level_offset;
        if (target_level_int < 0) {
          throw std::invalid_argument("update-level-offset produces a negative target level");
        }
        auto target_level = static_cast<uint32_t>(target_level_int);
        auto update_plain =
            cc->MakeCKKSPackedPlaintext(update_values, 1, target_level, nullptr, state_slots);
        update_plain->SetLength(static_cast<size_t>(state_slots));
        encrypted_updates.push_back(cc->Encrypt(keys.publicKey, update_plain));
        client_plaintext_public_weight_multiplies += state_slots;
      }
    }

    const double setup_seconds = seconds_since(setup_start);
    const auto eval_start = now();
    int unity_multiplies = 0;

    for (int t = 0; t < config.seq_len; ++t) {
      auto decayed = cc->EvalMult(h_cipher, decays[static_cast<size_t>(t)]);
      Ciphertext<DCRTPoly> update;
      if (config.input_mode == "server-bx") {
        update =
            cc->EvalMult(encrypted_inputs[static_cast<size_t>(t)], b_plaintexts[static_cast<size_t>(t)]);
      } else {
        update = encrypted_updates[static_cast<size_t>(t)];
      }
      align_levels(cc, decayed, update, unity_multiplies);
      h_cipher = cc->EvalAdd(decayed, update);
    }

    const double eval_seconds = seconds_since(eval_start);

    Plaintext decrypted;
    cc->Decrypt(keys.secretKey, h_cipher, &decrypted);
    decrypted->SetLength(static_cast<size_t>(state_slots));
    auto h_decrypted = decrypted->GetRealPackedValue();
    h_decrypted.resize(static_cast<size_t>(state_slots));

    double max_abs_error = 0.0;
    for (int i = 0; i < state_slots; ++i) {
      max_abs_error =
          std::max(max_abs_error, std::abs(h_decrypted[static_cast<size_t>(i)] - h_expected[static_cast<size_t>(i)]));
    }

    std::cout << "{";
    std::cout << "\"backend\":\"fideslib-gpu\",";
    std::cout << "\"encrypted\":true,";
    std::cout << "\"stage\":\"0\",";
    std::cout << "\"name\":\"fideslib-static-mimo-recurrence\",";
    std::cout << "\"model\":{";
    std::cout << "\"seq_len\":" << config.seq_len << ",";
    std::cout << "\"d_state\":" << config.d_state << ",";
    std::cout << "\"mimo_rank\":" << config.mimo_rank << ",";
    std::cout << "\"state_slots\":" << state_slots << ",";
    std::cout << "\"input_mode\":\"" << config.input_mode << "\",";
    std::cout << "\"update_level_offset\":" << config.update_level_offset;
    std::cout << "},";
    std::cout << "\"ckks\":{";
    std::cout << "\"multiplicative_depth\":" << config.multiplicative_depth << ",";
    std::cout << "\"scaling_mod_size\":" << config.scaling_mod_size << ",";
    std::cout << "\"first_mod_size\":" << config.first_mod_size << ",";
    std::cout << "\"ring_dimension\":" << cc->GetRingDimension() << ",";
    std::cout << "\"batch_size\":" << state_slots << ",";
    std::cout << "\"final_level\":" << h_cipher->GetLevel();
    std::cout << "},";
    std::cout << "\"timing\":{";
    std::cout << "\"setup_seconds\":" << setup_seconds << ",";
    std::cout << "\"eval_seconds\":" << eval_seconds << ",";
    std::cout << "\"latency_sec_per_token\":" << eval_seconds / static_cast<double>(config.seq_len);
    std::cout << "},";
    std::cout << "\"operation_counts\":{";
    std::cout << "\"ct_ct_mul\":0,";
    const auto server_input_multiplies = config.input_mode == "server-bx" ? config.seq_len : 0;
    std::cout << "\"ct_pt_mul\":" << (config.seq_len + server_input_multiplies + unity_multiplies)
              << ",";
    std::cout << "\"add\":" << config.seq_len << ",";
    std::cout << "\"rotations\":0,";
    std::cout << "\"bootstraps\":0,";
    std::cout << "\"encrypt\":" << (config.seq_len + 1) << ",";
    std::cout << "\"decrypt\":1,";
    std::cout << "\"level_alignment_unity_multiplies\":" << unity_multiplies << ",";
    std::cout << "\"client_plaintext_public_weight_multiplies\":"
              << client_plaintext_public_weight_multiplies;
    std::cout << "},";
    std::cout << "\"max_abs_error\":" << max_abs_error << ",";
    std::cout << "\"expected_final_state\":";
    print_vector(std::cout, h_expected, state_slots);
    std::cout << ",\"decrypted_final_state\":";
    print_vector(std::cout, h_decrypted, state_slots);
    std::cout << ",\"next_bottleneck\":\"ct-pt decay multiplies before packed readout/SSD scan\"";
    std::cout << "}" << std::endl;
  } catch (const std::exception& exc) {
    std::cerr << "stage0_static_mimo failed: " << exc.what() << std::endl;
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
