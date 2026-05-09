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
  std::string readout_mode = "rank-reduce";
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
    } else if (arg == "--readout-mode") {
      config.readout_mode = value;
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
  if (config.readout_mode != "none" && config.readout_mode != "rank-reduce") {
    throw std::invalid_argument("readout-mode must be none or rank-reduce");
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

auto make_c_vector(int slots) -> std::vector<double> {
  std::vector<double> values(slots);
  for (int i = 0; i < slots; ++i) {
    values[i] = 0.13 + 0.005 * static_cast<double>((3 * i + 1) % 13);
  }
  return values;
}

auto make_readout_rotations(int d_state, int mimo_rank) -> std::vector<int32_t> {
  std::vector<int32_t> rotations;
  for (int step = 1; step < d_state; step *= 2) {
    rotations.push_back(static_cast<int32_t>(step));
  }
  for (int rank = 1; rank < mimo_rank; ++rank) {
    const int shift = rank * d_state - rank;
    if (shift != 0) {
      rotations.push_back(static_cast<int32_t>(shift));
    }
  }
  std::sort(rotations.begin(), rotations.end());
  rotations.erase(std::unique(rotations.begin(), rotations.end()), rotations.end());
  return rotations;
}

void print_vector(std::ostream& out, const std::vector<double>& values, int length) {
  out << "[";
  for (int i = 0; i < length; ++i) {
    if (i > 0) {
      out << ",";
    }
    const auto value = values.at(static_cast<size_t>(i));
    if (std::isfinite(value)) {
      out << value;
    } else {
      out << "null";
    }
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

auto make_reduce_mask(int d_state, int mimo_rank, int step) -> std::vector<double> {
  const int slots = d_state * mimo_rank;
  std::vector<double> mask(static_cast<size_t>(slots), 0.0);
  for (int rank = 0; rank < mimo_rank; ++rank) {
    for (int n = 0; n < d_state; ++n) {
      if (n + step < d_state && n % (2 * step) == 0) {
        mask[static_cast<size_t>(rank * d_state + n)] = 1.0;
      }
    }
  }
  return mask;
}

auto make_scatter_mask(int d_state, int mimo_rank, int rank) -> std::vector<double> {
  const int slots = d_state * mimo_rank;
  std::vector<double> mask(static_cast<size_t>(slots), 0.0);
  mask[static_cast<size_t>(rank * d_state)] = 1.0;
  return mask;
}

auto rank_reduce_readout(
    const CryptoContext<DCRTPoly>& cc,
    const Ciphertext<DCRTPoly>& h_cipher,
    Plaintext& c_plain,
    const std::vector<int>& reduce_steps,
    std::vector<Plaintext>& reduce_masks,
    const std::vector<int>& scatter_shifts,
    std::vector<Plaintext>& scatter_masks,
    int d_state,
    int mimo_rank,
    int& readout_ct_pt_multiplies,
    int& readout_adds,
    int& readout_rotations) -> Ciphertext<DCRTPoly> {
  auto reduced = cc->EvalMult(h_cipher, c_plain);
  ++readout_ct_pt_multiplies;

  for (size_t mask_index = 0; mask_index < reduce_steps.size(); ++mask_index) {
    const int step = reduce_steps[mask_index];
    auto rotated = cc->EvalRotate(reduced, step);
    ++readout_rotations;
    auto masked = cc->EvalMult(rotated, reduce_masks[mask_index]);
    ++readout_ct_pt_multiplies;
    reduced = cc->EvalAdd(reduced, masked);
    ++readout_adds;
  }

  Ciphertext<DCRTPoly> output;
  for (int rank = 0; rank < mimo_rank; ++rank) {
    auto term = cc->EvalMult(reduced, scatter_masks[static_cast<size_t>(rank)]);
    ++readout_ct_pt_multiplies;
    const int shift = scatter_shifts[static_cast<size_t>(rank)];
    if (shift != 0) {
      term = cc->EvalRotate(term, shift);
      ++readout_rotations;
    }
    if (rank == 0) {
      output = term;
    } else {
      output = cc->EvalAdd(output, term);
      ++readout_adds;
    }
  }
  return output;
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
    const auto readout_rotation_keys =
        config.readout_mode == "rank-reduce"
            ? make_readout_rotations(config.d_state, config.mimo_rank)
            : std::vector<int32_t>{};
    if (!readout_rotation_keys.empty()) {
      cc->EvalRotateKeyGen(keys.secretKey, readout_rotation_keys);
    }
    cc->LoadContext(keys.publicKey);

    auto h_expected = make_initial_state(state_slots);
    const auto c_vector = make_c_vector(state_slots);
    auto h_plain = cc->MakeCKKSPackedPlaintext(h_expected);
    h_plain->SetLength(static_cast<size_t>(state_slots));
    auto h_cipher = cc->Encrypt(keys.publicKey, h_plain);
    auto c_plain = cc->MakeCKKSPackedPlaintext(c_vector);
    c_plain->SetLength(static_cast<size_t>(state_slots));
    std::vector<int> readout_reduce_steps;
    std::vector<Plaintext> readout_reduce_masks;
    std::vector<int> readout_scatter_shifts;
    std::vector<Plaintext> readout_scatter_masks;
    if (config.readout_mode == "rank-reduce") {
      for (int step = 1; step < config.d_state; step *= 2) {
        readout_reduce_steps.push_back(step);
        auto mask_plain = cc->MakeCKKSPackedPlaintext(
            make_reduce_mask(config.d_state, config.mimo_rank, step));
        mask_plain->SetLength(static_cast<size_t>(state_slots));
        readout_reduce_masks.push_back(mask_plain);
      }
      for (int rank = 0; rank < config.mimo_rank; ++rank) {
        readout_scatter_shifts.push_back(rank * config.d_state - rank);
        auto mask_plain = cc->MakeCKKSPackedPlaintext(
            make_scatter_mask(config.d_state, config.mimo_rank, rank));
        mask_plain->SetLength(static_cast<size_t>(state_slots));
        readout_scatter_masks.push_back(mask_plain);
      }
    }
    std::vector<double> expected_output(static_cast<size_t>(config.mimo_rank), 0.0);

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
    int readout_ct_pt_multiplies = 0;
    int readout_adds = 0;
    int readout_rotation_count = 0;
    Ciphertext<DCRTPoly> output_cipher;
    bool has_output_cipher = false;

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

      if (config.readout_mode == "rank-reduce") {
        output_cipher = rank_reduce_readout(
            cc,
            h_cipher,
            c_plain,
            readout_reduce_steps,
            readout_reduce_masks,
            readout_scatter_shifts,
            readout_scatter_masks,
            config.d_state,
            config.mimo_rank,
            readout_ct_pt_multiplies,
            readout_adds,
            readout_rotation_count);
        has_output_cipher = true;
      }
    }

    const double eval_seconds = seconds_since(eval_start);

    Plaintext decrypted;
    cc->Decrypt(keys.secretKey, h_cipher, &decrypted);
    decrypted->SetLength(static_cast<size_t>(state_slots));
    auto h_decrypted = decrypted->GetRealPackedValue();
    h_decrypted.resize(static_cast<size_t>(state_slots));

    std::vector<double> output_decrypted(static_cast<size_t>(config.mimo_rank), 0.0);
    double output_max_abs_error = 0.0;
    bool output_has_nonfinite = false;
    if (has_output_cipher) {
      Plaintext output_plain;
      cc->Decrypt(keys.secretKey, output_cipher, &output_plain);
      output_plain->SetLength(static_cast<size_t>(state_slots));
      auto output_slots = output_plain->GetRealPackedValue();
      for (int rank = 0; rank < config.mimo_rank; ++rank) {
        double expected = 0.0;
        for (int n = 0; n < config.d_state; ++n) {
          const int slot = rank * config.d_state + n;
          expected += c_vector[static_cast<size_t>(slot)] * h_expected[static_cast<size_t>(slot)];
        }
        expected_output[static_cast<size_t>(rank)] = expected;
        output_decrypted[static_cast<size_t>(rank)] = output_slots[static_cast<size_t>(rank)];
        if (!std::isfinite(output_decrypted[static_cast<size_t>(rank)])) {
          output_has_nonfinite = true;
          output_max_abs_error = 1e300;
        } else {
          output_max_abs_error = std::max(
              output_max_abs_error,
              std::abs(output_decrypted[static_cast<size_t>(rank)] - expected));
        }
      }
    }

    double state_max_abs_error = 0.0;
    bool state_has_nonfinite = false;
    for (int i = 0; i < state_slots; ++i) {
      if (!std::isfinite(h_decrypted[static_cast<size_t>(i)])) {
        state_has_nonfinite = true;
        state_max_abs_error = 1e300;
      } else {
        state_max_abs_error = std::max(
            state_max_abs_error,
            std::abs(h_decrypted[static_cast<size_t>(i)] - h_expected[static_cast<size_t>(i)]));
      }
    }
    const double max_abs_error = std::max(state_max_abs_error, output_max_abs_error);

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
    std::cout << "\"update_level_offset\":" << config.update_level_offset << ",";
    std::cout << "\"readout_mode\":\"" << config.readout_mode << "\"";
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
    std::cout << "\"ct_pt_mul\":"
              << (config.seq_len + server_input_multiplies + unity_multiplies +
                  readout_ct_pt_multiplies)
              << ",";
    std::cout << "\"add\":" << (config.seq_len + readout_adds) << ",";
    std::cout << "\"rotations\":" << readout_rotation_count << ",";
    std::cout << "\"bootstraps\":0,";
    std::cout << "\"encrypt\":" << (config.seq_len + 1) << ",";
    std::cout << "\"decrypt\":" << (has_output_cipher ? 2 : 1) << ",";
    std::cout << "\"level_alignment_unity_multiplies\":" << unity_multiplies << ",";
    std::cout << "\"readout_ct_pt_multiplies\":" << readout_ct_pt_multiplies << ",";
    std::cout << "\"readout_rotations\":" << readout_rotation_count << ",";
    std::cout << "\"client_plaintext_public_weight_multiplies\":"
              << client_plaintext_public_weight_multiplies;
    std::cout << "},";
    std::cout << "\"max_abs_error\":" << max_abs_error << ",";
    std::cout << "\"state_max_abs_error\":" << state_max_abs_error << ",";
    std::cout << "\"output_max_abs_error\":" << output_max_abs_error << ",";
    std::cout << "\"state_has_nonfinite\":" << (state_has_nonfinite ? "true" : "false") << ",";
    std::cout << "\"output_has_nonfinite\":" << (output_has_nonfinite ? "true" : "false") << ",";
    std::cout << "\"expected_final_state\":";
    print_vector(std::cout, h_expected, state_slots);
    std::cout << ",\"decrypted_final_state\":";
    print_vector(std::cout, h_decrypted, state_slots);
    std::cout << ",\"expected_final_output\":";
    print_vector(std::cout, expected_output, config.mimo_rank);
    std::cout << ",\"decrypted_final_output\":";
    print_vector(std::cout, output_decrypted, config.mimo_rank);
    std::cout << ",\"readout_rotation_keys\":";
    std::cout << "[";
    for (size_t i = 0; i < readout_rotation_keys.size(); ++i) {
      if (i > 0) {
        std::cout << ",";
      }
      std::cout << readout_rotation_keys[i];
    }
    std::cout << "]";
    if (config.readout_mode == "rank-reduce") {
      std::cout << ",\"next_bottleneck\":\"rank-reduce readout rotations before SSD scan\"";
    } else {
      std::cout << ",\"next_bottleneck\":\"ct-pt decay multiplies before packed readout/SSD scan\"";
    }
    std::cout << "}" << std::endl;
  } catch (const std::exception& exc) {
    std::cerr << "stage0_static_mimo failed: " << exc.what() << std::endl;
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}
