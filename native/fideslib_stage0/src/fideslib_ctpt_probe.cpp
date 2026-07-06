// Standalone FIDESlib GPU micro-op probe: mean latency of ct-pt multiply,
// ct-ct multiply (with relinearization), rotation, and addition at a chosen
// ring dimension. Emits one JSON object with the measured means in ms.
//
// Context setup mirrors stage1_rank_gate_fideslib.cpp (ring/depth/scale/keys/
// LoadContext ordering).

#include <fideslib.hpp>

#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

using namespace fideslib;

// GPU work is enqueued asynchronously; without a device sync the timing loop
// only measures launch overhead. cudart is already on the link line via
// fideslib; declare the one symbol we need instead of pulling CUDA headers.
extern "C" int cudaDeviceSynchronize(void);

namespace {

struct Config {
  std::string output_json;
  int ring_dim = 65536;
  int multiplicative_depth = 20;
  int scaling_mod_size = 40;
  int first_mod_size = 60;
  int iterations = 200;
  int warmup = 5;
  std::string security = "not-set";
  std::string secret_key_dist = "sparse-ternary";
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
    if (arg == "--output-json") {
      config.output_json = value;
    } else if (arg == "--ring-dim") {
      config.ring_dim = parse_int(arg, value);
    } else if (arg == "--depth" || arg == "--multiplicative-depth") {
      config.multiplicative_depth = parse_int(arg, value);
    } else if (arg == "--scaling-mod-size") {
      config.scaling_mod_size = parse_int(arg, value);
    } else if (arg == "--first-mod-size") {
      config.first_mod_size = parse_int(arg, value);
    } else if (arg == "--iterations") {
      config.iterations = parse_int(arg, value);
    } else if (arg == "--warmup") {
      config.warmup = parse_int(arg, value);
    } else if (arg == "--security") {
      config.security = value;
    } else if (arg == "--secret-key-dist") {
      config.secret_key_dist = value;
    } else {
      throw std::invalid_argument(std::string("unknown argument: ") + std::string(arg));
    }
  }
  if (config.ring_dim <= 0 || (config.ring_dim & (config.ring_dim - 1)) != 0) {
    throw std::invalid_argument("ring-dim must be a positive power of two");
  }
  if (config.multiplicative_depth <= 0 || config.scaling_mod_size <= 0 ||
      config.first_mod_size <= 0 || config.iterations <= 0 || config.warmup < 0) {
    throw std::invalid_argument("invalid probe parameters");
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

auto now() -> std::chrono::steady_clock::time_point { return std::chrono::steady_clock::now(); }

auto seconds_since(std::chrono::steady_clock::time_point start) -> double {
  return std::chrono::duration<double>(now() - start).count();
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
  std::cout << payload << std::endl;
}

}  // namespace

auto main(int argc, char* argv[]) -> int {
  try {
    const auto args = parse_args(argc, argv);
    const int batch_size = args.ring_dim / 2;

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
    cc->EvalRotateKeyGen(keys.secretKey, {1});
    cc->LoadContext(keys.publicKey);

    std::vector<double> values_a(static_cast<std::size_t>(batch_size), 0.0);
    std::vector<double> values_b(static_cast<std::size_t>(batch_size), 0.0);
    for (int slot = 0; slot < batch_size; ++slot) {
      values_a[static_cast<std::size_t>(slot)] = 0.001 * ((slot % 1024) - 512);
      values_b[static_cast<std::size_t>(slot)] = 0.0005 * ((slot % 512) - 256);
    }
    auto plain_a = cc->MakeCKKSPackedPlaintext(values_a);
    plain_a->SetLength(static_cast<std::size_t>(batch_size));
    auto plain_b = cc->MakeCKKSPackedPlaintext(values_b);
    plain_b->SetLength(static_cast<std::size_t>(batch_size));
    auto ct_a = cc->Encrypt(keys.publicKey, plain_a);
    auto ct_b = cc->Encrypt(keys.publicKey, plain_b);

    // Every measured op reads the fresh inputs and discards the result, so the
    // operand level stays constant across iterations.
    // Amortized cost: sync once before starting the clock and once after the
    // batch, so queued launches pipeline as they would inside the kernel.
    auto measure_ms = [&](auto&& work) -> double {
      for (int i = 0; i < args.warmup; ++i) {
        work();
      }
      cudaDeviceSynchronize();
      const auto start = now();
      for (int i = 0; i < args.iterations; ++i) {
        work();
      }
      cudaDeviceSynchronize();
      return seconds_since(start) * 1000.0 / args.iterations;
    };

    std::cerr << "[fideslib_ctpt_probe] ring_dim=" << args.ring_dim
              << " depth=" << args.multiplicative_depth
              << " iterations=" << args.iterations << std::endl;
    const double ct_pt_mul_ms = measure_ms([&]() { auto result = cc->EvalMult(ct_a, plain_b); });
    std::cerr << "[fideslib_ctpt_probe] ct_pt_mul_ms=" << ct_pt_mul_ms << std::endl;
    const double ct_ct_mul_ms = measure_ms([&]() { auto result = cc->EvalMult(ct_a, ct_b); });
    std::cerr << "[fideslib_ctpt_probe] ct_ct_mul_ms=" << ct_ct_mul_ms << std::endl;
    const double rotation_ms = measure_ms([&]() { auto result = cc->EvalRotate(ct_a, 1); });
    std::cerr << "[fideslib_ctpt_probe] rotation_ms=" << rotation_ms << std::endl;
    const double add_ms = measure_ms([&]() { auto result = cc->EvalAdd(ct_a, ct_b); });
    std::cerr << "[fideslib_ctpt_probe] add_ms=" << add_ms << std::endl;

    std::ostringstream out;
    out << "{";
    out << "\"ring_dim\":" << args.ring_dim << ",";
    out << "\"ct_pt_mul_ms\":" << ct_pt_mul_ms << ",";
    out << "\"ct_ct_mul_ms\":" << ct_ct_mul_ms << ",";
    out << "\"rotation_ms\":" << rotation_ms << ",";
    out << "\"add_ms\":" << add_ms << ",";
    out << "\"batch_size\":" << batch_size << ",";
    out << "\"multiplicative_depth\":" << args.multiplicative_depth << ",";
    out << "\"scaling_mod_size\":" << args.scaling_mod_size << ",";
    out << "\"first_mod_size\":" << args.first_mod_size << ",";
    out << "\"iterations\":" << args.iterations << ",";
    out << "\"security\":\"" << args.security << "\",";
    out << "\"secret_key_dist\":\"" << args.secret_key_dist << "\"";
    out << "}";
    write_payload(args.output_json, out.str());
    return EXIT_SUCCESS;
  } catch (const std::exception& exc) {
    std::cerr << "fideslib_ctpt_probe failed: " << exc.what() << std::endl;
    return EXIT_FAILURE;
  }
}
