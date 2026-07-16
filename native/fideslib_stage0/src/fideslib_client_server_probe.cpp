// Process-separated CKKS handoff probe: client keygen/encrypt, server evaluate,
// client decrypt. The server artifact directory never contains the secret key.

#include <fideslib.hpp>

#include <CKKS/Ciphertext.cuh>
#include <CKKS/openfhe-interface/RawCiphertext.cuh>
#include <ciphertext-ser.h>
#include <cryptocontext-ser.h>
#include <openfhe.h>
#include <scheme/ckksrns/ckksrns-ser.h>

#include "fideslib_handoff.hpp"

#include <algorithm>
#include <any>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

using namespace fideslib;
namespace fs = std::filesystem;

namespace {

using fhemamba::handoff::copy_context_device_metadata;
using fhemamba::handoff::deserialize_ciphertext;
using fhemamba::handoff::serialize_ciphertext;
using fhemamba::handoff::serialize_context;

constexpr int kRingDim = 4096;
constexpr int kDepth = 8;
constexpr int kSlots = 16;

auto require(bool ok, const std::string& message) -> void {
  if (!ok) {
    throw std::runtime_error(message);
  }
}

auto client_init(const fs::path& root) -> void {
  const auto client = root / "client";
  const auto server = root / "server";
  const auto exchange = root / "exchange";
  fs::create_directories(client);
  fs::create_directories(server);
  fs::create_directories(exchange);

  CCParams<CryptoContextCKKSRNS> params;
  params.SetSecretKeyDist(SPARSE_TERNARY);
  params.SetSecurityLevel(HEStd_NotSet);
  params.SetRingDim(kRingDim);
  params.SetScalingTechnique(FLEXIBLEAUTO);
  params.SetFirstModSize(60);
  params.SetScalingModSize(50);
  params.SetMultiplicativeDepth(kDepth);
  params.SetBatchSize(kSlots);
  params.SetDevices({0});
  params.SetCiphertextAutoload(false);

  auto cc = GenCryptoContext(params);
  cc->Enable(PKE);
  cc->Enable(KEYSWITCH);
  cc->Enable(LEVELEDSHE);
  auto keys = cc->KeyGen();
  cc->EvalMultKeyGen(keys.secretKey);
  cc->EvalRotateKeyGen(keys.secretKey, {1});

  serialize_context(client / "context.bin", cc);
  fs::copy_file(client / "context.bin", server / "context.bin",
                fs::copy_options::overwrite_existing);
  copy_context_device_metadata(client / "context.bin", server / "context.bin");
  require(fideslib::Serial::SerializeToFile((server / "public-key.bin").string(),
                                            keys.publicKey, SerType::BINARY),
          "failed to serialize public key");
  require(fideslib::Serial::SerializeToFile((client / "secret-key.bin").string(),
                                            keys.secretKey, SerType::BINARY),
          "failed to serialize secret key");
  {
    std::ofstream stream(server / "eval-mult.bin", std::ios::binary);
    require(cc->SerializeEvalMultKey(stream, SerType::BINARY),
            "failed to serialize eval-mult key");
  }
  {
    std::ofstream stream(server / "eval-rotation.bin", std::ios::binary);
    require(cc->SerializeEvalAutomorphismKey(stream, SerType::BINARY),
            "failed to serialize rotation key");
  }

  std::vector<double> input(kSlots);
  for (int i = 0; i < kSlots; ++i) {
    input[static_cast<std::size_t>(i)] = 0.05 * (i + 1);
  }
  auto plaintext = cc->MakeCKKSPackedPlaintext(input, 1, 0, nullptr, kSlots);
  auto ciphertext = cc->Encrypt(keys.publicKey, plaintext);
  serialize_ciphertext(exchange / "input.ct", cc, keys.publicKey, ciphertext);
  std::cout << "client-init complete secret_key_location="
            << (client / "secret-key.bin") << '\n';
}

auto server_eval(const fs::path& root) -> void {
  const auto server = root / "server";
  const auto exchange = root / "exchange";
  require(!fs::exists(server / "secret-key.bin"),
          "server artifact directory must not contain a secret key");

  CryptoContext<DCRTPoly> cc;
  PublicKey<DCRTPoly> public_key;
  require(fideslib::Serial::DeserializeFromFile((server / "context.bin").string(), cc,
                                                SerType::BINARY),
          "failed to deserialize context");
  require(fideslib::Serial::DeserializeFromFile((server / "public-key.bin").string(),
                                                public_key, SerType::BINARY),
          "failed to deserialize public key");
  {
    std::ifstream stream(server / "eval-mult.bin", std::ios::binary);
    require(cc->DeserializeEvalMultKey(stream, SerType::BINARY),
            "failed to deserialize eval-mult key");
  }
  {
    std::ifstream stream(server / "eval-rotation.bin", std::ios::binary);
    require(cc->DeserializeEvalAutomorphismKey(stream, SerType::BINARY),
            "failed to deserialize rotation key");
  }
  cc->LoadContext(public_key);

  auto input = deserialize_ciphertext(exchange / "input.ct", cc);
  auto squared = cc->EvalMult(input, input);
  auto rotated = cc->EvalRotate(squared, 1);
  auto output = cc->EvalAdd(squared, rotated);
  serialize_ciphertext(exchange / "output.ct", cc, public_key, output);
  std::cout << "server-eval complete secret_key_loaded=false\n";
}

auto client_decrypt(const fs::path& root) -> void {
  const auto client = root / "client";
  const auto server = root / "server";
  const auto exchange = root / "exchange";
  CryptoContext<DCRTPoly> cc;
  PrivateKey<DCRTPoly> secret_key;
  require(fideslib::Serial::DeserializeFromFile((client / "context.bin").string(), cc,
                                                SerType::BINARY),
          "failed to deserialize client context");
  require(fideslib::Serial::DeserializeFromFile((client / "secret-key.bin").string(),
                                                secret_key, SerType::BINARY),
          "failed to deserialize secret key");
  auto output = deserialize_ciphertext(exchange / "output.ct", cc);
  Plaintext plaintext;
  cc->Decrypt(secret_key, output, &plaintext);
  plaintext->SetLength(kSlots);
  const auto values = plaintext->GetRealPackedValue();
  double max_error = 0.0;
  for (int i = 0; i < kSlots; ++i) {
    const double x = 0.05 * (i + 1);
    const double next = 0.05 * ((i + 1) % kSlots + 1);
    max_error = std::max(max_error,
                         std::abs(values[static_cast<std::size_t>(i)] - x * x - next * next));
  }
  int server_secret_key_files = 0;
  for (const auto& entry : fs::recursive_directory_iterator(server)) {
    if (entry.is_regular_file() &&
        entry.path().filename().string().find("secret") != std::string::npos) {
      ++server_secret_key_files;
    }
  }
  const bool passed = max_error < 1e-5 && server_secret_key_files == 0;
  std::ofstream result(root / "result.json");
  require(result.good(), "failed to open client/server result artifact");
  result << std::setprecision(12)
         << "{\n"
         << "  \"version\": \"0.4.5\",\n"
         << "  \"stage\": \"fideslib-client-server-probe\",\n"
         << "  \"repo_commit\": \"working-tree\",\n"
         << "  \"status\": \"" << (passed ? "passed" : "failed") << "\",\n"
         << "  \"passed\": " << (passed ? "true" : "false") << ",\n"
         << "  \"parameters\": {\"ring_dimension\": " << kRingDim
         << ", \"multiplicative_depth\": " << kDepth
         << ", \"slots\": " << kSlots << ", \"security\": \"not-set\"},\n"
         << "  \"measurements\": {\"max_abs_error\": " << max_error
         << ", \"server_secret_key_files\": " << server_secret_key_files << "},\n"
         << "  \"operation_counts\": {\"rotations\": 1, \"ct_pt_mul\": 0, "
            "\"ct_ct_mul\": 1, \"adds\": 1, \"bootstraps\": 0},\n"
         << "  \"measurement_scope\": {\n"
         << "    \"client_server_process_separation\": true,\n"
         << "    \"server_secret_key_loaded\": false,\n"
         << "    \"full_model_correctness_claimed\": false,\n"
         << "    \"claim\": \"Separate-process CKKS keygen/encrypt, server evaluation, "
            "and client decrypt serialization probe; not the full Mamba pipeline.\"\n"
         << "  }\n"
         << "}\n";
  result.close();
  std::cout << "client-decrypt max_abs_error=" << max_error << '\n';
  require(passed, "client/server ciphertext round trip or key separation check failed");
}

}  // namespace

auto main(int argc, char* argv[]) -> int {
  try {
    if (argc != 3) {
      throw std::invalid_argument(
          "usage: fideslib_client_server_probe client-init|server-eval|client-decrypt DIR");
    }
    const std::string mode = argv[1];
    const fs::path root = argv[2];
    if (mode == "client-init") {
      client_init(root);
    } else if (mode == "server-eval") {
      server_eval(root);
    } else if (mode == "client-decrypt") {
      client_decrypt(root);
    } else {
      throw std::invalid_argument("unknown mode: " + mode);
    }
    return 0;
  } catch (const std::exception& exc) {
    std::cerr << "fideslib_client_server_probe: " << exc.what() << '\n';
    return 1;
  }
}
