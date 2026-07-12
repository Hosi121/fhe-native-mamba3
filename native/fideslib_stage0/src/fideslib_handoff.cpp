#include "fideslib_handoff.hpp"

#include <CKKS/Ciphertext.cuh>
#include <CKKS/openfhe-interface/RawCiphertext.cuh>
#include <ciphertext-ser.h>
#include <openfhe.h>
#include <scheme/ckksrns/ckksrns-ser.h>

#include <any>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <utility>
#include <vector>

using namespace fideslib;
namespace fs = std::filesystem;

namespace fhemamba::handoff {

void require_serialized(bool ok, const std::string& message) {
  if (!ok) {
    throw std::runtime_error(message);
  }
}

void serialize_ciphertext(const fs::path& path, const CryptoContext<DCRTPoly>& cc,
                          const PublicKey<DCRTPoly>& public_key,
                          Ciphertext<DCRTPoly>& ciphertext) {
  if (ciphertext->loaded) {
    cc->Synchronize();
    ciphertext->EnsureLazyCPUCopy();
    auto& cpu_context =
        std::any_cast<lbcrypto::CryptoContext<lbcrypto::DCRTPoly>&>(cc->cpu);
    auto& cpu_ciphertext =
        std::any_cast<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>&>(ciphertext->cpu);
    auto gpu_ciphertext = std::static_pointer_cast<FIDESlib::CKKS::Ciphertext>(
        cc->GetDeviceCiphertext(ciphertext->gpu));
    FIDESlib::CKKS::RawCipherText raw;
    gpu_ciphertext->store(raw);
    const auto cpu_towers =
        cpu_ciphertext->GetElements()[0].GetAllElements().size();
    if (cpu_towers < static_cast<std::size_t>(raw.numRes)) {
      std::vector<double> zero(1, 0.0);
      auto plain = cpu_context->MakeCKKSPackedPlaintext(
          zero, 1, cc->multiplicative_depth - gpu_ciphertext->getLevel());
      const auto& public_key_impl =
          std::any_cast<const lbcrypto::PublicKey<lbcrypto::DCRTPoly>&>(
              public_key->pimpl);
      cpu_ciphertext = cpu_context->Encrypt(public_key_impl, plain);
    }
    FIDESlib::CKKS::GetOpenFHECipherText(cpu_ciphertext, raw);
  }
  const auto& cpu_ciphertext =
      std::any_cast<const lbcrypto::Ciphertext<lbcrypto::DCRTPoly>&>(
          ciphertext->cpu);
  require_serialized(
      lbcrypto::Serial::SerializeToFile(path.string(), cpu_ciphertext,
                                        lbcrypto::SerType::BINARY),
      "failed to serialize ciphertext: " + path.string());
}

auto deserialize_ciphertext(const fs::path& path,
                            const CryptoContext<DCRTPoly>& cc)
    -> Ciphertext<DCRTPoly> {
  lbcrypto::Ciphertext<lbcrypto::DCRTPoly> cpu_ciphertext;
  require_serialized(
      lbcrypto::Serial::DeserializeFromFile(path.string(), cpu_ciphertext,
                                            lbcrypto::SerType::BINARY),
      "failed to deserialize ciphertext: " + path.string());
  auto context_copy = cc;
  auto ciphertext =
      std::make_shared<CiphertextImpl<DCRTPoly>>(std::move(context_copy));
  ciphertext->cpu =
      std::make_any<lbcrypto::Ciphertext<lbcrypto::DCRTPoly>>(
          std::move(cpu_ciphertext));
  return ciphertext;
}

void serialize_context(const fs::path& path,
                       const CryptoContext<DCRTPoly>& cc) {
  require_serialized(
      fideslib::Serial::SerializeToFile(path.string(), cc, SerType::BINARY),
      "failed to serialize context: " + path.string());
}

void copy_context_device_metadata(const fs::path& source,
                                  const fs::path& destination) {
  fs::copy_file(source.string() + ".dev", destination.string() + ".dev",
                fs::copy_options::overwrite_existing);
}

}  // namespace fhemamba::handoff
