#pragma once

#include <fideslib.hpp>

#include <filesystem>
#include <string>

namespace fhemamba::handoff {

using namespace fideslib;

void require_serialized(bool ok, const std::string& message);
void serialize_ciphertext(const std::filesystem::path& path,
                          const CryptoContext<DCRTPoly>& cc,
                          const PublicKey<DCRTPoly>& public_key,
                          Ciphertext<DCRTPoly>& ciphertext);
auto deserialize_ciphertext(const std::filesystem::path& path,
                            const CryptoContext<DCRTPoly>& cc)
    -> Ciphertext<DCRTPoly>;
void serialize_context(const std::filesystem::path& path,
                       const CryptoContext<DCRTPoly>& cc);
void copy_context_device_metadata(const std::filesystem::path& source,
                                  const std::filesystem::path& destination);

}  // namespace fhemamba::handoff
