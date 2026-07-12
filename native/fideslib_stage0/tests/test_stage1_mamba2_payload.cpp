#include "stage1_mamba2_payload.hpp"

#include <filesystem>
#include <fstream>
#include <functional>
#include <stdexcept>

namespace fs = std::filesystem;

namespace {

void require_invalid(const std::function<void()>& operation) {
  try {
    operation();
  } catch (const std::runtime_error&) {
    return;
  }
  throw std::runtime_error("expected runtime_error");
}

}  // namespace

auto main() -> int {
  using fhemamba::stage1::M1Payload;
  using fhemamba::stage1::read_chain_payload;
  using fhemamba::stage1::require_same_layer_dims;

  M1Payload expected;
  expected.d_model = 768;
  expected.d_inner = 1536;
  expected.num_heads = 24;
  expected.head_dim = 64;
  expected.state_size = 64;
  expected.n_groups = 1;
  expected.conv_kernel = 4;
  expected.conv_dim = 1664;
  expected.proj_dim = 3352;
  require_same_layer_dims(expected, expected, 0);

  auto mismatch = expected;
  mismatch.state_size = 32;
  require_invalid([&] { require_same_layer_dims(expected, mismatch, 7); });

  const auto root = fs::temp_directory_path() / "fhemamba-stage1-payload-test";
  fs::remove_all(root);
  fs::create_directories(root);
  {
    std::ofstream meta(root / "chain.json");
    meta << R"({"format":"wrong"})";
  }
  require_invalid([&] { read_chain_payload(root.string(), false); });
  {
    std::ofstream meta(root / "chain.json");
    meta << R"({"format":"fhemamba-m2-chain-v1","n_layers":2,)"
            R"("n_test_tokens":1,"final_norm_eps":0.00001,)"
            R"("layer_dirs":["layer_00"],"tensors":{}})";
  }
  require_invalid([&] { read_chain_payload(root.string(), false); });
  fs::remove_all(root);
  return 0;
}
