#pragma once

#include <cstddef>
#include <map>
#include <string>
#include <vector>

namespace fhemamba::stage1 {

struct PolySpec {
  std::string kind;
  std::vector<double> coeffs;
  double lo = 0.0;
  double hi = 0.0;
  int squarings = 0;
  int iterations = 0;
  double damping = 1.0;
  double guess = 0.0;
};


struct M1Payload {
  int d_model = 0;
  int d_inner = 0;
  int num_heads = 0;
  int head_dim = 0;
  int state_size = 0;
  int n_groups = 0;
  int conv_kernel = 0;
  int conv_dim = 0;
  int proj_dim = 0;
  int n_test_tokens = 0;
  double eps_block = 0.0;
  double eps_gated = 0.0;
  // Calibration-only carried-lineage magnitude maxima. < 0 means absent or
  // untrusted (older exports measured the evaluation prompt itself).
  double state_abs_max = -1.0;
  std::vector<double> state_head_abs_max;
  double fifo_abs_max = -1.0;
  std::map<std::string, double> checkpoint_abs_max;
  std::map<std::string, PolySpec> polys;
  std::map<std::string, std::vector<double>> tensors;
  std::map<std::string, std::vector<int>> shapes;
};


struct ChainPayload {
  int n_layers = 0;
  int n_test_tokens = 0;
  double final_norm_eps = 0.0;
  std::vector<std::string> layer_dirs;
  std::vector<double> final_norm_w;
  std::vector<double> input_embeddings;  // (tokens, d_model) row-major
  std::vector<double> expected_final;    // (tokens, d_model) row-major
  std::vector<double> expected_poly_final;  // identical polynomial circuit
  bool has_autoregressive = false;
  int autoregressive_prompt_tokens = 0;
  int autoregressive_generate_tokens = 0;
  int autoregressive_server_evaluations = 0;
  int autoregressive_vocab_size = 0;
  std::vector<int> autoregressive_expected_generated_ids;
  std::vector<double> autoregressive_embeddings;
  std::vector<double> autoregressive_expected_poly_final;
  std::vector<double> autoregressive_expected_exact_final;
  std::vector<double> client_embedding_w;
  // Empty means the checkpoint ties lm_head to client_embedding_w.
  std::vector<double> client_lm_head_w;
  std::vector<double> client_lm_head_b;
};


auto read_m1_payload(const std::string& dir) -> M1Payload;
auto read_chain_payload(const std::string& dir, bool load_autoregressive_assets)
    -> ChainPayload;
void require_same_layer_dims(const M1Payload& expected,
                             const M1Payload& actual,
                             std::size_t index);

}  // namespace fhemamba::stage1
