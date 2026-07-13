#include "stage1_mamba2_payload.hpp"

#include <cstdlib>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace fhemamba::stage1 {

auto read_text_file(const std::string& path) -> std::string {
  std::ifstream input(path);
  if (!input) {
    throw std::runtime_error("failed to open " + path);
  }
  std::ostringstream buffer;
  buffer << input.rdbuf();
  return buffer.str();
}

auto find_key_value_pos(const std::string& text, const std::string& key, std::size_t from = 0)
    -> std::size_t {
  const std::string needle = "\"" + key + "\"";
  const auto key_pos = text.find(needle, from);
  if (key_pos == std::string::npos) {
    return std::string::npos;
  }
  auto pos = key_pos + needle.size();
  while (pos < text.size() && (text[pos] == ' ' || text[pos] == '\n' || text[pos] == '\t' ||
                               text[pos] == '\r')) {
    ++pos;
  }
  if (pos >= text.size() || text[pos] != ':') {
    return find_key_value_pos(text, key, key_pos + needle.size());
  }
  ++pos;
  while (pos < text.size() && (text[pos] == ' ' || text[pos] == '\n' || text[pos] == '\t' ||
                               text[pos] == '\r')) {
    ++pos;
  }
  return pos;
}

auto json_number(const std::string& text, const std::string& key) -> double {
  const auto pos = find_key_value_pos(text, key);
  if (pos == std::string::npos) {
    throw std::runtime_error("meta.json key not found: " + key);
  }
  char* end = nullptr;
  const double value = std::strtod(text.c_str() + pos, &end);
  if (end == text.c_str() + pos) {
    throw std::runtime_error("meta.json key is not a number: " + key);
  }
  return value;
}

auto json_number_or(const std::string& text, const std::string& key, double fallback) -> double {
  const auto pos = find_key_value_pos(text, key);
  if (pos == std::string::npos) {
    return fallback;
  }
  char* end = nullptr;
  const double value = std::strtod(text.c_str() + pos, &end);
  if (end == text.c_str() + pos) {
    return fallback;
  }
  return value;
}

auto json_string(const std::string& text, const std::string& key) -> std::string {
  const auto pos = find_key_value_pos(text, key);
  if (pos == std::string::npos || pos >= text.size() || text[pos] != '"') {
    throw std::runtime_error("meta.json string key not found: " + key);
  }
  const auto close = text.find('"', pos + 1);
  if (close == std::string::npos) {
    throw std::runtime_error("meta.json unterminated string for key: " + key);
  }
  return text.substr(pos + 1, close - pos - 1);
}

auto json_balanced(const std::string& text, const std::string& key, char open, char close)
    -> std::string {
  const auto pos = find_key_value_pos(text, key);
  if (pos == std::string::npos || pos >= text.size() || text[pos] != open) {
    throw std::runtime_error("meta.json object/array key not found: " + key);
  }
  int depth = 0;
  for (std::size_t i = pos; i < text.size(); ++i) {
    if (text[i] == open) {
      ++depth;
    } else if (text[i] == close) {
      --depth;
      if (depth == 0) {
        return text.substr(pos, i - pos + 1);
      }
    }
  }
  throw std::runtime_error("meta.json unbalanced value for key: " + key);
}

auto json_number_list(const std::string& array_text) -> std::vector<double> {
  std::vector<double> values;
  const char* cursor = array_text.c_str();
  const char* end = cursor + array_text.size();
  while (cursor < end) {
    const char character = *cursor;
    if (character == '-' || (character >= '0' && character <= '9')) {
      char* next = nullptr;
      const double value = std::strtod(cursor, &next);
      if (next != cursor) {
        values.push_back(value);
        cursor = next;
        continue;
      }
    }
    ++cursor;
  }
  return values;
}


auto read_bin_tensor(const std::string& dir, const std::string& name,
                     const std::vector<int>& shape) -> std::vector<double> {
  std::size_t count = 1;
  for (const int dim : shape) {
    count *= static_cast<std::size_t>(dim);
  }
  const std::string path = dir + "/" + name + ".bin";
  std::ifstream input(path, std::ios::binary | std::ios::ate);
  if (!input) {
    throw std::runtime_error("failed to open " + path);
  }
  const auto bytes = static_cast<std::size_t>(input.tellg());
  if (bytes != count * sizeof(float)) {
    throw std::runtime_error("tensor size mismatch for " + name + ": expected " +
                             std::to_string(count * sizeof(float)) + " bytes, found " +
                             std::to_string(bytes));
  }
  input.seekg(0);
  std::vector<float> raw(count);
  input.read(reinterpret_cast<char*>(raw.data()), static_cast<std::streamsize>(bytes));
  if (!input) {
    throw std::runtime_error("failed to read " + path);
  }
  return {raw.begin(), raw.end()};
}

auto parse_poly_spec(const std::string& object_text) -> PolySpec {
  PolySpec spec;
  spec.kind = json_string(object_text, "kind");
  if (find_key_value_pos(object_text, "coeffs") != std::string::npos) {
    spec.coeffs = json_number_list(json_balanced(object_text, "coeffs", '[', ']'));
    spec.lo = json_number(object_text, "lo");
    spec.hi = json_number(object_text, "hi");
  }
  spec.squarings = static_cast<int>(json_number_or(object_text, "squarings", 0.0));
  spec.iterations = static_cast<int>(json_number_or(object_text, "iterations", 0.0));
  spec.damping = json_number_or(object_text, "damping", 1.0);
  spec.guess = json_number_or(object_text, "guess", 0.0);
  return spec;
}

auto read_m1_payload(const std::string& dir) -> M1Payload {
  M1Payload payload;
  const auto meta = read_text_file(dir + "/meta.json");
  if (json_string(meta, "format") != "fhemamba-m1-v1") {
    throw std::runtime_error("unexpected payload format (want fhemamba-m1-v1)");
  }
  const auto dims = json_balanced(meta, "dims", '{', '}');
  payload.d_model = static_cast<int>(json_number(dims, "d_model"));
  payload.d_inner = static_cast<int>(json_number(dims, "d_inner"));
  payload.num_heads = static_cast<int>(json_number(dims, "num_heads"));
  payload.head_dim = static_cast<int>(json_number(dims, "head_dim"));
  payload.state_size = static_cast<int>(json_number(dims, "state_size"));
  payload.n_groups = static_cast<int>(json_number(dims, "n_groups"));
  payload.conv_kernel = static_cast<int>(json_number(dims, "conv_kernel"));
  payload.conv_dim = static_cast<int>(json_number(dims, "conv_dim"));
  const auto eps = json_balanced(meta, "eps", '{', '}');
  payload.eps_block = json_number(eps, "block_norm");
  payload.eps_gated = json_number(eps, "gated_norm");
  payload.n_test_tokens = static_cast<int>(json_number(meta, "n_test_tokens"));
  if (find_key_value_pos(meta, "carried_bounds") != std::string::npos) {
    const auto carried = json_balanced(meta, "carried_bounds", '{', '}');
    // Values without an explicit independent-calibration provenance came
    // from the evaluation prompt in the first exporter implementation. Do
    // not let those benchmark-leaked bounds change the encrypted circuit.
    if (find_key_value_pos(carried, "source") != std::string::npos &&
        json_string(carried, "source") == "calibration_text") {
      payload.state_abs_max = json_number(carried, "state_abs_max");
      if (find_key_value_pos(carried, "state_head_abs_max") != std::string::npos) {
        payload.state_head_abs_max = json_number_list(
            json_balanced(carried, "state_head_abs_max", '[', ']'));
        if (payload.state_head_abs_max.size() !=
            static_cast<std::size_t>(payload.num_heads)) {
          throw std::runtime_error("state_head_abs_max length must equal num_heads");
        }
      }
      payload.fifo_abs_max = json_number(carried, "fifo_abs_max");
      if (find_key_value_pos(carried, "checkpoint_abs_max") != std::string::npos) {
        const auto checkpoints =
            json_balanced(carried, "checkpoint_abs_max", '{', '}');
        for (const auto* name : {"residual", "proj", "conv_silu", "dt", "y",
                                 "output", "gated_variance", "gated_newton"}) {
          if (find_key_value_pos(checkpoints, name) != std::string::npos) {
            payload.checkpoint_abs_max[name] = json_number(checkpoints, name);
          }
        }
      }
    }
  }

  const auto polys = json_balanced(meta, "polys", '{', '}');
  for (const auto* name : {"conv_silu", "gate_silu", "dt_softplus", "decay_exp",
                           "rms_invsqrt", "gated_rms_invsqrt"}) {
    payload.polys[name] = parse_poly_spec(json_balanced(polys, name, '{', '}'));
  }

  const auto tensors = json_balanced(meta, "tensors", '{', '}');
  for (const auto* name : {"in_proj_w", "conv_w", "conv_b", "dt_bias", "a_log", "d_skip",
                           "block_norm_w", "gated_norm_w", "out_proj_w", "test_layer_input",
                           "test_layer_output"}) {
    const auto shape_values = json_number_list(json_balanced(tensors, name, '[', ']'));
    std::vector<int> shape;
    shape.reserve(shape_values.size());
    for (const double value : shape_values) {
      shape.push_back(static_cast<int>(value));
    }
    payload.shapes[name] = shape;
    payload.tensors[name] = read_bin_tensor(dir, name, shape);
  }
  for (const auto* name : {"test_layer_output_poly", "test_state_output",
                           "test_state_output_poly", "autoregressive_poly_layer_output",
                           "autoregressive_poly_state_output",
                           "autoregressive_poly_decay_output",
                           "autoregressive_poly_state_update",
                           "autoregressive_poly_state_decayed"}) {
    if (find_key_value_pos(tensors, name) == std::string::npos) {
      continue;
    }
    const auto shape_values =
        json_number_list(json_balanced(tensors, name, '[', ']'));
    std::vector<int> shape;
    shape.reserve(shape_values.size());
    for (const double value : shape_values) {
      shape.push_back(static_cast<int>(value));
    }
    payload.shapes[name] = shape;
    payload.tensors[name] = read_bin_tensor(dir, name, shape);
  }
  for (const auto* name : {"test_state_output", "test_state_output_poly",
                           "autoregressive_poly_state_output",
                           "autoregressive_poly_state_update",
                           "autoregressive_poly_state_decayed"}) {
    const auto found = payload.shapes.find(name);
    if (found == payload.shapes.end()) {
      continue;
    }
    const auto& shape = found->second;
    const bool autoregressive =
        std::string_view(name).find("autoregressive_poly_") == 0;
    if (shape.size() != 4 || shape[0] < 1 ||
        (!autoregressive && shape[0] > payload.n_test_tokens) ||
        shape[1] != payload.num_heads || shape[2] != payload.head_dim ||
        shape[3] != payload.state_size) {
      throw std::runtime_error(std::string(name) +
                               " shape must be (tokens, heads, head_dim, state_size)");
    }
  }
  const auto autoregressive_decay =
      payload.shapes.find("autoregressive_poly_decay_output");
  if (autoregressive_decay != payload.shapes.end()) {
    const auto& shape = autoregressive_decay->second;
    if (shape.size() != 2 || shape[0] < 1 || shape[1] != payload.num_heads) {
      throw std::runtime_error(
          "autoregressive_poly_decay_output shape must be (tokens, heads)");
    }
  }
  const auto autoregressive_boundary =
      payload.shapes.find("autoregressive_poly_layer_output");
  if (autoregressive_boundary != payload.shapes.end()) {
    const auto& shape = autoregressive_boundary->second;
    if (shape.size() != 2 || shape[0] < 1 || shape[1] != payload.d_model) {
      throw std::runtime_error(
          "autoregressive_poly_layer_output shape must be (tokens, d_model)");
    }
  }
  payload.proj_dim = payload.shapes.at("in_proj_w").at(0);
  if (payload.proj_dim != payload.d_inner + payload.conv_dim + payload.num_heads) {
    throw std::runtime_error("in_proj packing with d_mlp != 0 is not supported");
  }
  if (payload.d_inner != payload.num_heads * payload.head_dim) {
    throw std::runtime_error("d_inner must equal num_heads * head_dim");
  }
  if (payload.n_groups != 1) {
    throw std::runtime_error("only n_groups == 1 payloads are supported");
  }
  if (payload.conv_dim != payload.d_inner + 2 * payload.state_size) {
    throw std::runtime_error("conv_dim must equal d_inner + 2 * state_size");
  }
  return payload;
}

// ---------------------------------------------------------------------------
// M2 chain payload: chain.json + root tensors + layer_XX/ m1 payloads.
// ---------------------------------------------------------------------------

auto json_string_list(const std::string& array_text) -> std::vector<std::string> {
  std::vector<std::string> values;
  std::size_t cursor = 1;  // skip '['
  while (cursor < array_text.size()) {
    const auto open = array_text.find('"', cursor);
    if (open == std::string::npos) {
      break;
    }
    const auto close = array_text.find('"', open + 1);
    if (close == std::string::npos) {
      break;
    }
    values.push_back(array_text.substr(open + 1, close - open - 1));
    cursor = close + 1;
  }
  return values;
}


auto read_chain_payload(const std::string& dir,
                        bool load_autoregressive_assets) -> ChainPayload {
  ChainPayload chain;
  const auto meta = read_text_file(dir + "/chain.json");
  if (json_string(meta, "format") != "fhemamba-m2-chain-v1") {
    throw std::runtime_error("unexpected chain payload format (want fhemamba-m2-chain-v1)");
  }
  chain.n_layers = static_cast<int>(json_number(meta, "n_layers"));
  chain.n_test_tokens = static_cast<int>(json_number(meta, "n_test_tokens"));
  chain.final_norm_eps = json_number(meta, "final_norm_eps");
  chain.layer_dirs = json_string_list(json_balanced(meta, "layer_dirs", '[', ']'));
  if (chain.n_layers <= 0 ||
      chain.layer_dirs.size() != static_cast<std::size_t>(chain.n_layers)) {
    throw std::runtime_error("chain.json layer_dirs does not match n_layers");
  }
  const auto tensors = json_balanced(meta, "tensors", '{', '}');
  for (const auto* name : {"final_norm_w", "chain_input_embeddings", "chain_expected_final"}) {
    const auto shape_values = json_number_list(json_balanced(tensors, name, '[', ']'));
    std::vector<int> shape;
    shape.reserve(shape_values.size());
    for (const double value : shape_values) {
      shape.push_back(static_cast<int>(value));
    }
    auto values = read_bin_tensor(dir, name, shape);
    if (std::string_view(name) == "final_norm_w") {
      chain.final_norm_w = std::move(values);
    } else if (std::string_view(name) == "chain_input_embeddings") {
      chain.input_embeddings = std::move(values);
    } else {
      chain.expected_final = std::move(values);
    }
  }
  if (find_key_value_pos(tensors, "chain_expected_poly_final") != std::string::npos) {
    const auto shape_values = json_number_list(
        json_balanced(tensors, "chain_expected_poly_final", '[', ']'));
    std::vector<int> shape;
    shape.reserve(shape_values.size());
    for (const double value : shape_values) {
      shape.push_back(static_cast<int>(value));
    }
    chain.expected_poly_final =
        read_bin_tensor(dir, "chain_expected_poly_final", shape);
  }
  const auto autoregressive_pos = find_key_value_pos(meta, "autoregressive");
  if (autoregressive_pos != std::string::npos && meta[autoregressive_pos] == '{') {
    chain.has_autoregressive = true;
    const auto autoregressive = json_balanced(meta, "autoregressive", '{', '}');
    if (json_string(autoregressive, "protocol") != "client-in-loop-greedy-v1") {
      throw std::runtime_error("unsupported autoregressive payload protocol");
    }
    chain.autoregressive_prompt_tokens =
        static_cast<int>(json_number(autoregressive, "prompt_tokens"));
    chain.autoregressive_generate_tokens =
        static_cast<int>(json_number(autoregressive, "generate_tokens"));
    chain.autoregressive_server_evaluations =
        static_cast<int>(json_number(autoregressive, "server_evaluations"));
    for (const double value : json_number_list(
             json_balanced(autoregressive, "poly_generated_ids", '[', ']'))) {
      chain.autoregressive_expected_generated_ids.push_back(static_cast<int>(value));
    }
    if (!load_autoregressive_assets) {
      return chain;
    }
    const auto read_root_tensor = [&](const std::string& name,
                                      std::vector<int>* output_shape = nullptr) {
      const auto shape_values =
          json_number_list(json_balanced(tensors, name, '[', ']'));
      std::vector<int> shape;
      shape.reserve(shape_values.size());
      for (const double value : shape_values) {
        shape.push_back(static_cast<int>(value));
      }
      if (output_shape != nullptr) {
        *output_shape = shape;
      }
      return read_bin_tensor(dir, name, shape);
    };
    std::vector<int> embedding_shape;
    chain.client_embedding_w =
        read_root_tensor("client_embedding_w", &embedding_shape);
    if (embedding_shape.size() != 2) {
      throw std::runtime_error("client_embedding_w must be rank 2");
    }
    chain.autoregressive_vocab_size = embedding_shape[0];
    if (find_key_value_pos(tensors, "client_lm_head_w") != std::string::npos) {
      chain.client_lm_head_w = read_root_tensor("client_lm_head_w");
    }
    if (find_key_value_pos(tensors, "client_lm_head_b") != std::string::npos) {
      chain.client_lm_head_b = read_root_tensor("client_lm_head_b");
    }
    chain.autoregressive_embeddings =
        read_root_tensor("autoregressive_poly_embeddings");
    chain.autoregressive_expected_poly_final =
        read_root_tensor("autoregressive_poly_expected_final");
    chain.autoregressive_expected_exact_final =
        read_root_tensor("autoregressive_exact_expected_final");
  }
  return chain;
}

void require_same_layer_dims(const M1Payload& expected, const M1Payload& actual,
                             std::size_t index) {
  if (expected.d_model == actual.d_model && expected.d_inner == actual.d_inner &&
      expected.num_heads == actual.num_heads && expected.head_dim == actual.head_dim &&
      expected.state_size == actual.state_size && expected.n_groups == actual.n_groups &&
      expected.conv_kernel == actual.conv_kernel && expected.conv_dim == actual.conv_dim &&
      expected.proj_dim == actual.proj_dim) {
    return;
  }
  throw std::runtime_error("chain layer payload dims mismatch at index " +
                           std::to_string(index));
}

// Per-layer host-side plaintext constants (folded weights, masks, poly

}  // namespace fhemamba::stage1
