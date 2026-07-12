#include "stage1_mamba2_artifact.hpp"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <system_error>

namespace fs = std::filesystem;

namespace fhemamba::stage1 {

namespace {

void write_double_json(std::ostringstream& out, double value) {
  if (std::isfinite(value)) {
    out << value;
    return;
  }
  out << "\"" << (std::isnan(value) ? "NaN" : (value > 0 ? "Inf" : "-Inf"))
      << "\"";
}

}  // namespace

void write_payload(const std::string& output_json, const std::string& payload) {
  if (output_json.empty()) {
    std::cout << payload << std::endl;
    return;
  }

  const fs::path destination(output_json);
  auto temporary = destination;
  temporary += ".tmp";
  {
    std::ofstream output(temporary, std::ios::trunc);
    if (!output) {
      throw std::runtime_error("failed to open output-json temporary path: " +
                               temporary.string());
    }
    output << payload << '\n';
    output.flush();
    if (!output) {
      throw std::runtime_error("failed to write output-json temporary path: " +
                               temporary.string());
    }
  }

  std::error_code error;
  fs::rename(temporary, destination, error);
  if (error) {
    fs::remove(temporary);
    throw std::runtime_error("failed to publish output-json path: " + error.message());
  }
}

auto payload_file_exists(const std::string& output_json) -> bool {
  if (output_json.empty()) {
    return false;
  }
  std::error_code error;
  const auto size = fs::file_size(output_json, error);
  return !error && size > 0;
}

auto json_escape(std::string_view value) -> std::string {
  constexpr char kHex[] = "0123456789abcdef";
  std::string output;
  output.reserve(value.size());
  for (const unsigned char character : value) {
    switch (character) {
      case '\\':
        output += "\\\\";
        break;
      case '"':
        output += "\\\"";
        break;
      case '\b':
        output += "\\b";
        break;
      case '\f':
        output += "\\f";
        break;
      case '\n':
        output += "\\n";
        break;
      case '\r':
        output += "\\r";
        break;
      case '\t':
        output += "\\t";
        break;
      default:
        if (character < 0x20) {
          output += "\\u00";
          output.push_back(kHex[(character >> 4) & 0x0f]);
          output.push_back(kHex[character & 0x0f]);
        } else {
          output.push_back(static_cast<char>(character));
        }
        break;
    }
  }
  return output;
}

void write_int_set_json(std::ostringstream& out, const std::set<int>& values) {
  out << "[";
  bool first = true;
  for (const int value : values) {
    if (!first) {
      out << ",";
    }
    first = false;
    out << value;
  }
  out << "]";
}

void write_double_vector_json(std::ostringstream& out,
                              const std::vector<double>& values) {
  out << "[";
  bool first = true;
  for (const double value : values) {
    if (!first) {
      out << ",";
    }
    first = false;
    write_double_json(out, value);
  }
  out << "]";
}

void write_int_vector_json(std::ostringstream& out,
                           const std::vector<int>& values) {
  out << "[";
  bool first = true;
  for (const int value : values) {
    if (!first) {
      out << ",";
    }
    first = false;
    out << value;
  }
  out << "]";
}

void write_double_map_json(std::ostringstream& out,
                           const std::map<std::string, double>& values) {
  out << "{";
  bool first = true;
  for (const auto& [name, value] : values) {
    if (!first) {
      out << ",";
    }
    first = false;
    out << "\"" << json_escape(name) << "\":";
    write_double_json(out, value);
  }
  out << "}";
}

void write_int_map_json(std::ostringstream& out,
                        const std::map<std::string, int>& values) {
  out << "{";
  bool first = true;
  for (const auto& [name, value] : values) {
    if (!first) {
      out << ",";
    }
    first = false;
    out << "\"" << json_escape(name) << "\":" << value;
  }
  out << "}";
}

void write_artifact_prefix(std::ostringstream& out, const Config& args) {
  out << "\"version\":\"" << json_escape(args.artifact_version) << "\",";
  out << "\"repo_commit\":\"" << json_escape(args.repo_commit) << "\",";
  out << "\"binary_sha256\":\"" << json_escape(args.binary_sha256) << "\",";
  out << "\"stage\":\"stage1-mamba2-decode-fideslib\",";
  out << "\"backend\":\"fideslib-gpu\",";
  out << "\"encrypted\":true,";
  out << "\"config\":{\"input_mode\":\"fhemamba-m1-payload\"},";
}

void write_runtime_failure_payload(const Config& args,
                                   std::string_view phase,
                                   std::string_view message) {
  if (args.output_json.empty()) {
    return;
  }
  std::ostringstream out;
  out << "{";
  write_artifact_prefix(out, args);
  out << "\"status\":\"failed\",";
  out << "\"passed\":false,";
  out << "\"failure_phase\":\"" << json_escape(phase) << "\",";
  out << "\"error_message\":\"" << json_escape(message) << "\",";
  out << "\"parameters\":{";
  out << "\"ring_dimension\":" << args.ring_dim << ",";
  out << "\"multiplicative_depth\":" << args.multiplicative_depth << ",";
  out << "\"scaling_mod_size\":" << args.scaling_mod_size << ",";
  out << "\"tokens\":" << args.tokens << ",";
  out << "\"process_role\":\"" << json_escape(args.process_role) << "\",";
  out << "\"autoregressive_client_loop\":"
      << (args.autoregressive_client_loop ? "true" : "false") << ",";
  out << "\"level_align_mode\":\"" << json_escape(args.level_align_mode) << "\",";
  out << "\"meta_bts_residual_align_mode\":\""
      << json_escape(args.meta_bts_residual_align_mode) << "\",";
  out << "\"bootstrap_before_token\":";
  write_int_set_json(out, args.bootstrap_before_token);
  out << ",\"debug_client_reencrypt_before_token\":";
  write_int_set_json(out, args.debug_client_reencrypt_before_token);
  out << ",\"refresh_recurrent_state_post\":"
      << (args.refresh_recurrent_state_post ? "true" : "false");
  out << ",\"refresh_recurrent_state_post_layers\":";
  write_int_set_json(out, args.refresh_recurrent_state_post_layers);
  out << "},";
  out << "\"measurement_scope\":{";
  out << "\"non_success_probe\":true,";
  out << "\"full_model_correctness_claimed\":false,";
  out << "\"claim\":\"Native FIDESlib Mamba-2 decode kernel failed before final decrypt; "
         "this artifact preserves the failure phase for collection.\"";
  out << "}";
  out << "}";
  write_payload(args.output_json, out.str());
}

}  // namespace fhemamba::stage1
