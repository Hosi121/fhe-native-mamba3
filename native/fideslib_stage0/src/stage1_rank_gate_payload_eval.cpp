#include "stage1_rank_gate_eval.hpp"

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Args {
  std::string input;
  std::string input_chain;
  double atol = 1e-9;
};

auto parse_args(int argc, char** argv) -> Args {
  Args args;
  for (int index = 1; index < argc; ++index) {
    const std::string item = argv[index];
    if (item == "--input" && index + 1 < argc) {
      args.input = argv[++index];
    } else if (item == "--input-chain" && index + 1 < argc) {
      args.input_chain = argv[++index];
    } else if (item == "--atol" && index + 1 < argc) {
      args.atol = std::stod(argv[++index]);
    } else {
      throw std::runtime_error(
          "usage: stage1_rank_gate_payload_eval (--input PATH | --input-chain PATHS) "
          "[--atol FLOAT]");
    }
  }
  if (args.input.empty() == args.input_chain.empty()) {
    throw std::runtime_error("exactly one of --input or --input-chain is required");
  }
  return args;
}

auto split_paths(const std::string& value) -> std::vector<std::string> {
  std::vector<std::string> paths;
  std::string current;
  for (const char character : value) {
    if (character == ',') {
      if (!current.empty()) {
        paths.push_back(current);
        current.clear();
      }
      continue;
    }
    current.push_back(character);
  }
  if (!current.empty()) {
    paths.push_back(current);
  }
  if (paths.empty()) {
    throw std::runtime_error("--input-chain must contain at least one path");
  }
  return paths;
}

void print_json(
    const stage1::RankGatePayload& payload,
    const stage1::RankGateEvalResult& result,
    double atol) {
  std::cout << "{";
  std::cout << "\"stage\":\"stage1-rank-gate-payload-native-eval\",";
  std::cout << "\"status\":\"" << (result.max_abs_error <= atol ? "passed" : "failed") << "\",";
  std::cout << "\"passed\":" << (result.max_abs_error <= atol ? "true" : "false") << ",";
  std::cout << "\"parameters\":{";
  std::cout << "\"d_model\":" << payload.config.d_model << ",";
  std::cout << "\"d_state\":" << payload.config.d_state << ",";
  std::cout << "\"mimo_rank\":" << payload.config.mimo_rank << ",";
  std::cout << "\"rank_pad\":" << payload.config.rank_pad << ",";
  std::cout << "\"layer_index\":" << payload.config.layer_index << ",";
  std::cout << "\"prompt_token\":" << payload.config.prompt_token;
  std::cout << "},";
  std::cout << "\"measurements\":{";
  std::cout << "\"array_count\":" << payload.arrays.size() << ",";
  std::cout << "\"atol\":" << atol << ",";
  std::cout << "\"max_abs_error\":" << result.max_abs_error << ",";
  std::cout << "\"conv_pre_max_abs_error\":" << result.conv_pre_max_abs_error << ",";
  std::cout << "\"rank_input_max_abs_error\":" << result.rank_input_max_abs_error << ",";
  std::cout << "\"gate_pre_max_abs_error\":" << result.gate_pre_max_abs_error << ",";
  std::cout << "\"gate_max_abs_error\":" << result.gate_max_abs_error << ",";
  std::cout << "\"skip_update_max_abs_error\":" << result.skip_update_max_abs_error;
  std::cout << "},";
  std::cout << "\"measurement_scope\":{";
  std::cout << "\"benchmark\":false,";
  std::cout << "\"native_handoff_payload\":true,";
  std::cout << "\"pre_recurrence_rank_gate_only\":true,";
  std::cout << "\"recurrence_tail_executed\":false,";
  std::cout << "\"fideslib_encrypted_execution\":false,";
  std::cout << "\"full_model_correctness_claimed\":false";
  std::cout << "}";
  std::cout << "}\n";
}

void print_chain_json(
    const std::vector<stage1::RankGatePayload>& payloads,
    const stage1::RankGateChainHandoffEvalResult& result,
    double atol) {
  std::cout << "{";
  std::cout << "\"stage\":\"stage1-rank-gate-payload-chain-native-eval\",";
  std::cout << "\"status\":\"" << (result.max_abs_error <= atol ? "passed" : "failed") << "\",";
  std::cout << "\"passed\":" << (result.max_abs_error <= atol ? "true" : "false") << ",";
  std::cout << "\"parameters\":{";
  std::cout << "\"payload_count\":" << payloads.size() << ",";
  std::cout << "\"layer_indices\":[";
  for (std::size_t index = 0; index < payloads.size(); ++index) {
    if (index > 0) {
      std::cout << ",";
    }
    std::cout << payloads[index].config.layer_index;
  }
  std::cout << "]";
  std::cout << "},";
  std::cout << "\"measurements\":{";
  std::cout << "\"atol\":" << atol << ",";
  std::cout << "\"max_abs_error\":" << result.max_abs_error << ",";
  std::cout << "\"rank_gate_max_abs_error\":" << result.rank_gate_max_abs_error << ",";
  std::cout << "\"model_layout_handoff_max_abs_error\":"
            << result.model_layout_handoff_max_abs_error;
  std::cout << "},";
  std::cout << "\"measurement_scope\":{";
  std::cout << "\"benchmark\":false,";
  std::cout << "\"native_handoff_payload\":true,";
  std::cout << "\"model_layout_handoff_reference\":true,";
  std::cout << "\"pre_recurrence_rank_gate_only\":true,";
  std::cout << "\"recurrence_tail_executed\":false,";
  std::cout << "\"fideslib_encrypted_execution\":false,";
  std::cout << "\"full_model_correctness_claimed\":false";
  std::cout << "}";
  std::cout << "}\n";
}

}  // namespace

auto main(int argc, char** argv) -> int {
  try {
    const auto args = parse_args(argc, argv);
    if (!args.input_chain.empty()) {
      std::vector<stage1::RankGatePayload> payloads;
      for (const auto& path : split_paths(args.input_chain)) {
        payloads.push_back(stage1::read_rank_gate_payload(path));
      }
      const auto result = stage1::evaluate_rank_gate_payload_chain_handoff(payloads);
      print_chain_json(payloads, result, args.atol);
      return result.max_abs_error <= args.atol ? EXIT_SUCCESS : EXIT_FAILURE;
    }
    const auto payload = stage1::read_rank_gate_payload(args.input);
    const auto result = stage1::evaluate_rank_gate_payload(payload);
    print_json(payload, result, args.atol);
    return result.max_abs_error <= args.atol ? EXIT_SUCCESS : EXIT_FAILURE;
  } catch (const std::exception& exc) {
    std::cerr << "stage1_rank_gate_payload_eval failed: " << exc.what() << "\n";
    return EXIT_FAILURE;
  }
}
