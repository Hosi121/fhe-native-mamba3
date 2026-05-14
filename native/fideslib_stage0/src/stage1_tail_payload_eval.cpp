#include "stage1_tail_eval.hpp"

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

namespace {

struct Args {
  std::string input;
  double atol = 1e-9;
};

auto parse_args(int argc, char** argv) -> Args {
  Args args;
  for (int index = 1; index < argc; ++index) {
    const std::string item = argv[index];
    if (item == "--input" && index + 1 < argc) {
      args.input = argv[++index];
    } else if (item == "--atol" && index + 1 < argc) {
      args.atol = std::stod(argv[++index]);
    } else {
      throw std::runtime_error("usage: stage1_tail_payload_eval --input PATH [--atol FLOAT]");
    }
  }
  if (args.input.empty()) {
    throw std::runtime_error("--input is required");
  }
  return args;
}

void print_json(const stage1::TailPayload& payload, const stage1::TailEvalResult& result, double atol) {
  std::cout << "{";
  std::cout << "\"stage\":\"stage1-tail-payload-native-eval\",";
  std::cout << "\"status\":\"" << (result.max_abs_error <= atol ? "passed" : "failed") << "\",";
  std::cout << "\"passed\":" << (result.max_abs_error <= atol ? "true" : "false") << ",";
  std::cout << "\"parameters\":{";
  std::cout << "\"d_model\":" << payload.config.d_model << ",";
  std::cout << "\"d_state\":" << payload.config.d_state << ",";
  std::cout << "\"mimo_rank\":" << payload.config.mimo_rank << ",";
  std::cout << "\"rank_pad\":" << payload.config.rank_pad << ",";
  std::cout << "\"layer_index\":" << payload.config.layer_index << ",";
  std::cout << "\"prompt_token\":" << payload.config.prompt_token << ",";
  std::cout << "\"dt_rank\":" << payload.config.dt_rank;
  std::cout << "},";
  std::cout << "\"measurements\":{";
  std::cout << "\"array_count\":" << payload.arrays.size() << ",";
  std::cout << "\"atol\":" << atol << ",";
  std::cout << "\"max_abs_error\":" << result.max_abs_error << ",";
  std::cout << "\"state_new_max_abs_error\":" << result.state_new_max_abs_error << ",";
  std::cout << "\"readout_rank_max_abs_error\":" << result.readout_rank_max_abs_error << ",";
  std::cout << "\"rank_output_max_abs_error\":" << result.rank_output_max_abs_error << ",";
  std::cout << "\"rank_payload_max_abs_error\":" << result.rank_payload_max_abs_error << ",";
  std::cout << "\"output_model_max_abs_error\":" << result.output_model_max_abs_error;
  std::cout << "},";
  std::cout << "\"measurement_scope\":{";
  std::cout << "\"benchmark\":false,";
  std::cout << "\"native_handoff_payload\":true,";
  std::cout << "\"pre_recurrence_tail_only\":true,";
  std::cout << "\"fideslib_encrypted_execution\":false,";
  std::cout << "\"full_model_correctness_claimed\":false";
  std::cout << "}";
  std::cout << "}\n";
}

}  // namespace

auto main(int argc, char** argv) -> int {
  try {
    const auto args = parse_args(argc, argv);
    const auto payload = stage1::read_tail_payload(args.input);
    const auto result = stage1::evaluate_tail_payload(payload);
    print_json(payload, result, args.atol);
    return result.max_abs_error <= args.atol ? EXIT_SUCCESS : EXIT_FAILURE;
  } catch (const std::exception& exc) {
    std::cerr << "stage1_tail_payload_eval failed: " << exc.what() << "\n";
    return EXIT_FAILURE;
  }
}
