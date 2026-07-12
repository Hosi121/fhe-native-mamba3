#include "stage1_mamba2_config.hpp"

#include <functional>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

auto parse(std::vector<std::string> args) -> fhemamba::stage1::Config {
  std::vector<char*> argv;
  argv.reserve(args.size());
  for (auto& arg : args) {
    argv.push_back(arg.data());
  }
  return fhemamba::stage1::parse_args(static_cast<int>(argv.size()), argv.data());
}

void require(bool condition, const char* message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

void require_invalid(const std::function<void()>& operation) {
  try {
    operation();
  } catch (const std::invalid_argument&) {
    return;
  }
  throw std::runtime_error("expected invalid_argument");
}

}  // namespace

auto main() -> int {
  const auto defaults = parse({"stage1", "--input", "payload"});
  require(defaults.input == "payload", "input was not parsed");
  require(defaults.process_role == "inline", "unexpected process role default");
  require(defaults.ring_dim == 131072, "unexpected ring dimension default");
  require(defaults.multiplicative_depth == 44, "unexpected depth default");

  const auto server = parse({
      "stage1",
      "--input-chain",
      "chain",
      "--process-role",
      "server-eval",
      "--handoff-dir",
      "handoff",
      "--output-json",
      "result.json",
  });
  require(server.input_chain == "chain", "chain input was not parsed");
  require(server.process_role == "server-eval", "server role was not parsed");

  const auto explicit_false =
      parse({"stage1", "--input", "payload", "--debug-decrypt", "false"});
  require(!explicit_false.debug_decrypt, "false boolean was not parsed");

  require_invalid([] { parse({"stage1"}); });
  require_invalid([] {
    parse({"stage1", "--input", "one", "--input-chain", "two"});
  });
  require_invalid([] {
    parse({"stage1", "--input", "payload", "--process-role", "server-eval"});
  });
  require_invalid([] {
    parse({
        "stage1",
        "--input",
        "payload",
        "--process-role",
        "server-eval",
        "--handoff-dir",
        "handoff",
        "--output-json",
        "result.json",
        "--debug-decrypt",
        "1",
    });
  });
  require_invalid([] {
    parse({"stage1", "--input", "payload", "--ring-dim", "65535"});
  });
  require_invalid([] {
    parse({"stage1", "--input", "payload", "--tokens", "4junk"});
  });
  require_invalid([] {
    parse({"stage1", "--input", "payload", "--tolerance", "nan"});
  });
  require_invalid([] {
    parse({"stage1", "--input", "payload", "--debug-decrypt", "yes"});
  });
  require_invalid([] {
    parse({"stage1", "--input", "payload", "--bsgs-replicas", "2junk"});
  });
  require_invalid([] {
    parse({
        "stage1",
        "--input",
        "payload",
        "--process-role",
        "server-eval",
        "--handoff-dir",
        "handoff",
        "--output-json",
        "result.json",
        "--debug-refresh-probes",
        "1",
    });
  });
  require_invalid([] {
    parse({
        "stage1",
        "--input",
        "payload",
        "--tokens",
        "3",
        "--bootstrap-before-token",
        "1",
        "--debug-client-reencrypt-before-token",
        "1",
    });
  });
  return 0;
}
