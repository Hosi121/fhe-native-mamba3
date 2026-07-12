#include "stage1_mamba2_process.hpp"

#include <filesystem>
#include <fstream>
#include <functional>
#include <stdexcept>

namespace fs = std::filesystem;

namespace {

void require(bool condition, const char* message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

void require_failure(const std::function<void()>& operation) {
  try {
    operation();
  } catch (const std::runtime_error&) {
    return;
  }
  throw std::runtime_error("expected runtime_error");
}

}  // namespace

auto main() -> int {
  using namespace fhemamba::stage1;

  const auto root = fs::temp_directory_path() / "fhemamba-stage1-process-test";
  fs::remove_all(root);
  const auto paths = handoff_paths(root.string());
  require(paths.client == root / "client", "client path mismatch");
  require(paths.server == root / "server", "server path mismatch");
  require(paths.exchange == root / "exchange", "exchange path mismatch");

  prepare_client_handoff(paths);
  for (const auto& path : {paths.root, paths.client, paths.server,
                           paths.exchange}) {
    const auto permissions = fs::status(path).permissions();
    require((permissions & fs::perms::group_all) == fs::perms::none,
            "handoff path is accessible by the group");
    require((permissions & fs::perms::others_all) == fs::perms::none,
            "handoff path is accessible by other users");
  }
  require(count_server_secret_files(paths) == 0,
          "empty server handoff reported a secret");

  {
    std::ofstream secret(paths.server / "Secret-Key.BIN");
    secret << "test";
  }
  require(count_server_secret_files(paths) == 1,
          "case-insensitive secret audit failed");
  require_failure([&] { require_server_has_no_secret_files(paths); });
  require_failure([&] { prepare_client_handoff(paths); });

  fs::remove_all(root);
  fs::create_directories(root);
  prepare_client_handoff(paths);
  require(fs::is_empty(paths.server), "fresh server handoff is not empty");
  fs::remove_all(root);
  return 0;
}
