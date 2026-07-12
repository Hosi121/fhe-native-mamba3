#include "stage1_mamba2_process.hpp"

#include <algorithm>
#include <cctype>
#include <stdexcept>
#include <string>

namespace fs = std::filesystem;

namespace fhemamba::stage1 {

namespace {

auto contains_secret(std::string filename) -> bool {
  std::transform(filename.begin(), filename.end(), filename.begin(),
                 [](unsigned char character) {
                   return static_cast<char>(std::tolower(character));
                 });
  return filename.find("secret") != std::string::npos;
}

void restrict_to_owner(const fs::path& path) {
  fs::permissions(path, fs::perms::owner_all, fs::perm_options::replace);
}

}  // namespace

auto handoff_paths(const std::string& root) -> HandoffPaths {
  HandoffPaths paths;
  paths.root = fs::path(root);
  paths.client = paths.root / "client";
  paths.server = paths.root / "server";
  paths.exchange = paths.root / "exchange";
  return paths;
}

void prepare_client_handoff(const HandoffPaths& paths) {
  if (fs::exists(paths.root)) {
    if (!fs::is_directory(paths.root)) {
      throw std::runtime_error("handoff root exists but is not a directory: " +
                               paths.root.string());
    }
    if (!fs::is_empty(paths.root)) {
      throw std::runtime_error("refusing to reuse non-empty handoff root: " +
                               paths.root.string());
    }
  }

  fs::create_directories(paths.client);
  fs::create_directories(paths.server);
  fs::create_directories(paths.exchange);
  restrict_to_owner(paths.root);
  restrict_to_owner(paths.client);
  restrict_to_owner(paths.server);
  restrict_to_owner(paths.exchange);
}

auto count_server_secret_files(const HandoffPaths& paths) -> int {
  if (!fs::exists(paths.server)) {
    return 0;
  }
  int count = 0;
  for (const auto& entry : fs::recursive_directory_iterator(paths.server)) {
    if (entry.is_regular_file() &&
        contains_secret(entry.path().filename().string())) {
      ++count;
    }
  }
  return count;
}

void require_server_has_no_secret_files(const HandoffPaths& paths) {
  const int count = count_server_secret_files(paths);
  if (count != 0) {
    throw std::runtime_error("server handoff contains " + std::to_string(count) +
                             " secret-key-named file(s)");
  }
}

}  // namespace fhemamba::stage1
