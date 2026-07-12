#pragma once

#include <filesystem>
#include <string>

namespace fhemamba::stage1 {

struct HandoffPaths {
  std::filesystem::path root;
  std::filesystem::path client;
  std::filesystem::path server;
  std::filesystem::path exchange;
};

auto handoff_paths(const std::string& root) -> HandoffPaths;
void prepare_client_handoff(const HandoffPaths& paths);
auto count_server_secret_files(const HandoffPaths& paths) -> int;
void require_server_has_no_secret_files(const HandoffPaths& paths);

}  // namespace fhemamba::stage1
