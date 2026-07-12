#include "stage1_mamba2_artifact.hpp"

#include <filesystem>
#include <fstream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>

namespace fs = std::filesystem;

namespace {

void require(bool condition, const char* message) {
  if (!condition) {
    throw std::runtime_error(message);
  }
}

auto read_file(const fs::path& path) -> std::string {
  std::ifstream input(path);
  return {std::istreambuf_iterator<char>(input),
          std::istreambuf_iterator<char>()};
}

}  // namespace

auto main() -> int {
  using namespace fhemamba::stage1;

  const std::string controls =
      std::string("\"\\\b\f\n\r\t") + static_cast<char>(1);
  require(json_escape(controls) == "\\\"\\\\\\b\\f\\n\\r\\t\\u0001",
          "JSON control characters were not escaped");

  std::ostringstream vector;
  write_double_vector_json(
      vector,
      {1.5, std::numeric_limits<double>::quiet_NaN(),
       std::numeric_limits<double>::infinity(),
       -std::numeric_limits<double>::infinity()});
  require(vector.str() == "[1.5,\"NaN\",\"Inf\",\"-Inf\"]",
          "non-finite vector values produced invalid JSON");

  std::ostringstream map;
  write_double_map_json(
      map, {{"bad\nkey", std::numeric_limits<double>::quiet_NaN()}});
  require(map.str() == "{\"bad\\nkey\":\"NaN\"}",
          "non-finite map values produced invalid JSON");

  const auto root = fs::temp_directory_path() / "fhemamba-stage1-artifact-test";
  fs::remove_all(root);
  fs::create_directories(root);
  const auto output = root / "result.json";
  write_payload(output.string(), "{\"generation\":1}");
  require(read_file(output) == "{\"generation\":1}\n",
          "artifact payload was not written");
  require(!fs::exists(output.string() + ".tmp"),
          "temporary artifact was not renamed");
  write_payload(output.string(), "{\"generation\":2}");
  require(read_file(output) == "{\"generation\":2}\n",
          "artifact replacement was not atomic");

  Config config;
  config.input = "payload";
  config.output_json = output.string();
  config.artifact_version = std::string("test") + static_cast<char>(1);
  config.binary_sha256 = std::string(64, 'a');
  write_runtime_failure_payload(config, "phase\nname", "message\ttext");
  const auto failure = read_file(output);
  require(failure.find("\"status\":\"failed\"") != std::string::npos,
          "failure artifact lacks status");
  require(failure.find("\"binary_sha256\":\"" + std::string(64, 'a') + "\"") !=
              std::string::npos,
          "failure artifact lacks the binary hash");
  require(failure.find("test\\u0001") != std::string::npos,
          "failure artifact contains an unescaped control character");
  require(failure.find("phase\\nname") != std::string::npos,
          "failure phase was not escaped");

  fs::remove_all(root);
  return 0;
}
