#pragma once

#include "stage1_mamba2_config.hpp"

#include <map>
#include <set>
#include <sstream>
#include <string>
#include <string_view>
#include <vector>

namespace fhemamba::stage1 {

void write_payload(const std::string& output_json, const std::string& payload);
auto payload_file_exists(const std::string& output_json) -> bool;
auto json_escape(std::string_view value) -> std::string;
void write_int_set_json(std::ostringstream& out, const std::set<int>& values);
void write_double_vector_json(std::ostringstream& out,
                              const std::vector<double>& values);
void write_int_vector_json(std::ostringstream& out,
                           const std::vector<int>& values);
void write_double_map_json(std::ostringstream& out,
                           const std::map<std::string, double>& values);
void write_int_map_json(std::ostringstream& out,
                        const std::map<std::string, int>& values);
void write_artifact_prefix(std::ostringstream& out, const Config& args);
void write_runtime_failure_payload(const Config& args,
                                   std::string_view phase,
                                   std::string_view message);

}  // namespace fhemamba::stage1
