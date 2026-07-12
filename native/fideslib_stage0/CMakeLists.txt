cmake_minimum_required(VERSION 3.25.2)

project(fideslib_stage0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(FHE_STAGE0_BUILD_KERNEL "Build the FIDESlib Stage 0 kernel" ON)
option(FHE_STAGE0_BUILD_TESTS "Build FIDESlib-free Stage 0 layout tests" OFF)

add_library(stage0_layout INTERFACE)
target_include_directories(stage0_layout INTERFACE include)

add_library(stage1_tail_payload INTERFACE)
target_include_directories(stage1_tail_payload INTERFACE include)

add_executable(stage1_tail_payload_eval src/stage1_tail_payload_eval.cpp)
target_link_libraries(stage1_tail_payload_eval PRIVATE stage1_tail_payload)

add_library(stage1_rank_gate_payload INTERFACE)
target_include_directories(stage1_rank_gate_payload INTERFACE include)

add_library(stage1_mamba2_config STATIC src/stage1_mamba2_config.cpp)
target_include_directories(stage1_mamba2_config PUBLIC src)

add_executable(stage1_rank_gate_payload_eval src/stage1_rank_gate_payload_eval.cpp)
target_link_libraries(
  stage1_rank_gate_payload_eval PRIVATE stage1_rank_gate_payload stage1_tail_payload)

if(FHE_STAGE0_BUILD_KERNEL)
  find_package(fideslib REQUIRED CONFIG)
  find_package(CUDAToolkit REQUIRED)
  find_package(OpenFHE REQUIRED CONFIG)

  if(fideslib_FOUND)
    message(STATUS "FOUND PACKAGE FIDESlib")
    message(STATUS "FIDESlib Version: ${fideslib_VERSION}")
    message(STATUS "FIDESlib header files location: ${FIDESLIB_INCLUDE_PATH}")
    message(STATUS "FIDESlib lib files location: ${FIDESLIB_LIBRARIES_PATH}")
    message(STATUS "FIDESlib C++ compiler: ${FIDESLIB_CXX_COMPILER}")
  else()
    message(FATAL_ERROR "PACKAGE FIDESlib NOT FOUND")
  endif()

  set(CMAKE_CXX_COMPILER ${FIDESLIB_CXX_COMPILER})
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${FIDESLIB_CXX_FLAGS}")

  add_executable(stage0_static_mimo src/stage0_static_mimo.cpp)
  target_link_libraries(stage0_static_mimo PRIVATE fideslib::fideslib stage0_layout)

  add_executable(stage1_bootstrap_probe src/stage1_bootstrap_probe.cpp)
  target_link_libraries(stage1_bootstrap_probe PRIVATE fideslib::fideslib)

  add_executable(stage1_rotation_probe src/stage1_rotation_probe.cpp)
  target_link_libraries(stage1_rotation_probe PRIVATE fideslib::fideslib)

  add_executable(stage1_tail_fideslib src/stage1_tail_fideslib.cpp)
  target_link_libraries(stage1_tail_fideslib PRIVATE fideslib::fideslib stage1_tail_payload)

  add_executable(stage1_rank_gate_fideslib src/stage1_rank_gate_fideslib.cpp)
  target_link_libraries(
    stage1_rank_gate_fideslib PRIVATE fideslib::fideslib stage1_rank_gate_payload stage1_tail_payload)

  add_executable(
    stage1_mamba2_decode_fideslib
    src/stage1_mamba2_decode_fideslib.cpp)
  target_link_libraries(
    stage1_mamba2_decode_fideslib PRIVATE fideslib::fideslib CUDA::cudart stage1_mamba2_config stage1_rank_gate_payload stage1_tail_payload)
  target_include_directories(
    stage1_mamba2_decode_fideslib PRIVATE
    ${OpenFHE_INCLUDE}
    ${OpenFHE_INCLUDE}/core
    ${OpenFHE_INCLUDE}/pke
    ${OpenFHE_INCLUDE}/binfhe
    ${OpenFHE_INCLUDE}/third-party/include)

  add_executable(fideslib_client_server_probe src/fideslib_client_server_probe.cpp)
  target_link_libraries(
    fideslib_client_server_probe PRIVATE fideslib::fideslib CUDA::cudart)
  target_include_directories(
    fideslib_client_server_probe PRIVATE
    ${OpenFHE_INCLUDE}
    ${OpenFHE_INCLUDE}/core
    ${OpenFHE_INCLUDE}/pke
    ${OpenFHE_INCLUDE}/binfhe
    ${OpenFHE_INCLUDE}/third-party/include)

  add_executable(fideslib_ctpt_probe src/fideslib_ctpt_probe.cpp)
  target_link_libraries(fideslib_ctpt_probe PRIVATE fideslib::fideslib)
endif()

if(FHE_STAGE0_BUILD_TESTS)
  enable_testing()
  add_executable(test_stage0_layout tests/test_stage0_layout.cpp)
  target_link_libraries(test_stage0_layout PRIVATE stage0_layout)
  add_test(NAME stage0_layout COMMAND test_stage0_layout)

  add_executable(test_stage1_tail_payload tests/test_stage1_tail_payload.cpp)
  target_link_libraries(test_stage1_tail_payload PRIVATE stage1_tail_payload)
  add_test(NAME stage1_tail_payload COMMAND test_stage1_tail_payload)

  add_executable(test_stage1_rank_gate_payload tests/test_stage1_rank_gate_payload.cpp)
  target_link_libraries(
    test_stage1_rank_gate_payload PRIVATE stage1_rank_gate_payload stage1_tail_payload)
  add_test(NAME stage1_rank_gate_payload COMMAND test_stage1_rank_gate_payload)

  add_executable(
    test_stage1_mamba2_config
    tests/test_stage1_mamba2_config.cpp)
  target_link_libraries(test_stage1_mamba2_config PRIVATE stage1_mamba2_config)
  add_test(NAME stage1_mamba2_config COMMAND test_stage1_mamba2_config)
endif()
