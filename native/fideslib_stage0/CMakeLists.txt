cmake_minimum_required(VERSION 3.25.2)

project(fideslib_stage0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

option(FHE_STAGE0_BUILD_KERNEL "Build the FIDESlib Stage 0 kernel" ON)
option(FHE_STAGE0_BUILD_TESTS "Build FIDESlib-free Stage 0 layout tests" OFF)

add_library(stage0_layout INTERFACE)
target_include_directories(stage0_layout INTERFACE include)

if(FHE_STAGE0_BUILD_KERNEL)
  find_package(fideslib REQUIRED CONFIG)

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
endif()

if(FHE_STAGE0_BUILD_TESTS)
  enable_testing()
  add_executable(test_stage0_layout tests/test_stage0_layout.cpp)
  target_link_libraries(test_stage0_layout PRIVATE stage0_layout)
  add_test(NAME stage0_layout COMMAND test_stage0_layout)
endif()
