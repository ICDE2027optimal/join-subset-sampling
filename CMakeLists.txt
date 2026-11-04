cmake_minimum_required(VERSION 3.20)
project(product_subset_sampling LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build mode" FORCE)
endif()

add_library(sampling_headers INTERFACE)
target_include_directories(sampling_headers INTERFACE
  ${PROJECT_SOURCE_DIR}/include)
target_include_directories(sampling_headers SYSTEM INTERFACE
  ${PROJECT_SOURCE_DIR}/third_party/ac-library)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  target_compile_options(sampling_headers INTERFACE -Wall -Wextra -Wpedantic)
endif()

add_executable(product_sample src/sample.cpp)
target_link_libraries(product_sample PRIVATE sampling_headers)

option(BUILD_EXPERIMENTS "Build paper measurement drivers" ON)
if(BUILD_EXPERIMENTS)
  foreach(target formal_driver materialized_explicit_driver)
    add_executable(${target} src/${target}.cpp)
    target_link_libraries(${target} PRIVATE sampling_headers)
  endforeach()
endif()

include(CTest)
if(BUILD_TESTING)
  foreach(target core_correctness orient_correctness loader_correctness
                 materialized_explicit_test rank_scan_controls_test)
    add_executable(${target} tests/${target}.cpp)
    target_link_libraries(${target} PRIVATE sampling_headers)
    if(target STREQUAL "loader_correctness")
      add_test(NAME ${target} COMMAND ${target} ${PROJECT_SOURCE_DIR}/tests/fixtures/tiny)
    else()
      add_test(NAME ${target} COMMAND ${target})
    endif()
    set_tests_properties(${target} PROPERTIES TIMEOUT 120)
  endforeach()
  find_package(Python3 3.10 COMPONENTS Interpreter REQUIRED)
  add_test(NAME sampling_cli COMMAND ${Python3_EXECUTABLE}
    ${PROJECT_SOURCE_DIR}/tests/test_cli.py --binary $<TARGET_FILE:product_sample> -v)
  set_tests_properties(sampling_cli PROPERTIES TIMEOUT 120)
  add_test(NAME artifact_tools COMMAND ${Python3_EXECUTABLE}
    ${PROJECT_SOURCE_DIR}/tests/test_artifact_tools.py -v)
  set_tests_properties(artifact_tools PROPERTIES TIMEOUT 120)
endif()
