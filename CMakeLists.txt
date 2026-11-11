cmake_minimum_required(VERSION 3.16)
project(CumulantCov
    VERSION 1.0.0
    DESCRIPTION "Header-only C++17 Delta-method and multinomial covariance estimation for cumulants"
    LANGUAGES CXX
)

option(CUMULANT_COV_BUILD_EXAMPLES "Build CumulantCov usage examples" ON)
option(CUMULANT_COV_BUILD_TESTS    "Build CumulantCov unit test suite" ON)

# Header-only INTERFACE target
add_library(CumulantCov INTERFACE)
add_library(CumulantCov::CumulantCov ALIAS CumulantCov)

target_include_directories(CumulantCov INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:include>
)
target_compile_features(CumulantCov INTERFACE cxx_std_17)

# Example executable
if(CUMULANT_COV_BUILD_EXAMPLES)
    add_executable(cumulant_demo examples/demo.cpp)
    target_link_libraries(cumulant_demo PRIVATE CumulantCov::CumulantCov)
    target_compile_options(cumulant_demo PRIVATE -O3 -Wall -Wextra -Wpedantic)
endif()

# Unit tests via CTest
if(CUMULANT_COV_BUILD_TESTS)
    enable_testing()
    add_executable(test_cumulant_cov tests/test_cumulant_cov.cpp)
    target_link_libraries(test_cumulant_cov PRIVATE CumulantCov::CumulantCov)
    target_compile_options(test_cumulant_cov PRIVATE -O3 -Wall -Wextra -Wpedantic)
    add_test(NAME CumulantCovTestSuite COMMAND test_cumulant_cov)
endif()