cmake_minimum_required(VERSION 3.16)
project(blend_bench CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)   # default Release = -O3 -DNDEBUG
endif()

include(FetchContent)
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "" FORCE)
set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "" FORCE)
FetchContent_Declare(
  benchmark
  GIT_REPOSITORY https://github.com/google/benchmark.git
  GIT_TAG v1.9.1
)
FetchContent_MakeAvailable(benchmark)

add_executable(blend_bench bench.cpp)
target_link_libraries(blend_bench PRIVATE benchmark::benchmark)

# coverage: every atlas mode has a table-free SIMD kernel, verified vs the atlas
add_executable(blend_coverage coverage.cpp)
target_compile_options(blend_coverage PRIVATE -O3)
# keep the scalar baseline genuinely scalar (explicit NEON intrinsics are unaffected),
# so the benchmark measures hand-SIMD vs scalar rather than hand-SIMD vs auto-SIMD.
target_compile_options(blend_bench PRIVATE -O3 -fno-vectorize -fno-slp-vectorize)
