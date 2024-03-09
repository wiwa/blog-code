cmake_minimum_required(VERSION 3.27)

enable_language(CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_SIZEOF_VOID_P EQUAL 8)
  message(FATAL_ERROR "Build MUST be 64-bit: cmake . -DCMAKE_GENERATOR_PLATFORM=x64")
endif()

### Project

project(Someproj LANGUAGES CXX CUDA)
cmake_path(GET PROJECT_SOURCE_DIR PARENT_PATH PARENT_DIR)
set(PROJECT_SOURCE_DIR "${PARENT_DIR}/src")

message("[CMakeInfo]")
message("[CMakeInfo] PROJECT_SOURCE_DIR: ${PROJECT_SOURCE_DIR}")
message("[CMakeInfo] CMAKE_BINARY_DIR: ${CMAKE_BINARY_DIR}")
message("[CMakeInfo] CMAKE_SOURCE_DIR: ${CMAKE_SOURCE_DIR}")
message("[CMakeInfo]")

set(SRC "${PROJECT_SOURCE_DIR}")

### Build options

option(COMPILE_WARNINGS "COMPILE_WARNINGS" OFF)

### Env variables and Cmake settings

if(NOT DEFINED ENV{CUDA_HOME})
  if(DEFINED ENV{CONDA_PREFIX})
    message("Setting CUDA_HOME to CONDA_PREFIX: $ENV{CONDA_PREFIX}")
    set(ENV{CUDA_HOME} "$ENV{CONDA_PREFIX}")
  endif()
endif()

message("Using CUDA_HOME: $ENV{CUDA_HOME}")

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 86 89)
endif()
set_property(GLOBAL PROPERTY THRUST_INCLUDE_CUB_CMAKE ON)

if (UNIX)
  message("Detected UNIX")
  set(CMAKE_CUDA_COMPILER "$ENV{CUDA_HOME}/bin/nvcc")
  set(CMAKE_CUDA_COMPILER_TOOLKIT_ROOT "$ENV{CUDA_HOME}")
  set(CMAKE_CUDA_COMPILER_LIBRARY_ROOT "$ENV{CUDA_HOME}/lib64")
  set(CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES "$ENV{CUDA_HOME}/include")
  set(CUDA_CUDART "$ENV{CUDA_HOME}/lib64/libcudart.so")
  message("Using CMAKE_CUDA_COMPILER_LIBRARY_ROOT=${CMAKE_CUDA_COMPILER_LIBRARY_ROOT}")
endif (UNIX)
if (WIN32)
message("Detected WIN32")
  set(CMAKE_GENERATOR_PLATFORM x64)
endif (WIN32)

### Dependencies

# https://github.com/cpm-cmake/CPM.cmake/wiki/Downloading-CPM.cmake-in-CMake
# download CPM.cmake
file(
  DOWNLOAD
  https://github.com/cpm-cmake/CPM.cmake/releases/download/v0.38.3/CPM.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/cmake/CPM.cmake
  EXPECTED_HASH SHA256=cc155ce02e7945e7b8967ddfaff0b050e958a723ef7aad3766d368940cb15494
)
include(${CMAKE_CURRENT_BINARY_DIR}/cmake/CPM.cmake)

CPMAddPackage(
  NAME cxxopts
  GITHUB_REPOSITORY jarro2783/cxxopts
  VERSION 3.2.0
  OPTIONS
    "CXXOPTS_BUILD_EXAMPLES Off"
    "CXXOPTS_BUILD_TESTS Off"
)

### Compile flags

if (NOT COMPILE_WARNINGS)
  # set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-all")
  add_definitions(-w)
endif()

### Macros and Functions

macro(SetupCuda TARGET)
  set_target_properties(${TARGET} PROPERTIES LINKER_LANGUAGE CUDA)
  set_target_properties(${TARGET} PROPERTIES CUDA_ARCHITECTURES "86")
  set_target_properties(${TARGET} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

  # https://forums.developer.nvidia.com/t/passing-flags-to-nvcc-via-cmake/75768/2
  target_compile_options(${TARGET}
                        PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                        --generate-line-info
                        --use_fast_math
                        --relocatable-device-code=true
                        --extended-lambda
                        --expt-relaxed-constexpr
                        >)

  if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(${TARGET}
                          PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:
                          -g
                          -G
                          -DDEBUG=1
                          >)
  endif()
endmacro()

### Targets

add_executable(
  BenchSort
  ${SRC}/bench_sort.cu
  ${SRC}/data.cuh
  ${SRC}/sort_impl.cuh
  ${SRC}/utils.cuh
)
SetupCuda(BenchSort)
target_link_libraries(BenchSort cxxopts)
