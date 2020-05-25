cmake_minimum_required(VERSION 3.10)

if(POLICY CMP0074)
  cmake_policy(SET CMP0074 NEW)
endif()

project(HW6_SSA LANGUAGES CXX CUDA)

find_package(CUDA 9.2 REQUIRED)

cuda_add_executable(
    ssa
    src/main.cpp
    src/ssa.cu
    src/kernels.cu
    src/test.cu
    src/utils.cpp)

target_compile_features(ssa PUBLIC cxx_std_14)
target_link_libraries(ssa curand)

# Choose RelWithDebInfo mode as default.
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE "RelWithDebInfo" CACHE STRING
    "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel." FORCE)
endif()
