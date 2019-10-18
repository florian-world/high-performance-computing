cmake_minimum_required(VERSION 2.8)

set(CMAKE_CXX_FLAGS_RELEASE "-O3")

project(brownian_motion)
add_executable(${PROJECT_NAME} "main.cpp")


find_package(OpenMP)
target_link_libraries(${PROJECT_NAME} PUBLIC OpenMP::OpenMP_CXX)
