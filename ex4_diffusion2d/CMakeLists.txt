cmake_minimum_required(VERSION 2.8)


set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(diffusion2d)

set( project_sources
   diffusion2d.cpp
   main.cpp
)

# Create executable with all necessary source files
add_executable(${PROJECT_NAME}
  ${project_sources}
)

find_package(OpenMP REQUIRED)
find_package(MPI REQUIRED)

target_include_directories(${PROJECT_NAME} PUBLIC MPI::MPI_CXX OpenMP::OpenMP_CXX)
target_link_libraries(${PROJECT_NAME} PUBLIC MPI::MPI_CXX OpenMP::OpenMP_CXX)


set( project_headers
    diffusion2d.h
    timer.h
)

add_custom_target(headers SOURCES ${project_headers})
