cmake_minimum_required(VERSION 2.8)

set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(adi)

set( project_sources
   diffusionADI.cpp
)

# Create executable with all necessary source files
add_executable(${PROJECT_NAME}
  ${project_sources}
)

find_package(OpenMP)
target_link_libraries(${PROJECT_NAME} PUBLIC OpenMP::OpenMP_CXX)

