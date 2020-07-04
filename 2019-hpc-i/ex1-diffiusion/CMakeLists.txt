cmake_minimum_required(VERSION 2.8)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(diffusion)

set(CMAKE_INCLUDE_CURRENT_DIR ON)

set( project_sources
   main.cpp
   cacheflusher.cpp
   diffusionsolver.cpp
)

# Create executable with all necessary source files
add_executable(${PROJECT_NAME}
  ${project_sources}
)

set( project_headers
    cacheflusher.h
    diffusionsolver.h
)

add_custom_target(headers SOURCES ${project_headers})
