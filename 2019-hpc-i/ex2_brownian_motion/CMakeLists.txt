cmake_minimum_required(VERSION 2.8)

set(CMAKE_CXX_FLAGS_RELEASE "-O3")

project(brownian_motion)




set( project_sources
   main.cpp
   cacheflusher.cpp
)

# Create executable with all necessary source files
add_executable(${PROJECT_NAME}
  ${project_sources}
)

find_package(OpenMP)
target_link_libraries(${PROJECT_NAME} PUBLIC OpenMP::OpenMP_CXX)

set( project_headers
    cacheflusher.h
)

add_custom_target(headers SOURCES ${project_headers})
