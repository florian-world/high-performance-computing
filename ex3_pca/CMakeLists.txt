cmake_minimum_required(VERSION 2.8)



list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)

set(CMAKE_CXX_FLAGS_RELEASE "-O3")

project(pca)




set( project_sources
   main.cpp
   cacheflusher.cpp
)

# Create executable with all necessary source files
add_executable(${PROJECT_NAME}
  ${project_sources}
)


set(BLA_DEFINITIONS)

if ("$ENV{MKLROOT}" STREQUAL "")
    message(FATAL_ERROR "MKLROOT is not set. Please source the Intel MKL mklvars.sh file.")
endif()

# user defined options for MKL
option(MKL_USE_parallel "Use MKL parallel" True)
option(MKL_USE_sdl "Single Dynamic Library or static/dynamic" False)
set(MKL_USE_interface "lp64" CACHE STRING "for Intel(R)64 compatible arch: ilp64/lp64 or for ia32 arch: cdecl/stdcall")

if (BLA_VENDOR MATCHES "_seq")
    set(MKL_USE_parallel OFF)
else()
    set(MKL_USE_parallel ON)
endif()


find_package(OpenMP)
find_package(MKL REQUIRED)
include_directories(${MKL_INCLUDE_DIR})
target_link_libraries(${PROJECT_NAME} ${MKL_LIBRARIES})
target_link_libraries(${PROJECT_NAME} PUBLIC OpenMP::OpenMP_CXX)

set( project_headers
    cacheflusher.h
)

add_custom_target(headers SOURCES ${project_headers})
