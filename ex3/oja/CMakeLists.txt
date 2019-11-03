cmake_minimum_required(VERSION 2.8)


set(CMAKE_CXX_FLAGS_RELEASE "-O3")

project(oja)

set(ON_EULER False)

set( project_sources
   main.cpp
   perceptron.cpp
   ../code/utils.cpp
)

# Create executable with all necessary source files
add_executable(${PROJECT_NAME}
  ${project_sources}
)


find_package(OpenMP)
target_link_libraries(${PROJECT_NAME} PUBLIC OpenMP::OpenMP_CXX)


# on euler
if (${ON_EULER})
  set(MKL_CXXFLAGS "-DMKL_ILP64 -m64 -I$ENV{MKLROOT}/include")
  set(MKL_LDFLAGS "-Wl,--start-group $ENV{MKLROOT}/lib/intel64/libmkl_intel_ilp64.a $ENV{MKLROOT}/lib/intel64/libmkl_gnu_thread.a $ENV{MKLROOT}/lib/intel64/libmkl_core.a -Wl,--end-group -lgomp -lpthread -lm -ldl")
  target_compile_options(${PROJECT_NAME} PUBLIC ${MKL_CXXFLAGS})
  target_link_libraries(${PROJECT_NAME} PUBLIC ${MKL_LDFLAGS})
else()
  list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
  include_directories(${MKL_INCLUDE_DIR})
  # user defined options for MKL
  option(MKL_USE_parallel "Use MKL parallel" True)
  option(MKL_USE_sdl "Single Dynamic Library or static/dynamic" False)
  set(MKL_USE_interface "lp64" CACHE STRING "for Intel(R)64 compatible arch: ilp64/lp64 or for ia32 arch: cdecl/stdcall")
  find_package(MKL REQUIRED)
  target_link_libraries(${PROJECT_NAME} ${MKL_LIBRARIES})
endif()

set(project_headers
    ../code/utils.h
    perceptron.h
)

add_custom_target(headers SOURCES ${project_headers})
