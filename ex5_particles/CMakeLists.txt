cmake_minimum_required(VERSION 2.8)

set(CMAKE_CXX_FLAGS_RELEASE "-O3")
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(particles)

set( project_sources
	 ArrayOfParticles.h
   Profiler.h
   SerialParticlesIterator.h
# 	 main.cpp put in executable to destinguish with mpi version
)

# Create executable with all necessary source files
add_executable(${PROJECT_NAME}
	main.cpp
  ${project_sources}
)

find_package(OpenMP REQUIRED)
find_package(MPI REQUIRED)

target_include_directories(${PROJECT_NAME} PUBLIC OpenMP::OpenMP_CXX)
target_link_libraries(${PROJECT_NAME} PUBLIC OpenMP::OpenMP_CXX)


add_executable(
 ${PROJECT_NAME}_mpi
 main_mpi.cpp
 ${project_sources}
)

target_include_directories(${PROJECT_NAME}_mpi PUBLIC MPI::MPI_CXX OpenMP::OpenMP_CXX)
target_link_libraries(${PROJECT_NAME}_mpi PUBLIC MPI::MPI_CXX OpenMP::OpenMP_CXX)
