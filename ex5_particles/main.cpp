#include <fstream>
#include <iostream>
#include <sstream>
#include <cmath>

#include "Profiler.h"

#include "SerialParticlesIterator_parallel.h"

#include <omp.h>


using namespace SerialParticlesIterator;

int exec_main (int argc, char ** argv, int num_threads);


int main(int argc, char ** argv) {
  exec_main(argc,argv,1);
  int steps = 4;
  for (int i = steps; i <= 24; i+= steps) {
    exec_main(argc,argv,i);
  }
//  exec_main(argc,argv,8);
}

int exec_main (int argc, char ** argv, int num_threads)
{
  omp_set_num_threads(num_threads);
  std::stringstream csvPerfSS;

  Profiler profiler;
  int mpi_rank=0, mpi_size=1;

#define WEAK_SCALING
#ifdef WEAK_SCALING
  size_t global_n_particles = 3600 * static_cast<size_t>(round(sqrt(num_threads)));
#else
  size_t global_n_particles = 36;
#endif

  if (argc > 1) global_n_particles = std::stoul(argv[1]);
  if (mpi_rank==0)
  {
    printf("Simulating %lu particles.\n", global_n_particles);
    if (argc < 2) printf("To change N particles run as '%s N'\n", argv[0]);
  }

  const size_t n_particles = global_n_particles;
  const value_t extent_x   = 1.0;
  const value_t vol_p = 1.0 / global_n_particles;
  const value_t start_x = - 0.5;
  const value_t end_x = start_x + extent_x;

  // time integration setup:
  const value_t end_time = 2.5;
  const value_t print_freq = 0.1;
  value_t print_time = 0;
  const value_t dt = 0.0001;

  // initialize particles: position and circulation
  std::function<value_t(value_t)> gamma_init_1D = [&] (const value_t x)
  {
    return vol_p * 4 * x / std::sqrt(1 - 4 * x * x);
  };
  ArrayOfParticles particles = initialize_particles_1D(
        n_particles, start_x, end_x, gamma_init_1D);

  value_t time = 0;
  size_t step = 0;
  while (time < end_time)
  {
//    std::cout << "At time step: " << time << std::endl;
    // 0. init velocities to zero
    profiler.start("clear");
    reset_velocities(particles);
    profiler.stop("clear");

    // 1. compute local
    profiler.start("compute");
    compute_interaction(particles, particles);
    profiler.stop("compute");

    // 2. with new velocities, advect particles positions:
    profiler.start("advect");
    advect_euler(particles, dt);
    profiler.stop("advect");

    // 3. debug measure: make sure circulation stays constant
    profiler.start("sum gamma");
    value_t total_circulation = sum_circulation(particles);
    profiler.stop("sum gamma");

    if ((time+dt) > print_time)
    {
      print_time += print_freq;
      {   // print to file
        profiler.start("fwrite");
        const std::string config = particles.convert2csv();
        char fname[128]; sprintf(fname, "config_%06lu.csv", step);
        FILE * fout = fopen(fname, "wb");
        fwrite (config.c_str(), sizeof(char), config.size(), fout);
        fclose (fout);
        profiler.stop("fwrite");
      }

      if (mpi_rank == 0)
      {
        if(time > 0) profiler.printStatAndReset(false);
        printf("Iteration %lu - time %f - Total Circulation %f\n",
               step, time, total_circulation);

        if (time > 0) {
          profiler.writeCsvLine(csvPerfSS);
        } else {
          profiler.writeCsvHeaders(csvPerfSS);
        }
      }
    }

    // advance time counters:
    time += dt;
    step ++;
  }




#ifdef WEAK_SCALING
  char fname[128]; sprintf(fname, "perf_weak_nthreads_%02d.csv", num_threads);
#else
  char fname[128]; sprintf(fname, "perf_strong_nthreads_%02d.csv", num_threads);
#endif
  std::ofstream outFile;
  outFile.open(fname);
  outFile << csvPerfSS.rdbuf();
  outFile.close();

  return 0;
}
