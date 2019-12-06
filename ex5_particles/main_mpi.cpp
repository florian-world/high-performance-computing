#include <fstream>
#include <openmpi/mpi.h>

#include "Profiler.h"

#include "SerialParticlesIterator.h"
using namespace SerialParticlesIterator;

enum Tags {
  X = 19284,
  Y,
  Gamma
};

int main (int argc, char ** argv)
{
  Profiler profiler;
  std::stringstream csvPerfSS;
  int mpi_rank=0, mpi_size=1;
  size_t global_n_particles = 360;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &mpi_rank);
  MPI_Comm_size(MPI_COMM_WORLD, &mpi_size);

  if (argc > 1) global_n_particles = std::stoul(argv[1]);
  if (mpi_rank==0)
  {
    printf("Simulating %lu particles with %d ranks.\n", global_n_particles, mpi_size);
    if (argc < 2) printf("To change N particles run as '%s N'\n", argv[0]);
  }


  assert(global_n_particles % static_cast<size_t>(mpi_size) == 0);
  const size_t n_particles = global_n_particles/static_cast<size_t>(mpi_size);
  const value_t extent_x   = 1.0 / mpi_size;
  const value_t vol_p = 1.0 / global_n_particles;
  const value_t start_x = - 0.5 + extent_x*mpi_rank;
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
  ArrayOfParticles particles_mpi1 = initialize_particles_1D(
        n_particles, start_x, end_x, gamma_init_1D);
  ArrayOfParticles particles_mpi2 = initialize_particles_1D(
        n_particles, start_x, end_x, gamma_init_1D);

  value_t time = 0;
  size_t step = 0;
  while (time < end_time)
  {
    // 0. init velocities to zero
    profiler.start("clear");
    reset_velocities(particles);
    profiler.stop("clear");

    profiler.start("compute");

    for (int i = 1; i < mpi_size; ++i) {
      auto rankTo = (mpi_rank + i) % mpi_size;
      auto rankFrom = (mpi_rank + mpi_size - i) % mpi_size;

      assert(n_particles == particles.size());

      MPI_Request mpiReqs[6];

      auto &particles_mpi = (i%2==0) ? particles_mpi1 : particles_mpi2;
      auto &particles_calc = i == 1 ? particles : ((i%2==0) ? particles_mpi2 : particles_mpi1);

      MPI_Isend(particles.pos_x(), n_particles, MPI_VALUE_T, rankTo, Tags::X, MPI_COMM_WORLD, &mpiReqs[0]);
      MPI_Irecv(particles_mpi.pos_x(), n_particles, MPI_VALUE_T, rankFrom, Tags::X, MPI_COMM_WORLD, &mpiReqs[3]);

      MPI_Isend(particles.pos_y(), n_particles, MPI_VALUE_T, rankTo, Tags::Y, MPI_COMM_WORLD, &mpiReqs[1]);
      MPI_Irecv(particles_mpi.pos_y(), n_particles, MPI_VALUE_T, rankFrom, Tags::Y, MPI_COMM_WORLD, &mpiReqs[4]);

      MPI_Isend(particles.gamma(), n_particles, MPI_VALUE_T, rankTo, Tags::Gamma, MPI_COMM_WORLD, &mpiReqs[2]);
      MPI_Irecv(particles_mpi.gamma(), n_particles, MPI_VALUE_T, rankFrom, Tags::Gamma, MPI_COMM_WORLD, &mpiReqs[5]);

      compute_interaction(particles_calc, particles, i == 1);

      // synchronize to be able to use the buffers again...
      MPI_Waitall(6, mpiReqs, MPI_STATUSES_IGNORE);

      if (i == mpi_size-1) {
        // compute last one directly here
        compute_interaction(particles_mpi, particles, i == 1);
      }
    }
    profiler.stop("compute");

    // 2. with new velocities, advect particles positions:
    profiler.start("advect");
    advect_euler(particles, dt);
    profiler.stop("advect");

    // 3. debug measure: make sure circulation stays constant
    profiler.start("sum gamma");
    value_t total_circulation = sum_circulation(particles);
    MPI_Allreduce(&total_circulation, &total_circulation, 1, MPI_VALUE_T, MPI_SUM, MPI_COMM_WORLD);
    profiler.stop("sum gamma");

    if ((time+dt) > print_time)
    {
      print_time += print_freq;

      {   // print to file
        profiler.start("fwrite");

        const std::string config = particles.convert2csv();
        char fname[128]; sprintf(fname, "config_%06lu.csv", step);

        MPI_File file;
        MPI_File_open(MPI_COMM_WORLD, fname, MPI_MODE_CREATE | MPI_MODE_WRONLY, MPI_INFO_NULL, &file);
        MPI_File_write_ordered(file, config.c_str(), config.size(), MPI_CHAR, MPI_STATUS_IGNORE);
        MPI_File_close(&file);

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

  if (mpi_rank == 0) {
    char fname[128]; sprintf(fname, "perf_mpi_r_%d_n_%zu.csv", mpi_size, global_n_particles);
    std::ofstream outFile;
    outFile.open(fname);
    outFile << csvPerfSS.rdbuf();
    outFile.close();
  }

  MPI_Finalize();

  return 0;
}
