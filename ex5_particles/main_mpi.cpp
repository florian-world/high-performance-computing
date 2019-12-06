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
    ArrayOfParticles particles_mpi = initialize_particles_1D(
        n_particles, start_x, end_x, gamma_init_1D);

    value_t time = 0;
    size_t step = 0;
    while (time < end_time)
    {
        // 0. init velocities to zero
        profiler.start("clear");
        reset_velocities(particles);
        profiler.stop("clear");

        // 1. compute local
        profiler.start("compute");
        compute_interaction(particles, particles);
        profiler.stop("compute");


        for (int i = 1; i < mpi_size; ++i) {
          auto rankTo = (mpi_rank + i) % mpi_size;
          auto rankFrom = (mpi_rank + mpi_size - i) % mpi_size;
//          auto toSendX = i > 0 ? particles_mpi.pos_x() : particles.pos_x();
//          auto toSendY = i > 0 ? particles_mpi.pos_y() : particles.pos_y();
//          auto toSendGamma = i > 0 ? particles_mpi.gamma() : particles.gamma();

          assert(n_particles == particles.size());

          MPI_Sendrecv(particles.pos_x(), n_particles, MPI_VALUE_T, rankTo, Tags::X,
                       particles_mpi.pos_x(), n_particles, MPI_VALUE_T, rankFrom, Tags::X, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
          MPI_Sendrecv(particles.pos_y(), n_particles, MPI_VALUE_T, rankTo, Tags::Y,
                       particles_mpi.pos_y(), n_particles, MPI_VALUE_T, rankFrom, Tags::Y, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
          MPI_Sendrecv(particles.gamma(), n_particles, MPI_VALUE_T, rankTo, Tags::Gamma,
                       particles_mpi.gamma(), n_particles, MPI_VALUE_T, rankFrom, Tags::Gamma, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
          compute_interaction(particles_mpi, particles, false);
        }

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
                if(time > 0) profiler.printStatAndReset();
                printf("Iteration %lu - time %f - Total Circulation %f\n",
                       step, time, total_circulation);
            }
        }

        // advance time counters:
        time += dt;
        step ++;
    }

  MPI_Finalize();

	return 0;
}
