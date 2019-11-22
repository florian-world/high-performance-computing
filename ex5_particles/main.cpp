#include <fstream>

#include "Profiler.h"

#include "SerialParticlesIterator.h"


using namespace SerialParticlesIterator;


int main (int argc, char ** argv)
{
    Profiler profiler;
    int mpi_rank=0, mpi_size=1;
    size_t global_n_particles = 5040;

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
                if(time > 0) profiler.printStatAndReset();
                printf("Iteration %lu - time %f - Total Circulation %f\n",
                       step, time, total_circulation);
            }
        }

        // advance time counters:
        time += dt;
        step ++;
    }

	return 0;
}
