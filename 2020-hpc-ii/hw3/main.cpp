#include "wave.h"

int main(int argc, char **argv) {
  int provided;
  MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided);

  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  if (argc != 4) {
    if (rank == 0) {
      std::cout << " Incorrect number of inputs. Run as:\n";
      std::cout << " mpirun -n x ./main N y t \n";
      std::cout << " where \n";
      std::cout << " N= number of grid points per direction\n";
      std::cout << " y= number of ranks per direction (=(x)^(1/3))\n";
      std::cout << " t= final time of simulation \n";
      std::cout << " Aborting...\n";
    }
    int err = 1;
    MPI_Abort(MPI_COMM_WORLD, err);
  }

  int points = std::stoi(argv[1]);
  int procs_per_dim = std::stoi(argv[2]);
  double t_end = std::stof(argv[3]);

  if (size != procs_per_dim * procs_per_dim * procs_per_dim) {
    if (rank == 0)
      std::cout << " Incorrect number of ranks per direction. Aborting... \n";
    int err = 2;
    MPI_Abort(MPI_COMM_WORLD, err);
  }

  if (points % procs_per_dim != 0) {
    if (rank == 0)
      std::cout << " Grid points per direction must be divisible by the number "
                   "of ranks per direction. Aborting... \n";
    int err = 3;
    MPI_Abort(MPI_COMM_WORLD, err);
  }

  int nthreads = omp_get_max_threads();
  int threads_per_dim = pow(nthreads, 1.0 / 3.0);
  if (nthreads != threads_per_dim * threads_per_dim * threads_per_dim) {
    if (rank == 0)
      std::cout
          << " Number of OPENMP threads must be a cubic number. Aborting... \n";
    int err = 4;
    MPI_Abort(MPI_COMM_WORLD, err);
  }

  WaveEquation Simulation(points, procs_per_dim);

  Simulation.run(t_end);

  MPI_Finalize();
  return 0;
}
