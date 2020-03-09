#include "wave.h"

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);

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

  int points = atoi(argv[1]);
  int procs_per_dim = atoi(argv[2]);
  double t_end = atof(argv[3]);

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

  WaveEquation *Simulation = new WaveEquation(points, procs_per_dim);

  double start = MPI_Wtime();
  Simulation->run(t_end);
  double finish = MPI_Wtime();

  if (rank == 0)
    std::cout << "Total time = " << finish - start << "\n";

  delete Simulation;

  MPI_Finalize();
  return 0;
}