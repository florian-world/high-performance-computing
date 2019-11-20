#include <openmpi/mpi.h>
#include <iostream>

#include "diffusion2d.h"
#include "timer.h"

using namespace hpcse;

//#define _PERF_

int main(int argc, char *argv[]) {
  if (argc < 5) {
      std::cerr << "Usage: " << argv[0] << " D L N dt\n";
      return 1;
  }

  // TODO: Start-up the MPI environment and determine this process' rank ID as
  // well as the total number of processes (=ranks) involved in the
  // communicator

  int rank, procs;

  // *** start MPI part ***

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &procs);

  // *** end MPI part ***

  const double D = std::stod(argv[1]);
  const double L = std::stod(argv[2]);
  const int N = std::stoi(argv[3]);
  const double dt = std::stod(argv[4]);

  Diffusion2d system(D, L, N, dt, rank, procs);
  system.compute_diagnostics(0);

  Timer t;
  t.start();
  for (int step = 0; step < 10000; ++step) {
      system.advance();
#ifndef _PERF_
      system.compute_diagnostics(dt * step);
#endif
  }
  t.stop();

  if (rank == 0)
      std::cout << "Timing: " << N << ' ' << t.get_timing() << '\n';

  system.compute_histogram_hybrid();

#ifndef _PERF_
  if (rank == 0)
      system.write_diagnostics("diagnostics_mpi.dat");
#endif

  // TODO: Shutdown the MPI environment
  // *** start MPI part ***
  MPI_Finalize();
  // *** end MPI part ***

  return 0;
}
