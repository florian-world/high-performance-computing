#include <stdio.h>
#include <stdlib.h>
#include <random>
#include <fstream>
#include <iostream>
#include <cmath>
#include <omp.h>

#include "cacheflusher.h"


// evolution time
double tmax = 0.2;
// number of bins in histogram
size_t nb = 50;
// range of histogram
double xmin = -2;
double xmax = 2;

// hh: histogram, number of particles in each bin
// name: filename
void WriteHistogram(const std::vector<double>& hh, std::string name) {
  std::ofstream o(name);
  size_t nb = hh.size();
  for (size_t j = 0; j < nb; ++j) {
    double x = xmin + (xmax - xmin) * (j + 0.5) / nb;
    o << x << " " << hh[j] << "\n";
  }
}

// Returns current wall-clock time in seconds
double GetWtime() {
  return omp_get_wtime();
}

// TODO 3: parallelize with OpenMP
// Returns histogram of positions xx in range [xmin,xmax] with nb bins
std::vector<double> GetHistogram(const std::vector<double>& xx) {
  std::vector<double> hh(nb, 0);
  for (size_t i = 0; i < xx.size(); ++i) {
    int j = (xx[i] - xmin) / (xmax - xmin) * nb;
    j = std::max(0, std::min(int(nb) - 1, j));
    hh[j] += 1;
  }
  return hh;
}

int main(int argc, char *argv[]) {
  if (argc < 2 || argc > 3 || std::string(argv[1]) == "-h") {
    fprintf(stderr, "open threads: %d\n", omp_get_num_threads()); // REMOVE ME
    fprintf(stderr, "usage: %s N M\n", argv[0]);
    fprintf(stderr, "Brownian motion of N paritcles in M steps in time");
    return 1;
  }

  // number of particles
  size_t N = atoi(argv[1]);
  // number of steps
  size_t M = atoi(argv[2]);
  // time step
  double dt = tmax / M;

  CacheFlusher cf;


  for (int k = 1; k <= 24; ++k) {
    cf.flush();
    omp_set_num_threads(k);

    std::vector<double> xx(N);
    double wt0, wt1;



    // seed
  #pragma omp parallel shared(xx)
    {
      std::default_random_engine gen;

      auto random = gen();
      // Seed generator
      int seed = (omp_get_thread_num() + 1) * random % gen.max();
      gen.seed(seed);

      // parallel initialization
      std::uniform_real_distribution<double> dis(-0.5, 0.5);

  #pragma omp for
      for (size_t i = 0; i < N; ++i) {
        xx[i] = dis(gen);
      }

  #pragma omp single
      {
        WriteHistogram(GetHistogram(xx), "hist_0.dat");

        // Perform M steps of random walks and measure time
        wt0 = GetWtime();
      }

      std::normal_distribution<double> dis2(0., std::sqrt(dt));

  #pragma omp for
      for (size_t i = 0; i < N; ++i) {
        for (size_t m = 0; m < M; ++m) {
          xx[i] += dis2(gen);
        }
      }

  #pragma omp single
      {
        wt1 = GetWtime();
        double wtime_walk = wt1 - wt0;

        // Compute histogram and measure time
        wt0 = GetWtime();
        auto hh = GetHistogram(xx);
        wt1 = GetWtime();
        double wtime_hist = wt1 - wt0;

        WriteHistogram(hh, "hist_1.dat");

        printf("walk & hist with %2d threads: %.16f %.16f\n", omp_get_num_threads(), wtime_walk, wtime_hist);
//        printf("hist with %2d threads: \n", omp_get_num_threads(), );
        fflush(stdout);
      }
    }
  }
}
