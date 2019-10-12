#include <stdio.h>
#include <stdlib.h>
#include <random>
#include <fstream>
#include <iostream>
#include <cmath>
// TODO 1: add OpenMP headers


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

// TODO 1: implement using OpenMP
// Returns current wall-clock time in seconds
double GetWtime() {
  return 0;
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

  // Seed generator
  int seed = 19;
  std::default_random_engine gen(seed);

  // Initialize particles
  std::vector<double> xx(N);
  {
    std::uniform_real_distribution<double> dis(-0.5, 0.5);
    for (size_t i = 0; i < N; ++i) {
      xx[i] = dis(gen);
    }
  }

  WriteHistogram(GetHistogram(xx), "hist_0.dat");

  double wt0, wt1;

  // Perform M steps of random walks and measure time
  // TODO 2: parallelize with OpenMP
  double t = 0;
  wt0 = GetWtime();
  {
    for (size_t m = 0; m < M; ++m) {
      std::normal_distribution<double> dis(0., std::sqrt(dt));
      for (size_t i = 0; i < N; ++i) {
        xx[i] += dis(gen);
      }
      t += dt * M;
    }
  }
  wt1 = GetWtime();
  double wtime_walk = wt1 - wt0;

  // Compute histogram and measure time
  wt0 = GetWtime();
  auto hh = GetHistogram(xx);
  wt1 = GetWtime();
  double wtime_hist = wt1 - wt0;

  WriteHistogram(hh, "hist_1.dat");

  printf("walk: %.16f\n", wtime_walk);
  printf("hist: %.16f\n", wtime_hist);
}
