#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>
#include <chrono>

#include "ArgumentParser.hpp"
#include "SSA_CPU.hpp"

typedef std::chrono::system_clock Clock;

using namespace std;


int main(int argc, const char ** argv)
{
  if (argc != 6) {
    fprintf(stderr, "Usage: %s k1 k2 k3 k4 n\n", argv[0]);
    exit(1);
  }

  auto k1 = (double) atof(argv[1]);
  auto k2 = (double) atof(argv[2]);
  auto k3 = (double) atof(argv[3]);
  auto k4 = (double) atof(argv[4]);
  int n = atoi(argv[5]);

  printf("Running with k1 = %f, k2 = %f, k3 = %f k4 = %f\n\n", k1, k2, k3, k4);

  double sum1 = 0.0;
  double sum2 = 0.0;

  for (int i = 0; i < n; ++i) {
    SSA_CPU ssa(10, 2000, 5.0, 0.1, // omega, numSamples, T, dt
              k1, k2, k3, k4);

    ssa();

    auto S1 = ssa.getS1();
    auto S2 = ssa.getS2();

    sum1 += S1;
    sum2 += S2;

    printf("Output %2d: S1 = %f, S2 = %f\n", (i+1), S1, S2);
  }

  printf("\n Average: S1 = %f, S2 = %f\n\n", sum1/n, sum2/n);


  return 0;
}
