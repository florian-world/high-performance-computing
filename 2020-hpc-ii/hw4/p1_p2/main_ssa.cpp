#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>
#include <chrono>

#include "ArgumentParser.hpp"
#include "SSA_CPU.hpp"
#include "SSABenchmark.hpp"

typedef std::chrono::system_clock Clock;

using namespace std;


int main(int argc, const char ** argv)
{
  ArgumentParser parser(argc, argv);

  // params:
  // -omega -> number of molecules -> i.e. result of i means i*omega mol. (def = 1)
  // -samples -> how many samples of stoch. algo. to run (def = 2000)
  // -T -> end time for algo (def = 5)
  // -dt -> size of bin for trajectory binning (def = 0.1)
  // -seed -> seed for random number generator (def = random)
  // -runs -> how many repetitions for benchmarking (def = 100)

  auto now = Clock::now();
  auto ms = std::chrono::time_point_cast<std::chrono::milliseconds>(now);
  auto fraction = now - ms;

  // get params with def. values
  int omega = parser("-omega").asInt(10);
  int samples = parser("-samples").asInt(2000);
  double T = parser("-T").asDouble(5.0);
  double dt = parser("-dt").asDouble(0.1);
  int rnd_seed = parser("-seed").asInt(fraction.count());
  int runs = parser("-runs").asInt(100);

  srand48(rnd_seed);
  SSABenchmark::timeIt<SSA_CPU>(omega, samples, T, dt, runs);

  return 0;
}
