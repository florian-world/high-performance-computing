#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <cmath>
#include <assert.h>
#include <algorithm>
#include <vector>
#include <fstream>
#include <iomanip>
#include <iostream>

class SSABase
{
protected:
  /** Multiplier for number of molecules. */
  const int omega;
  /** Number of samples of stochastic algo to run. */
  const int numSamples;
  /** End time for algorithm. */
  const double tend;
  /** Width of trajectory bin. */
  const double bin_dt;

  /** Trajectories (binned) for S1*omega and S2*omega (trajS(i) = avg of S for t in [i,i+1]*bin_dt). */
  std::vector<double> trajS1;
  std::vector<double> trajS2;
  std::vector<int> ntraj;

  /** Store timing result here. */
  double time;

public:
  SSABase(int _omega, int num_samples, double T, double bin_dt)
    : omega(_omega)
    , numSamples(num_samples)
    , tend(T)
    , bin_dt(bin_dt)
    , trajS1(static_cast<unsigned int>(std::ceil(T/bin_dt)))
    , trajS2(static_cast<unsigned int>(std::ceil(T/bin_dt)))
    , ntraj(static_cast<unsigned int>(std::ceil(T/bin_dt)))
    , time(0.0)
  { }

  /** Return time taken by start/stopTiming. */
  double getTime() {
    return time;
  }
  /** Get omega. */
  double getOmega() {
    return omega;
  }

  /**
   * Dump trajectories to file (nbins rows and 2 columns).
   */
  void dumpTrajToFile(std::string filename) {
    // Open file stream
    std::ofstream outfile(filename.c_str());

    // Dump it
    const int nbins = trajS1.size();
    int totalevals = 0;
    for (int i = 0; i < nbins; ++i) {
      // must rescale
      outfile << i*bin_dt+bin_dt/2 << " "
              << (trajS1[i] / omega) << " "
              << (trajS2[i] / omega) << std::endl;
      totalevals += ntraj[i];
    }
    std::cout << "Average number of time steps per sample: " << double(totalevals)/numSamples << std::endl;
  }
};
