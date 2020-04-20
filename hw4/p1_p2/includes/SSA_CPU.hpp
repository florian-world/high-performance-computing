/*
 * Execute dimetization with 0815 SSA on CPU.
 */

#pragma once

#include <numeric>
#include <vector>

#include "SSABase.hpp"
#include "Timer.hpp"

using namespace std;

class SSA_CPU : public SSABase
{
private:
  double k[4]; // rates

protected:
  Timer timer;
  int pass;

public:
  SSA_CPU(const int omega, const int num_samples, const double T, const double bin_dt)
    : SSABase(omega,num_samples,T,bin_dt)
    ,pass(0)
  {
    k[0] = 1;
    k[1] = 1;
    k[2] = 0.2/omega;
    k[3] = 20.0*omega;
  }

  void operator()();
  void normalize_bins();

  void setRates(double k1, double k2, double k3, double k4) {
    k[0] = k1;
    k[1] = k2;
    k[2] = k3/omega;
    k[3] = k4*omega;
  }
  double getS1() const {
    return trajS1.back()/omega ;
  };
  double getS2() const {
    return trajS2.back()/omega ;
  };

  double getTransfers() const;
  double getFlops() const;

  void startTiming() {
    timer.start();
  }
  void stopTiming() {
    time += timer.stop();
  }
};
