#ifndef _DIRECT_HPP_
#define _DIRECT_HPP_

#include "korali.hpp"
#include "SSA_CPU.hpp"

#define SQUARE(x) ((x)*(x))

void direct(korali::Sample& k)
{
  double k1 = k["Parameters"][0];
  double k2 = k["Parameters"][1];
  double k3 = k["Parameters"][2];
  double k4 = k["Parameters"][3];

  SSA_CPU ssa(10, 2000, 5.0, 0.1, // omega, numSamples, T, dt
              k1, k2, k3, k4);

  ssa();

  auto S1 = ssa.getS1();
  auto S2 = ssa.getS2();

  // TODO: TASK 2b)
  //    - Initialize SSA_CPU class
  //    - set the rates k1, k2, k3, and k4
  //    - run
  //    - get S1 and  S2
  //    - calculate objective function

  double sse = SQUARE(S1 - 15) + SQUARE(S2 - 5);

  k["F(x)"] = -sse;
}


#endif
