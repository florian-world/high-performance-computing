#ifndef _DIRECT_HPP_
#define _DIRECT_HPP_

#include "korali.hpp"
#include "SSA_CPU.hpp"

void direct(korali::Sample& k)
{
  double k1 = k["Parameters"][0];
  double k2 = k["Parameters"][1];
  double k3 = k["Parameters"][2];
  double k4 = k["Parameters"][3];

  // TODO: TASK 2b)
  //    - Initialize SSA_CPU class
  //    - set the rates k1, k2, k3, and k4
  //    - run
  //    - get S1 and S2
  //    - calculate objective function

  double sse = 0.0; // TODO

  k["F(x)"] = -sse;
}


#endif
