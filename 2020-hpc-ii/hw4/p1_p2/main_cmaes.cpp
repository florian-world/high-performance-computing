#include "includes/objective.hpp"
#include "korali.hpp"

#include "SSA_CPU.hpp"

int main(int argc, char* argv[])
{
// Configuring Problem.
  auto e = korali::Experiment();
  e["Problem"]["Type"] = "Optimization/Stochastic";
  e["Problem"]["Objective Function"] = &direct;

// Defining the problem's variables.
  e["Variables"][0]["Name"] = "k1";
  e["Variables"][0]["Lower Bound"] = 0.0;
  e["Variables"][0]["Upper Bound"] = 5.0;

  e["Variables"][1]["Name"] = "k2";
  e["Variables"][1]["Lower Bound"] = 0.0;
  e["Variables"][1]["Upper Bound"] = 5.0;

  e["Variables"][2]["Name"] = "k3";
  e["Variables"][2]["Lower Bound"] = 0.0;
  e["Variables"][2]["Upper Bound"] = 5.0;

  e["Variables"][3]["Name"] = "k4";
  e["Variables"][3]["Lower Bound"] = 0.0;
  e["Variables"][3]["Upper Bound"] = 50.0;


// Configuring CMA-ES parameters
  e["Solver"]["Type"] = "CMAES";
  e["Solver"]["Population Size"] = 8;
  e["Solver"]["Termination Criteria"]["Min Value Difference Threshold"] = 1e-7;
  e["Solver"]["Termination Criteria"]["Max Generations"] = 100;

  // Running Korali
  auto k = korali::Engine();
  k.run(e);
}
