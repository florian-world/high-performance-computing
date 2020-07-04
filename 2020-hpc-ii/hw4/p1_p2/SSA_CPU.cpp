#include "SSA_CPU.hpp"
#ifdef _OPENMP
#include <omp.h>
#endif
#include <cmath>

void SSA_CPU::operator()()
{
  // number of reactions
  const int m = 4;
  // number of species
  const int n = 2;
  // initial conditions
  const int S0[n] = {4*omega,0};

  const int niters = static_cast<int>(tend*1000);

  double * const r48  = new double[2*niters*numSamples];
  double * const curT = new double[numSamples];
  double * const x0 = new double[numSamples];
  double * const x1 = new double[numSamples];

  // NUMA aware initialization (first touch)
#ifdef _OPENMP
  #pragma omp parallel for
#endif
  for (int s=0; s<numSamples; s++)
  {
    curT[s] = 0.0;
    x0[s] = 0.0;
    x1[s] = 0.0;
    for (int iter=0; iter<niters; iter++)
    {
      r48[2*s*niters + iter*2    ] = 0.;
      r48[2*s*niters + iter*2 + 1] = 0.;
    }
  }

  bool bNotDone = true;
  pass = 0;

  while (bNotDone)
  {
#ifdef _OPENMP
    #pragma omp parallel for
#endif
    for (int i=0; i<niters*2*numSamples; i++)
      r48[i] = drand48();

    startTiming();
#ifdef _OPENMP
    int num_threads;
    #pragma omp parallel
    #pragma omp single
    {
      num_threads = omp_get_num_threads();
    }
#else
    const int num_threads = 1;
#endif

    const int nbins = trajS1.size();
    double * const trajS1L = new double[nbins*num_threads];
    double * const trajS2L = new double[nbins*num_threads];
    int    * const ntrajL  = new int[nbins*num_threads];

    // NUMA aware initialization (first touch)
#ifdef _OPENMP
    #pragma omp parallel for
#endif
    for(int t=0; t <num_threads; ++t)
    {
      for(int b=0; b <nbins; ++b)
      {
        trajS1L[t*nbins+b] = 0.0;
        trajS2L[t*nbins+b] = 0.0;
        ntrajL[t*nbins+b] = 0;                                    
      }
    }

    // printf("num_threads = %d, nbins = %d, numSamples = %d\n", num_threads, nbins, numSamples);

#ifdef _OPENMP
    #pragma omp parallel for
#endif
    for(int s = 0; s < numSamples; ++s)
    {
#ifdef _OPENMP
      const int thread_no = omp_get_thread_num();
#else
      const int thread_no = 0;
#endif
      // local version of trajectory bins
      const int nbins = trajS1.size();

      // init
      double time;
      double Sa;
      double Sb;
      if (pass>0 && bNotDone)
      {
        time = curT[s];
        Sa = x0[s];
        Sb = x1[s];
      }
      else
      {
        time = 0.0;
        Sa = S0[0];
        Sb = S0[1];
      }
      // propensities
      double a[m];

      // time stepping
      int iter = 0;
      while (time <= tend && iter<niters)
      {

        // store trajectory
        const int ib = static_cast<int>(time / bin_dt);         // 1 FLOP
        trajS1L[ib+thread_no*nbins] += Sa;
        trajS2L[ib+thread_no*nbins] += Sb;                      // 2 FLOP, 2 WRITE
        ++ntrajL[ib+thread_no*nbins];                           // 1 WRITE

        // TODO: Task 1a) (STEP 0) 
        //          - compute propensities a[0], a[1], .., a[3] and a0
        //          - use values Sa and Sb, and values stored in k[4], check initialization in SSA_CPU.hpp

        a[0] = k[0] * Sa;
        a[1] = k[1] * Sb;
        a[2] = k[2] * Sa * Sb;
        a[3] = k[3];                                            // 4 FLOP

        // compute cumulative sum
        for (int i = 1; i < m; ++i)
          a[i] += a[i-1];                                       // 3 FLOP

        // a0 simply last a[i]:
        const double& a0 = a[m-1];

        // TODO: Task 1a) (STEP 1)
        //          - sample tau using the inverse sampling method and increment time, use uniform random numbers initialized in r48

        const double& r1 = r48[2*s*niters + iter*2];
        const double& r2 = r48[2*s*niters + iter*2 + 1];        // 2 READ

        // tau ~ exp(a_0), so the rate of the exponential distribution is a_0

        time -= log1p(-r1) / a0;                                // 4 FLOP, considering: time - tau, 1-r1, log(), /a0

        // TODO: Task 1a) (STEP 2)
        //          - sample a reaction, use uniform random numbers initialized in r48

        double reaction = a0 * r2;                              // 1 FLOP

        // TODO: Task 1a) (STEP 3)
        //          - increment Sa, Sb

        // remind that a[i] contains the cmulative sum
        const bool react1 = reaction < a[0];
        const bool react2 = reaction >= a[0] && reaction < a[1];
        const bool react3 = reaction >= a[1] && reaction < a[2];
        const bool react4 = reaction >= a[2];

        Sa += -react1 + react3;
        Sb += -react2 -react3 +react4;                          // 5  FLOP
        iter++;
      }

      curT[s] = time;
      x0[s] = Sa;
      x1[s] = Sb;

      bNotDone = time <= tend && Sa!=0 && Sb!=0;
    }

    for(int t = 0; t < num_threads; ++t)
    {
      for (int i = 0; i < nbins; ++i) {
        trajS1[i] += trajS1L[i+t*nbins];
        trajS2[i] += trajS2L[i+t*nbins];
        ntraj[i] += ntrajL[i+t*nbins];                          // bins * (3 FLOP, 3 READ, 3 WRITE)         (assuming trajS1L, trajS2L, ntrajL) in cache
      }
    }

    delete[] ntrajL;
    delete[] trajS2L;
    delete[] trajS1L;
    stopTiming();

    pass++;
  }

  delete[] x1;
  delete[] x0;
  delete[] curT;
  delete[] r48;

  normalize_bins();
}

void SSA_CPU::normalize_bins()
{
  assert( trajS2.size() == trajS1.size() );
  assert( ntraj.size() == trajS1.size() );
  const int nbins = trajS1.size();

#ifdef _OPENMP
  #pragma omp parallel for
#endif
  for(int i=0; i < nbins; ++i)
  {
    trajS1[i]/=ntraj[i];
    trajS2[i]/=ntraj[i];                                        // 2 FLOP, 3 READ, 2 WRITE ---> NOTE: do not count this, not inside while(bNotDone)
  }
}
  
double SSA_CPU::getTransfers() const
{
  // TODO: (Optional) Task 1c)
  //          - return number of read writes in [BYTES]

#ifdef _OPENMP
  int num_threads;
  #pragma omp parallel
  #pragma omp single
  {
    num_threads = omp_get_num_threads();
  }
#else
  const int num_threads = 1;
#endif

  const int num_iter = std::accumulate(ntraj.begin(), ntraj.end(), 0); // already accounts for *numSamples
  const int& num_bins = ntraj.size();

  const int transfers_per_iter = 2*sizeof(double); // READ random

  const int transfers_per_bin_and_thread = (2+2)*sizeof(double) + (1+1)*sizeof(int); // 3 reads of trajS1[i], trajS2[i] and ntraj[i] + 3 writes

  return num_threads * num_bins * transfers_per_bin_and_thread + num_iter * transfers_per_iter;
}

double SSA_CPU::getFlops() const
{
  const int num_iter = std::accumulate(ntraj.begin(), ntraj.end(), 0); // already accounts for *numSamples
  const int& num_bins = ntraj.size();
  
#ifdef _OPENMP
  int num_threads;
  #pragma omp parallel
  #pragma omp single
  {
    num_threads = omp_get_num_threads();
  }
#else
  const int num_threads = 1;
#endif


  const int flops_per_iter = 20; // summing the flops inside while(... iter<niters)
  const int flops_per_bin_and_thread = 3; // summing up bins, normalization outside while loop  
  
  // TODO: (Optional) Task 1c)
  //          - return number of floating point operations
  return num_iter * flops_per_iter + flops_per_bin_and_thread * num_bins * num_threads;
}
