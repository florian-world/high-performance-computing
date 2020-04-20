#include <iostream>
#include <fstream>
#include <cmath>

class SSABenchmark
{
public:
  template< typename SSAType>
  static void timeIt(int omega, int num_samples, double T, double dt, int runs)
  {
#ifdef _OPENMP
    #pragma omp parallel
#endif
    {
      printf ("Printing one line per active thread....\n");
    }

    // init
    vector<double> times(runs);
    vector<double> flops(runs);
    vector<double> trfs(runs);
    vector<double> perfs(runs);
    vector<double> rs(runs);

    printf("\n==========================================================================\n");
    for (int i=0; i<runs; i++)
    {
      SSAType ssa(omega, num_samples, T, dt);
      // run it
      ssa();
      ssa.dumpTrajToFile("output.txt");

      // check timing and performance
      const double time  = ssa.getTime();
      const double flop  = ssa.getFlops();
      const double trf   = ssa.getTransfers();
      const double r     = flop/trf;

      times[i] = time;
      flops[i] = flop;
      trfs[i]  = trf;
      perfs[i] = (flop/time/1e9);
      rs[i]    = r;

      std::cerr<<"time: "<< time <<", "<< flop <<" FLOP, Trf: "<< trf <<" Byte, R: "<<r<<" FLOP/Byte, perf: "<<flop/time/1e9<<" GFLOP/s"<<std::endl;
    }

    sort(times.begin(),times.end());
    sort(flops.begin(),flops.end());
    sort(trfs.begin(),trfs.end());
    sort(perfs.begin(),perfs.end());
    sort(rs.begin(),rs.end());

    const int i05 = static_cast<int>(runs*.05);
    const int i40 = static_cast<int>(runs*.4);
    const int i60 = static_cast<int>(runs*.6);
    const int i95 = static_cast<int>(runs*.95);

    std::cerr<<"==========================================================================\n";
    std::cerr<<"(05th perc.) time: "<< times[i05] <<", "<< flops[i05] <<" FLOP, Trf: "<< trfs[i05] <<" Byte, R: "<<rs[i05]<<" FLOP/Byte, perf: "<<perfs[i05]<<" GFLOP/s\n";
    std::cerr<<"(40th perc.) time: "<< times[i40] <<", "<< flops[i40] <<" FLOP, Trf: "<< trfs[i40] <<" Byte, R: "<<rs[i40]<<" FLOP/Byte, perf: "<<perfs[i40]<<" GFLOP/s\n";
    std::cerr<<"(60th perc.) time: "<< times[i60] <<", "<< flops[i60] <<" FLOP, Trf: "<< trfs[i60] <<" Byte, R: "<<rs[i60]<<" FLOP/Byte, perf: "<<perfs[i60]<<" GFLOP/s\n";
    std::cerr<<"(95th perc.) time: "<< times[i95] <<", "<< flops[i95] <<" FLOP, Trf: "<< trfs[i95] <<" Byte, R: "<<rs[i95]<<" FLOP/Byte, perf: "<<perfs[i95]<<" GFLOP/s\n";
    std::cerr<<"=========================================================================="<<std::endl;

  }
};
