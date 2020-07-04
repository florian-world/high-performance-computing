#include <cmath>
#include <vector>

class SSA_GPU
{
public:
    SSA_GPU(int omega, int numSamples, double T, double dtBin, unsigned seed, int itersPerPass) :
        omega(omega),
        numSamples(numSamples),
        endTime(T),
        dtBin(dtBin),
        numBins((int)std::ceil(T / dtBin)),
        numItersPerPass(itersPerPass),
        seed(seed),
        trajSa(numBins),
        trajSb(numBins),
        trajNumSteps(numBins)
    { }

    void run();

    /// Dump trajectories to file (nbins rows and 2 columns).
    void dumpTrajectoryToFile(const char *filename);

private:
    const int omega;            /// Multiplier for number of molecules.
    const int numSamples;       /// Number of samples of stochastic algo to run.
    const double endTime;       /// End time for the simulation.
    const double dtBin;         /// Width of trajectory bin.
    const int numBins;          /// Number of time bins.
    const int numItersPerPass;  /// Number of iterations per pass.
    const unsigned seed;        /// Random number seed.

    /** Trajectories (binned) for Sa*omega and Sb*omega (trajSa(i) = avg of Sa for t in [i,i+1]*dtBin). */
    std::vector<double> trajSa;
    std::vector<double> trajSb;
    std::vector<int> trajNumSteps;
};
