#include "argument_parser.h"
#include "ssa.h"
#include "test.h"

int main(int argc, const char **argv)
{
    // Run tests.
    testReduceIsDoneKernel();

    ArgumentParser parser(argc, argv);

    // Parameters:
    // -omega   -> number of molecules -> i.e. result of i means i*omega mol.
    // -samples -> how many samples of stoch. algo. to run
    // -T       -> end time for algo
    // -dt      -> size of bin for trajectory binning
    // -seed    -> seed for random number generator
    // -runs    -> how many repetitions for benchmarking

    // Read command-line arguments.
    int omega        = parser("-omega", 10);
    int samples      = parser("-samples", 200'000);
    double T         = parser("-T", 5.0);
    double dtBin     = parser("-dt", 0.1);
    unsigned seed    = (unsigned)parser("-seed", 12345);
    int runs         = parser("-runs", 100);
    int itersPerPass = parser("-iters", 1000);

    // Run the code.
    SSA_GPU ssa(omega, samples, T, dtBin, seed, itersPerPass);
    ssa.run();
    ssa.dumpTrajectoryToFile("output.txt");

    return 0;
}
