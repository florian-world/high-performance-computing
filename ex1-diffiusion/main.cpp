#include <iostream>
#include <cmath>
#include <vector>
#include <numeric>
#include <chrono> // use std::chrono to benchmark your code

#include "cacheflusher.h"
#include "diffusionsolver.h"

using hrc = std::chrono::high_resolution_clock;



int main()
{
    std::cerr << "Execution started..." << std::endl;
    CacheFlusher cf;


    DiffusionSolver solver;

    const int N = 10;
    std::vector<double> measurements;

    for (int i = 0; i < N; ++i) {
        // initialize to u_0
        solver.init();

        cf.flush();
        auto start = hrc::now();

        solver.solve(5000.0);

        auto end = hrc::now();

        std::chrono::duration<double> duration = end-start;
        measurements.push_back(duration.count());

        if (i > 0)
            std::cout << "; ";

        std::cout << duration.count();
        std::cerr << "Duration: " << duration.count() << std::endl;
    }

    std::cout << std::endl;

    std::cout << measurements.size() << std::endl;


    auto mean = std::accumulate(measurements.begin(), measurements.end(), 0.0)/measurements.size();

//    auto mean = std::accumulate(measurements.begin(), measurements.end(), 0.0)/measurements.size();

    std::cout << "Mean computing time; " << mean << std::endl;
    std::cerr << "Mean computing time: " << mean << "s" << std::endl;

    //solver.printCSVLine();

    return 0;
}
