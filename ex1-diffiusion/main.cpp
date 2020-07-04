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
//    std::cerr << "Execution started..." << std::endl;
    CacheFlusher cf;

    DiffusionSolver solver;

    const int N = 10;
    std::vector<double> measurements;
    measurements.reserve(10);

    for (int i = 0; i < N; ++i) {
        // initialize to u_0
        solver.init();

        cf.flush();
        auto start = hrc::now();
        solver.solve(50000.0);
        auto end = hrc::now();

        std::chrono::duration<double> duration = end-start;
        measurements.push_back(duration.count());

        if (i > 0)
            std::cout << "; ";

        std::cout << duration.count();
    }
    std::cout << std::endl;
    std::cout << measurements.size() << std::endl;
    auto mean = std::accumulate(measurements.begin(), measurements.end(), 0.0)/measurements.size();
    std::cout << "Mean computing time; " << mean << std::endl;



    for (int i = 0; i < N; ++i) {
        // initialize to u_0
        solver.init();

//        cf.flush();
        auto start = hrc::now();
        solver.solve(50000.0);
        auto end = hrc::now();

        std::chrono::duration<double> duration = end-start;
        measurements.push_back(duration.count());

        if (i > 0)
            std::cout << "; ";

        std::cout << duration.count();
    }
    std::cout << std::endl;
    std::cout << measurements.size() << std::endl;
    mean = std::accumulate(measurements.begin(), measurements.end(), 0.0)/measurements.size();
    std::cout << "Mean computing time; " << mean << std::endl;

    //solver.printCSVLine();

    return 0;
}
