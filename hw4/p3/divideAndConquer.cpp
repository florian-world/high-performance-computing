/**********************************************************************
 * Code: UPC++ - Homework 3
 * Author: Vlachas Pantelis (pvlachas@ethz.ch)
 * ETH Zuerich - HPCSE II (Spring 2020)
 **********************************************************************/

// Loading necessary libraries
#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <math.h>

#include <upcxx/upcxx.hpp>
#include "factor.h"

// using ofstream constructors.
#include <iostream>
#include <fstream>  

#define NUM_FACTORS 240000

int main(int argc, char* argv[])
{
    // Measuring the total time needed.
    auto start = std::chrono::system_clock::now();

    // Intializing UPCXX
    upcxx::init();
    int rankId    = upcxx::rank_me();
    int rankCount = upcxx::rank_n();

    if (rankId == 0){
        printf("Approximating the value of PI with %d series coefficients.\n", NUM_FACTORS);
    }
    int nFactors = NUM_FACTORS;

    // TODO: Specify the variables factorsPerRank, initFactor and endFactor
    int factorsPerRank = 0;
    int initFactor = 0;
    int endFactor  = 0;

    // TODO: Initialize a global pointer factorArray. Rank zero has to initialize the array. Do not forget to finally broadcast the global pointer to all ranks from rank 0 so that all ranks have access to the same global adress space

    upcxx::global_ptr<double> factorArray;
    if (rankId == 0) factorArray = upcxx::new_array<double>(rankCount);

    upcxx::broadcast(&factorArray, 1, 0).wait(); // broadcast 1 array from rank 0

    // TODO: After broadcasting the array, each rank needs to compute the portion of the factors it is assigned, and then \textbf{place} the result back to the \texttt{factorArray}. Do not use RPCs in this question, use the \texttt{upcxx::rput} command.
    upcxx::future<> fut_all = upcxx::make_future();

    int numLocal = NUM_FACTORS / rankCount;
    int idxStart = 1 + rankId * numLocal;
    int idxStop = idxStart + numLocal;

    // printf("In rank %d computiong from %d to %d\n", numLocal, idxStart, idxStop);

    double localFactor = 0.0;

    for(int k = idxStart; k < idxStop; ++k){
        localFactor += FACTOR(k);
    }

    auto future = upcxx::rput(&localFactor, factorArray + rankId, 1);

    fut_all = upcxx::when_all(fut_all, future);

    fut_all.wait(); // take time when all futures are done

    auto end = std::chrono::system_clock::now();
    double rankTime = std::chrono::duration<double>(end-start).count();

    // Saving to a separate file for each rank
    std::stringstream filename;
    filename << "./Results/divide_and_conquer_time_rank_" << rankId << ".txt";
    std::string filenameStr = filename.str();
    std::ofstream outfile(filenameStr);
    outfile << rankTime << "\n" << std::endl;
    outfile.close();

    // TODO ?:


    // TODO: Finally, rank zero needs to compute the approximate value $\tilde{\pi}$ and save it to the results file, along with the total time. \textbf{Downcast} the global pointer to a local one and use it to compute the final approximation.
    if (rankId == 0)
    {

        double pi_approx = 0.0;

        auto localArray = factorArray.local();

        for (int i = 0; i < rankCount; ++i) {
            pi_approx += localArray[i];
        }

        pi_approx = 4 * pi_approx;

        // Reporting the result
        printf("PI approximate: %.17g\n", pi_approx);
        printf("PI: %.10f\n", M_PI);
        double error = abs(pi_approx - M_PI);
        printf("Absolute error: %.17g\n", error);

        // Computing the total time
        auto end = std::chrono::system_clock::now();
        double totalTime = std::chrono::duration<double>(end-start).count();
        printf("Total Running Time: %.17gs\n", totalTime);

        // Saving the result and the total time
        std::ofstream outfile ("./Results/divide_and_conquer.txt");
        outfile << pi_approx << "," << totalTime << "\n" << std::endl;
        outfile.close();
    }

    // Finalize UPCXX
    upcxx::finalize();

    return 0;
}
