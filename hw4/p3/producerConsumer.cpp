/**********************************************************************
 * Code: UPC++ - Homework 3
 * Author: Vlachas Pantelis (pvlachas@ethz.ch)
 * ETH Zuerich - HPCSE II (Spring 2020)
 **********************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <chrono>
#include <math.h>
#include <upcxx/upcxx.hpp>
#include "factor.h"

// using ofstream constructors.
#include <iostream>
#include <fstream>  

#include <queue>

#define NUM_FACTORS 240000

std::queue<int> workers;

upcxx::global_ptr<double> factorArray;

// Dummy function initialization
void master(int rankCount);
void worker(int rankId);

bool continueFactorEvaluations;

int main(int argc, char* argv[])
{
    auto start = std::chrono::system_clock::now();

    // Intializing UPCXX
    upcxx::init();
    int rankId    = upcxx::rank_me();
    int rankCount = upcxx::rank_n();

    continueFactorEvaluations = true;

    if (rankId == 0){
        printf("Approximating the value of PI with %d series coefficients.\n", NUM_FACTORS);
    }
    int nFactors = NUM_FACTORS;

    // factorArray is now initialized as a global array. (please see some lines above)
    // TODO: Similarly to the divide and conquer strategy, the master needs to initialize the array with the upcxx::new_array<double> command. Then the array needs to be broadcasted so that all ranks have access and point to the shared address space.





    if (rankId == 0) master(rankCount);
    else worker(rankId);
    upcxx::barrier();

    // TODO: Finally, the master rank needs to compute the approximate value $\tilde{\pi}$ and save it to the results file, along with the total time. \textbf{Downcast} the global pointer to a local one and use it to compute the final approximation. (Similarly to the divide and conquer strategy)
    if (rankId == 0)
    {

        double pi_approx = 0.0;





        printf("PI approximate: %.17g\n", pi_approx);
        printf("PI: %.10f\n", M_PI);
        double error = abs(pi_approx - M_PI);
        printf("Absolute error: %.17g\n", error);
        
        auto end = std::chrono::system_clock::now();
        double rankTime = std::chrono::duration<double>(end-start).count();
        
        // Saving to a separate file for the master rank
        std::stringstream filename;
        filename << "./Results/producer_consumer" << ".txt";
        // assign to std::string
        std::string filenameStr = filename.str();
        std::ofstream outfile(filenameStr);
        outfile << pi_approx << "," << rankTime << "\n" << std::endl;
        outfile.close();
        printf("Total Running Time: %.17gs\n", rankTime);

        // Saving to a separate file for each rank
        std::stringstream filenameTime;
        filenameTime << "./Results/producer_consumer_time_rank_" << rankId << ".txt";
        // assign to std::string
        std::string filenameTimeStr = filenameTime.str();
        std::ofstream outfile2(filenameTimeStr);
        outfile2 << rankTime << "\n" << std::endl;
        outfile2.close();
    }


    // Finalize UPCXX
    upcxx::finalize();

    return 0;
}

void workerComputeFactor(int rankId, int factorId)
{
    // TODO: Compute factor


    // TODO: As a worker, use a RPC to update the master's factorArray with the computed factor, and notify the master that you are again available to compute by pushing your rankId to the queue 






}

void master(int rankCount)
{
    // Creating a queue with all the ranks, without master rank !
    for (int rank = 1; rank < rankCount; rank++) workers.push(rank);

    // Sending the factors to be computed in the ranks
    for (int factorId = NUM_FACTORS-1; factorId >= 0; --factorId)
    {
        // Waiting for a rank to become available
        // When no rank available call progress()
        while(workers.empty()) upcxx::progress();
        //TODO: Whenever any worker is available, the master has to get his \texttt{workerId}, by popping out the first \texttt{workerId} from the queue of the available workers.

        // TODO: After identifying an available worker, the master needs to send an RPC, the task that the consumer has to complete. This is no other than the workerComputeFactor(workerId, factorId)

    }
    // Master is notifying the workers to stop the evaluation of factors
    for (int i = 1; i < rankCount; i++) upcxx::rpc_ff(i, [](){continueFactorEvaluations = false;});
}


void worker(int rankId)
{
    // Worker is calling progress(); until the master sends an RPC notifying to stop the evaluation of any more factors
    auto t0 = std::chrono::system_clock::now();
    while(continueFactorEvaluations)
    {
        upcxx::progress();
    }
    auto t1 = std::chrono::system_clock::now();
    double rankTime = std::chrono::duration<double>(t1-t0).count();

    // Saving the runtime of each rank to a separate file
    std::stringstream filename;
    filename << "./Results/producer_consumer_time_rank_" << rankId << ".txt";
    // assign to std::string
    std::string filenameStr = filename.str();
    std::ofstream outfile(filenameStr);
    outfile << rankTime << "\n" << std::endl;
    outfile.close();
}







