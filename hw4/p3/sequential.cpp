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
#include <vector>

// Library of the factor F(n)
#include "factor.h"

// ofstream constructors.
#include <iostream>
#include <fstream>  

#define NUM_FACTORS 240000

int main(int argc, char* argv[])
{
    printf("Approximating the value of PI with %d series coefficients.\n", NUM_FACTORS);

    // Measuring the total time needed.
    auto start = std::chrono::system_clock::now();

    std::vector<double> factors(NUM_FACTORS);

    for(int k=1; k<=NUM_FACTORS; ++k){
        factors[k-1] = FACTOR(k);
    }

    auto end = std::chrono::system_clock::now();

    double pi_approx = 0.0;
    for(int k=1; k<=NUM_FACTORS; ++k){
        pi_approx += factors[k-1];
    }
    pi_approx = 4. * pi_approx;

    printf("PI approximate: %.17g\n", pi_approx);
    printf("PI: %.17g\n", M_PI);
    double error = abs(pi_approx - M_PI);
    printf("Absolute error: %.17g\n", error);

    double totalTime = std::chrono::duration<double>(end-start).count();
    printf("Total Running Time: %.17g s\n", totalTime);

    // Saving the results to a file
    std::ofstream outfile1 ("./Results/sequential_time.txt");
    outfile1 << totalTime << "\n" << std::endl;
    outfile1.close();

    std::ofstream outfile2 ("./Results/sequential.txt");
    outfile2 << pi_approx << ", " << totalTime << "\n" << std::endl;
    outfile2.close();

    return 0;
}
