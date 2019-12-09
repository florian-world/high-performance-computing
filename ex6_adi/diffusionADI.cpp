#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <omp.h>
#include <string>
#include <vector>

// sample collection for diagnostics
struct Diagnostics {
    double time;
    double heat;
    Diagnostics(double time, double heat) : time(time), heat(heat) {}
};

class Diffusion2D
{
public:
    Diffusion2D(const double D, const double L, const int N, const double dt)
        : D_(D), L_(L), N_(N), dt_(dt)
    {
        // Real space grid spacing.
        dr_ = L_ / (N_ - 1);

        // Actual dimension of a row (+2 for the ghost cells).
        real_N_ = N + 2;

        // Total number of cells.
        Ntot_ = (N_ + 2) * (N_ + 2);

        phi_.resize(Ntot_, 0.);
        rhs_.resize(Ntot_, 0.);

        // Initialize field on grid
        initialize_phi();

        // Common pre-factor
        R_ = D * dt / (2. * dr_ * dr_);

        // TODO:
        // Initialize diagonals of the coefficient
        // matrix A, where Ax=b is the corresponding
        // system to be solved
    }

    void advance()
    {
        // TODO:
        // Implement the ADI scheme for diffusion
        // and parallelize with OpenMP

        // ADI Step 1: Update rows at half timestep
        // Solve implicit system with Thomas algorithm

        // ADI: Step 2: Update columns at full timestep
        // Solve implicit system with Thomas algorithm
    }

    void compute_diagnostics(const double t)
    {
        double heat = 0.0;

        // TODO:
        // Compute the integral of phi_ in the computational domain

#ifndef NDEBUG
        std::cout << "t = " << t << " heat = " << heat << '\n';
#endif
        diag_.push_back(Diagnostics(t, heat));
    }

    void write_diagnostics(const std::string &filename) const
    {
        std::ofstream out_file(filename, std::ios::out);
        for (const Diagnostics &d : diag_)
            out_file << d.time << '\t' << d.heat << '\n';
        out_file.close();
    }

private:
    void initialize_phi()
    {
        // Initialize phi(x, y, t=0)
        double bound = 0.25 * L_;

        // TODO:
        // Initialize field phi based on the
        // prescribed initial conditions
        // and parallelize with OpenMP
        for (int i = 1; i < real_N_ - 1; ++i)     // rows
            for (int j = 1; j < real_N_ - 1; ++j) // columns
            {
            }
    }

    double D_, L_;
    int N_, Ntot_, real_N_;
    double dr_, dt_;
    double R_;
    std::vector<double> phi_, rhs_;
    std::vector<Diagnostics> diag_;
    std::vector<double> a_, b_, c_;
};

// No additional code required from this point on
int main(int argc, char *argv[])
{
    if (argc < 5) {
        std::cerr << "Usage: " << argv[0] << " D L N dt\n";
        return 1;
    }

#pragma omp parallel
#pragma omp master
        std::cout << "Running with " << omp_get_num_threads() << " threads\n";

    const double D = std::stod(argv[1]);  // diffusion constant
    const double L = std::stod(argv[2]);  // domain side (length)
    const int N = std::stoul(argv[3]);    // number of grid points per dimension
    const double dt = std::stod(argv[4]); // timestep

    Diffusion2D system(D, L, N, dt);

    const auto start = std::chrono::system_clock::now();
    for (int step = 0; step < 10000; ++step) {
        system.advance();
        system.compute_diagnostics(dt * step);
    }
    const auto end = std::chrono::system_clock::now();
    auto elapsed =
        std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

    std::cout << "Timing: "
              << "N=" << N << " elapsed=" << elapsed.count() << " ms" << '\n';

    system.write_diagnostics("diagnostics.dat");

    return 0;
}
