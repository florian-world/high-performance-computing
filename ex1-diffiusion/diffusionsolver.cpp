//#define _USE_MATH_DEFINES
#include "diffusionsolver.h"

#include <cmath>
#include <iostream>

DiffusionSolver::DiffusionSolver()
    : m_u(N)
{

}

void DiffusionSolver::init()
{
//    std::cerr << "Initializing diffusion vector, params: " << std::endl
//              << "N = " << N << std::endl
//              << "L = " << L << std::endl
//              << "DX = " << DX << std::endl
//              << "ALPHA = " << ALPHA << std::endl
//              << "DT = " << DT << std::endl;

    for (int i = 0; i < N; ++i) {
        // do not accumulate summation errors
        auto x = i*DX;
        m_u[i] = std::sin(2*M_PI/L*x);
    }
}

void DiffusionSolver::solve(double t_f)
{
    int nit = std::ceil(t_f/DT);

//    std::cerr << "Starting " << nit << " iterations now" << std::endl;

    for (int n = 0; n < nit; ++n) {
        doStep();
    }
}

void DiffusionSolver::printCSVLine()
{
    bool first = true;
    for (const auto& ui : m_u) {
        if (!first) std::cout << ";";
        std::cout << ui;
        first = false;
    }
    std::cout << std::endl;
}

void DiffusionSolver::doStep()
{
    static const double coeff = DT * ALPHA / (DX * DX);
    double u_old_0 = m_u[0];
    double u_old_i_prev = m_u[N-1];
    for (int i = 0; i < N-1; ++i) {
        double u_new = m_u[i] + coeff * (u_old_i_prev - 2*m_u[i] + m_u[i+1]);
        u_old_i_prev = m_u[i];
        m_u[i] = u_new;
    }

    m_u[N-1] = m_u[N-1] + coeff * (u_old_i_prev - 2*m_u[N-1] + u_old_0);
}
