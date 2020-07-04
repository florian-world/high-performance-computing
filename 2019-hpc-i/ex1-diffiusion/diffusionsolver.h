#ifndef DIFFUSIONSOLVER_H
#define DIFFUSIONSOLVER_H

#include <vector>

class DiffusionSolver
{
public:
    DiffusionSolver();

    static const int N = 20000;
    constexpr static const double L = 1000.0;
    constexpr static const double DX = L/N;
    constexpr static const double ALPHA = 10e-4;
    constexpr static const double DT = DX*DX/(4*ALPHA);

    /**
     * @brief init initializes values to U0
     */
    void init();

    void solve(double t_f);

    void printCSVLine();

private:
    std::vector<double> m_u;

    void doStep();
};

#endif // DIFFUSIONSOLVER_H
