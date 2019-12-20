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
  enum Direction {
      X = 0,
      Y = 1
    };
public:
  Diffusion2D(const double D, const double L, const size_t N, const double dt)
    : m_L(L), m_N(N)
  {
    // Real space grid spacing.
    m_dr = m_L / (m_N - 1);

    // Actual dimension of a row (+2 for the ghost cells).
    m_realN = N + 2;

    // Total number of cells.
    m_Ntot = (m_N + 2) * (m_N + 2);

    m_phi.resize(m_Ntot, 0.);
    m_rhs.resize(m_Ntot, 0.);

    // Initialize field on grid
    initialize_phi();

    // Common pre-factor
    m_C = D * dt / (2. * m_dr * m_dr);

    // TODO:
    // Initialize diagonals of the coefficient
    // matrix A, where Ax=b is the corresponding
    // system to be solved
    m_a.resize(m_N, -m_C);
    m_b.resize(m_N, 1. + 2.*m_C);
    m_c.resize(m_N, -m_C);
  }

  void midpointDerivative (Direction dir) {
      // picks the right difference in indicies
      auto idxDiff = sub2ind(0, 1, dir);
  #pragma omp parallel for
      for (size_t iy=1; iy<m_realN-1; iy++) {
        for (size_t ix=1; ix<m_realN-1; ix++) {
          // always sum up in the same way
          auto k = sub2ind(ix, iy, Direction::Y);
          auto k1 = k - idxDiff;
          auto k2 = k + idxDiff;

          m_rhs[k] = m_phi[k] + m_C * (m_phi[k1] - 2.0*m_phi[k] + m_phi[k2]);
        }
      }
    }

  void advance()
  {
    // TODO:
    // Implement the ADI scheme for diffusion
    // and parallelize with OpenMP

    // ADI Step 1: Update rows at half timestep
    // Solve implicit system with Thomas algorithm
    midpointDerivative(Direction::Y);
#pragma omp parallel for
    for (size_t iy=1; iy<m_realN-1; iy++) //rows
      thomasSolver(Direction::X, iy);

    // ADI: Step 2: Update columns at full timestep
    // Solve implicit system with Thomas algorithm
    midpointDerivative(Direction::X);
#pragma omp parallel for
    for (size_t ix=1; ix<m_realN-1; ix++) //columns
      thomasSolver(Direction::Y, ix);
  }

  void compute_diagnostics(const double t)
  {
    double heat = 0.0;

    // TODO:
    // Compute the integral of phi_ in the computational domain
#pragma omp parallel for reduction(+:heat)
    for (size_t i = 1; i < m_realN-1; ++i)
      for (size_t j = 1; j < m_realN-1; ++j)
        heat += m_dr * m_dr * m_phi[sub2ind(i, j)];

#ifndef NDEBUG
    std::cout << "t = " << t << " heat = " << heat << '\n';
#endif
    m_diag.push_back(Diagnostics(t, heat));
  }

  void write_diagnostics(const std::string &filename) const
  {
    std::ofstream out_file(filename, std::ios::out);
    for (const Diagnostics &d : m_diag)
      out_file << d.time << '\t' << d.heat << '\n';
    out_file.close();
  }

private:
  void initialize_phi()
  {
    // TODO:
    // Initialize field phi based on the
    // prescribed initial conditions
    // and parallelize with OpenMP

    // Initialize phi(x, y, t=0)
    double bound = 0.25 * m_L;

    auto start = - m_L/2.0; // (-s, -s)

#pragma omp parallel for
    for (size_t i = 1; i < m_realN - 1; ++i)     // rows
      for (size_t j = 1; j < m_realN - 1; ++j) // columns
      {
        auto k = sub2ind(i, j);
        if (std::abs(start + (i-1)*m_dr) < bound && std::abs(start + (j-1)*m_dr) < bound)
          m_phi[k] = 1.0;
        else
          m_phi[k] = 0.0;
      }
  }

  double m_L;
  size_t m_N, m_Ntot, m_realN;
  double m_dr;
  double m_C;
  std::vector<double> m_phi, m_rhs;
  std::vector<Diagnostics> m_diag;
  std::vector<double> m_a, m_b, m_c;

  size_t sub2ind(const size_t a, const size_t b, const Direction dir = Direction::X)
  {
    switch(dir) {
    case Direction::X:
      return a * m_realN + b;
    case Direction::Y:
      return b * m_realN + a;
    }
  }


  void thomasSolver(const Direction dir, const size_t nid)
  {
    std::vector<double> rhs(m_N);
    std::vector<double> rhsp(m_N);
    std::vector<double> cp(m_N);

    rhs[0] = m_rhs[sub2ind(dir, nid)];

    cp[0] = m_c[0]/m_b[0];
    rhsp[0] = rhs[0]/m_b[0];

    for (size_t i=1; i<m_N-1; ++i) {
      size_t k = sub2ind(nid, i+1,dir);
      rhs[i]  = m_rhs[k];
      cp[i] = m_c[i] / (m_b[i] - m_a[i] * cp[i-1]);
      rhsp[i] = (rhs[i] - m_a[i]*rhsp[i-1]) / (m_b[i] - m_a[i] * cp[i-1]);
    }

    rhsp[m_N-1] = (rhs[m_N-1] - m_a[m_N-1]*rhsp[m_N-2]) / (m_b[m_N-1] - m_a[m_N-1] * cp[m_N-1]);
    m_phi[sub2ind(nid, m_realN-2, dir)] = rhsp[m_N-1];

    for (ssize_t i=static_cast<ssize_t>(m_N)-2; i>=0; --i) {
      auto idx = static_cast<size_t>(i);
      auto k  = sub2ind(nid, idx+1, dir);
      auto k1 = sub2ind(nid, idx+2, dir);
      m_phi[k] = rhsp[idx] - cp[idx] * m_phi[k1];
    }
  }
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
  {
    std::cout << "Running with " << omp_get_num_threads() << " threads\n";
  }

  const double D = std::stod(argv[1]);  // diffusion constant
  const double L = std::stod(argv[2]);  // domain side (length)
  const int N = std::stoul(argv[3]);    // number of grid points per dimension
  const double dt = std::stod(argv[4]); // timestep

#pragma omp master
  std::cout << "D = " << D << "; L = " << L << "; N = " << N << "; dt = " << dt << std::endl;

  Diffusion2D system(D, L, N, dt);

  const auto start = std::chrono::system_clock::now();
  for (int step = 0; step < static_cast<int>(std::round(10.0/dt)); ++step) {
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
