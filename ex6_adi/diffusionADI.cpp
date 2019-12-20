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
  Diffusion2D(const double D, const double L, const int N, const double dt)
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
    m_R = D * dt / (2. * m_dr * m_dr);

    // TODO:
    // Initialize diagonals of the coefficient
    // matrix A, where Ax=b is the corresponding
    // system to be solved
    m_a.resize(m_realN-2, -m_R);
    m_b.resize(m_realN-2, 1. + 2.*m_R);
    m_c.resize(m_realN-2, -m_R);
  }

  void advance()
  {
    // TODO:
    // Implement the ADI scheme for diffusion
    // and parallelize with OpenMP
#pragma omp parallel for
    for (size_t iy=1; iy<m_realN-1; iy++) //rows
      for (size_t ix=1; ix<m_realN-1; ix++) //columns
      {
        size_t k  =  iy    * m_realN + ix;
        size_t k1 = (iy-1) * m_realN + ix;
        size_t k2 = (iy+1) * m_realN + ix;
        m_rhs[k] = m_phi[k] + m_R * (m_phi[k1] - 2.*m_phi[k] + m_phi[k2]);
      }
#pragma omp parallel for
    for (int iy=1; iy<m_realN-1; iy++) //rows
      thomas(Direction::X, iy);

    // ADI Step 1: Update rows at half timestep
    // Solve implicit system with Thomas algorithm
#pragma omp parallel for
    for (size_t iy=1; iy<m_realN-1; iy++) //rows
      for (size_t ix=1; ix<m_realN-1; ix++) //columns
      {
        size_t k  = iy * m_realN + ix;
        size_t k1 = iy * m_realN + (ix-1);
        size_t k2 = iy * m_realN + (ix+1);
        m_rhs[k] = m_phi[k] + m_R * (m_phi[k1] - 2.*m_phi[k] + m_phi[k2]);
      }

    // ADI: Step 2: Update columns at full timestep
    // Solve implicit system with Thomas algorithm

#pragma omp parallel for
    for (size_t iy=1; iy<m_realN-1; iy++) //rows
      for (size_t ix=1; ix<m_realN-1; ix++) //columns
      {
        size_t k  = iy * m_realN + ix;
        size_t k1 = iy * m_realN + (ix-1);
        size_t k2 = iy * m_realN + (ix+1);
        m_rhs[k] = m_phi[k] + m_R * (m_phi[k1] - 2.*m_phi[k] + m_phi[k2]);
      }
#pragma omp parallel for
    for (size_t ix=1; ix<m_realN-1; ix++) //columns
      thomas(Direction::Y, ix);
  }

  void compute_diagnostics(const double t)
  {
    double heat = 0.0;

    // TODO:
    // Compute the integral of phi_ in the computational domain
#pragma omp parallel for reduction(+:heat)
    for (int i = 1; i < m_realN-1; ++i)
      for (int j = 1; j < m_realN-1; ++j)
        heat += m_dr * m_dr * m_phi[i * m_realN + j];

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
    // Initialize phi(x, y, t=0)
    double bound = 0.25 * m_L;

    // TODO:
    // Initialize field phi based on the
    // prescribed initial conditions
    // and parallelize with OpenMP
#pragma omp parallel for
    for (size_t i = 1; i < m_realN - 1; ++i)     // rows
      for (size_t j = 1; j < m_realN - 1; ++j) // columns
      {
        size_t k = i*m_realN + j;
        if (std::abs((i-1)*m_dr - m_L/2.) < bound && std::abs((j-1)*m_dr - m_L/2.) < bound)
          m_phi[k] = 1.;
        else
          m_phi[k] = 0.;
      }
  }

  double m_L;
  size_t m_N, m_Ntot, m_realN;
  double m_dr;
  double m_R;
  std::vector<double> m_phi, m_rhs;
  std::vector<Diagnostics> m_diag;
  std::vector<double> m_a, m_b, m_c;

  size_t sub2ind(const Direction dir, const size_t a, const size_t b)
  {
    switch(dir) {
    case Direction::X:
      return a * m_realN + b;
    case Direction::Y:
      return b * m_realN + a;
    }
  }


  void thomas(const Direction dir, const size_t nid)
  {
    std::vector<double> d_(m_N);  // right hand side
    std::vector<double> cp_(m_N); // c prime
    std::vector<double> dp_(m_N); // d prime

    // compute modified coefficients
    d_[0] = dir==0 ? m_rhs[nid*m_realN] : m_rhs[nid];
    cp_[0] = m_c[0]/m_b[0];
    dp_[0] = d_[0]/m_b[0];
    for (size_t i=1; i<m_N-1; i++)
    {
      size_t k = sub2ind(dir, nid, i+1);
      d_[i]  = m_rhs[k];
      cp_[i] = m_c[i] / (m_b[i] - m_a[i] * cp_[i-1]);
      dp_[i] = (d_[i] - m_a[i]*dp_[i-1]) / (m_b[i] - m_a[i] * cp_[i-1]);
    }
    size_t i = m_N-1;
    size_t k = sub2ind(dir, nid, i+1);
    dp_[i] = (d_[i] - m_a[i]*dp_[i-1]) / (m_b[i] - m_a[i] * cp_[i-1]);

    // back substitution phase
    k = sub2ind(dir, nid, m_realN-2);
    m_phi[k] = dp_[m_N-1];
    for (int i=m_N-2; i>=0; i--) {
      size_t k  = sub2ind(dir, nid, i+1);
      size_t k1 = sub2ind(dir, nid, i+2);
      m_phi[k] = dp_[i] - cp_[i] * m_phi[k1];
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
