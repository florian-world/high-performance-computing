#include "diffusion2d.h"

extern "C" {
#include <stdio.h>
}

#include <iostream>

#include <fstream>

using namespace hpcse;

Diffusion2d::Diffusion2d(const double D, const double L, const int N, const double dt, const int rank, const int procs) : m_D(D), m_L(L), m_N(N), m_dt(dt), m_rank(rank), m_procs(procs)
{
  /* Real space grid spacing */
  m_dr = m_L / (m_N - 1);

  /* Stencil factor */
  m_fac = m_dt * m_D / (m_dr * m_dr);

  /* Number of rows per process */
  m_localN = m_N / m_procs;

  /* Small correction for the last process */
  if (m_rank == procs - 1)
    m_localN += m_N % m_procs;

  /* Actual dimension of a row (+2 for the ghost cells) */
  m_realN = m_N + 2;

  /* Total number of cells */
  m_Ntot = (m_localN + 2) * (m_N + 2);

  m_rho.resize(m_Ntot, 0.0);
  m_rho_tmp.resize(m_Ntot, 0.0);

  /* Check that the timestep satisfies the restriction for stability */
  if (m_rank == 0) {
    std::cout << "timestep from stability condition is " << m_dr * m_dr / (4. * m_D) << '\n';
  }

  initialize_density();
}

void Diffusion2d::advance()
{
  // TODO: Implement Blocking MPI communication to exchange the ghost
  // cells on a periodic domain required to compute the central finite
  // differences below.

  // *** start MPI part ***
  // ...
  // *** end MPI part ***


  /* Central differences in space, forward Euler in time, Dirichlet BCs */
  for (int i = 1; i <= m_localN; ++i) {
    for (int j = 1; j <= m_N; ++j) {
      m_rho_tmp[i*m_realN + j] = m_rho[i*m_realN + j] + m_fac * ( + m_rho[i*m_realN + (j+1)]
          + m_rho[i*m_realN + (j-1)]
          + m_rho[(i+1)*m_realN + j]
          + m_rho[(i-1)*m_realN + j]
          - 4.*m_rho[i*m_realN + j]
          );
    }
  }

  /* Use swap instead of rho_ = rho_tmp__. This is much more efficient,
             because it does not copy element by element, just replaces storage
             pointers. */
  using std::swap;
  swap(m_rho_tmp, m_rho);
}

void Diffusion2d::compute_diagnostics(const double t)
{
  double heat = 0.0;

  /* Integration to compute total heat */
  for(int i = 1; i <= m_localN; ++i)
    for(int j = 1; j <= m_N; ++j)
      heat += m_rho[i*m_realN + j] * m_dr * m_dr;

  // TODO: Sum total heat from all ranks

  // *** start MPI part ***
  // ...
  // *** end MPI part ***

  if (m_rank == 0) {
    std::cout << "t = " << t << " heat = " << heat << '\n';
    m_diag.push_back(Diagnostics(t, heat));
  }
}

void Diffusion2d::write_diagnostics(const std::string &filename) const
{
  std::ofstream out_file(filename, std::ios::out);
  for (const Diagnostics &d : m_diag)
    out_file << d.time << '\t' << d.heat << '\n';
  out_file.close();
}

void Diffusion2d::compute_histogram_hybrid()
{
  /* Number of histogram bins */
  int M = 10;
  int hist[M] = {0};

  /* Find max and min density values */
  double max_rho, min_rho;
  max_rho = m_rho[1*m_realN + 1];
  min_rho = m_rho[1*m_realN + 1];

  for(int i = 1; i <= m_localN; ++i)
    for(int j = 1; j <= m_N; ++j) {
      if (m_rho[i*m_realN + j] > max_rho) max_rho = m_rho[i*m_realN + j];
      if (m_rho[i*m_realN + j] < min_rho) min_rho = m_rho[i*m_realN + j];
    }

  // TODO: Compute the global min and max heat values on this rank and
  // store the result in min_rho and max_rho, respectively.
  double lmin_rho = min_rho;
  double lmax_rho = max_rho;

  // *** start MPI part ***
  // ...
  // *** end MPI part ***

  double epsilon = 1e-8;
  double dh = (max_rho - min_rho + epsilon) / M;

  for(int i = 1; i <= m_localN; ++i)
    for(int j = 1; j <= m_N; ++j) {
      unsigned int bin = (m_rho[i*m_realN + j] - min_rho) / dh;
      hist[bin]++;
    }


  int g_hist[M];

  // TODO: Compute the sum of the histogram bins over all ranks and store
  // the result in the array g_hist.  Only rank 0 must print the result.

  // *** start MPI part ***
  // ...
  // *** end MPI part ***

  if (m_rank == 0)
  {
    printf("=====================================\n");
    printf("Output of compute_histogram_hybrid():\n");
    int gl = 0;
    for (int i = 0; i < M; i++) {
      printf("bin[%d] = %d\n", i, g_hist[i]);
      gl += g_hist[i];
    }
    printf("Total elements = %d\n", gl);
  }

}

void Diffusion2d::initialize_density()
{
  /* Initialization of the density distribution */
  int gi; // global index
  double bound = 0.25 * m_L;

  for (int i = 1; i <= m_localN; ++i) {
    gi = m_rank * (m_N / m_procs) + i;	// convert local index to global index
    for (int j = 1; j <= m_N; ++j) {
      if (std::abs((gi-1)*m_dr - m_L/2.) < bound && std::abs((j-1)*m_dr - m_L/2.) < bound) {
        m_rho[i*m_realN + j] = 1;
      } else {
        m_rho[i*m_realN + j] = 0;
      }
    }
  }
}
