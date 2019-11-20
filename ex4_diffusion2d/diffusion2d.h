#ifndef DIFFUSION2D_H
#define DIFFUSION2D_H

#include <vector>
#include <string>

namespace hpcse {

struct Diagnostics
{
  double time;
  double heat;
  Diagnostics(double time, double heat) : time(time), heat(heat) {}
};

class Diffusion2d
{
public:
  Diffusion2d(const double D, const double L, const int N, const double dt, const int rank, const int procs);

  enum CommTags {
    Upper = 5712,
    Lower,
  };

  void advance();
  void compute_diagnostics(const double t);
  void write_diagnostics(const std::string &filename) const;
  void compute_histogram_hybrid(); //end public

private:
  void initialize_density();

  double m_D, m_L;
  int m_N, m_Ntot, m_localN, m_realN;
  double m_dr, m_dt, m_fac;
  int m_rank, m_procs;

  std::vector<double> m_rho, m_rho_tmp;
  std::vector<Diagnostics> m_diag;
};

}

#endif // DIFFUSION2D_H
