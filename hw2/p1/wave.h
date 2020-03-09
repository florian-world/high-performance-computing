#pragma once

#include <cmath>
#include <iomanip>
#include <iostream>
#include <mpi.h>
#include <sstream>
#include <string>

//#define USE_HDF5 // comment this line if you do not want hdf5 file output
                 // (suitable for visualization with Paraview)

#ifdef USE_HDF5
#include <hdf5.h>
#endif


#define L 1.0 // domain size (unit cube) with Ntot x Ntot x Ntot grid points

struct WaveEquation {
  int N;     // grid points per direction for this rank
  double h;  // grid spacing (dx = dy = dz = h)
  double dt; // timestep
  double t;  // current time
  double *u; // solution vector
  double *u_old;
  double *u_new;

  int Ntot;          // total grid points per direction
  int procs_per_dim; // ranks per direction

  int size; // =total ranks = procs_per_dim*procs_per_dim*procs_per_dim
  int rank;

  int rank_plus[3];  // neighboring ranks
  int rank_minus[3]; //

  MPI_Comm cart_comm; // should be a Cartesian topology communicator

  double origin[3]; // physical space (x0,x1,x2) coordinates of 1st grid point
                    // of this rank
  int coords[3]; // index space coordinates of this rank

  double aux;

  ~WaveEquation();
  WaveEquation(int a_N, int a_procs_per_dim);
  void FindCoordinates();
  double Initial_Condition(double x0, double x1, double x2);
  void UpdateGridPoint(int i0, int i1, int i2);
  void Print(int kt);
  void pack_face(double *pack, int array_of_sizes[3], int array_of_subsizes[3],
                 int array_of_starts[3]);
  void unpack_face(double *pack, int array_of_sizes[3],
                   int array_of_subsizes[3], int array_of_starts[3]);
  void run(double t_end);
};
