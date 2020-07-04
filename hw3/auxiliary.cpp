#include "wave.h"

//************************
// DO NOT MODIFY THIS FILE
//************************

WaveEquation::WaveEquation(int a_N, int a_procs_per_dim) {
  MPI_Comm_size(MPI_COMM_WORLD, &size);
  procs_per_dim = a_procs_per_dim;
  Ntot = a_N;
  h = L / a_N;
  N = a_N / procs_per_dim;

  // the chosen numerical method is stable if  dt <= h/sqrt(3)
  dt = h / sqrt(3.0);
  aux = dt * dt / h / h;

  nthreads = omp_get_max_threads();
  threads_per_dim = pow(nthreads, 1. / 3.);

  assert(N % threads_per_dim == 0);

  FindCoordinates();

  origin[0] = coords[0] * N * h;
  origin[1] = coords[1] * N * h;
  origin[2] = coords[2] * N * h;
  u = new double[(N + 2) * (N + 2) * (N + 2)];

  u_old = new double[(N + 2) * (N + 2) * (N + 2)];
  u_new = new double[(N + 2) * (N + 2) * (N + 2)];

  pack = new double *[6 * threads_per_dim * threads_per_dim];
  unpack = new double *[6 * threads_per_dim * threads_per_dim];

// ********************************************************************************
// Question 1: Follow this example to parallelize the loop that updates grid
// points
// ********************************************************************************
#pragma omp parallel
  {
    int tid = omp_get_thread_num();
    int ti0, ti1, ti2;
    thread_coordinates(tid, threads_per_dim, ti0, ti1, ti2);

    int nloc = N / threads_per_dim;

    int i0_min = ti0 * nloc;
    int i1_min = ti1 * nloc;
    int i2_min = ti2 * nloc;

    for (int i0 = i0_min; i0 < i0_min + nloc; i0++) {
      double x0 = origin[0] + i0 * h + 0.5 * h;
      for (int i1 = i1_min; i1 < i1_min + nloc; i1++) {
        double x1 = origin[1] + i1 * h + 0.5 * h;
        for (int i2 = i2_min; i2 < i2_min + nloc; i2++) {
          double x2 = origin[2] + i2 * h + 0.5 * h;
          int m = (i0 + 1) * (N + 2) * (N + 2) + (i1 + 1) * (N + 2) + (i2 + 1);
          u[m] = Initial_Condition(x0, x1, x2);
          u_new[m] = u[m];
          u_old[m] = u[m];
          // assuming that u_old = u is equivalent to du/dt(t=0) = 0
        }
      }
    }
  }
}

WaveEquation::~WaveEquation() {
  delete[] u;
  delete[] u_old;
  delete[] u_new;
}

void WaveEquation::FindCoordinates() {
  int p = procs_per_dim;

  cart_comm = MPI_COMM_WORLD;
  MPI_Comm_rank(cart_comm, &rank);

  coords[0] = rank / (p * p);

  coords[1] = (rank - coords[0] * (p * p)) / p;

  coords[2] = (rank - coords[0] * (p * p) - coords[1] * p) % p;

  int coor_0_plus = (coords[0] + 1 + p) % p;
  int coor_1_plus = (coords[1] + 1 + p) % p;
  int coor_2_plus = (coords[2] + 1 + p) % p;

  int coor_0_minus = (coords[0] - 1 + p) % p;
  int coor_1_minus = (coords[1] - 1 + p) % p;
  int coor_2_minus = (coords[2] - 1 + p) % p;

  rank_plus[0] = (p * p) * coor_0_plus + coords[1] * p + coords[2];
  rank_plus[1] = coords[0] * p * p + p * coor_1_plus + coords[2];
  rank_plus[2] = coords[0] * p * p + coords[1] * p + coor_2_plus;

  rank_minus[0] = (p * p) * coor_0_minus + coords[1] * p + coords[2];
  rank_minus[1] = coords[0] * p * p + p * coor_1_minus + coords[2];
  rank_minus[2] = coords[0] * p * p + coords[1] * p + coor_2_minus;
}

void WaveEquation::pack_face(double *pack, int array_of_sizes[3],
                             int array_of_subsizes[3], int array_of_starts[3]) {
  int p0 = array_of_subsizes[0];
  int p1 = array_of_subsizes[1];
  int p2 = array_of_subsizes[2];

  int n1 = array_of_sizes[1];
  int n2 = array_of_sizes[2];

  for (int i0 = array_of_starts[0]; i0 < array_of_starts[0] + p0; i0++)
    for (int i1 = array_of_starts[1]; i1 < array_of_starts[1] + p1; i1++)
      for (int i2 = array_of_starts[2]; i2 < array_of_starts[2] + p2; i2++) {
        int i = (i0 - array_of_starts[0]) * p1 * p2 +
                (i1 - array_of_starts[1]) * p2 + (i2 - array_of_starts[2]);

        pack[i] = *(u + i0 * n1 * n2 + i1 * n2 + i2);
      }
}

void WaveEquation::unpack_face(double *pack, int array_of_sizes[3],
                               int array_of_subsizes[3],
                               int array_of_starts[3]) {
  int p0 = array_of_subsizes[0];
  int p1 = array_of_subsizes[1];
  int p2 = array_of_subsizes[2];

  int n1 = array_of_sizes[1];
  int n2 = array_of_sizes[2];

  for (int i0 = array_of_starts[0]; i0 < array_of_starts[0] + p0; i0++)
    for (int i1 = array_of_starts[1]; i1 < array_of_starts[1] + p1; i1++)
      for (int i2 = array_of_starts[2]; i2 < array_of_starts[2] + p2; i2++) {
        int i = (i0 - array_of_starts[0]) * p1 * p2 +
                (i1 - array_of_starts[1]) * p2 + (i2 - array_of_starts[2]);

        *(u + i0 * n1 * n2 + i1 * n2 + i2) = pack[i];
      }
}

void WaveEquation::thread_coordinates(int tid, int threads_per_dim, int &ti0,
                                      int &ti1, int &ti2) {
  ti0 = tid / (threads_per_dim * threads_per_dim);
  ti1 = (tid - ti0 * threads_per_dim * threads_per_dim) / threads_per_dim;
  ti2 = tid - ti0 * threads_per_dim * threads_per_dim - ti1 * threads_per_dim;
  assert(tid ==
         ti0 * threads_per_dim * threads_per_dim + ti1 * threads_per_dim + ti2);
}

double WaveEquation::Initial_Condition(double x0, double x1, double x2) {
  double r = (x0 - 0.5) * (x0 - 0.5) + (x1 - 0.5) * (x1 - 0.5) +
             (x2 - 0.5) * (x2 - 0.5);
  return exp(-r / 0.1);
}

void WaveEquation::Print(int kt = 0) {
#ifdef USE_HDF5

  std::string name = "Wave__" + std::to_string(kt);
  if (rank == 0) {
    FILE *xmf = 0;
    xmf = fopen((name + ".xmf").c_str(), "w");

    std::stringstream s_head;

    s_head << "<?xml version=\"1.0\" ?>\n";
    s_head << "<!DOCTYPE Xdmf SYSTEM \"Xdmf.dtd\" []>\n";
    s_head << "<Xdmf Version=\"2.0\">\n";
    s_head << " <Domain>\n";

    s_head << "     <Time Value=\"" << std::setprecision(10) << std::setw(10)
           << t << "\"/>\n\n";

    fprintf(xmf, (s_head.str()).c_str());

    std::stringstream s;
    s << "   <Grid GridType=\"Uniform\">\n";
    s << "     <Topology TopologyType=\"3DCoRectMesh\" Dimensions=\" "
      << Ntot + 1 << " " << Ntot + 1 << " " << Ntot + 1 << "\"/>\n\n";
    s << "       <Geometry GeometryType=\"ORIGIN_DXDYDZ\">\n";
    s << "          <DataItem Dimensions=\"3\" NumberType=\"Double\" "
         "Precision=\"8\" Format=\"XML\">\n";
    s << "                 " << 0 << " " << 0 << " " << 0 << "\n";
    s << "          </DataItem>\n";
    s << "          <DataItem Dimensions=\"3\" NumberType=\"Double\" "
         "Precision=\"8\" Format=\"XML\">\n";
    s << "                 " << std::setprecision(10) << std::setw(15) << h
      << " " << std::setw(15) << h << " " << std::setw(15) << h << "\n";
    s << "          </DataItem>\n";
    s << "     </Geometry>\n\n";

    s << "     <Attribute Name=\"data\" AttributeType=\" "
      << "Scalar"
      << "\" Center=\"Cell\">\n";
    s << "       <DataItem Dimensions=\" " << Ntot << " " << Ntot << " " << Ntot
      << " " << std::setw(10) << 1 << "\" NumberType=\"Float\" Precision=\" "
      << sizeof(double) << "\" Format=\"HDF\">\n";
    std::stringstream name_ss;
    name_ss << "data"; // << std::setfill('0') << std::setw(10) << mm ;
    std::string tmp = name_ss.str();
    s << "        " << (name + ".h5").c_str() << ":/" << tmp << "\n";
    s << "       </DataItem>\n";
    s << "     </Attribute>\n";

    s << "   </Grid>\n";

    std::string st = s.str();

    fprintf(xmf, st.c_str());

    std::stringstream s_tail;
    s_tail << " </Domain>\n";
    s_tail << "</Xdmf>\n";

    fprintf(xmf, (s_tail.str()).c_str());
    fclose(xmf);
  }

  hid_t file_id, dataset_id, fspace_id, fapl_id, mspace_id;

  H5open();
  fapl_id = H5Pcreate(H5P_FILE_ACCESS);

  H5Pset_fapl_mpio(fapl_id, cart_comm, MPI_INFO_NULL);

  file_id =
      H5Fcreate((name + ".h5").c_str(), H5F_ACC_TRUNC, H5P_DEFAULT, fapl_id);

  H5Pclose(fapl_id);

  double *array_all = new double[N * N * N];

  hsize_t count[4] = {static_cast<unsigned int>(N),
                      static_cast<unsigned int>(N),
                      static_cast<unsigned int>(N), 1};
  hsize_t dims[4] = {static_cast<unsigned int>(Ntot),
                     static_cast<unsigned int>(Ntot),
                     static_cast<unsigned int>(Ntot), 1};
  hsize_t offset[4] = {static_cast<unsigned int>(coords[0] * N),
                       static_cast<unsigned int>(coords[1] * N),
                       static_cast<unsigned int>(coords[2] * N), 0};

  for (int i0 = 1; i0 <= N; i0++) {
    for (int i1 = 1; i1 <= N; i1++) {
      for (int i2 = 1; i2 <= N; i2++) {
        array_all[(i2 - 1) + (i1 - 1) * N + (i0 - 1) * N * N] =
            u[i2 + i1 * (N + 2) + i0 * (N + 2) * (N + 2)];
      }
    }
  }
  fapl_id = H5Pcreate(H5P_DATASET_XFER);
  H5Pset_dxpl_mpio(fapl_id, H5FD_MPIO_COLLECTIVE);
  fspace_id = H5Screate_simple(4, dims, NULL);

  dataset_id = H5Dcreate(file_id, "data", H5T_NATIVE_DOUBLE, fspace_id,
                         H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
  fspace_id = H5Dget_space(dataset_id);

  H5Sselect_hyperslab(fspace_id, H5S_SELECT_SET, offset, NULL, count, NULL);

  mspace_id = H5Screate_simple(4, count, NULL);

  H5Dwrite(dataset_id, H5T_NATIVE_DOUBLE, mspace_id, fspace_id, fapl_id,
           array_all);

  H5Sclose(mspace_id);
  H5Sclose(fspace_id);
  H5Dclose(dataset_id);
  H5Pclose(fapl_id);
  H5Fclose(file_id);
  H5close();

  delete[] array_all;
#else

  std::string name = "Wave__" + std::to_string(kt) + ".txt";
  MPI_File file;
  MPI_File_open(cart_comm, name.c_str(), MPI_MODE_RDWR | MPI_MODE_CREATE,
                MPI_INFO_NULL, &file);

  std::stringstream ss;

  for (int i0 = 0; i0 < N; i0++) {
    double x0 = origin[0] + i0 * h + 0.5 * h;
    for (int i1 = 0; i1 < N; i1++) {
      double x1 = origin[1] + i1 * h + 0.5 * h;
      for (int i2 = 0; i2 < N; i2++) {
        double x2 = origin[2] + i2 * h + 0.5 * h;
        ss << x0 << " " << x1 << " " << x2 << " "
           << u[(i2 + 1) + (i1 + 1) * (N + 2) + (i0 + 1) * (N + 2) * (N + 2)]
           << "\n";
      }
    }
  }

  std::string asciiData = ss.str();

  MPI_Offset my_length = asciiData.size() * sizeof(char);

  MPI_Offset offset;
  MPI_Exscan(&my_length, &offset, 1, MPI_OFFSET, MPI_SUM, cart_comm);

  MPI_File_write_at_all(file, offset, asciiData.data(), asciiData.size(),
                        MPI_CHAR, MPI_STATUS_IGNORE);
  MPI_File_close(&file);

#endif
}

void WaveEquation::UpdateGridPoint(int i0, int i1, int i2) {
  int m = i0 * (N + 2) * (N + 2) + i1 * (N + 2) + i2;
  int ip1 = (i0 + 1) * (N + 2) * (N + 2) + (i1) * (N + 2) + (i2);
  int im1 = (i0 - 1) * (N + 2) * (N + 2) + (i1) * (N + 2) + (i2);
  int jp1 = (i0) * (N + 2) * (N + 2) + (i1 + 1) * (N + 2) + (i2);
  int jm1 = (i0) * (N + 2) * (N + 2) + (i1 - 1) * (N + 2) + (i2);
  int kp1 = (i0) * (N + 2) * (N + 2) + (i1) * (N + 2) + (i2 + 1);
  int km1 = (i0) * (N + 2) * (N + 2) + (i1) * (N + 2) + (i2 - 1);
  u_new[m] =
      2.0 * u[m] - u_old[m] +
      aux * (u[ip1] + u[im1] + u[jp1] + u[jm1] + u[kp1] + u[km1] - 6.0 * u[m]);
}