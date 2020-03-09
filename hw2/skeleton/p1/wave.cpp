#include "wave.h"

/********************************************************************/ 
/* Subquestion a: change the following function and use a Cartesian */ 
/* topology to find coords[3], rank_plus[3] and rank_minus[3]       */
/********************************************************************/
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

// This function is not needed when custom datatypes are used
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

// This function is not needed when custom datatypes are used
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





/********************************************************************/ 
/* Subquestion b: you should no longer need the functions pack_face */
/* and unpack_face nor should you need to allocate memory by using  */
/* double *pack[6] and double *unpack[6].                           */
/********************************************************************/
void WaveEquation::run(double t_end) {

  t = 0;

  /********************************************************************/ 
  /* Subquestion b: you need to define 12 custom datatypes.           */
  /* For sending data, six datatypes (one per face) are required.     */
  /* For receiving data, six more datatypes are required.             */
  /* You should use MPI_Type_create_subarray for those datatypes.     */
  /********************************************************************/


  /* Subquestion b: You can delete this part when the subquestion is completed. */
  /************************************************************************************************/
  double *pack[6];
  pack[0] = new double[N * N];
  pack[1] = new double[N * N];
  pack[2] = new double[N * N];
  pack[3] = new double[N * N];
  pack[4] = new double[N * N];
  pack[5] = new double[N * N];
  double *unpack[6];
  unpack[0] = new double[N * N];
  unpack[1] = new double[N * N];
  unpack[2] = new double[N * N];
  unpack[3] = new double[N * N];
  unpack[4] = new double[N * N];
  unpack[5] = new double[N * N];
  /************************************************************************************************/






  /* Subquestion b: Create and commit custom datatypes here */
  /************************************************************************************************/
  MPI_Datatype SEND_FACE_PLUS [3];
  MPI_Datatype SEND_FACE_MINUS[3]; 

  MPI_Datatype RECV_FACE_PLUS [3];
  MPI_Datatype RECV_FACE_MINUS[3]; 
  /************************************************************************************************/




  int count = 0;
  do {
    if (count % 100 == 0) {
      if (rank == 0)
        std::cout << count << " t=" << t << "\n";
      Print(count);
    }


  /* Subquestion b: You can delete this part when the subquestion is completed. */
  /************************************************************************************************/
    int array_of_sizes[3] = {N + 2, N + 2, N + 2};

    {
      int array_of_subsizes[3] = {1, N, N};
      int array_of_starts[3] = {N, 1, 1};
      pack_face(pack[0], array_of_sizes, array_of_subsizes, array_of_starts);
    }

    {
      int array_of_subsizes[3] = {1, N, N};
      int array_of_starts[3] = {1, 1, 1};
      pack_face(pack[1], array_of_sizes, array_of_subsizes, array_of_starts);
    }

    {
      int array_of_subsizes[3] = {N, 1, N};
      int array_of_starts[3] = {1, N, 1};
      pack_face(pack[2], array_of_sizes, array_of_subsizes, array_of_starts);
    }

    {
      int array_of_subsizes[3] = {N, 1, N};
      int array_of_starts[3] = {1, 1, 1};
      pack_face(pack[3], array_of_sizes, array_of_subsizes, array_of_starts);
    }

    {
      int array_of_subsizes[3] = {N, N, 1};
      int array_of_starts[3] = {1, 1, N};
      pack_face(pack[4], array_of_sizes, array_of_subsizes, array_of_starts);
    }

    {
      int array_of_subsizes[3] = {N, N, 1};
      int array_of_starts[3] = {1, 1, 1};
      pack_face(pack[5], array_of_sizes, array_of_subsizes, array_of_starts);
    }
  /************************************************************************************************/





    MPI_Request request[12];

    /* Subquestion b: Replace the sends and receives with ones that correspond to custom datatypes*/
    /**********************************************************************************************/
    MPI_Irecv(unpack[0], N * N, MPI_DOUBLE, rank_minus[0], 100, cart_comm,&request[0]);
    MPI_Isend(pack  [0], N * N, MPI_DOUBLE, rank_plus [0], 100, cart_comm,&request[1]);

    MPI_Irecv(unpack[1], N * N, MPI_DOUBLE, rank_plus [0], 101, cart_comm, &request[2]);
    MPI_Isend(pack  [1], N * N, MPI_DOUBLE, rank_minus[0], 101, cart_comm, &request[3]);

    MPI_Irecv(unpack[2], N * N, MPI_DOUBLE, rank_minus[1], 200, cart_comm, &request[4]);
    MPI_Isend(pack  [2], N * N, MPI_DOUBLE, rank_plus [1], 200, cart_comm, &request[5]);

    MPI_Irecv(unpack[3], N * N, MPI_DOUBLE, rank_plus [1], 201, cart_comm, &request[6]);
    MPI_Isend(pack  [3], N * N, MPI_DOUBLE, rank_minus[1], 201, cart_comm, &request[7]);

    MPI_Irecv(unpack[4], N * N, MPI_DOUBLE, rank_minus[2], 300, cart_comm, &request[8]);
    MPI_Isend(pack  [4], N * N, MPI_DOUBLE, rank_plus [2], 300, cart_comm, &request[9]);

    MPI_Irecv(unpack[5], N * N, MPI_DOUBLE, rank_plus [2], 301, cart_comm, &request[10]);
    MPI_Isend(pack  [5], N * N, MPI_DOUBLE, rank_minus[2], 301, cart_comm, &request[11]);
    /**********************************************************************************************/

    // Wait for communication to finish
    MPI_Waitall(12, &request[0], MPI_STATUSES_IGNORE);




  /* Subquestion b: You can delete this part when the subquestion is completed. */
  /************************************************************************************************/
    {
      int array_of_subsizes[3] = {1, N, N};
      int array_of_starts[3] = {0, 1, 1};
      unpack_face(unpack[0], array_of_sizes, array_of_subsizes,
                  array_of_starts);
    }
    {
      int array_of_subsizes[3] = {1, N, N};
      int array_of_starts[3] = {N + 1, 1, 1};
      unpack_face(unpack[1], array_of_sizes, array_of_subsizes,
                  array_of_starts);
    }
    {
      int array_of_subsizes[3] = {N, 1, N};
      int array_of_starts[3] = {1, 0, 1};
      unpack_face(unpack[2], array_of_sizes, array_of_subsizes,
                  array_of_starts);
    }
    {
      int array_of_subsizes[3] = {N, 1, N};
      int array_of_starts[3] = {1, N + 1, 1};
      unpack_face(unpack[3], array_of_sizes, array_of_subsizes,
                  array_of_starts);
    }
    {
      int array_of_subsizes[3] = {N, N, 1};
      int array_of_starts[3] = {1, 1, 0};
      unpack_face(unpack[4], array_of_sizes, array_of_subsizes,
                  array_of_starts);
    }
    {
      int array_of_subsizes[3] = {N, N, 1};
      int array_of_starts[3] = {1, 1, N + 1};
      unpack_face(unpack[5], array_of_sizes, array_of_subsizes,
                  array_of_starts);
    }
   /************************************************************************************************/


    for (int i0 = 1; i0 <= N; i0++)
      for (int i1 = 1; i1 <= N; i1++)
        for (int i2 = 1; i2 <= N; i2++)
          UpdateGridPoint(i0, i1, i2);

    double *temp = u_old;
    u_old = u;
    u = u_new;
    u_new = temp;
    t += dt;
    count++;
  } while (t < t_end);

  double s = 0;
  double Checksum = 0;
  for (int k = 1; k <= N; k++)
    for (int j = 1; j <= N; j++)
      for (int i = 1; i <= N; i++) {
        int m = k + j * (N + 2) + i * (N + 2) * (N + 2);
        s += u[m] * u[m];
      }

  MPI_Reduce(&s, &Checksum, 1, MPI_DOUBLE, MPI_SUM, 0, cart_comm);
  if (rank == 0)
    std::cout << "Checksum = " << Checksum << "\n";

  delete[] pack[5];
  delete[] pack[4];
  delete[] pack[3];
  delete[] pack[2];
  delete[] pack[1];
  delete[] pack[0];
  delete[] unpack[5];
  delete[] unpack[4];
  delete[] unpack[3];
  delete[] unpack[2];
  delete[] unpack[1];
  delete[] unpack[0];

  /* Subquestion b: You should free the custom datatypes and the communicator here. */

  
}
