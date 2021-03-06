#include "wave.h"

#include <cassert>

/********************************************************************/ 
/* Subquestion a: change the following function and use a Cartesian */ 
/* topology to find coords[3], rank_plus[3] and rank_minus[3]       */
/********************************************************************/

void WaveEquation::setValueOnGrid(int i0, int i1, int i2, double value)
{
  u[(i0) * (N + 2) * (N + 2) + (i1) * (N + 2) + (i2)] = value;
}

void WaveEquation::FindCoordinates() {

  int nums[3] = {0,0,0};
  int periodic[3] = {false, false, false};
  MPI_Dims_create(size, 3, nums); // split the nodes automatically

  MPI_Cart_create(MPI_COMM_WORLD, 3, nums, periodic, true, &cart_comm);
  MPI_Comm_rank(cart_comm, &rank);
  if (rank == 0) {
    std::cout << "Grid: " << nums[0] << ", " << nums[1] << ", " << nums[2] << std::endl;
  }

  MPI_Cart_shift(cart_comm, 0, 1, &rank_minus[0], &rank_plus[0]);
  MPI_Cart_shift(cart_comm, 1, 1, &rank_minus[1], &rank_plus[1]);
  MPI_Cart_shift(cart_comm, 2, 1, &rank_minus[2], &rank_plus[2]);
  MPI_Cart_coords(cart_comm, rank, 3, coords);
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

  /* Subquestion b: Create and commit custom datatypes here */
  /************************************************************************************************/
  MPI_Datatype SEND_FACE_PLUS [3];
  MPI_Datatype SEND_FACE_MINUS[3]; 

  MPI_Datatype RECV_FACE_PLUS [3];
  MPI_Datatype RECV_FACE_MINUS[3];

  // same for all
  int sizes[3] = {N+2, N+2, N+2};
  int subsizes[3][3] = {{1, N, N},
                        {N, 1, N},
                        {N, N, 1}};

  /** SEND **/
  // same for all faces
  int sendStartsMinus[3] = {1,1,1};
  int sendStartsPlus[3][3] = {{N, 1, 1},
                              {1, N, 1},
                              {1, 1, N}};

  MPI_Type_create_subarray(3, sizes, subsizes[0], sendStartsMinus, MPI_ORDER_C, MPI_DOUBLE, &SEND_FACE_MINUS[0]);
  MPI_Type_create_subarray(3, sizes, subsizes[1], sendStartsMinus, MPI_ORDER_C, MPI_DOUBLE, &SEND_FACE_MINUS[1]);
  MPI_Type_create_subarray(3, sizes, subsizes[2], sendStartsMinus, MPI_ORDER_C, MPI_DOUBLE, &SEND_FACE_MINUS[2]);
  MPI_Type_create_subarray(3, sizes, subsizes[0], sendStartsPlus[0], MPI_ORDER_C, MPI_DOUBLE, &SEND_FACE_PLUS[0]);
  MPI_Type_create_subarray(3, sizes, subsizes[1], sendStartsPlus[1], MPI_ORDER_C, MPI_DOUBLE, &SEND_FACE_PLUS[1]);
  MPI_Type_create_subarray(3, sizes, subsizes[2], sendStartsPlus[2], MPI_ORDER_C, MPI_DOUBLE, &SEND_FACE_PLUS[2]);

  /** RECEIVE **/
  int recvStartsMinus[3][3] = {{0, 1, 1},
                               {1, 0, 1},
                               {1, 1, 0}};
  int recvStartsPlus[3][3] = {{N + 1, 1, 1},
                              {1, N + 1, 1},
                              {1, 1, N + 1}};

  MPI_Type_create_subarray(3, sizes, subsizes[0], recvStartsMinus[0], MPI_ORDER_C, MPI_DOUBLE, &RECV_FACE_MINUS[0]);
  MPI_Type_create_subarray(3, sizes, subsizes[1], recvStartsMinus[1], MPI_ORDER_C, MPI_DOUBLE, &RECV_FACE_MINUS[1]);
  MPI_Type_create_subarray(3, sizes, subsizes[2], recvStartsMinus[2], MPI_ORDER_C, MPI_DOUBLE, &RECV_FACE_MINUS[2]);
  MPI_Type_create_subarray(3, sizes, subsizes[0], recvStartsPlus[0], MPI_ORDER_C, MPI_DOUBLE, &RECV_FACE_PLUS[0]);
  MPI_Type_create_subarray(3, sizes, subsizes[1], recvStartsPlus[1], MPI_ORDER_C, MPI_DOUBLE, &RECV_FACE_PLUS[1]);
  MPI_Type_create_subarray(3, sizes, subsizes[2], recvStartsPlus[2], MPI_ORDER_C, MPI_DOUBLE, &RECV_FACE_PLUS[2]);

  MPI_Type_commit(&SEND_FACE_PLUS[0]); MPI_Type_commit(&SEND_FACE_PLUS[1]); MPI_Type_commit(&SEND_FACE_PLUS[2]);
  MPI_Type_commit(&SEND_FACE_MINUS[0]); MPI_Type_commit(&SEND_FACE_MINUS[1]); MPI_Type_commit(&SEND_FACE_MINUS[2]);
  MPI_Type_commit(&RECV_FACE_PLUS[0]); MPI_Type_commit(&RECV_FACE_PLUS[1]); MPI_Type_commit(&RECV_FACE_PLUS[2]);
  MPI_Type_commit(&RECV_FACE_MINUS[0]); MPI_Type_commit(&RECV_FACE_MINUS[1]); MPI_Type_commit(&RECV_FACE_MINUS[2]);

  /************************************************************************************************/

  /** Set non-periodic boundary conditions **/

  // for previous version of problem statement:
//  for (int i1 = 0; i1 < N; i1++)
//    for (int i2 = 0; i2 < N; i2++) {
//      if (rank_minus[0] == MPI_PROC_NULL) setValueOnGrid(0,i1,i2, exp(-10));
//      if (rank_plus[0] == MPI_PROC_NULL) setValueOnGrid(N+1,i1,i2, exp(-10));
//      if (rank_minus[1] == MPI_PROC_NULL) setValueOnGrid(i1,0,i2, exp(-10));
//      if (rank_plus[1] == MPI_PROC_NULL) setValueOnGrid(i1,N+1,i2, exp(-10));
//      if (rank_minus[2] == MPI_PROC_NULL) setValueOnGrid(i1,i2,0, exp(-10));
//      if (rank_plus[2] == MPI_PROC_NULL) setValueOnGrid(i1,i2,N+1, exp(-10));
//    }


  int count = 0;
  do {
    if (count % 5 == 0) {
      if (rank == 0)
        std::cout << count << " t=" << t << "\n";
      Print(count);
    }

    MPI_Request request[12];

    /* Subquestion b: Replace the sends and receives with ones that correspond to custom datatypes*/
    /**********************************************************************************************/
    MPI_Irecv(u, 1, RECV_FACE_MINUS[0], rank_minus[0], 100, cart_comm, &request[0]);
    MPI_Isend(u, 1, SEND_FACE_PLUS[0], rank_plus [0], 100, cart_comm, &request[1]);

    MPI_Irecv(u, 1, RECV_FACE_PLUS[0], rank_plus [0], 101, cart_comm, &request[2]);
    MPI_Isend(u, 1, SEND_FACE_MINUS[0], rank_minus[0], 101, cart_comm, &request[3]);

    MPI_Irecv(u, 1, RECV_FACE_MINUS[1], rank_minus[1], 200, cart_comm, &request[4]);
    MPI_Isend(u, 1, SEND_FACE_PLUS[1], rank_plus [1], 200, cart_comm, &request[5]);

    MPI_Irecv(u, 1, RECV_FACE_PLUS[1], rank_plus [1], 201, cart_comm, &request[6]);
    MPI_Isend(u, 1, SEND_FACE_MINUS[1], rank_minus[1], 201, cart_comm, &request[7]);

    MPI_Irecv(u, 1, RECV_FACE_MINUS[2], rank_minus[2], 300, cart_comm, &request[8]);
    MPI_Isend(u, 1, SEND_FACE_PLUS[2], rank_plus [2], 300, cart_comm, &request[9]);

    MPI_Irecv(u, 1, RECV_FACE_PLUS[2], rank_plus [2], 301, cart_comm, &request[10]);
    MPI_Isend(u, 1, SEND_FACE_MINUS[2], rank_minus[2], 301, cart_comm, &request[11]);
    /**********************************************************************************************/

    // Wait for communication to finish
    MPI_Waitall(12, &request[0], MPI_STATUSES_IGNORE);

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

  /* Subquestion b: You should free the custom datatypes and the communicator here. */

  MPI_Type_free(&SEND_FACE_PLUS[0]); MPI_Type_free(&SEND_FACE_PLUS[1]); MPI_Type_free(&SEND_FACE_PLUS[2]);
  MPI_Type_free(&SEND_FACE_MINUS[0]); MPI_Type_free(&SEND_FACE_MINUS[1]); MPI_Type_free(&SEND_FACE_MINUS[2]);
  MPI_Type_free(&RECV_FACE_PLUS[0]); MPI_Type_free(&RECV_FACE_PLUS[1]); MPI_Type_free(&RECV_FACE_PLUS[2]);
  MPI_Type_free(&RECV_FACE_MINUS[0]); MPI_Type_free(&RECV_FACE_MINUS[1]); MPI_Type_free(&RECV_FACE_MINUS[2]);

  MPI_Comm_free(&cart_comm);
}
