#include "wave.h"

//********************************************************
//YOU ONLY NEED TO MODIFY THIS FILE TO COMPLETE HOMEWORK 3
//********************************************************

void WaveEquation::pack_all() {
  // ********************************************************************************
  // Question 2: the number of allocated packs is 6 * threads_per_dim ^ 2
  //             as there are six faces per process and each face corresponds to
  //             threads_per_dim ^ 2 threads. Each pack is associated (numbered)
  //             with a unique ID (variable pid), that is a function of the face
  //             and the thread id (variable tid). Those id's are provided.
  //             Create a parallel region in this function and modify the inputs
  //             of pack_face accordingly, so that each thread packs the data in
  //             its own subdomain. Note that if threads_per_dim = 1, the id's
  //             of each pack reduce to the numbers 0,1,2,3,4,5.
  // ********************************************************************************
  int array_of_sizes[3] = {N + 2, N + 2, N + 2};
  int nloc = N / threads_per_dim;
  int p = threads_per_dim;
  int tid = omp_get_thread_num();
  int t0, t1, t2;
  thread_coordinates(tid, threads_per_dim, t0, t1, t2);

  if (t0 == p - 1) {
    int pid = 0 * threads_per_dim * threads_per_dim + t1 * threads_per_dim + t2;
  }
  if (t0 == 0) {
    int pid = 1 * threads_per_dim * threads_per_dim + t1 * threads_per_dim + t2;
  }
  if (t1 == p - 1) {
    int pid = 2 * threads_per_dim * threads_per_dim + t0 * threads_per_dim + t2;
  }
  if (t1 == 0) {
    int pid = 3 * threads_per_dim * threads_per_dim + t0 * threads_per_dim + t2;
  }
  if (t2 == p - 1) {
    int pid = 4 * threads_per_dim * threads_per_dim + t0 * threads_per_dim + t1;
  }
  if (t2 == 0) {
    int pid = 5 * threads_per_dim * threads_per_dim + t0 * threads_per_dim + t1;
  }

  // You will need to remove this part, when you fill the if statements above
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
}

void WaveEquation::unpack_all() {
  // ********************************************************************************
  // Question 2: the number of allocated unpacks is 6 * threads_per_dim ^ 2
  //             as there are six faces per process and each face corresponds to
  //             threads_per_dim ^ 2 threads. Each unpack is associated
  //             (numbered) with a unique ID (variable pid), that is a function
  //             of the face and the thread id (variable tid). Those id's are
  //             provided.
  //
  //             Each unpack should correspond to the appropriate pack (they
  //             must have the same id).
  //
  //             Create a parallel region in this function and modify the inputs
  //             of unpack_face accordingly, so that each thread unpacks the
  //             data in its own subdomain. Note that if threads_per_dim = 1,
  //             the id's of each unpack reduce to the numbers 0,1,2,3,4,5.
  // ********************************************************************************

  int array_of_sizes[3] = {N + 2, N + 2, N + 2};
  int nloc = N / threads_per_dim;
  int p = threads_per_dim;
  int tid = omp_get_thread_num();
  int t0, t1, t2;
  thread_coordinates(tid, threads_per_dim, t0, t1, t2);

  if (t0 == 0) {
    int pid = 0 * threads_per_dim * threads_per_dim + t1 * threads_per_dim + t2;
  }
  if (t0 == p - 1) {
    int pid = 1 * threads_per_dim * threads_per_dim + t1 * threads_per_dim + t2;
  }
  if (t1 == 0) {
    int pid = 2 * threads_per_dim * threads_per_dim + t0 * threads_per_dim + t2;
  }
  if (t1 == p - 1) {
    int pid = 3 * threads_per_dim * threads_per_dim + t0 * threads_per_dim + t2;
  }
  if (t2 == 0) {
    int pid = 4 * threads_per_dim * threads_per_dim + t0 * threads_per_dim + t1;
  }
  if (t2 == p - 1) {
    int pid = 5 * threads_per_dim * threads_per_dim + t0 * threads_per_dim + t1;
  }

  // You will need to remove this part, when you fill the if statements above
  {
    int array_of_subsizes[3] = {1, N, N};
    int array_of_starts[3] = {0, 1, 1};
    unpack_face(unpack[0], array_of_sizes, array_of_subsizes, array_of_starts);
  }
  {
    int array_of_subsizes[3] = {1, N, N};
    int array_of_starts[3] = {N + 1, 1, 1};
    unpack_face(unpack[1], array_of_sizes, array_of_subsizes, array_of_starts);
  }
  {
    int array_of_subsizes[3] = {N, 1, N};
    int array_of_starts[3] = {1, 0, 1};
    unpack_face(unpack[2], array_of_sizes, array_of_subsizes, array_of_starts);
  }
  {
    int array_of_subsizes[3] = {N, 1, N};
    int array_of_starts[3] = {1, N + 1, 1};
    unpack_face(unpack[3], array_of_sizes, array_of_subsizes, array_of_starts);
  }
  {
    int array_of_subsizes[3] = {N, N, 1};
    int array_of_starts[3] = {1, 1, 0};
    unpack_face(unpack[4], array_of_sizes, array_of_subsizes, array_of_starts);
  }
  {
    int array_of_subsizes[3] = {N, N, 1};
    int array_of_starts[3] = {1, 1, N + 1};
    unpack_face(unpack[5], array_of_sizes, array_of_subsizes, array_of_starts);
  }
}

void WaveEquation::run(double t_end) {
  t = 0;

  // ********************************************************************************
  // Question 2: the number of allocated unpacks is 6 * threads_per_dim ^ 2
  //             as there are six faces per process and each face corresponds to
  //             threads_per_dim ^ 2 threads. You will need to modify the
  //             variable n so that each pack/unpack has the correct size (n =
  //             N^2 is correct only when a single thread is used).
  // ********************************************************************************
  int n = N * N;
  for (int i = 0; i < 6 * threads_per_dim * threads_per_dim; i++) {
    pack[i] = new double[n];
    unpack[i] = new double[n];
  }

  double total_time = 0;

  int count = 0;
  MPI_Barrier(cart_comm);
  double time_start = MPI_Wtime();

  // ********************************************************************************
  // Question 3: The parallel region should start outside the 'while' loop.
  // ********************************************************************************

  do {

    // Question 4: a pragma omp master might make screen output better
    if (count % 100 == 0) {
      if (rank == 0)
        std::cout << count << " t=" << t << "\n";
      // Print(count); //saving data really slows down the code
    }

    pack_all();

    // ********************************************************************************
    // Question 2: multiple threads send and receive messages, according to their
    //             thread id and 'thread coordinates' (t0,t1,t2).
    //             Be careful to correctly match the message tags pid_send and
    //             pid_recv that also correspond to the pack/unpack arrays.
    // ********************************************************************************
    int nloc = N / threads_per_dim;
    int p = threads_per_dim;
    int tid = omp_get_thread_num();
    int t0, t1, t2;
    thread_coordinates(tid, threads_per_dim, t0, t1, t2);
    int pid_send, pid_recv;

    std::vector<MPI_Request> local_request;
    if (t0 == 0) {
      local_request.resize(local_request.size() + 2);

      pid_recv =
          0 * threads_per_dim * threads_per_dim + t1 * threads_per_dim + t2;

      pid_send =
          1 * threads_per_dim * threads_per_dim + t1 * threads_per_dim + t2;
    }
    if (t0 == p - 1) {
      local_request.resize(local_request.size() + 2);

      pid_recv =
          1 * threads_per_dim * threads_per_dim + t1 * threads_per_dim + t2;

      pid_send =
          0 * threads_per_dim * threads_per_dim + t1 * threads_per_dim + t2;
    }
    if (t1 == 0) {
    }
    if (t1 == p - 1) {
    }
    if (t2 == 0) {
    }
    if (t2 == p - 1) {
    }
    // uncomment when you complete question 2
    // MPI_Waitall(local_request.size(),
    // local_request.data(),MPI_STATUSES_IGNORE);

    // ********************************************************************************
    // Question 4: Some computation can be carried out before waiting for
    // communication to finish
    // ********************************************************************************

    // Remove these when you complete question 2
    MPI_Request request[12];
    MPI_Irecv(unpack[0], N * N, MPI_DOUBLE, rank_minus[0], 100, cart_comm,&request[0]);
    MPI_Isend(  pack[0], N * N, MPI_DOUBLE, rank_plus [0], 100, cart_comm,&request[1]);

    MPI_Irecv(unpack[1], N * N, MPI_DOUBLE, rank_plus [0], 101, cart_comm,&request[2]);
    MPI_Isend(  pack[1], N * N, MPI_DOUBLE, rank_minus[0], 101, cart_comm,&request[3]);

    MPI_Irecv(unpack[2], N * N, MPI_DOUBLE, rank_minus[1], 200, cart_comm,&request[4]);
    MPI_Isend(  pack[2], N * N, MPI_DOUBLE, rank_plus [1], 200, cart_comm,&request[5]);

    MPI_Irecv(unpack[3], N * N, MPI_DOUBLE, rank_plus [1], 201, cart_comm,&request[6]);
    MPI_Isend(  pack[3], N * N, MPI_DOUBLE, rank_minus[1], 201, cart_comm,&request[7]);

    MPI_Irecv(unpack[4], N * N, MPI_DOUBLE, rank_minus[2], 300, cart_comm,&request[8]);
    MPI_Isend(  pack[4], N * N, MPI_DOUBLE, rank_plus [2], 300, cart_comm,&request[9]);

    MPI_Irecv(unpack[5], N * N, MPI_DOUBLE, rank_plus [2], 301, cart_comm,&request[10]);
    MPI_Isend(  pack[5], N * N, MPI_DOUBLE, rank_minus[2], 301, cart_comm,&request[11]);

    // Remove this when you complete question 2
    MPI_Waitall(12, &request[0], MPI_STATUSES_IGNORE);

    unpack_all();

    // ********************************************************************************
    // Question 1: parallelize this loop with OPENMP, similarly to the loop
    // found in
    //             auxiliary.cpp in the WaveEquation struct constructor.
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
      int i0_max = i0_min + nloc;
      int i1_max = i1_min + nloc;
      int i2_max = i2_min + nloc;
      for (int i0 = 1 + i0_min; i0 < i0_max + 1; i0++)
        for (int i1 = 1 + i1_min; i1 < i1_max + 1; i1++)
          for (int i2 = 1 + i2_min; i2 < i2_max + 1; i2++)
            UpdateGridPoint(i0, i1, i2);
    }
    // ********************************************************************************

    // ********************************************************************************
    // Question 3: You will need to add the following barrier (why?)
    // ********************************************************************************
    //#pragma omp barrier

    std::swap(u_new, u);
    std::swap(u_new, u_old);
    t += dt;
    count++;

  } while (t < t_end);

  MPI_Barrier(cart_comm);
  total_time = MPI_Wtime() - time_start;

  double total_time_max;
  MPI_Reduce(&total_time, &total_time_max, 1, MPI_DOUBLE, MPI_MAX, 0,
             cart_comm);
  if (rank == 0) {
    std::cout << "Total time = " << total_time_max << "\n";
  }
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

  for (int i = 6 * threads_per_dim * threads_per_dim - 1; i >= 0; i--) {
    delete[] pack[i];
    delete[] unpack[i];
  }
  delete[] pack;
  delete[] unpack;
}
