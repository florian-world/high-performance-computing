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
    int array_of_subsizes[3] = {1, nloc, nloc};
    int array_of_starts[3] = {N, 1+t1*nloc, 1+t2*nloc};
    pack_face(pack[pid], array_of_sizes, array_of_subsizes, array_of_starts);
  }
  if (t0 == 0) {
    int pid = 1 * threads_per_dim * threads_per_dim + t1 * threads_per_dim + t2;
    int array_of_subsizes[3] = {1, nloc, nloc};
    int array_of_starts[3] = {1, 1+t1*nloc, 1+t2*nloc};
    pack_face(pack[pid], array_of_sizes, array_of_subsizes, array_of_starts);
  }
  if (t1 == p - 1) {
    int pid = 2 * threads_per_dim * threads_per_dim + t0 * threads_per_dim + t2;
    int array_of_subsizes[3] = {nloc, 1, nloc};
    int array_of_starts[3] = {1+t0*nloc, N, 1+t2*nloc};
    pack_face(pack[pid], array_of_sizes, array_of_subsizes, array_of_starts);
  }
  if (t1 == 0) {
    int pid = 3 * threads_per_dim * threads_per_dim + t0 * threads_per_dim + t2;
    int array_of_subsizes[3] = {nloc, 1, nloc};
    int array_of_starts[3] = {1+t0*nloc, 1, 1+t2*nloc};
    pack_face(pack[pid], array_of_sizes, array_of_subsizes, array_of_starts);
  }
  if (t2 == p - 1) {
    int pid = 4 * threads_per_dim * threads_per_dim + t0 * threads_per_dim + t1;
    int array_of_subsizes[3] = {nloc, nloc, 1};
    int array_of_starts[3] = {1+t0*nloc, 1+t1*nloc, N};
    pack_face(pack[pid], array_of_sizes, array_of_subsizes, array_of_starts);
  }
  if (t2 == 0) {
    int pid = 5 * threads_per_dim * threads_per_dim + t0 * threads_per_dim + t1;
    int array_of_subsizes[3] = {nloc, nloc, 1};
    int array_of_starts[3] = {1+t0*nloc, 1+t1*nloc, 1};
    pack_face(pack[pid], array_of_sizes, array_of_subsizes, array_of_starts);
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
    int array_of_subsizes[3] = {1, nloc, nloc};
    int array_of_starts[3] = {0, 1+t1*nloc, 1+t2*nloc};
    unpack_face(unpack[pid], array_of_sizes, array_of_subsizes, array_of_starts);
  }
  if (t0 == p - 1) {
    int pid = 1 * threads_per_dim * threads_per_dim + t1 * threads_per_dim + t2;
    int array_of_subsizes[3] = {1, nloc, nloc};
    int array_of_starts[3] = {N+1, 1+t1*nloc, 1+t2*nloc};
    unpack_face(unpack[pid], array_of_sizes, array_of_subsizes, array_of_starts);
  }
  if (t1 == 0) {
    int pid = 2 * threads_per_dim * threads_per_dim + t0 * threads_per_dim + t2;
    int array_of_subsizes[3] = {nloc, 1, nloc};
    int array_of_starts[3] = {1+t0*nloc, 0, 1+t2*nloc};
    unpack_face(unpack[pid], array_of_sizes, array_of_subsizes, array_of_starts);
  }
  if (t1 == p - 1) {
    int pid = 3 * threads_per_dim * threads_per_dim + t0 * threads_per_dim + t2;
    int array_of_subsizes[3] = {nloc, 1, nloc};
    int array_of_starts[3] = {1+t0*nloc, N+1, 1+t2*nloc};
    unpack_face(unpack[pid], array_of_sizes, array_of_subsizes, array_of_starts);
  }
  if (t2 == 0) {
    int pid = 4 * threads_per_dim * threads_per_dim + t0 * threads_per_dim + t1;
    int array_of_subsizes[3] = {nloc, nloc, 1};
    int array_of_starts[3] = {1+t0*nloc, 1+t1*nloc, 0};
    unpack_face(unpack[pid], array_of_sizes, array_of_subsizes, array_of_starts);
  }
  if (t2 == p - 1) {
    int pid = 5 * threads_per_dim * threads_per_dim + t0 * threads_per_dim + t1;
    int array_of_subsizes[3] = {nloc, nloc, 1};
    int array_of_starts[3] = {1+t0*nloc, 1+t1*nloc, N+1};
    unpack_face(unpack[pid], array_of_sizes, array_of_subsizes, array_of_starts);
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
  int n = N * N / (threads_per_dim*threads_per_dim);
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

#pragma omp parallel
  {
    do {
      // Question 4: a pragma omp master might make screen output better
      #pragma omp master
      {
        if (count % 10 == 0) {
          if (rank == 0)
            std::cout << count << " t=" << t << std::endl;
          // Print(count); //saving data really slows down the code
        }
      }


      // ********************************************************************************
      // Question 2: multiple threads send and receive messages, according to their
      //             thread id and 'thread coordinates' (t0,t1,t2).
      //             Be careful to correctly match the message tags pid_send and
      //             pid_recv that also correspond to the pack/unpack arrays.
      // ********************************************************************************
      // call pack_all inside the parallel region (easier refactoring later)
      pack_all();
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

        MPI_Irecv(unpack[pid_recv], nloc * nloc, MPI_DOUBLE, rank_minus[0], pid_recv, cart_comm, &local_request[local_request.size()-2]);
        MPI_Isend(  pack[pid_send], nloc * nloc, MPI_DOUBLE, rank_plus [0], pid_send, cart_comm, &local_request[local_request.size()-1]);
      }
      if (t0 == p - 1) {
        local_request.resize(local_request.size() + 2);


        pid_recv =
            1 * threads_per_dim * threads_per_dim + t1 * threads_per_dim + t2;

        pid_send =
            0 * threads_per_dim * threads_per_dim + t1 * threads_per_dim + t2;

        MPI_Irecv(unpack[pid_recv], nloc * nloc, MPI_DOUBLE, rank_plus [0], pid_recv, cart_comm, &local_request[local_request.size()-2]);
        MPI_Isend(  pack[pid_send], nloc * nloc, MPI_DOUBLE, rank_minus[0], pid_send, cart_comm, &local_request[local_request.size()-1]);
      }
      if (t1 == 0) {
        local_request.resize(local_request.size() + 2);

        pid_recv =
            2 * threads_per_dim * threads_per_dim + t0 * threads_per_dim + t2;

        pid_send =
            3 * threads_per_dim * threads_per_dim + t0 * threads_per_dim + t2;

        MPI_Irecv(unpack[pid_recv], nloc * nloc, MPI_DOUBLE, rank_minus[1], pid_recv, cart_comm, &local_request[local_request.size()-2]);
        MPI_Isend(  pack[pid_send], nloc * nloc, MPI_DOUBLE, rank_plus [1], pid_send, cart_comm, &local_request[local_request.size()-1]);
      }
      if (t1 == p - 1) {
        local_request.resize(local_request.size() + 2);

        pid_recv =
            3 * threads_per_dim * threads_per_dim + t0 * threads_per_dim + t2;

        pid_send =
            2 * threads_per_dim * threads_per_dim + t0 * threads_per_dim + t2;

        MPI_Irecv(unpack[pid_recv], nloc * nloc, MPI_DOUBLE, rank_plus [1], pid_recv, cart_comm, &local_request[local_request.size()-2]);
        MPI_Isend(  pack[pid_send], nloc * nloc, MPI_DOUBLE, rank_minus[1], pid_send, cart_comm, &local_request[local_request.size()-1]);
      }
      if (t2 == 0) {
        local_request.resize(local_request.size() + 2);

        pid_recv =
            4 * threads_per_dim * threads_per_dim + t0 * threads_per_dim + t1;

        pid_send =
            5 * threads_per_dim * threads_per_dim + t0 * threads_per_dim + t1;

        MPI_Irecv(unpack[pid_recv], nloc * nloc, MPI_DOUBLE, rank_minus[2], pid_recv, cart_comm, &local_request[local_request.size()-2]);
        MPI_Isend(  pack[pid_send], nloc * nloc, MPI_DOUBLE, rank_plus [2], pid_send, cart_comm, &local_request[local_request.size()-1]);
      }
      if (t2 == p - 1) {
        local_request.resize(local_request.size() + 2);

        pid_recv =
            5 * threads_per_dim * threads_per_dim + t0 * threads_per_dim + t1;

        pid_send =
            4 * threads_per_dim * threads_per_dim + t0 * threads_per_dim + t1;

        MPI_Irecv(unpack[pid_recv], nloc * nloc, MPI_DOUBLE, rank_plus [2], pid_recv, cart_comm, &local_request[local_request.size()-2]);
        MPI_Isend(  pack[pid_send], nloc * nloc, MPI_DOUBLE, rank_minus[2], pid_send, cart_comm, &local_request[local_request.size()-1]);
      }


      // ********************************************************************************
      // Question 4: Some computation can be carried out before waiting for
      // communication to finish
      // ********************************************************************************


      // ********************************************************************************
      // Question 1: parallelize this loop with OPENMP, similarly to the loop
      // found in
      //             auxiliary.cpp in the WaveEquation struct constructor.
      // ********************************************************************************

      int i0_min = t0 * nloc;
      int i1_min = t1 * nloc;
      int i2_min = t2 * nloc;
      int i0_max = i0_min + nloc;
      int i1_max = i1_min + nloc;
      int i2_max = i2_min + nloc;
      for (int i0 = i0_min + (t0 == 0 ? 2 : 1); i0 < i0_max + (t0 == p-1 ? 0 : 1); i0++)
        for (int i1 = i1_min + (t1 == 0 ? 2 : 1); i1 < i1_max + (t1 == p-1 ? 0 : 1); i1++)
          for (int i2 = i2_min + (t2 == 0 ? 2 : 1); i2 < i2_max + (t2 == p-1 ? 0 : 1); i2++)
            UpdateGridPoint(i0, i1, i2);


      MPI_Waitall(local_request.size(), local_request.data(),MPI_STATUSES_IGNORE);
      unpack_all();

      for (int i1 = 1 + i1_min; i1 < i1_max + 1; i1++)
        for (int i2 = 1 + i2_min; i2 < i2_max + 1; i2++) {
          if (t0 == 0) UpdateGridPoint(1+i0_min, i1, i2);
          if (t0 == p-1) UpdateGridPoint(i0_max, i1, i2);
        }

      for (int i0 = 1 + i0_min; i0 < i0_max + 1; i0++)
        for (int i2 = 1 + i2_min; i2 < i2_max + 1; i2++) {
          if (t1 == 0) UpdateGridPoint(i0, 1+i1_min, i2);
          if (t1 == p-1) UpdateGridPoint(i0, i1_max, i2);
        }

      for (int i0 = 1 + i0_min; i0 < i0_max + 1; i0++)
        for (int i1 = 1 + i1_min; i1 < i1_max + 1; i1++) {
          if (t2 == 0) UpdateGridPoint(i0, i1, 1+i2_min);
          if (t2 == p-1) UpdateGridPoint(i0, i1, i2_max);
        }

      // ********************************************************************************

      // ********************************************************************************
      // Question 3: You will need to add the following barrier (why?)
      // ********************************************************************************
      #pragma omp barrier // to make sure all calculation is done before it's swapped

      #pragma omp single
      {
        std::swap(u_new, u);
        std::swap(u_new, u_old);
        t += dt;
        count++;
      }

      // implicit barrier is important here to get the exit criteria right (!) [e.g. omp master does not have one]

    } while (t < t_end);
  }

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
