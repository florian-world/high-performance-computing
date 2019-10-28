#include <iostream>
#include <omp.h>
#include <mkl.h>

#include "cacheflusher.h"


int main(int argc, char *argv[]) {
//  if (argc < 2 || argc > 3 || std::string(argv[1]) == "-h") {
//    fprintf(stderr, "usage: %s N M\n", argv[0]);
//    fprintf(stderr, "Brownian motion of N paritcles in M steps in time");
//    return 1;
//  }

  const int vlen=198;
  char mklversion[vlen];
  mkl_get_version_string(mklversion, vlen);

  std::cout << "OMP Version:" << _OPENMP << std::endl;
  std::cout << "MKL Version:" << mklversion << std::endl;
}
