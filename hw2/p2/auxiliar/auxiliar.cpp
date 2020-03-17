#include "auxiliar.hpp"
#include <math.h>
#include <mpi.h>

void initializeMatrices(double* A, double* B, const int n, const int N, const int rankx, const int ranky)
{
 //initialize value and align for A, B
 double *aptr, *bptr;
 aptr = A; bptr = B;
 for(int i = n*rankx; i < n*(rankx+1); i++)
  for(int j = n*ranky; j < n*(ranky+1); j++)
  {
    *aptr = 1.0/((i +ranky*n)%N + j +1);
    *bptr = 1.0/(i+ (j+rankx*n)%N +1);
    aptr++; bptr++;
  }
}

void verifyMatMulResults(double* C, const int n, const int N, const int rankx, const int ranky, double execTime)
{
 if (rankx == 0 && ranky == 0) printf("Verifying Result... ");

 double tolerance = 1e-6;
 int error = 0;
 double* cptr = C;
 for(int i = n*rankx; i < n*(rankx+1) && !error; i++)
  for(int j = n*ranky; j < n*(ranky+1) && !error; j++, cptr++)
  {
   double tmp = 0;
   for(int k = 0; k < N; k++) tmp += 1.0/((i+k+1)*(k+j+1));
   error = fabs(*cptr-tmp) > tolerance;
  }
 int tempErr = error;
 MPI_Reduce(&tempErr, &error, 1, MPI_INT, MPI_MAX, 0, MPI_COMM_WORLD);

 if (rankx == 0 && ranky == 0)
 {
    if (error) { printf("\n[Error] Verification Failed!\n");  exit(-1); }

    double gflops = (((2e-9)*N)*N)*N/execTime;
    printf("Passed! \n");
    printf("Execution time: %.3fs \n",execTime);
    printf("GFlop/s: %.4f \n",gflops);
 }
}

