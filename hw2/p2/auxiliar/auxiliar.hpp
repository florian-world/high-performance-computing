#ifndef _HPCSE2_AUXILIAR_
#define _HPCSE2_AUXILIAR_

void verifyMatMulResults(double* C, const int n, const int N, const int rankx, const int ranky, double execTime);
void initializeMatrices(double* A, double* B, const int n, const int N, const int rankx, const int ranky);

#endif
