#ifndef UTILS_H_UY1M7JFH
#define UTILS_H_UY1M7JFH

#include <iostream>
#include <fstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <cmath>
#include <cassert> /* assert */

namespace utils{

  double* loadDataset(const std::string & data_path, const int N, const int D);

  int writeRowMajorMatrixToFile(const std::string &data_path,
                                const double *M,
                                const int rows,
                                const int cols);

  int writeColMajorMatrixToFile(const std::string &data_path,
                                const double *M,
                                const int rows,
                                const int cols);

  void copyWeights(double* W1, const double* const W2, const int SIZE);

  void printArrayMatrix(const double* const array, const int sizeX, const int sizeY);


  double computeArrayMatrixNorm(const double* const M, const int D);
  std::vector<double> computeComponentNorms(const double* const M, const int num_comp, const int D);

  void plotComponentNorms(const double* const weights, const int num_comp, const int D);


  double computeArrayMatrixDifferenceNorm(const double* const M1, const double* const M2, const int D);



  int writeVectorToFile(const std::string &data_path,
                        const std::vector<double> &array);

  void reverseArray(double* data, const int D);

  void transposeData(double* data_T, const double* const data, const int N, const int D);

  void computeMean(double* mean, const double* const data_T, const int N, const int D);

  void computeVar(double* std, const double* const mean, const double* const data_T, const int N, const int D);

  void computeStd(double* std, const double* const mean, const double* const data_T, const int N, const int D);

  void standardizeColMajor(double* data_T, const double* const mean, const double* const std, const int N, const int D);

  void centerDataColMajor(double* data_T, const double* const mean, const int N, const int D);

  void standardizeRowMajor(double* data, const double* const mean, const double* const std, const int N, const int D);

  void centerDataRowMajor(double* data, const double* const mean, const int N, const int D);

  void constructCovariance(double* C, const double* const data_T, const int N, const int D);

  void getEigenvectors(double* V, const double* const C, const int NC, const int D);

  void reduceDimensionality(double* data_red, const double* const V, const double* const data_T, const int N, const int D, const int NC);

  void reconstructDatasetRowMajor(double* data_rec, const double* const V, const double* const data_red, const int N, const int D, const int NC);

  void reconstructDatasetColMajor(double* data_rec, const double* const V, const double* const data_red, const int N, const int D, const int NC);

  void inverseStandarizeDatasetRowMajor(double* data_rec, const double* const mean, const double* const std, const int N, const int D);

  void inverseCenterDatasetRowMajor(double* data_rec, const double* const mean, const int N, const int D);

}

#endif /* UTILS_H_UY1M7JFH */
