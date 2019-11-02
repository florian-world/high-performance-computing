#include "utils.h"

#include <mkl_cblas.h>

using namespace std;

#ifndef SQUARE
#define SQUARE(BASE) ((BASE) * (BASE))
#endif

#ifndef RMJ
#define RMJ(i,j,rows,cols) ((i)*(cols)+(j))
#endif
#ifndef CMJ
#define CMJ(i,j,rows,cols) ((i)*(j)*(rows))
#endif

double *utils::loadDataset(const std::string &data_path, const int N, const int D)
{
  // Input file stream object to read data
  std::ifstream inputFile;
  inputFile.open(data_path);

  const std::string delimiter = ",";
  std::string line;
  std::string strnum;

  size_t pos = 0;
  std::string token;

  double *data = new double[N * D];

  int stillToLoad = 0;
  int n = 0;
  // parse line by line
  while (std::getline(inputFile, line))
  {
    // Input dimension
    int d = 0;
    while ((pos = line.find(delimiter)) != std::string::npos) {
      token = line.substr(0, pos);
      double number = std::stod(token);
      data[n*D + d] = number;
      d++;
      line.erase(0, pos + delimiter.length());
      stillToLoad = N-n;
      if(stillToLoad<0){
        break;
      }
    }
    if(stillToLoad<=0){
      break;
    }
    // Last element:
    token = line;
    double number = std::stod(token);
    data[n*D + d] = number;
    d++;
    n++;
  }
  std::cout << "Number of datapoints loaded: " << n << '\n';
  return data;
}

int utils::writeRowMajorMatrixToFile(const std::string &data_path, const double *M, const int rows, const int cols)
{
  // Row major encoding M(i,j)=M[i*cols+j]
  std::ofstream oFile;
  oFile.open(data_path);
  for(int i=0; i<rows;++i){
    for(int j=0; j<cols;++j){
      oFile << M[i*cols+j];
      if(j<cols-1){
        oFile << ",";
      }
    }
    oFile << "\n";
  }
  oFile.close();

  return 0;
}

int utils::writeColMajorMatrixToFile(const std::string &data_path, const double *M, const int rows, const int cols)
{
  // Col major encoding M(i,j)=M[j + i*rows]
  std::ofstream oFile;
  oFile.open(data_path);
  for(int i=0; i<rows;++i){
    for(int j=0; j<cols;++j){
      oFile << M[j*rows+i];
      if(j<cols-1){
        oFile << ",";
      }
    }
    oFile << "\n";
  }
  oFile.close();
  return 0;
}

void utils::copyWeights(double *W1, const double * const W2, const int SIZE)
{
  for(int i=0; i<SIZE;++i){
    W1[i] = W2[i];
  }
}

void utils::printArrayMatrix(const double * const array, const int sizeX, const int sizeY)
{
  std::cout << "ARRAY:\n";
  for(int i=0; i<sizeX;++i){
    for(int o=0; o<sizeY;++o){
      std::cout << array[o + i * sizeY];
      if(o<sizeY-1){
        std::cout << ",";
      }
    }
    std::cout << "\n";
  }
  std::cout << "\n";
}

double utils::computeArrayMatrixNorm(const double * const M, const int D)
{
  double norm_ = 0.0;
  for(int i=0; i<D;++i){
    assert(M[i]==M[i]);
    assert(!isinf(M[i]));
    norm_ += std::pow(M[i],2);
  }
  norm_=std::sqrt(norm_);
  return norm_;
}

std::vector<double> utils::computeComponentNorms(const double * const M, const int num_comp, const int D)
{
  std::vector<double> comp_norms(num_comp, 0.0);
  for(int m=0; m<num_comp;++m){
    for(int i=0; i<D;++i){
      comp_norms[m]+=std::pow(M[m + i*num_comp],2);
    }
    comp_norms[m] = std::sqrt(comp_norms[m]);
  }
  return comp_norms;
}

void utils::plotComponentNorms(const double * const weights, const int num_comp, const int D)
{
  std::vector<double> comp_norms = computeComponentNorms(weights, num_comp, D);
  std::cout << "Perceptrons component norms |W|: \n";
  for(size_t m=0; m<comp_norms.size(); ++m)
  {
    std::cout << "["<<m<<"]= " << comp_norms[m]  << ", ";
  }
  std::cout << std::endl;
}

double utils::computeArrayMatrixDifferenceNorm(const double * const M1, const double * const M2, const int D)
{
  double norm_ = 0.0;
  for(int i=0; i<D;++i){
    assert(M1[i]==M1[i]);
    assert(M2[i]==M2[i]);

    norm_ += std::pow(M1[i] - M2[i],2);
  }
  norm_=std::sqrt(norm_);
  return norm_;
}

int utils::writeVectorToFile(const std::string &data_path, const std::vector<double> &array)
{
  std::ofstream oFile;
  oFile.open(data_path);
  for (size_t o = 0; o < array.size(); ++o) {
    oFile << array[o];
    if (o < array.size() - 1) {
      oFile << ",";
    }
  }
  oFile << "\n";
  oFile.close();
  return 0;
}

void utils::reverseArray(double *data, const int D)
{
  int DD = (int)std::ceil((double)D / 2.0);
  for (int j = 0; j < DD; ++j) {
    double temp = data[j];
    data[j] = data[D-j-1];
    data[D-j-1] = temp;
  }
}

void utils::transposeData(double *data_T, const double * const data, const int N, const int D)
{
  // Data are given in the form data(n,d)=data[n*D+d]
  // This function transposes the data to data(n,d)=data_T[d*N+n]
  // for the purpose of more efficient memory layout
  // (e.g. calculation of the mean)

  // TODO:

  for (int n = 0; n < N; ++n) {
    for (int d = 0; d < D; ++d) {
      data_T[d*N+n] = data[n*D+d];
    }
  }
  // :TODO
}

void utils::computeMean(double *mean, const double * const data_T, const int N, const int D)
{
  // Calculation of the mean (over samples) of the dataset
  // data(n,d)=data_T[d*N+n]

  // TODO:
#pragma omp parallel for
  for (int d = 0; d < D; ++d) {
    mean[d] = data_T[d*N];
    for (int n = 1; n < N; ++n) {
      // avoid loosing precision due to huge sums -> calculate step wise
      mean[d] = (n*mean[d] + data_T[d*N+n]) / (n+1);
    }
  }

  // :TODO
}

void utils::computeStd(double *std, const double * const mean, const double * const data_T, const int N, const int D)
{
  // Calculation of the mean (over samples) of the dataset
  // data(n,d)=data_T[d*N+n]

  // TODO:
#pragma omp parallel for
  for (int d = 0; d < D; ++d) {
    std[d] = SQUARE(mean[d] - data_T[d*N]);
    for (int n = 1; n < N; ++n) {
      // avoid loosing precision due to huge sums -> calculate step wise
      std[d] = (n*std[d] + SQUARE(mean[d] - data_T[d*N+n])) / (n+1);
    }
    // Bessel's correction -> not needed here to comply with PYTHON results
//    std[d] = std[d] * N / (N-1);
    std[d] = std::sqrt(std[d]);
  }


  // :TODO
}

void utils::standardizeColMajor(double *data_T, const double * const mean, const double * const std, const int N, const int D)
{
  std::cout << "Scaling - zero mean, unit variance." << std::endl;
  // COL MAJOR IMPLEMENTATION
  // Data normalization (or standardization)
  // Transormation of the data to zero mean unit variance.
  // data(n,d)=data_T[d*N+n]

  // TODO:

#pragma omp parallel for
  for (int d = 0; d < D; ++d) {
    for (int n = 0; n < N; ++n) {
      data_T[d*N+n] = (data_T[d*N+n] - mean[d]) / std[d];
    }
  }

  // :TODO
}

void utils::centerDataColMajor(double *data_T, const double * const mean, const int N, const int D)
{
  std::cout << "Centering data..." << std::endl;
  // COL MAJOR IMPLEMENTATION
  // data(n,d)=data_T[d*N + n]

  // TODO:

#pragma omp parallel for
  for (int d = 0; d < D; ++d) {
    for (int n = 0; n < N; ++n) {
      data_T[d*N+n] = data_T[d*N+n] - mean[d];
    }
  }

  // :TODO
}

void utils::standardizeRowMajor(double *data, const double * const mean, const double * const std, const int N, const int D)
{
  std::cout << "Scaling - zero mean, unit variance." << std::endl;
  // ROW MAJOR IMPLEMENTATION
  // Data normalization (or standardization)
  // Transormation of the data to zero mean unit variance.
  // data(n,d)=data[n*D+d]
#pragma omp parallel for
  for (int n = 0; n < N; ++n) {
    for (int d = 0; d < D; ++d) {
      data[n*D+d] = (data[n*D+d] - mean[d]) / std[d];
    }
  }
}

void utils::centerDataRowMajor(double *data, const double * const mean, const int N, const int D)
{
  std::cout << "Centering data..." << std::endl;
  // ROW MAJOR IMPLEMENTATION
  // data(n,d)=data[n*D+d]
#pragma omp parallel for
  for (int n = 0; n < N; ++n) {
    for (int d = 0; d < D; ++d) {
      data[n*D + d] = data[n*D + d] - mean[d];
    }
  }
}

void utils::constructCovariance(double *C, const double * const data_T, const int N, const int D)
{
  // Construct the covariance matrix (DxD) of the data.
  // data(n,d)=data_T[d*N+n]
  // For the covariance follow the row major notation
  // C(j,k)=C[j*D+k]

  // TODO:

  // note that the output is column major, but it does not matter since it will be symmetric by design
  double alpha = 1.0 / (N-1);

  cblas_dgemm(CBLAS_LAYOUT::CblasColMajor, CBLAS_TRANSPOSE::CblasTrans, CBLAS_TRANSPOSE::CblasNoTrans,
              D, D, N, // m n k
              alpha, data_T, N, data_T, N,
              0.0, C, D);

  // :TODO
}

void utils::getEigenvectors(double *V, const double * const C, const int NC, const int D)
{
  // Extracting the last rows from matrix C containig the PCA components (eigenvectors of the covariance matrix) that explain the highest variance.
  // Be carefull to extract them in order of descenting variance.
  // C(j,d)=C[j*D+d] # ROW MAJOR
  // V(k,d)=V[k*D+d] # ROW MAJOR

  for (int n = 0; n < NC; ++n) {
    for (int d = 0; d < D; ++d) {
      // read last rows of C (eigenvectors with highest eigenvalue)
      V[RMJ(n,d,NC,D)] = C[RMJ(D-n-1,d,D,D)];
    }
  }

  // TODO
}

void utils::reduceDimensionality(double *data_red, const double * const V, const double * const data_T, const int N, const int D, const int NC)
{
  // data(n,d)=data_T[d*N+n] (transposed dataset)
  // V(k,d)=V[k*D+d]
  // data_red(n,k)=data_red[n*NC + k], K<<D

#pragma omp parallel for
  for (int n = 0; n < N; ++n) // Iterate through all data
  {
    for (int k = 0; k < NC; ++k) // Iterate through all components
    {
      double sum = 0.0;
      for (int d = 0; d < D; ++d) // Iterate through the dimensions
      {
        sum += V[k*D + d] * data_T[d*N + n];
      }
      data_red[n*NC + k] = sum;
    }
  }
}

void utils::reconstructDatasetRowMajor(double *data_rec, const double * const V, const double * const data_red, const int N, const int D, const int NC)
{
  // ROW MAJOR
  // V(c,d)=V[d + c*D]
  // data_red(n,c)=data_red[c + n*NC], C<<D   # ROW MAJOR
  // data_rec(n,d)=data_rec[d + n*D]          # ROW MAJOR

  // TODO:

  cblas_dgemm(CBLAS_LAYOUT::CblasRowMajor, CBLAS_TRANSPOSE::CblasNoTrans, CBLAS_TRANSPOSE::CblasNoTrans,
              N, D, NC, // m n k
              1.0, data_red, NC, V, D,
              0.0, data_rec, D);

  // :TODO
}

void utils::reconstructDatasetColMajor(double *data_rec, const double * const V, const double * const data_red, const int N, const int D, const int NC)
{
  // COL MAJOR:
  // V(c,d)=V[c + d*NC]  # COL MAJOR
  // (since weights are given by weights[o + i*nOutputs] where o=c components)

  // ROW MAJOR:
  // data_red(n,c) = data_red[c + n*NC] (output[o + k*nOutputs])
  // data_rec(n,d) = data_rec[d + n*D] ALSO ROW MAJOR

#pragma omp parallel for
  for (int n = 0; n < N; ++n) // Iterate through all data
  {
    for (int d = 0; d < D; ++d) // Iterate through all dimensions
    {
      double sum = 0.0;
      for (int c = 0; c < NC; ++c) // Iterate through components
      {
        // TODO: Fill the line here
        sum += V[c + d*NC] * data_red[c + n*NC];
        // TODO:
      }
      data_rec[d + n*D] = sum;
    }
  }
}

void utils::inverseStandarizeDatasetRowMajor(double *data_rec, const double * const mean, const double * const std, const int N, const int D)
{
  // ROW MAJOR
  // data_rec(n,d)=data_rec[d + n*D]          # ROW MAJOR
  for (int n = 0; n < N; ++n) // Iterate through all data
  {
    for (int d = 0; d < D; ++d) // Iterate through all dimensions
    {
      data_rec[n*D + d] = data_rec[n*D + d] * std[d] + mean[d];
    }
  }
}

void utils::inverseCenterDatasetRowMajor(double *data_rec, const double * const mean, const int N, const int D)
{
  // ROW MAJOR
  // data_rec(n,d)=data_rec[d + n*D]          # ROW MAJOR
  for (int n = 0; n < N; ++n) // Iterate through all data
  {
    for (int d = 0; d < D; ++d) // Iterate through all dimensions
    {
      data_rec[n*D + d] = data_rec[n*D + d] + mean[d];
    }
  }
}
