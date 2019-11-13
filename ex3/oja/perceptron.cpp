
#include "perceptron.h"

#include "../code/utils.h"

#include <iostream>
#include <iomanip>
#include <mkl_cblas.h>
#include <omp.h>
#include <mkl.h>


void Perceptron::initializeWeights()
{
  if (weight_init.compare("allsame") == 0){
    std::cout << "Initializing all weights with 1/sqrt(nInputs), |w|=1." << std::endl;
    double fac = 1.0/std::sqrt(nInputs);
    for(int i=0; i<nInputs;++i){
      for(int o=0; o<nOutputs;++o){
        weights[o + i*nOutputs] = fac;
        weights_prev[o + i*nOutputs] = fac;
      }
    }
  }
  else if (weight_init.compare("normal") == 0){
    std::cout << "Initializing all weights with random normal." << std::endl;
    std::default_random_engine generator;
    std::normal_distribution<double> distribution(0.0,1.0);
    for(int i=0; i<nInputs;++i){
      for(int o=0; o<nOutputs;++o){
        double number = distribution(generator);
        weights[o + i*nOutputs] = number;
        weights_prev[o + i*nOutputs] = number;
      }
    }
  }
}

double *Perceptron::forward(const double * const input, const int batch_size) {
  // Input dimension [batch_size, nInputs]
  assert(batch_size > 0);
  assert(batch_size <= max_batch_size);


  mkl_set_num_threads(8);
  cblas_dgemm(CBLAS_LAYOUT::CblasRowMajor, CBLAS_TRANSPOSE::CblasNoTrans, CBLAS_TRANSPOSE::CblasNoTrans,
              batch_size, nOutputs, nInputs, // m n k
              1.0, input, nInputs, weights, nOutputs,
              0.0, output, nOutputs);

//  memset(output, 0.0, sizeof(double) * batch_size * nOutputs);
//  for (int k = 0; k < batch_size; ++k) {
//    for (int i = 0; i < nInputs; ++i) {
//      for (int o = 0; o < nOutputs; ++o) {
//        // TODO:
//         output[o + k * nOutputs] += weights[i*nOutputs+o]*input[k*nInputs+i];
//        // :TODO
//      }
//    }
//  }
  return output;
}

double *Perceptron::hebbsRuleGradient(const double * const input, const int batch_size) {
  forward(input, batch_size);
  memset(gradient, 0.0, sizeof(double) * nOutputs * nInputs);
  for (int k = 0; k < batch_size; ++k) {
    for (int i = 0; i < nInputs; ++i) {
      for (int o = 0; o < nOutputs; ++o) {
        gradient[o + i * nOutputs] +=
            output[o + k * nOutputs] * input[i + k * nInputs];
      }
    }
  }

  for (int i = 0; i < nInputs * nOutputs; ++i) {
    gradient[i] = gradient[i] / batch_size;
  }
  return gradient;
}

void Perceptron::ojasRuleGradient(const double * const input, const int batch_size) {
  forward(input, batch_size);
  memset(gradient, 0.0, sizeof(double) * nOutputs * nInputs);
  // TODO:

  for (int k = 0; k < batch_size; ++k) {
    for (int i = 0; i < nInputs; ++i) {
      for (int o = 0; o < nOutputs; ++o) {
        gradient[o + i * nOutputs] +=
            output[o + k * nOutputs] * (input[i + k * nInputs] - output[o + k * nOutputs]*weights[i*nOutputs+o]);
      }
    }
  }

  // :TODO

  for (int i = 0; i < nInputs * nOutputs; ++i) {
    gradient[i] = gradient[i] / batch_size;
  }
}

void Perceptron::sangersRuleGradient(const double * const input, const int batch_size) {
  // input [batch_size, nInputs]
  // output [batch_size, nOutputs]
  // weights [nInputs, nOutputs]

  forward(input, batch_size);
  memset(gradient, 0.0, sizeof(double) * nOutputs * nInputs);
  // TODO:

//#pragma omp parallel
  for (int k = 0; k < batch_size; ++k) {
    for (int i = 0; i < nInputs; ++i) {
      double sum_weights = 0.0;
      for (int o = 0; o < nOutputs; ++o) {
        sum_weights += weights[i*nOutputs + o] * output[k*nOutputs + o];

        gradient[o + i * nOutputs] +=
            output[o + k * nOutputs] * (input[i + k * nInputs] - sum_weights);

        // FORWARD: output[o] += weights[i*nOutputs+o]*input[i]; // sum over i --> w^T * x
      }
    }
  }

  // :TODO

  for (int i = 0; i < nInputs * nOutputs; ++i) {
    gradient[i] = gradient[i] / batch_size;
  }
}

void Perceptron::normalizeGradient() {
  double norm_ = 0.0;
  for(int i=0;i<nInputs*nOutputs;++i)
  {
    norm_+= std::pow(gradient[i],2);
  }
  norm_=std::sqrt(norm_);
  for(int i=0;i<nInputs*nOutputs;++i)
  {
    gradient[i]/= norm_;
  }
}

void Perceptron::normalizeComponentWeights()
{
  auto norms = utils::computeComponentNorms(weights, nOutputs, nInputs);
  for (int o = 0; o < nOutputs; ++o) {
    for (int i = 0; i < nInputs; ++i) {
      weights[i*nOutputs+o] /= norms[o];
    }
  }

//  for(int i=0; i<nInputs;++i){
//    double norm_ = 0.0;
//    for(int o=0; o<nOutputs;++o){
//      norm_ += weights[o + i*nOutputs] * weights[o + i*nOutputs];
//    }
//    norm_ = sqrt(norm_);
//    for(int o=0; o<nOutputs;++o){
//      weights[o + i*nOutputs] /= norm_;
//    }
//  }
}

void Perceptron::printGradientNorm() {
  double norm_ = 0.0;
  for(int i=0; i<nInputs*nOutputs;++i){
    norm_ += std::pow(gradient[i],2);
  }
  norm_=std::sqrt(norm_);
  std::cout << "Gradient norm = " << norm_ << std::endl;
}

void Perceptron::updateParams(const double learning_rate) {
  double * temp = weights_prev;
  weights_prev = weights;
  weights = temp;
  for(int i=0; i<nInputs;++i){
    for(int o=0; o<nOutputs;++o){
      weights[o + i*nOutputs] += gradient[o + i*nOutputs] * learning_rate;
    }
  }


//  double norm_ = 0.0;
//  for(int i=0;i<nInputs*nOutputs;++i)
//  {
//    norm_+= std::pow(weights[i],2);
//  }
//  norm_=std::sqrt(norm_);
//  for(int i=0;i<nInputs*nOutputs;++i)
//  {
//    weights[i]/= norm_;
//  }
}

void Perceptron::computeEigenvalues(const double * const input, const int batch_size) {
  double* old_output = new double[nOutputs * batch_size];
  double* old_weights = new double[nOutputs * nInputs];
  memcpy(old_output, output, nOutputs*batch_size*sizeof(double));
  memcpy(old_weights, weights, nOutputs*nInputs*sizeof(double));


//  printWeights();
  normalizeComponentWeights();
//  printWeights();

  // The eigenvalues are given by the standard deviation at the output
  forward(input, batch_size);
  memset(eigenvalues, 0.0, sizeof(double) * nOutputs);
  memset(mean, 0.0, sizeof(double) * nOutputs);

  // TODO:

  auto output_t = new double[nOutputs * batch_size];

  utils::transposeData(output_t, output, batch_size, nOutputs);
  utils::computeMean(mean, output_t, batch_size, nOutputs);
  utils::computeVar(eigenvalues, mean, output_t, batch_size, nOutputs);


  memcpy(output, old_output, nOutputs*batch_size*sizeof(double));
  memcpy(weights, old_weights, nOutputs*nInputs*sizeof(double));
  delete[] output_t;
  delete[] old_output;
  delete[] old_weights;
  // :TODO
}

void Perceptron::printWeights()
{
  std::cout << "WEIGHTS:" << std::endl;
//  for(int i=0; i<nInputs;++i){
//    double norm_ = 0.0;
//    for(int o=0; o<nOutputs;++o){
//      std::cout << weights[o + i*nOutputs];
//      if(i<nInputs-1){
//        std::cout << ",";
//      }
//      norm_ += weights[o + i*nOutputs] * weights[o + i*nOutputs];
//    }
//    norm_ = sqrt(norm_);
//    std::cout << "\n Norm=" << norm_ << std::endl;
//    std::cout << "\n";
//  }
//  std::cout << "\n";


  for(int i = 0; i < nInputs; ++i) {
    for(int o = 0; o < nOutputs; ++o) {
      std::cout << std::right << std::setw(10) << weights[o + i*nOutputs];
    }
    std::cout << std::endl;
  }
  std::cout << std::endl;
}

void Perceptron::printEigenvalues()
{
  std::cout << "EIGENVALUES:" << std::endl;
  for(int o=0; o<nOutputs;++o){
    std::cout << eigenvalues[o];
    if(o<nOutputs-1){
      std::cout << ",";
    }
  }
  std::cout << std::endl;
}
