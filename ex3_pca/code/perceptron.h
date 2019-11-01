#ifndef PERCEPTRON_H_6XOBU3FM
#define PERCEPTRON_H_6XOBU3FM

#include <cassert> /* assert */
#include <cstring>
#include <math.h> /* sqrt */
#include <omp.h>
#include <math.h>       /* isinf, sqrt */
#include <random>
#include <string>

class Perceptron
{
public:
  const int nInputs;
  const int nOutputs;
  const int max_batch_size;
  double * weights;
  double * weights_prev;
  std::string weight_init;

  double *const output;
  double *const gradient;
  double *eigenvalues;
  double *mean;

private:
  double *const gradient_local;

public:
  Perceptron(int nInputs_, int nOutputs_, int max_batch_size_, std::string weight_init_)
      : nInputs(nInputs_), nOutputs(nOutputs_), max_batch_size(max_batch_size_), weight_init(weight_init_),
        weights(new double[nOutputs_ * nInputs_]()),
        weights_prev(new double[nOutputs_ * nInputs_]()),
        output(new double[nOutputs_ * max_batch_size_]()),
        gradient(new double[nOutputs_ * nInputs_]()),
        eigenvalues(new double[nOutputs_]()), mean(new double[nOutputs_]()), 
        gradient_local(new double[nOutputs_ * nInputs_]())
        {
    assert(nullptr != weights);
    assert(nullptr != weights_prev);
    assert(nullptr != output);
    assert(nullptr != gradient);
    assert(nullptr != eigenvalues);
    assert(nullptr != mean);
    assert(nullptr != gradient_local);
    initializeWeights();
  }

  ~Perceptron() {
    delete[] weights;
    delete[] weights_prev;
    delete[] output;
    delete[] gradient;
    delete[] eigenvalues;
    delete[] mean;
    delete[] gradient_local;
  }

  Perceptron() = delete;
  Perceptron(const Perceptron&) = delete;
  Perceptron(Perceptron&&) = delete;
  Perceptron& operator=(const Perceptron&) = delete;
  Perceptron& operator=(Perceptron&&) = delete;

  void initializeWeights()
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

  double *forward(const double *const input, const int batch_size) {
    // Input dimension [batch_size, nInputs]
    assert(batch_size > 0);
    assert(batch_size <= max_batch_size);
    memset(output, 0.0, sizeof(double) * batch_size * nOutputs);
    for (int k = 0; k < batch_size; ++k) {
      for (int i = 0; i < nInputs; ++i) {
        for (int o = 0; o < nOutputs; ++o) {
          // TODO:


          // :TODO
        }
      }
    }
    return output;
  }

  double *hebbsRuleGradient(const double *const input, const int batch_size) {
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


  void ojasRuleGradient(const double *const input, const int batch_size) {
    forward(input, batch_size);
    memset(gradient, 0.0, sizeof(double) * nOutputs * nInputs);
    // TODO:







    // :TODO

    for (int i = 0; i < nInputs * nOutputs; ++i) {
      gradient[i] = gradient[i] / batch_size;
    }
  }

  void sangersRuleGradient(const double *const input, const int batch_size) {
    forward(input, batch_size);
    memset(gradient, 0.0, sizeof(double) * nOutputs * nInputs);
    // TODO:










    // :TODO

    for (int i = 0; i < nInputs * nOutputs; ++i) {
      gradient[i] = gradient[i] / batch_size;
    }
  }

  void normalizeGradient() {
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

  void printGradientNorm() {
    double norm_ = 0.0;
    for(int i=0; i<nInputs*nOutputs;++i){
        norm_ += std::pow(gradient[i],2);
    }
    norm_=std::sqrt(norm_);
    std::cout << "Gradient norm = " << norm_ << std::endl;
  }


  void updateParams(const double learning_rate) {
    double * temp = weights_prev;
    weights_prev = weights;
    weights = temp;
    for(int i=0; i<nInputs;++i){
      for(int o=0; o<nOutputs;++o){
        weights[o + i*nOutputs] += gradient[o + i*nOutputs] * learning_rate;
      }
    }
  }

  void computeEigenvalues(const double *const input, const int batch_size) {
    // The eigenvalues are given by the standard deviation at the output
    forward(input, batch_size);
    memset(eigenvalues, 0.0, sizeof(double) * nOutputs);
    memset(mean, 0.0, sizeof(double) * nOutputs);

    // TODO:




















    // :TODO
  }

  void printWeights()
  {
    std::cout << "WEIGHTS:\n";
    for(int i=0; i<nInputs;++i){
      double norm_ = 0.0;
      for(int o=0; o<nOutputs;++o){
        std::cout << weights[o + i*nOutputs];
        if(i<nInputs-1){
            std::cout << ",";
          }
        norm_ += weights[o + i*nOutputs] * weights[o + i*nOutputs];
      }
      norm_ = sqrt(norm_);
      std::cout << "\n Norm=" << norm_ << std::endl;
      std::cout << "\n";
    }
    std::cout << "\n";
  }

};

#endif /* PERCEPTRON_H_6XOBU3FM */
