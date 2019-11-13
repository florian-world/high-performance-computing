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

  void initializeWeights();

  double *forward(const double *const input, const int batch_size);

  double *hebbsRuleGradient(const double *const input, const int batch_size);

  void ojasRuleGradient(const double *const input, const int batch_size);

  void sangersRuleGradient(const double *const input, const int batch_size);

  void normalizeGradient();
  void normalizeComponentWeights();

  void printGradientNorm();


  void updateParams(const double learning_rate);

  void computeEigenvalues(const double *const input, const int batch_size);

  void printWeights();

  void printEigenvalues();

};

#endif /* PERCEPTRON_H_6XOBU3FM */
