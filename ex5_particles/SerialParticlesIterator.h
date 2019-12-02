#pragma once

#include "ArrayOfParticles.h"

#include <functional>

#ifndef SQUARE
#define SQUARE(BASE) ((BASE) * (BASE))
#endif

namespace SerialParticlesIterator
{
ArrayOfParticles initialize_particles_1D(
    size_t n_particles, value_t start_pos_x, value_t end_pos_x,
    const std::function<value_t(value_t)> init_function)
{
  ArrayOfParticles particles(n_particles);
  const value_t dx = (end_pos_x - start_pos_x) / n_particles;

  for(size_t i=0; i<n_particles; ++i)
  {
    const value_t x = start_pos_x + dx * (i + 0.5);
    particles.pos_x(i) = x;
    particles.pos_y(i) = 0;
    particles.gamma(i) = init_function(x);
  }

  return particles;
}

void reset_velocities(ArrayOfParticles & particles)
{
  for (size_t i=0; i<particles.size(); ++i)
  {
    particles.vel_x(i) = 0;
    particles.vel_y(i) = 0;
  }
}

void compute_interaction(const ArrayOfParticles & sources,
                         ArrayOfParticles & targets)
{
  assert(sources.size() == targets.size());

#pragma omp parallel for
  for (size_t i = 0; i < sources.size(); ++i) {
    targets.vel_x(i) = 0.0;
    targets.vel_y(i) = 0.0;
    for (size_t j = 0; j < sources.size(); ++j) {
      if (i == j)
          continue; // exclude self interaction

      auto denominator = (SQUARE(sources.pos_x(i) - sources.pos_x(j)) + SQUARE(sources.pos_y(i) - sources.pos_y(j)));

      targets.vel_x(i) += sources.gamma(j) * M_PI * (- (sources.pos_y(i) - sources.pos_y(j))) / denominator;
      targets.vel_y(i) += sources.gamma(j) * M_PI * (sources.pos_x(i) - sources.pos_x(j)) / denominator;
    }
  }
}

void advect_euler(ArrayOfParticles & particles, const value_t dt)
{
  for (size_t i=0; i<particles.size(); ++i)
  {
    particles.pos_x(i) += particles.vel_x(i) * dt;
    particles.pos_y(i) += particles.vel_y(i) * dt;
  }
}

value_t sum_circulation(const ArrayOfParticles& particles)
{
  value_t total = 0;
  for (size_t i=0; i<particles.size(); ++i)
  {
    total += particles.gamma(i);
  }
  return total;
}
}
