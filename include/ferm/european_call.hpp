#ifndef FERM_HPP
#define FERM_HPP

#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <ferm/ferm_export.hpp>

namespace ferm {

FERM_EXPORT void european_call(float *d_s,
  unsigned N_PATHS,
  float S0,
  float K,
  float r,
  float v,
  float T,
  curandState_t *states,
  cudaStream_t stream);
}// namespace ferm

#endif
