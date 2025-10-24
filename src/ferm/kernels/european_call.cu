#include <curand.h>
#include <curand_kernel.h>
#include <curand_normal.h>
#include <ferm/european_call.hpp>

__global__ void european_call_kernel(
  float *d_s,
  unsigned N_PATHS,
  float S0,
  float K,
  float r,
  float v,
  float T,
  curandState_t *states) {
  const unsigned tid = threadIdx.x;
  const unsigned bid = blockIdx.x;
  const unsigned block_dim = blockDim.x;

  int s_idx = static_cast<int>(tid + bid * block_dim);

  if (s_idx >= static_cast<int>(N_PATHS)) { return; }

  float s_curr = S0;
  float s_adjust = s_curr * static_cast<float>(exp(T * (r - 0.5 * v * v)));

  float Z = curand_normal(&states[s_idx]);

  s_curr = s_adjust * exp(sqrt(v * v * T) * Z);

  float payoff = fmaxf(s_curr - K, 0.0F);
  d_s[s_idx] = exp(-r * T) * payoff;
}

FERM_EXPORT void european_call(
  float *d_s,
  unsigned N_PATHS,
  float S0,
  float K,
  float r,
  float v,
  float T,
  curandState_t *states,
  cudaStream_t stream) {
  dim3 block_dim(256U, 1U, 1U);
  dim3 grid_dim((N_PATHS + block_dim.x - 1U) / block_dim.x, 1U, 1U);

  european_call_kernel<<<grid_dim, block_dim, 0, stream>>>(
    d_s,
    N_PATHS,
    S0,
    K,
    r,
    v,
    T,
    states);
}
