#pragma once
#include <ferm/gpu_types.cuh>

struct HawkesParams {
    float    mu_bid;
    float    mu_ask;
    float    alpha_self;
    float    alpha_cross;
    float    beta;
    int32_t  steps;
    uint64_t seed;
};

// mid_paths: [N x steps] row-major
void gpu_hawkes_sim(
    GPUBook            *d_books,
    const HawkesParams &params,
    float              *d_mid_paths,
    int                 N,
    cudaStream_t        stream = nullptr
);
