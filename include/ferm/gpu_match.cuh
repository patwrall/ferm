#pragma once
#include <ferm/gpu_types.cuh>

// price < 0 = market
void gpu_match_batch(
    GPUBook       *d_books,
    const int32_t *d_prices,
    const int32_t *d_qtys,
    const int8_t  *d_sides,
    int32_t       *d_fills,
    int            N,
    cudaStream_t   stream = nullptr
);
