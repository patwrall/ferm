#include <ferm/gpu_hawkes.cuh>
#include <curand_kernel.h>

__global__ __launch_bounds__(256)
void hawkes_kernel(
    GPUBook            * __restrict__ books,
    const HawkesParams   p,
    float              * __restrict__ mid_paths,
    int N
) {
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= N) return;

    // Philox is faster to init than XORWOW and still high quality
    curandStatePhilox4_32_10_t rng;
    curand_init(p.seed, static_cast<uint64_t>(id), 0, &rng);

    float Rb  = 0.0f;
    float Ra  = 0.0f;
    int   mid = books[id].mid_tick;

    for (int step = 0; step < p.steps; ) {
        const float lb   = p.mu_bid + Rb;
        const float la   = p.mu_ask + Ra;
        const float lmax = lb + la;

        const float dt    = -__logf(curand_uniform(&rng)) / lmax;
        const float decay = __expf(-p.beta * dt);
        Rb *= decay;
        Ra *= decay;

        const float l_new = (p.mu_bid + Rb) + (p.mu_ask + Ra);
        if (curand_uniform(&rng) > l_new / lmax) continue;

        if (curand_uniform(&rng) < (p.mu_bid + Rb) / l_new) {
            books[id].bid[(mid - 1) & (BOOK_LEVELS - 1)]++;
            Rb += p.alpha_self;
            Ra += p.alpha_cross;
        } else {
            books[id].ask[(mid + 1) & (BOOK_LEVELS - 1)]++;
            Ra += p.alpha_self;
            Rb += p.alpha_cross;
        }

        mid_paths[id * p.steps + step] = static_cast<float>(mid) / TICK_SCALE;
        ++step;
    }
}

void gpu_hawkes_sim(
    GPUBook *d_books, const HawkesParams &params,
    float *d_mid_paths, int N, cudaStream_t stream
) {
    constexpr int block = 256;
    hawkes_kernel<<<(N + block - 1) / block, block, 0, stream>>>(
        d_books, params, d_mid_paths, N);
}
