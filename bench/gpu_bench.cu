#include <ferm/gpu_match.cuh>
#include <ferm/gpu_hawkes.cuh>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

static void check(cudaError_t e, const char *msg)
{
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", msg, cudaGetErrorString(e));
        std::exit(1);
    }
}

static float cuda_ms(cudaEvent_t start, cudaEvent_t stop)
{
    float ms = 0;
    cudaEventElapsedTime(&ms, start, stop);
    return ms;
}

static void bench_match(int N, int warmup, int iters)
{
    std::vector<GPUBook> h_books(static_cast<size_t>(N));
    for (auto &b : h_books) {
        std::memset(&b, 0, sizeof(b));
        b.mid_tick = 1000;
        for (int i = 0; i < 20; ++i) {
            b.ask[(1001 + i) & (BOOK_LEVELS - 1)] = 100;
            b.bid[(999  - i) & (BOOK_LEVELS - 1)] = 100;
        }
    }

    std::vector<int32_t> h_prices(static_cast<size_t>(N), -1); // market
    std::vector<int32_t> h_qtys(static_cast<size_t>(N), 5);
    std::vector<int8_t>  h_sides(static_cast<size_t>(N), 1);
    std::vector<int32_t> h_fills(static_cast<size_t>(N), 0);

    GPUBook  *d_books;  int32_t *d_prices, *d_qtys, *d_fills;  int8_t *d_sides;
    check(cudaMalloc(&d_books,  N * sizeof(GPUBook)),  "alloc books");
    check(cudaMalloc(&d_prices, N * sizeof(int32_t)), "alloc prices");
    check(cudaMalloc(&d_qtys,   N * sizeof(int32_t)), "alloc qtys");
    check(cudaMalloc(&d_sides,  N * sizeof(int8_t)),  "alloc sides");
    check(cudaMalloc(&d_fills,  N * sizeof(int32_t)), "alloc fills");

    check(cudaMemcpy(d_books,  h_books.data(),  N * sizeof(GPUBook),  cudaMemcpyHostToDevice), "H2D books");
    check(cudaMemcpy(d_prices, h_prices.data(), N * sizeof(int32_t), cudaMemcpyHostToDevice), "H2D prices");
    check(cudaMemcpy(d_qtys,   h_qtys.data(),   N * sizeof(int32_t), cudaMemcpyHostToDevice), "H2D qtys");
    check(cudaMemcpy(d_sides,  h_sides.data(),  N * sizeof(int8_t),  cudaMemcpyHostToDevice), "H2D sides");

    for (int i = 0; i < warmup; ++i)
        gpu_match_batch(d_books, d_prices, d_qtys, d_sides, d_fills, N);
    cudaDeviceSynchronize();

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0);
    for (int i = 0; i < iters; ++i)
        gpu_match_batch(d_books, d_prices, d_qtys, d_sides, d_fills, N);
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);

    const float total_ms = cuda_ms(t0, t1);
    const float per_launch_us = total_ms * 1000.0f / static_cast<float>(iters);
    const float ns_per_order  = total_ms * 1e6f / (static_cast<float>(iters) * static_cast<float>(N));

    std::printf("gpu_match_batch  N=%-8d  per_launch=%.1fus  ns/order=%.2f\n",
        N, per_launch_us, ns_per_order);

    cudaFree(d_books); cudaFree(d_prices); cudaFree(d_qtys);
    cudaFree(d_sides); cudaFree(d_fills);
    cudaEventDestroy(t0); cudaEventDestroy(t1);
}

static void bench_hawkes(int N, int steps, int warmup, int iters)
{
    std::vector<GPUBook> h_books(static_cast<size_t>(N));
    for (auto &b : h_books) {
        std::memset(&b, 0, sizeof(b));
        b.mid_tick = 1000;
    }

    HawkesParams p{ .mu_bid = 5.0f, .mu_ask = 5.0f,
                    .alpha_self = 3.0f, .alpha_cross = 1.0f,
                    .beta = 10.0f, .steps = steps, .seed = 42 };

    GPUBook *d_books; float *d_paths;
    check(cudaMalloc(&d_books, N * sizeof(GPUBook)), "alloc books");
    check(cudaMalloc(&d_paths, static_cast<size_t>(N) * static_cast<size_t>(steps) * sizeof(float)), "alloc paths");
    check(cudaMemcpy(d_books, h_books.data(), N * sizeof(GPUBook), cudaMemcpyHostToDevice), "H2D books");

    for (int i = 0; i < warmup; ++i)
        gpu_hawkes_sim(d_books, p, d_paths, N);
    cudaDeviceSynchronize();

    cudaEvent_t t0, t1;
    cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0);
    for (int i = 0; i < iters; ++i)
        gpu_hawkes_sim(d_books, p, d_paths, N);
    cudaEventRecord(t1);
    cudaEventSynchronize(t1);

    const float total_ms      = cuda_ms(t0, t1);
    const float per_launch_ms = total_ms / static_cast<float>(iters);
    const float sims_per_sec  = static_cast<float>(N) / (per_launch_ms * 1e-3f);

    std::printf("gpu_hawkes_sim   N=%-8d  steps=%-6d  per_launch=%.2fms  sims/sec=%.0f\n",
        N, steps, per_launch_ms, sims_per_sec);

    cudaFree(d_books); cudaFree(d_paths);
    cudaEventDestroy(t0); cudaEventDestroy(t1);
}

int main()
{
    int dev; cudaGetDevice(&dev);
    cudaDeviceProp prop{}; cudaGetDeviceProperties(&prop, dev);
    std::printf("device: %s\n\n", prop.name);

    bench_match(10'000,   5, 50);
    bench_match(100'000,  5, 50);
    bench_match(1'000'000, 3, 20);

    std::printf("\n");

    bench_hawkes(10'000,  1000, 3, 20);
    bench_hawkes(100'000, 1000, 3, 10);
}
