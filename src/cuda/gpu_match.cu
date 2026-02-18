#include <ferm/gpu_match.cuh>

__global__ void match_kernel(
    GPUBook       * __restrict__ books,
    const int32_t * __restrict__ prices,
    const int32_t * __restrict__ qtys,
    const int8_t  * __restrict__ sides,
    int32_t       * __restrict__ fills,
    int N
) {
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id >= N) return;

    GPUBook      &book      = books[id];
    int32_t       remaining = qtys[id];
    const int32_t lim       = prices[id];
    const bool    is_buy    = sides[id] > 0;
    const int     mid       = book.mid_tick;

    if (is_buy) {
        for (int lvl = mid; lvl < mid + BOOK_LEVELS / 2 && remaining > 0; ++lvl) {
            if (lim >= 0 && lvl > lim) break;
            const int     idx   = lvl & (BOOK_LEVELS - 1);
            const int32_t avail = book.ask[idx];
            if (avail <= 0) continue;
            const int32_t take  = min(remaining, avail);
            book.ask[idx] -= take;
            remaining     -= take;
        }
    } else {
        for (int lvl = mid; lvl > mid - BOOK_LEVELS / 2 && remaining > 0; --lvl) {
            if (lim >= 0 && lvl < lim) break;
            const int     idx   = lvl & (BOOK_LEVELS - 1);
            const int32_t avail = book.bid[idx];
            if (avail <= 0) continue;
            const int32_t take  = min(remaining, avail);
            book.bid[idx] -= take;
            remaining     -= take;
        }
    }

    fills[id] = qtys[id] - remaining;
}

void gpu_match_batch(
    GPUBook *d_books, const int32_t *d_prices, const int32_t *d_qtys,
    const int8_t *d_sides, int32_t *d_fills, int N, cudaStream_t stream
) {
    constexpr int block = 256;
    match_kernel<<<(N + block - 1) / block, block, 0, stream>>>(
        d_books, d_prices, d_qtys, d_sides, d_fills, N);
}
