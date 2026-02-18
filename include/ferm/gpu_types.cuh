#pragma once
#include <cuda_runtime.h>
#include <cstdint>

static constexpr int32_t BOOK_LEVELS = 512;
static constexpr int32_t TICK_SCALE  = 100;

struct __align__(128) GPUBook {
    int32_t bid[BOOK_LEVELS];
    int32_t ask[BOOK_LEVELS];
    int32_t mid_tick;
    int32_t pad[3];
};
