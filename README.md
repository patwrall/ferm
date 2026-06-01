# FERM

[![ci](https://github.com/patwrall/ferm/actions/workflows/ci.yml/badge.svg)](https://github.com/patwrall/ferm/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/patwrall/ferm/branch/main/graph/badge.svg)](https://codecov.io/gh/patwrall/ferm)
[![CodeQL](https://github.com/patwrall/ferm/actions/workflows/codeql-analysis.yml/badge.svg)](https://github.com/patwrall/ferm/actions/workflows/codeql-analysis.yml)

**F**inancial **E**ngine for **R**ealistic **M**icrostructure

GPU-accelerated limit order book with Hawkes-driven order flow. Built for microstructure research, execution studies, and RL env training.

## Architecture

**CPU (`src/ferm/`)** — price-time priority matching engine

- `OrderPool` — slab allocator with free-list reuse, 1M order capacity
- `Ladder<Comparator>` — sorted price levels via binary search on a flat vector
- `PriceLevel` — FIFO deque of order indices per price
- full price-time priority: limit/market, partial fills, cancel

**GPU (`src/cuda/`)**

- `gpu_match.cu` — batch order matching, one thread per (order, book) pair
- `gpu_hawkes.cu` — bivariate Hawkes LOB simulation; each thread drives one independent book
  - exponential kernel with recursive intensity update (no O(n) history sum)
  - Ogata thinning for exact event times
  - Philox RNG for fast init across large batches

`GPUBook` is a flat 512-level price grid (`__align__(128)`) — no pointer chasing, fits in L2.

## Benchmarks

CPU (Release, no sanitizers):

| operation | p50 | p99 |
|-----------|-----|-----|
| passive limit add | 170ns | 270ns |
| market buy (10 levels) | 140ns | 220ns |
| cancel (100-order book) | 480ns | 680ns |

GPU (RTX 3070):

| kernel | N | per launch | throughput |
|--------|---|-----------|------------|
| `gpu_match_batch` | 100K | 80µs | ~1.2B orders/sec |
| `gpu_hawkes_sim` (1K steps) | 10K | 2.3ms | 4.3M sims/sec |

## What's next

- O(1) cancel — current linear scan in level deque gets bad with depth, need order-id → position map
- warm-book CPU benchmarks — current measures cold construction overhead from `OrderPool::reserve`
- GPU matching with shared books + atomics for the non-independent case
- Hawkes calibration against real tick data (MLE or GMM)
- IOC/FOK order types

## Status

CPU matching engine working. GPU sim running. Mid-price tracking and spread reconstruction in progress.

## Getting Started

- [Dependency Setup](README_dependencies.md)
- [Building Details](README_building.md)
- [Docker](README_docker.md)
