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

## Status

CPU matching engine working. GPU sim running. Mid-price tracking and spread reconstruction in progress.

## Getting Started

- [Dependency Setup](README_dependencies.md)
- [Building Details](README_building.md)
- [Docker](README_docker.md)
