#include <ferm/OrderBook.hpp>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <numeric>
#include <print>
#include <vector>

using ns = std::chrono::nanoseconds;

static auto now() { return std::chrono::steady_clock::now(); }

static void report(const char *label, std::vector<int64_t> &samples)
{
    std::sort(samples.begin(), samples.end());
    const auto n   = static_cast<double>(samples.size());
    const auto sum = std::accumulate(samples.begin(), samples.end(), int64_t{});
    std::println("{:<40}  mean={:6}ns  p50={:6}ns  p99={:6}ns",
        label,
        static_cast<int64_t>(static_cast<double>(sum) / n),
        samples[static_cast<size_t>(n * 0.50)],
        samples[static_cast<size_t>(n * 0.99)]);
}

static void bench_passive_add(int rounds)
{
    std::vector<int64_t> samples;
    samples.reserve(static_cast<size_t>(rounds));

    for (int i = 0; i < rounds; ++i) {
        OrderBook book;
        // prime spread
        book.addOrder(Order{ OrderType::LIMIT, 1, Side::BUY,  9900, 100 });
        book.addOrder(Order{ OrderType::LIMIT, 2, Side::SELL, 10100, 100 });

        const auto t0 = now();
        book.addOrder(Order{ OrderType::LIMIT, 3, Side::BUY, 9950, 10 });
        samples.push_back(std::chrono::duration_cast<ns>(now() - t0).count());
    }
    report("passive limit add (empty book)", samples);
}

static void bench_aggressive_add(int rounds)
{
    std::vector<int64_t> samples;
    samples.reserve(static_cast<size_t>(rounds));

    for (int i = 0; i < rounds; ++i) {
        OrderBook book;
        for (int lvl = 0; lvl < 10; ++lvl)
            book.addOrder(Order{ OrderType::LIMIT,
                static_cast<Order::id_t>(lvl + 1),
                Side::SELL,
                static_cast<Order::price_t>(10000 + lvl * 10),
                50 });

        const auto t0 = now();
        book.addOrder(Order{ OrderType::MARKET, 999, Side::BUY, Order::PRICE_NA, 20 });
        samples.push_back(std::chrono::duration_cast<ns>(now() - t0).count());
    }
    report("market buy (10 ask levels, partial fill)", samples);
}

static void bench_cancel(int rounds)
{
    std::vector<int64_t> samples;
    samples.reserve(static_cast<size_t>(rounds));

    for (int i = 0; i < rounds; ++i) {
        OrderBook book;
        for (int j = 0; j < 100; ++j)
            book.addOrder(Order{ OrderType::LIMIT,
                static_cast<Order::id_t>(j + 1),
                Side::BUY,
                static_cast<Order::price_t>(10000 - j),
                10 });

        const auto t0 = now();
        book.cancel(50);
        samples.push_back(std::chrono::duration_cast<ns>(now() - t0).count());
    }
    report("cancel (100-order book, mid-queue)", samples);
}

static void bench_deep_sweep(int rounds)
{
    std::vector<int64_t> samples;
    samples.reserve(static_cast<size_t>(rounds));

    for (int i = 0; i < rounds; ++i) {
        OrderBook book;
        for (int lvl = 0; lvl < 50; ++lvl)
            book.addOrder(Order{ OrderType::LIMIT,
                static_cast<Order::id_t>(lvl + 1),
                Side::SELL,
                static_cast<Order::price_t>(10000 + lvl),
                10 });

        const auto t0 = now();
        book.addOrder(Order{ OrderType::MARKET, 999, Side::BUY, Order::PRICE_NA, 500 });
        samples.push_back(std::chrono::duration_cast<ns>(now() - t0).count());
    }
    report("market buy sweep (50 levels x 10 qty)", samples);
}

int main()
{
    constexpr int rounds = 100'000;
    std::println("rounds={}", rounds);
    std::println("");
    bench_passive_add(rounds);
    bench_aggressive_add(rounds);
    bench_cancel(rounds);
    bench_deep_sweep(rounds);
}
