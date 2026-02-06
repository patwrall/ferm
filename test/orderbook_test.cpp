#include <catch2/catch_test_macros.hpp>
#include <ferm/OrderBook.hpp>

static Order make(OrderType t, Order::id_t id, Side s, Order::price_t px, Order::quantity_t qty)
{
    return Order{ t, id, s, px, qty };
}

TEST_CASE("resting limit orders visible at best bid/ask", "[orderbook]")
{
    OrderBook book;
    book.addOrder(make(OrderType::LIMIT, 1, Side::BUY,  100, 10));
    book.addOrder(make(OrderType::LIMIT, 2, Side::SELL, 102, 10));

    REQUIRE(book.bestBid().has_value());
    REQUIRE(book.bestAsk().has_value());
    REQUIRE(book.bestBid()->getPrice() == 100);
    REQUIRE(book.bestAsk()->getPrice() == 102);
}

TEST_CASE("incoming limit crosses and fully fills resting order", "[orderbook]")
{
    OrderBook book;
    book.addOrder(make(OrderType::LIMIT, 1, Side::SELL, 100, 10));
    book.addOrder(make(OrderType::LIMIT, 2, Side::BUY,  100, 10));

    REQUIRE_FALSE(book.bestAsk().has_value());
    REQUIRE_FALSE(book.bestBid().has_value());
}

TEST_CASE("partial fill leaves remainder resting", "[orderbook]")
{
    OrderBook book;
    book.addOrder(make(OrderType::LIMIT, 1, Side::SELL, 100, 20));
    book.addOrder(make(OrderType::LIMIT, 2, Side::BUY,  100,  7));

    REQUIRE(book.bestAsk().has_value());
    REQUIRE(book.bestAsk()->getCurrentSize() == 13);
    REQUIRE_FALSE(book.bestBid().has_value());
}

TEST_CASE("market order fills against best available price", "[orderbook]")
{
    OrderBook book;
    book.addOrder(make(OrderType::LIMIT, 1, Side::SELL, 100, 5));
    book.addOrder(make(OrderType::LIMIT, 2, Side::SELL, 101, 5));

    book.addOrder(make(OrderType::MARKET, 3, Side::BUY, Order::PRICE_NA, 8));

    REQUIRE(book.bestAsk().has_value());
    REQUIRE(book.bestAsk()->getPrice() == 101);
    REQUIRE(book.bestAsk()->getCurrentSize() == 2);
}

TEST_CASE("cancel removes order from book", "[orderbook]")
{
    OrderBook book;
    book.addOrder(make(OrderType::LIMIT, 1, Side::BUY,   99, 10));
    book.addOrder(make(OrderType::LIMIT, 2, Side::BUY,  100,  5));
    book.addOrder(make(OrderType::LIMIT, 3, Side::SELL, 102, 10));

    book.cancel(2);
    REQUIRE(book.bestBid()->getPrice() == 99);

    book.cancel(3);
    REQUIRE_FALSE(book.bestAsk().has_value());
}

TEST_CASE("price-time priority: earlier order at same level fills first", "[orderbook]")
{
    OrderBook book;
    book.addOrder(make(OrderType::LIMIT, 1, Side::SELL, 100, 5));
    book.addOrder(make(OrderType::LIMIT, 2, Side::SELL, 100, 5));

    book.addOrder(make(OrderType::LIMIT, 3, Side::BUY, 100, 5));

    REQUIRE(book.bestAsk().has_value());
    REQUIRE(book.bestAsk()->getId() == 2);
}

TEST_CASE("cancel on unknown id is a no-op", "[orderbook]")
{
    OrderBook book;
    book.addOrder(make(OrderType::LIMIT, 1, Side::BUY, 100, 10));
    REQUIRE_NOTHROW(book.cancel(999));
    REQUIRE(book.bestBid().has_value());
}

TEST_CASE("market order sweeps multiple price levels", "[orderbook]")
{
    OrderBook book;
    book.addOrder(make(OrderType::LIMIT, 1, Side::SELL, 100, 3));
    book.addOrder(make(OrderType::LIMIT, 2, Side::SELL, 101, 3));
    book.addOrder(make(OrderType::LIMIT, 3, Side::SELL, 102, 3));

    book.addOrder(make(OrderType::MARKET, 4, Side::BUY, Order::PRICE_NA, 8));

    REQUIRE(book.bestAsk().has_value());
    REQUIRE(book.bestAsk()->getPrice() == 102);
    REQUIRE(book.bestAsk()->getCurrentSize() == 1);
}
