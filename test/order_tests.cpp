#include <catch2/catch_test_macros.hpp>
#include <ferm/Order.hpp>

TEST_CASE("Order class basic functionality", "[Order]")
{
  Order order_example(OrderType::LIMIT, 1, Side::BUY, 1000, 10);

  REQUIRE(order_example.getType() == OrderType::LIMIT);
  REQUIRE(order_example.getId() == 1);
  REQUIRE(order_example.getSide() == Side::BUY);
  REQUIRE(order_example.getPrice() == 1000);
  REQUIRE(order_example.getInitialSize() == 10);
  REQUIRE(order_example.getCurrentSize() == 10);
  REQUIRE(!order_example.isFilled());

  order_example.fill(5);
  REQUIRE(order_example.getCurrentSize() == 5);
  REQUIRE(!order_example.isFilled());

  order_example.fill(5);
  REQUIRE(order_example.getCurrentSize() == 0);
  REQUIRE(order_example.isFilled());
}


TEST_CASE("Order class error handling", "[Order]")
{
  REQUIRE_THROWS_AS(Order(OrderType::LIMIT, 2, Side::SELL, 500, 0), std::runtime_error);
  REQUIRE_THROWS_AS(Order(OrderType::LIMIT, 3, Side::SELL, 500, -5), std::runtime_error);

  REQUIRE_THROWS_AS(Order(OrderType::LIMIT, 4, Side::BUY, -100, 10), std::runtime_error);

  Order order_example(OrderType::MARKET, 5, Side::BUY, Order::PRICE_NA, 10);
  REQUIRE_THROWS_AS(order_example.fill(0), std::runtime_error);
  REQUIRE_THROWS_AS(order_example.fill(-3), std::runtime_error);
  REQUIRE_THROWS_AS(order_example.fill(15), std::runtime_error);
}

TEST_CASE("Market order price handling", "[Order]")
{
  Order market_order(OrderType::MARKET, 6, Side::SELL, Order::PRICE_NA, 20);
  REQUIRE(market_order.getPrice() == Order::PRICE_NA);
}

TEST_CASE("Limit order price validation", "[Order]")
{
  REQUIRE_THROWS_AS(Order(OrderType::LIMIT, 7, Side::BUY, -1, 10), std::runtime_error);
  REQUIRE_NOTHROW(Order(OrderType::LIMIT, 8, Side::BUY, 0, 10));
  REQUIRE_NOTHROW(Order(OrderType::LIMIT, 9, Side::BUY, 100, 10));
}

TEST_CASE("Filling order beyond current size", "[Order]")
{
  Order order_example(OrderType::LIMIT, 10, Side::SELL, 2000, 15);
  REQUIRE_THROWS_AS(order_example.fill(20), std::runtime_error);
}
