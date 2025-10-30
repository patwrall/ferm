#ifndef ORDER_HPP
#define ORDER_HPP

#include "OrderType.hpp"
#include <cmath>
#include <cstdint>
#include <limits>
#include <stdexcept>

enum class Side : uint8_t { BUY, SELL };

class Order
{
public:
  using price_t = std::int64_t;
  using quantity_t = std::int64_t;
  using id_t = std::uint64_t;

  static constexpr price_t PRICE_NA = std::numeric_limits<price_t>::min();

  Order(OrderType type, id_t id, Side side, price_t price, quantity_t initial_size)
    : type_{ type }, id_{ id }, side_{ side }, price_{ price }, initial_quantity_{ initial_size },
      current_quantity_{ initial_quantity_ }
  {
    if (initial_quantity_ <= 0) { throw std::runtime_error("Initial size must be positive"); }
    if (type == OrderType::LIMIT && price_ < 0) { throw std::runtime_error("Limit order price must be non-negative"); }
    if (type == OrderType::MARKET) { price_ = PRICE_NA; }
  }

  [[nodiscard]] OrderType getType() const noexcept { return type_; }
  [[nodiscard]] id_t getId() const noexcept { return id_; }
  [[nodiscard]] Side getSide() const noexcept { return side_; }
  [[nodiscard]] price_t getPrice() const noexcept { return price_; }
  [[nodiscard]] quantity_t getInitialSize() const noexcept { return initial_quantity_; }
  [[nodiscard]] quantity_t getCurrentSize() const noexcept { return current_quantity_; }
  [[nodiscard]] bool isFilled() const noexcept { return current_quantity_ == 0; }

  void fill(quantity_t size)
  {
    if (size <= 0) { throw std::runtime_error("Fill size must be positive"); }
    if (size > getCurrentSize()) { throw std::runtime_error("Fill size exceeds current size"); }

    current_quantity_ -= size;
  }

private:
  OrderType type_;
  id_t id_;
  Side side_;
  price_t price_;
  quantity_t initial_quantity_;
  quantity_t current_quantity_;
};

#endif
