#ifndef ORDER_HPP
#define ORDER_HPP

#include <cmath>
#include <cstdint>
#include <stdexcept>

enum class Side : uint8_t { BUY, SELL };

enum class OrderType : uint8_t { LIMIT, MARKET };

class Order
{
public:
  using OrderId = std::int64_t;
  using PriceType = std::int64_t;
  using SizeType = std::int64_t;

  static constexpr PriceType PRICE_NA = std::numeric_limits<PriceType>::min();

  Order(OrderType type, OrderId id, Side side, PriceType price, SizeType intial_size)
    : type_{ type }, id_{ id }, side_{ side }, price_{ price }, initial_size_{ intial_size },
      current_size_{ initial_size_ }
  {
    if (initial_size_ <= 0) throw std::runtime_error("Initial size must be positive");
    if (type == OrderType::LIMIT && price < 00) throw std::runtime_error("Limit order price must be non-negative");
    if (type == OrderType::MARKET) price = PRICE_NA;
  }

  [[nodiscard]] OrderType getType() const noexcept { return type_; }
  [[nodiscard]] OrderId getId() const noexcept { return id_; }
  [[nodiscard]] Side getSide() const noexcept { return side_; }
  [[nodiscard]] PriceType getPrice() const noexcept { return price_; }
  [[nodiscard]] SizeType getInitialSize() const noexcept { return initial_size_; }
  [[nodiscard]] SizeType getCurrentSize() const noexcept { return current_size_; }
  [[nodiscard]] bool isFilled() const noexcept { return current_size_ == 0; }

  void fill(SizeType size)
  {
    if (size <= 0) throw std::runtime_error("Fill size must be positive");
    if (size > getCurrentSize()) { throw std::runtime_error("Fill size exceeds current size"); }

    current_size_ -= size;
  }

private:
  OrderType type_;
  OrderId id_;
  Side side_;
  PriceType price_;
  SizeType initial_size_;
  SizeType current_size_;
};

#endif
