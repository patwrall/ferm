#ifndef ORDER_HPP
#define ORDER_HPP

#include <cmath>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>

enum class Side : uint8_t { BUY, SELL };

enum class OrderType : uint8_t { LIMIT, MARKET };

class Order
{
public:
  using OrderId = std::int64_t;
  using PriceType = std::int64_t;
  using SizeType = std::int64_t;

  Order(OrderType type, OrderId id, Side side, PriceType price, SizeType intial_size, SizeType current_size)
    : type_{ type }, id_{ id }, side_{ side }, price_{ price }, initial_size_{ intial_size },
      current_size_{ current_size }
  {}

  OrderType getType() const { return type_; }
  OrderId getId() const { return id_; }
  Side getSide() const { return side_; }
  PriceType getPrice() const { return price_; }
  SizeType getInitialSize() const { return initial_size_; }
  SizeType getCurrentSize() const { return current_size_; }
  bool isFilelled() const { return current_size_ == 0; }
  void fill(SizeType size)
  {
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

using OrderPtr = std::unique_ptr<Order>;
using OrderList = std::vector<Order *>;

#endif
