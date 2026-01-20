#include <ferm/OrderBook.hpp>
#include <memory>
#include <optional>

OrderBook::order_idx_t OrderBook::OrderPool::add(const Order &order)
{
  if (orders_.size() >= ORDERS_CAPACITY) { throw std::overflow_error("OrderPool has reached maximum capacity"); }

  if (!free_list_.empty()) {
    const auto idx = free_list_.back();

    free_list_.pop_back();
    orders_[idx] = order;
    active_[idx] = true;

    return idx;
  } else {
    orders_.push_back(order);
    active_.push_back(true);

    return orders_.size() - 1;
  }
}

void OrderBook::OrderPool::remove(OrderBook::order_idx_t idx)
{
  if (idx >= orders_.size() || !active_[idx]) { throw std::out_of_range("Invalid order index for removal"); }

  active_[idx] = false;

  free_list_.push_back(idx);
}

Order &OrderBook::OrderPool::get(const OrderBook::order_idx_t idx) { return orders_.at(idx); }

[[nodiscard]] const Order &OrderBook::OrderPool::get(const OrderBook::order_idx_t idx) const { return orders_.at(idx); }

[[nodiscard]] bool OrderBook::OrderPool::isActive(const OrderBook::order_idx_t idx) const noexcept
{
  if (idx >= orders_.size()) { return false; }

  return active_[idx];
}

std::unique_ptr<OrderBook::MatchResult> &OrderBook::match(const Aggressor &a)
{
  std::vector<OrderBook::Fill> fills;
  std::unique_ptr<MatchResult> res =
    std::make_unique<MatchResult>(MatchResult{ .fills = fills, .qty_remaining = static_cast<int>(a.size) });

  if (a.size <= 0) { return res; }

  auto walk_ladder = [&](auto &ladder) {
    // TODO: walk ladders
    for (auto &entry : ladder.entries()) {}
  };

  if (a.side == Side::BUY) {
    walk_ladder(asks_);
  } else {
    walk_ladder(bids_);
  }

  return res;
}

void OrderBook::addOrder(const Order &order)
{
  switch (order.getType()) {
  case OrderType::MARKET: {
    break;
  }
  case OrderType::LIMIT: {
    break;
  }
  default:
    // error
    break;
  }
}

void OrderBook::cancel(Order::id_t id) {}

[[nodiscard]] std::optional<Order> OrderBook::bestBid() const noexcept
{
  const auto *const best_level = bids_.bestLevel();
  if (static_cast<bool>(best_level)) {
    if (!best_level->empty()) {
      auto first_order_idx = best_level->order_idxs.front();
      return OrderBook::order_pool_.get(first_order_idx);
    }

    return std::nullopt;
  }

  return std::nullopt;
};

[[nodiscard]] std::optional<Order> OrderBook::bestAsk() const noexcept
{
  const auto *const best_level = asks_.bestLevel();
  if (static_cast<bool>(best_level)) {
    if (!best_level->empty()) {
      auto first_order_idx = best_level->order_idxs.front();
      return OrderBook::order_pool_.get(first_order_idx);
    }

    return std::nullopt;
  }

  return std::nullopt;
};
