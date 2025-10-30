#ifndef ORDERBOOK_HPP
#define ORDERBOOK_HPP

#include "Order.hpp"
#include <algorithm>
#include <cstdint>
#include <deque>
#include <optional>
#include <unordered_map>
#include <vector>

class OrderBook
{
public:
  // TODO: return events for logging, further processing etc
  void addMarketOrder(Order &order);
  void addLimitOrder(Order &order);
  void cancel(Order::id_t order_id);

  [[nodiscard]] std::optional<Order> bestBid() const noexcept;
  [[nodiscard]] std::optional<Order> bestAsk() const noexcept;

private:
  using order_idx_t = uint64_t;

  class OrderPool
  {
  public:
    constexpr static uint64_t ORDERS_CAPACITY = 1'000'000;
    constexpr static uint64_t FREE_LIST_CAPACITY = 100'000;

    OrderPool()
    {
      orders_.reserve(ORDERS_CAPACITY);
      active_.reserve(orders_.capacity());
    };

    order_idx_t add(const Order &order);
    void remove(order_idx_t idx);

    Order &get(const order_idx_t idx);
    [[nodiscard]] const Order &get(const order_idx_t idx) const;

    [[nodiscard]] bool isActive(const order_idx_t idx) const noexcept;

  private:
    std::vector<Order> orders_;
    std::vector<bool> active_;

    // Tombstone free list for reusing idxs
    std::vector<order_idx_t> free_list_;
  };

  struct PriceLevel
  {
    std::deque<order_idx_t> order_idxs;
    Order::quantity_t total_quantity{ 0 };

    void addOrder(const order_idx_t idx, const Order::quantity_t size)
    {
      order_idxs.push_back(idx);
      total_quantity += size;
    }

    order_idx_t popFront(const Order::quantity_t size)
    {
      auto idx = order_idxs.front();
      order_idxs.pop_front();
      total_quantity -= size;
      return idx;
    }

    void adjustDepth(const Order::quantity_t delta) { total_quantity += delta; }

    [[nodiscard]] Order::quantity_t depth() const noexcept { return total_quantity; }
    [[nodiscard]] bool empty() const noexcept { return order_idxs.empty(); }
  };


  template<typename Comparator> class Ladder
  {
  public:
    explicit Ladder(Comparator comp) : comp_{ comp } {}

    // insert or find existing
    PriceLevel &levelFor(const Order::price_t price);

    // find existing only
    PriceLevel *levelOf(const Order::price_t price);
    [[nodiscard]] const PriceLevel *levelOf(const Order::price_t price) const;

    [[nodiscard]] const PriceLevel *bestLevel() const;

    struct Entry
    {
      Order::price_t price;
      PriceLevel level;
    };

    [[nodiscard]] const std::vector<Entry> &entries() const noexcept { return entries_; }
    std::vector<Entry> &entries() noexcept { return entries_; }

    void eraseIfEmpty(Order::price_t price);

    [[nodiscard]] bool empty() const noexcept { return entries_.empty(); }

  private:
    Comparator comp_;
    std::vector<Entry> entries_;
  };

  using BidLadder = Ladder<std::greater<>>;
  using AskLadder = Ladder<std::less<>>;

  BidLadder bids_;
  AskLadder asks_;
  OrderPool order_pool_;

  // used for cancel
  std::unordered_map<Order::id_t, order_idx_t> id_to_index_;
};

template<typename Comparator>
typename OrderBook::PriceLevel &OrderBook::Ladder<Comparator>::levelFor(const Order::price_t price)
{
  auto compare = [this](const Entry &entry, const Order::price_t price) { return comp_(entry.price, price); };

  auto it = std::lower_bound(entries_.begin(), entries_.end(), price, compare);

  if (it != entries_.end() && it->price == price) { return it->level; }

  it = entries_.insert(it, Entry{ price, PriceLevel{} });
  return it->level;
}

template<typename Comparator>
typename OrderBook::PriceLevel *OrderBook::Ladder<Comparator>::levelOf(const Order::price_t price)
{
  auto compare = [this](const Entry &entry, const Order::price_t price) { return comp_(entry.price, price); };

  auto it = std::lower_bound(entries_.begin(), entries_.end(), price, compare);

  if (it != entries_.end() && it->price == price) { return &it->level; }

  return nullptr;
}

template<typename Comparator>
[[nodiscard]] const typename OrderBook::PriceLevel *OrderBook::Ladder<Comparator>::levelOf(
  const Order::price_t price) const
{
  auto compare = [this](const Entry &entry, const Order::price_t price) { return comp_(entry.price, price); };

  auto it = std::lower_bound(entries_.begin(), entries_.end(), price, compare);

  if (it != entries_.end() && it->price == price) { return &it->level; }

  return nullptr;
}

template<typename Comparator>
[[nodiscard]] const typename OrderBook::PriceLevel *OrderBook::Ladder<Comparator>::bestLevel() const
{
  return entries_.empty() ? nullptr : &entries_.front().level;
}

template<typename Comparator> void OrderBook::Ladder<Comparator>::eraseIfEmpty(Order::price_t price)
{
  auto compare = [this](const Entry &entry, const Order::price_t price) { return comp_(entry.price, price); };

  auto it = std::lower_bound(entries_.begin(), entries_.end(), price, compare);

  if (it == entries_.end() || it->price != price) { return; }

  if (it->level.empty()) { entries_.erase(it); }
}

#endif
