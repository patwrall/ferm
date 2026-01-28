#include <ferm/OrderBook.hpp>
#include <algorithm>

OrderBook::order_idx_t OrderBook::OrderPool::add(const Order &order)
{
    if (!free_list_.empty()) {
        const auto idx = free_list_.back();
        free_list_.pop_back();
        orders_[idx] = order;
        active_[idx] = true;
        return idx;
    }
    if (orders_.size() >= ORDERS_CAPACITY)
        throw std::overflow_error("OrderPool capacity exceeded");
    orders_.push_back(order);
    active_.push_back(true);
    return static_cast<order_idx_t>(orders_.size() - 1);
}

void OrderBook::OrderPool::remove(order_idx_t idx)
{
    active_[idx] = false;
    free_list_.push_back(idx);
}

Order &OrderBook::OrderPool::get(order_idx_t idx) { return orders_[idx]; }
const Order &OrderBook::OrderPool::get(order_idx_t idx) const { return orders_[idx]; }

bool OrderBook::OrderPool::isActive(order_idx_t idx) const noexcept
{
    return idx < orders_.size() && active_[idx];
}

OrderBook::MatchResult OrderBook::match(const Aggressor &a)
{
    MatchResult res{ .qty_remaining = a.size };
    res.fills.reserve(8);

    const auto fill_against = [&](auto &ladder, auto crosses) {
        while (res.qty_remaining > 0 && !ladder.empty()) {
            const Order::price_t best_px = ladder.entries().front().price;
            if (!crosses(best_px)) break;
            auto *lvl = ladder.levelOf(best_px);

            while (res.qty_remaining > 0 && !lvl->empty()) {
                const auto rest_idx = lvl->order_idxs.front();
                auto &rest = order_pool_.get(rest_idx);
                const auto qty = std::min(res.qty_remaining, rest.getCurrentSize());

                res.fills.push_back({ rest_idx, qty, best_px });
                res.qty_remaining -= qty;
                rest.fill(qty);

                if (rest.isFilled()) {
                    lvl->popFront(qty);
                    id_to_index_.erase(rest.getId());
                    order_pool_.remove(rest_idx);
                } else {
                    lvl->adjustDepth(-qty);
                }
            }

            if (lvl->empty()) ladder.eraseIfEmpty(best_px);
        }
    };

    if (a.side == Side::BUY)
        fill_against(asks_, [&](Order::price_t px) {
            return a.type == OrderType::MARKET || a.price >= px;
        });
    else
        fill_against(bids_, [&](Order::price_t px) {
            return a.type == OrderType::MARKET || a.price <= px;
        });

    return res;
}

void OrderBook::addOrder(const Order &order)
{
    const auto idx = order_pool_.add(order);
    id_to_index_[order.getId()] = idx;

    auto res = match({ order.getSide(), order.getPrice(), order.getType(),
                       order.getCurrentSize(), order.getId() });

    const auto filled = order.getCurrentSize() - res.qty_remaining;
    if (filled > 0) order_pool_.get(idx).fill(filled);

    if (res.qty_remaining > 0 && order.getType() == OrderType::LIMIT) {
        auto &lvl = (order.getSide() == Side::BUY)
            ? bids_.levelFor(order.getPrice())
            : asks_.levelFor(order.getPrice());
        lvl.push(idx, res.qty_remaining);
    } else {
        order_pool_.remove(idx);
        id_to_index_.erase(order.getId());
    }
}

void OrderBook::cancel(Order::id_t id)
{
    const auto it = id_to_index_.find(id);
    if (it == id_to_index_.end()) return;

    const auto idx = it->second;
    const auto &order = order_pool_.get(idx);
    const auto qty = order.getCurrentSize();
    const auto px = order.getPrice();

    const auto erase_from = [&](auto &ladder) {
        auto *lvl = ladder.levelOf(px);
        if (!lvl) return;
        auto &deq = lvl->order_idxs;
        if (const auto pos = std::find(deq.begin(), deq.end(), idx); pos != deq.end())
            deq.erase(pos);
        lvl->adjustDepth(-qty);
        ladder.eraseIfEmpty(px);
    };

    if (order.getSide() == Side::BUY) erase_from(bids_);
    else erase_from(asks_);

    order_pool_.remove(idx);
    id_to_index_.erase(it);
}

std::optional<Order> OrderBook::bestBid() const noexcept
{
    const auto *lvl = bids_.bestLevel();
    if (!lvl || lvl->empty()) return std::nullopt;
    return order_pool_.get(lvl->order_idxs.front());
}

std::optional<Order> OrderBook::bestAsk() const noexcept
{
    const auto *lvl = asks_.bestLevel();
    if (!lvl || lvl->empty()) return std::nullopt;
    return order_pool_.get(lvl->order_idxs.front());
}
