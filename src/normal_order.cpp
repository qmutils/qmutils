#include "qmutils/normal_order.h"

#include <iostream>

namespace qmutils {

Expression NormalOrderer::normal_order(const Term& term) {
  return term.coefficient() * normal_order_recursive(term.operators());
}

Expression NormalOrderer::normal_order(const Expression& expr) {
  Expression result;
  for (const auto& [ops, coeff] : expr.terms()) {
    result += coeff * normal_order_recursive(ops);
  }
  return result;
}

static constexpr int phase_to_sign(size_t phase) { return 1 - 2 * (phase & 1); }

Expression NormalOrderer::normal_order_recursive(const operators_type& ops) {
  if (ops.size() < 2) {
    return Expression(ops);
  }

  std::size_t ops_key = std::hash<operators_type>{}(ops);
  return normal_order_recursive(ops, ops_key);
}

Expression NormalOrderer::normal_order_recursive(const operators_type& ops,
                                                 size_t ops_hash) {
  if (auto cached_result = m_cache.get(ops_hash)) {
    m_cache_hits++;
    return cached_result.value();
  }
  m_cache_misses++;

  std::unique_ptr<operators_type> local_ops_ptr = m_pool.acquire();
  operators_type& local_ops = *local_ops_ptr;
  local_ops.assign(ops.begin(), ops.end());
  size_t global_phase = 0;

  for (size_t i = 1; i < local_ops.size(); ++i) {
    size_t j = i;
    while (j > 0 && local_ops[j] < local_ops[j - 1]) {
      size_t local_phase = Operator::parity(local_ops[j], local_ops[j - 1]);
      if (local_ops[j].commutes(local_ops[j - 1])) {
        std::swap(local_ops[j], local_ops[j - 1]);
        global_phase ^= local_phase;
        --j;
      } else {
        Expression result = phase_to_sign(global_phase) *
                            handle_non_commuting(local_ops, j - 1, local_phase);
        m_cache.put(ops_hash, result);
        m_pool.release(std::move(local_ops_ptr));
        return result;
      }
    }
  }

  Expression result(phase_to_sign(global_phase), local_ops);
  m_cache.put(ops_hash, result);
  m_pool.release(std::move(local_ops_ptr));
  return result;
}

Expression NormalOrderer::handle_non_commuting(const operators_type& ops,
                                               size_t index, size_t phase) {
  std::unique_ptr<operators_type> contracted_ptr = m_pool.acquire();
  operators_type& contracted = *contracted_ptr;
  contracted.assign(ops.begin(), ops.begin() + index);
  contracted.insert(contracted.end(), ops.begin() + index + 2, ops.end());

  std::unique_ptr<operators_type> swapped_ptr = m_pool.acquire();
  operators_type& swapped = *swapped_ptr;
  swapped.assign(ops.begin(), ops.end());
  std::swap(swapped[index], swapped[index + 1]);

  Expression result = normal_order_recursive(contracted) +
                      phase_to_sign(phase) * normal_order_recursive(swapped);
  m_pool.release(std::move(contracted_ptr));
  m_pool.release(std::move(swapped_ptr));
  return result;
}

void NormalOrderer::print_cache_stats() const {
  std::cout << "Total entries: " << m_cache.size() << std::endl;
  std::cout << "Cache hits: " << m_cache_hits << std::endl;
  std::cout << "Cache misses: " << m_cache_misses << std::endl;
  double hit_ratio =
      static_cast<double>(m_cache_hits) / (m_cache_hits + m_cache_misses);
  std::cout << "Hit ratio: " << hit_ratio << std::endl;
}

}  // namespace qmutils
