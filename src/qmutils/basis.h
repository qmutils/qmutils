#pragma once

#include <algorithm>
#include <optional>
#include <unordered_map>

#include "qmutils/assert.h"
#include "qmutils/operator.h"
#include "qmutils/term.h"

namespace qmutils {

static constexpr uint64_t qmutils_choose(uint64_t n, uint64_t m) {
  if (m > n) return 0;
  if (m == 0 || m == n) return 1;

  if (m > n - m) m = n - m;

  uint64_t result = 1;
  for (uint64_t i = 0; i < m; ++i) {
    result *= (n - i);
    result /= (i + 1);
  }

  return result;
}

class BasisBase {
 public:
  using operators_type = Term::container_type;

  BasisBase(size_t orbitals, size_t particles)
      : m_orbitals(orbitals), m_particles(particles) {}

  BasisBase(const BasisBase&) = default;
  BasisBase& operator=(const BasisBase&) = default;
  BasisBase(BasisBase&&) noexcept = default;
  BasisBase& operator=(BasisBase&&) noexcept = default;

  virtual ~BasisBase() = default;

  size_t orbitals() const noexcept { return m_orbitals; }
  size_t particles() const noexcept { return m_particles; }
  size_t size() const noexcept { return m_index_map.size(); }

  bool operator==(const BasisBase& other) const {
    return m_orbitals == other.m_orbitals && m_particles == other.m_particles &&
           m_index_map == other.m_index_map;
  }

  bool operator!=(const BasisBase& other) const { return !(*this == other); }

  bool contains(const operators_type& value) const {
    auto it = std::lower_bound(m_index_map.begin(), m_index_map.end(),
                               Term(value), term_sorter);
    return it != m_index_map.end() && it->operators() == value;
  }

  // TODO: needs testing
  ptrdiff_t index_of(const operators_type& value) const {
    QMUTILS_ASSERT(contains(value));
    auto it = std::lower_bound(m_index_map.begin(), m_index_map.end(),
                               Term(value), term_sorter);
    return std::distance(m_index_map.begin(), it);
  }

  void insert(const operators_type& value) {
    QMUTILS_ASSERT(!contains(value));
    Term term(value);
    auto it = std::lower_bound(m_index_map.begin(), m_index_map.end(), term,
                               term_sorter);
    m_index_map.insert(it, term);
  }

  auto begin() const noexcept { return m_index_map.begin(); }
  auto end() const noexcept { return m_index_map.end(); }

  const Term& at(size_t i) const noexcept {
    QMUTILS_ASSERT(i < m_index_map.size());
    return m_index_map[i];
  }

  template <typename Fn>
  void filter(Fn&& fn) {
    m_index_map.erase(std::remove_if(m_index_map.begin(), m_index_map.end(),
                                     std::forward<Fn>(fn)),
                      m_index_map.end());
  }

  void generate_basis();

  virtual void generate_combinations(operators_type& current,
                                     size_t first_orbital, size_t depth) = 0;

  static bool term_sorter(const Term& a, const Term& b) noexcept {
    return a.operators() < b.operators();
  }

 protected:
  std::vector<Term> m_index_map;
  size_t m_orbitals;
  size_t m_particles;
};

class FermionicBasis : public BasisBase {
 public:
  FermionicBasis(size_t orbitals, size_t particles,
                 std::optional<int> required_sz = std::nullopt);

  void generate_combinations(operators_type& current, size_t first_orbital,
                             size_t depth) override;

  void generate_spin_basis(size_t required_up, size_t required_down);
  void generate_combinations_with_sz(operators_type& current,
                                     size_t first_orbital, size_t remaining_up,
                                     size_t remaining_down);

 private:
  static float calculate_normalization(
      [[maybe_unused]] const operators_type& ops) {
    return 1.0f;
  }

  static constexpr uint64_t compute_basis_size(uint64_t orbitals,
                                               uint64_t particles) {
    return qmutils_choose(2 * orbitals, particles);
  }
};

class BosonicBasis : public BasisBase {
 public:
  BosonicBasis(size_t orbitals, size_t particles)
      : BasisBase(orbitals, particles) {
    QMUTILS_ASSERT(orbitals <= Operator::Boson::max_orbital_size());
    size_t basis_size = compute_basis_size(orbitals, particles);
    m_index_map.reserve(basis_size);
    generate_basis();
    QMUTILS_ASSERT(m_index_map.size() == basis_size);
    std::sort(m_index_map.begin(), m_index_map.end(), term_sorter);
  }

  void generate_combinations(operators_type& current, size_t first_orbital,
                             size_t depth) override;

 private:
  static float calculate_normalization(const operators_type& ops) {
    std::unordered_map<Operator, size_t> state_counts;
    state_counts.reserve(ops.size());
    for (const auto& op : ops) {
      state_counts[op]++;
    }
    float normalization = 1.0f;
    for (const auto& [op, count] : state_counts) {
      // gamma(n + 1) = n!
      normalization *= std::sqrt(std::tgamma(static_cast<float>(count) + 1.0f));
    }
    return 1.0f / normalization;
  }

  static constexpr uint64_t compute_basis_size(uint64_t orbitals,
                                               uint64_t particles) {
    return qmutils_choose(orbitals + particles - 1, particles);
  }
};

class HardCoreBosonicBasis : public BasisBase {
 public:
  HardCoreBosonicBasis(size_t orbitals, size_t particles)
      : BasisBase(orbitals, particles) {
    QMUTILS_ASSERT(orbitals <= Operator::Boson::max_orbital_size());
    size_t basis_size = compute_basis_size(orbitals, particles);
    m_index_map.reserve(basis_size);
    generate_basis();
    QMUTILS_ASSERT(m_index_map.size() == basis_size);
    std::sort(m_index_map.begin(), m_index_map.end(), term_sorter);
  }

  void generate_combinations(operators_type& current, size_t first_orbital,
                             size_t depth) override;

 private:
  static float calculate_normalization(
      [[maybe_unused]] const operators_type& ops) {
    return 1.0f;
  }

  static constexpr uint64_t compute_basis_size(uint64_t orbitals,
                                               uint64_t particles) {
    return qmutils_choose(orbitals, particles);
  }
};

using Basis = FermionicBasis;

}  // namespace qmutils
