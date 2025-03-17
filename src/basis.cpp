#include "qmutils/basis.h"

namespace qmutils {

void BasisBase::generate_basis() {
  operators_type current;
  current.reserve(m_particles);
  generate_combinations(current, 0, 0);
}

FermionicBasis::FermionicBasis(size_t orbitals, size_t particles,
                               std::optional<int> required_sz)
    : BasisBase(orbitals, particles) {
  QMUTILS_ASSERT(orbitals <= Operator::Fermion::max_orbital_size());
  QMUTILS_ASSERT(particles <= 2 * orbitals);

  if (required_sz.has_value()) {
    int total = static_cast<int>(particles);
    QMUTILS_ASSERT((total + required_sz.value()) % 2 == 0 &&
                   (total - required_sz.value()) % 2 == 0);

    size_t required_up = static_cast<size_t>((total + required_sz.value()) / 2);
    size_t required_down =
        static_cast<size_t>((total - required_sz.value()) / 2);
    QMUTILS_ASSERT(required_up + required_down == particles);

    QMUTILS_ASSERT(std::abs(required_sz.value()) <=
                   static_cast<int>(particles));

    size_t basis_size = qmutils_choose(m_orbitals, required_up) *
                        qmutils_choose(m_orbitals, required_down);
    m_index_map.reserve(basis_size);
    generate_spin_basis(required_up, required_down);
    QMUTILS_ASSERT(m_index_map.size() == basis_size);
  } else {
    size_t basis_size = compute_basis_size(orbitals, particles);
    m_index_map.reserve(basis_size);
    generate_basis();
    QMUTILS_ASSERT(m_index_map.size() == basis_size);
  }

  std::sort(m_index_map.begin(), m_index_map.end(), term_sorter);
}

void FermionicBasis::generate_spin_basis(size_t required_up,
                                         size_t required_down) {
  operators_type current;
  current.reserve(required_up + required_down);
  generate_combinations_with_sz(current, 0, required_up, required_down);
}

void FermionicBasis::generate_combinations_with_sz(operators_type& current,
                                                   size_t first_orbital,
                                                   size_t remaining_up,
                                                   size_t remaining_down) {
  if (remaining_up == 0 && remaining_down == 0) {
    operators_type current_copy(current);
    std::sort(current_copy.begin(), current_copy.end());
    m_index_map.emplace_back(calculate_normalization(current_copy),
                             current_copy);
    return;
  }

  for (size_t i = first_orbital; i < m_orbitals; i++) {
    if (remaining_up > 0) {
      Operator::Spin spin = Operator::Spin::Up;
      if (current.empty() || current.back().orbital() < i ||
          ((current.back().orbital() == i && spin > current.back().spin()))) {
        current.push_back(Operator::Fermion::creation(spin, i));
        generate_combinations_with_sz(current, i, remaining_up - 1,
                                      remaining_down);
        current.pop_back();
      }
    }
    if (remaining_down > 0) {
      Operator::Spin spin = Operator::Spin::Down;
      if (current.empty() || current.back().orbital() < i ||
          ((current.back().orbital() == i && spin > current.back().spin()))) {
        current.push_back(Operator::Fermion::creation(spin, i));
        generate_combinations_with_sz(current, i, remaining_up,
                                      remaining_down - 1);
        current.pop_back();
      }
    }
  }
}

void FermionicBasis::generate_combinations(operators_type& current,
                                           size_t first_orbital, size_t depth) {
  if (depth == m_particles) {
    operators_type current_copy(current);
    std::sort(current_copy.begin(), current_copy.end());
    m_index_map.emplace_back(calculate_normalization(current_copy),
                             current_copy);
    return;
  }

  for (size_t i = first_orbital; i < m_orbitals; i++) {
    for (int spin_index = 0; spin_index < 2; ++spin_index) {
      Operator::Spin spin = static_cast<Operator::Spin>(spin_index);
      if (current.empty() || current.back().orbital() < i ||
          ((current.back().orbital() == i && spin > current.back().spin()))) {
        current.push_back(Operator::Fermion::creation(spin, i));
        generate_combinations(current, i, depth + 1);
        current.pop_back();
      }
    }
  }
}

void BosonicBasis::generate_combinations(operators_type& current,
                                         size_t first_orbital, size_t depth) {
  if (depth == m_particles) {
    operators_type current_copy(current);
    std::sort(current_copy.begin(), current_copy.end());
    m_index_map.emplace_back(calculate_normalization(current_copy),
                             current_copy);
    return;
  }

  for (size_t i = first_orbital; i < m_orbitals; i++) {
    if (current.empty() || current.back().orbital() <= i) {
      current.push_back(Operator::Boson::creation(i));
      generate_combinations(current, i, depth + 1);
      current.pop_back();
    }
  }
}

void HardCoreBosonicBasis::generate_combinations(operators_type& current,
                                                 size_t first_orbital,
                                                 size_t depth) {
  if (depth == m_particles) {
    operators_type current_copy(current);
    std::sort(current_copy.begin(), current_copy.end());
    m_index_map.emplace_back(calculate_normalization(current_copy),
                             current_copy);
    return;
  }

  for (size_t i = first_orbital; i < m_orbitals; i++) {
    if (current.empty() || current.back().orbital() < i) {
      current.push_back(Operator::Boson::creation(i));
      generate_combinations(current, i, depth + 1);
      current.pop_back();
    }
  }
}

}  // namespace qmutils
