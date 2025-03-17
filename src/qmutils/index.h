#pragma once

#include <array>
#include <sstream>

#include "qmutils/operator.h"

namespace qmutils {

template <typename Coords>
std::string operator_to_string(const Operator& op, const Coords& coords) {
  std::ostringstream oss;
  oss << (op.type() == Operator::Type::Creation ? "c+" : "c") << "("
      << (op.spin() == Operator::Spin::Up ? "↑" : "↓") << ",[";

  for (size_t i = 0; i < coords.size(); ++i) {
    if (i > 0) oss << ",";
    oss << coords[i];
  }

  oss << "])";
  return oss.str();
}

template <typename Index>
std::string term_to_string(const Term& term, const Index& index) {
  std::ostringstream oss;
  oss << term.coefficient() << " ";
  for (Operator op : term.operators()) {
    oss << operator_to_string(op, index.from_orbital(op.orbital()));
  }
  return oss.str();
}

template <typename Index>
std::string expr_to_string(const Expression& expr, const Index& index) {
  std::ostringstream oss;
  bool first = true;
  for (const auto& [ops, coeff] : expr.terms()) {
    if (!first) {
      oss << " + ";
    }
    first = false;
    oss << coeff << " ";
    for (Operator op : ops) {
      oss << operator_to_string(op, index.from_orbital(op.orbital()));
    }
  }
  return oss.str();
}

template <size_t... Dims>
class StaticIndex {
 public:
  static constexpr size_t Dimensions = sizeof...(Dims);

  constexpr StaticIndex() {
    // Assume Fermion for now
    static_assert(total_size() <= Operator::Fermion::max_orbital_size(),
                  "Size exceeds maximum number of orbitals");
  }

  template <typename... Coords>
  [[nodiscard]] constexpr size_t to_orbital(Coords... coords) const {
    static_assert(sizeof...(coords) == Dimensions,
                  "Invalid number of coordinates");
    std::array<size_t, Dimensions> coordinates = {
        static_cast<size_t>(coords)...};

    for (size_t i = 0; i < Dimensions; ++i) {
      if (coordinates[i] >= m_dimensions[i]) {
        throw std::out_of_range("Coordinates out of bounds");
      }
    }

    size_t orbital = 0;
    size_t multiplier = 1;
    for (size_t i = 0; i < Dimensions; ++i) {
      orbital += static_cast<size_t>(coordinates[i] * multiplier);
      multiplier *= m_dimensions[i];
    }
    return orbital;
  }

  [[nodiscard]] constexpr std::array<size_t, Dimensions> from_orbital(
      size_t orbital) const {
    if (orbital >= total_size()) {
      throw std::out_of_range("Orbital index out of bounds");
    }

    std::array<size_t, Dimensions> coordinates;
    for (size_t i = 0; i < Dimensions; ++i) {
      coordinates[i] = orbital % m_dimensions[i];
      orbital /= m_dimensions[i];
    }
    return coordinates;
  }

  [[nodiscard]] constexpr size_t value_at(size_t orbital, size_t i) const {
    if (i >= Dimensions) {
      throw std::out_of_range("Dimension index out of bounds");
    }
    return from_orbital(orbital)[i];
  }

  constexpr const std::array<size_t, Dimensions>& dimensions() const {
    return m_dimensions;
  }

  constexpr size_t dimension(size_t i) const {
    if (i >= Dimensions) {
      throw std::out_of_range("Dimension index out of bounds");
    }
    return m_dimensions[i];
  }

  [[nodiscard]] std::string to_string(const Operator& op) const {
    return operator_to_string(op, from_orbital(op.orbital()));
  }

  [[nodiscard]] std::string to_string(const Term& term) const {
    return term_to_string(term, *this);
  }

  [[nodiscard]] std::string to_string(const Expression& expr) const {
    return expr_to_string(expr, *this);
  }
  static constexpr size_t total_size() { return (... * Dims); }

 private:
  static constexpr std::array<size_t, Dimensions> m_dimensions = {Dims...};
};

class DynamicIndex {
 public:
  DynamicIndex(std::vector<size_t> dimensions)
      : m_dimensions(std::move(dimensions)) {
    // Assume fermion for now
    if (total_size() > Operator::Fermion::max_orbital_size()) {
      throw std::out_of_range("Size exceeds maximum number of orbitals");
    }
  }

  [[nodiscard]] size_t to_orbital(
      const std::vector<size_t>& coordinates) const {
    if (coordinates.size() != m_dimensions.size()) {
      throw std::out_of_range("Invalid number of coordinates");
    }

    size_t orbital = 0;
    size_t multiplier = 1;
    for (size_t i = 0; i < m_dimensions.size(); ++i) {
      if (coordinates[i] >= m_dimensions[i]) {
        throw std::out_of_range("Coordinates out of bounds");
      }
      orbital += coordinates[i] * multiplier;
      multiplier *= m_dimensions[i];
    }
    return orbital;
  }

  [[nodiscard]] std::vector<size_t> from_orbital(size_t orbital) const {
    if (orbital >= total_size()) {
      throw std::out_of_range("Orbital index out of bounds");
    }

    std::vector<size_t> coordinates(m_dimensions.size());
    for (size_t i = 0; i < m_dimensions.size(); ++i) {
      coordinates[i] = orbital % m_dimensions[i];
      orbital /= m_dimensions[i];
    }
    return coordinates;
  }

  [[nodiscard]] size_t value_at(size_t orbital, size_t i) const {
    if (i >= m_dimensions.size()) {
      throw std::out_of_range("Dimension is out of bounds");
    }
    return from_orbital(orbital)[i];
  }

  const std::vector<size_t>& dimensions() const { return m_dimensions; }

  size_t dimension(size_t i) const {
    if (i >= m_dimensions.size()) {
      throw std::out_of_range("Dimension is out of bounds");
    }
    return m_dimensions[i];
  }

  size_t size() const { return total_size(); }

  [[nodiscard]] std::string to_string(const Operator& op) const {
    return operator_to_string(op, from_orbital(op.orbital()));
  }

  [[nodiscard]] std::string to_string(const Term& term) const {
    return term_to_string(term, *this);
  }

  [[nodiscard]] std::string to_string(const Expression& expr) const {
    return expr_to_string(expr, *this);
  }

 private:
  std::vector<size_t> m_dimensions;

  size_t total_size() const {
    size_t size = 1;
    for (size_t dim : m_dimensions) {
      size *= dim;
    }
    return size;
  }
};

}  // namespace qmutils
