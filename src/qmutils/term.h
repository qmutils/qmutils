#pragma once

#include <algorithm>
#include <complex>
#include <ostream>
#include <string>
#include <unordered_map>
#include <vector>

#include "qmutils/operator.h"
#include "xxhashct/xxh64.hpp"

namespace qmutils {

class Term {
 public:
  using coefficient_type = std::complex<float>;
  using container_type = std::vector<Operator>;

  Term() = default;

  explicit Term(Operator op) : m_coefficient(1.0f), m_operators({op}) {}

  explicit Term(const container_type& operators)
      : m_coefficient(1.0f), m_operators(operators) {}

  explicit Term(container_type&& operators) noexcept
      : m_coefficient(1.0f), m_operators(std::move(operators)) {}

  explicit Term(const coefficient_type& coeff) : m_coefficient(coeff) {}

  explicit Term(const coefficient_type& coeff, const container_type& operators)
      : m_coefficient(coeff), m_operators(operators) {}

  explicit Term(const coefficient_type& coeff,
                container_type&& operators) noexcept
      : m_coefficient(coeff), m_operators(std::move(operators)) {}

  Term(const Term& other) = default;

  Term(Term&& other) noexcept = default;

  Term& operator=(const Term& other) = default;

  Term& operator=(Term&& other) noexcept = default;

  const container_type& operators() const { return m_operators; }

  const coefficient_type& coefficient() const { return m_coefficient; }

  size_t size() const { return m_operators.size(); }

  Operator operator[](size_t index) const { return m_operators[index]; }

  Operator& operator[](size_t index) { return m_operators[index]; }

  Term& operator*=(const Term& other) {
    m_coefficient *= other.m_coefficient;
    m_operators.reserve(m_operators.size() + other.m_operators.size());
    m_operators.insert(m_operators.end(), other.m_operators.begin(),
                       other.m_operators.end());
    return *this;
  }

  Term& operator*=(Operator op) {
    m_operators.push_back(op);
    return *this;
  }

  Term& operator*=(const coefficient_type& scalar) {
    m_coefficient *= scalar;
    return *this;
  }

  bool operator==(const Term& other) const {
    return m_coefficient == other.m_coefficient &&
           m_operators == other.m_operators;
  }

  bool operator!=(const Term& other) const { return !(*this == other); }

  Term adjoint() const;

  Term flip_spin() const;

  std::string to_string() const;

  friend std::ostream& operator<<(std::ostream& os, const Term& term);

  friend Term operator*(Term term, const Term& other) {
    term *= other;
    return term;
  }

  friend Term operator*(Term term, Operator other) {
    term *= other;
    return term;
  }

  friend Term operator*(Term term, const Term::coefficient_type& scalar) {
    term *= scalar;
    return term;
  }

  friend Term operator*(const Term::coefficient_type& scalar, Term term) {
    return term * scalar;
  }

  struct Fermion;
  struct Boson;

  // Remove
  bool is_purely(const Operator::Statistics& s) const {
    return std::all_of(
        m_operators.begin(), m_operators.end(),
        [&](const Operator& op) { return op.statistics() == s; });
  }

  // Remove
  static bool is_diagonal(const Term::container_type& ops) {
    std::unordered_map<size_t, int> counts;

    for (const auto& op : ops) {
      auto key = op.data();
      key |= 1 << 14;  // set the creation bit
      if (op.type() == Operator::Type::Creation) {
        counts[key] += 1;
      } else {
        counts[key] -= 1;
      }
    }

    if (std::any_of(counts.begin(), counts.end(),
                    [](const auto& elm) { return elm.second != 0; })) {
      return false;
    }

    return true;
  }

  bool is_diagonal() const { return is_diagonal(m_operators); }

 private:
  coefficient_type m_coefficient;
  container_type m_operators;
};

struct Term::Fermion {
  static Term creation(Operator::Spin spin, size_t orbital) {
    return Term({Operator::Fermion::creation(spin, orbital)});
  }

  static Term annihilation(Operator::Spin spin, size_t orbital) {
    return Term({Operator::Fermion::annihilation(spin, orbital)});
  }

  static Term one_body(Operator::Spin spin1, size_t orbital1,
                       Operator::Spin spin2, size_t orbital2) {
    return Term({Operator::Fermion::creation(spin1, orbital1),
                 Operator::Fermion::annihilation(spin2, orbital2)});
  }

  static Term density(Operator::Spin spin, size_t orbital) {
    return Term({Operator::Fermion::creation(spin, orbital),
                 Operator::Fermion::annihilation(spin, orbital)});
  }

  static Term density_density(Operator::Spin spin1, size_t i,
                              Operator::Spin spin2, size_t j) {
    return Term::Fermion::density(spin1, i) * Term::Fermion::density(spin2, j);
  }
};

struct Term::Boson {
  static Term creation(size_t orbital) {
    return Term({Operator::Boson::creation(orbital)});
  }
  static Term annihilation(size_t orbital) {
    return Term({Operator::Boson::annihilation(orbital)});
  }
  static Term one_body(size_t orbital1, size_t orbital2) {
    return Term({Operator::Boson::creation(orbital1),
                 Operator::Boson::annihilation(orbital2)});
  }

  static Term density(size_t orbital) {
    return Term({Operator::Boson::creation(orbital),
                 Operator::Boson::annihilation(orbital)});
  }

  static Term density_density(size_t i, size_t j) {
    return Term::Boson::density(i) * Term::Boson::density(j);
  }
};
}  // namespace qmutils

template <>
struct std::hash<qmutils::Term::container_type> {
  size_t operator()(const qmutils::Term::container_type& container) const {
    const char* begin = reinterpret_cast<const char*>(&container.data()[0]);
    size_t len = container.size() * sizeof(qmutils::Operator);
    return xxh64::hash(begin, len, 0);
  }
};

template <>
struct std::hash<qmutils::Term> {
  size_t operator()(const qmutils::Term& term) const {
    size_t seed = std::hash<float>{}(term.coefficient().real()) ^
                  std::hash<float>{}(term.coefficient().imag());
    const char* begin =
        reinterpret_cast<const char*>(&term.operators().data()[0]);
    size_t len = term.operators().size() * sizeof(qmutils::Operator);
    return xxh64::hash(begin, len, seed);
  }
};
