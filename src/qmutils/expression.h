#pragma once

#include <algorithm>
#include <complex>
#include <unordered_map>

#include "qmutils/term.h"

namespace qmutils {

class Expression {
 public:
  using coefficient_type = Term::coefficient_type;
  using operators_type = Term::container_type;
  using terms_type = std::unordered_map<operators_type, coefficient_type>;

  Expression() = default;

  explicit Expression(const Term& term) {
    m_terms[term.operators()] = term.coefficient();
  }

  explicit Expression(Operator op) {
    m_terms[{op}] = coefficient_type(1.0f, 0.0f);
  }

  explicit Expression(const operators_type& ops) {
    m_terms[ops] = coefficient_type(1.0f, 0.0f);
  }

  explicit Expression(const coefficient_type& coeff,
                      const operators_type& ops) {
    m_terms[ops] = coeff;
  }

  Expression(const coefficient_type& scalar) { m_terms[{}] = scalar; }

  Expression(const Expression&) = default;

  Expression(Expression&&) noexcept = default;

  Expression& operator=(const Expression&) = default;

  Expression& operator=(Expression&&) noexcept = default;

  // Arithmetic operations
  Expression& operator+=(const Expression& other) {
    for (const auto& [ops, coeff] : other.m_terms) {
      m_terms[ops] += coeff;
    }
    normalize();
    return *this;
  }

  Expression& operator+=(const Term& other) {
    m_terms[other.operators()] += other.coefficient();
    normalize();
    return *this;
  }

  Expression& operator+=(Operator other) {
    m_terms[{other}] += coefficient_type(1.0);
    normalize();
    return *this;
  }

  Expression& operator+=(coefficient_type other) {
    m_terms[{}] += coefficient_type(other);
    normalize();
    return *this;
  }

  Expression& operator-=(const Expression& other) {
    for (const auto& [ops, coeff] : other.m_terms) {
      m_terms[ops] -= coeff;
    }
    normalize();
    return *this;
  }

  Expression& operator-=(const Term& other) {
    m_terms[other.operators()] -= other.coefficient();
    normalize();
    return *this;
  }

  Expression& operator*=(const Expression& other) {
    terms_type new_terms;
    for (const auto& [ops1, coeff1] : m_terms) {
      for (const auto& [ops2, coeff2] : other.m_terms) {
        operators_type new_ops;
        new_ops.reserve(ops1.size() + ops2.size());
        new_ops.insert(new_ops.end(), ops1.begin(), ops1.end());
        new_ops.insert(new_ops.end(), ops2.begin(), ops2.end());
        new_terms[std::move(new_ops)] += coeff1 * coeff2;
      }
    }
    m_terms = std::move(new_terms);
    normalize();
    return *this;
  }

  Expression& operator*=(const Term& term) {
    terms_type new_terms;
    for (const auto& [ops, coeff] : m_terms) {
      operators_type new_ops;
      new_ops.reserve(ops.size() + term.operators().size());
      new_ops.insert(new_ops.end(), ops.begin(), ops.end());
      new_ops.insert(new_ops.end(), term.operators().begin(),
                     term.operators().end());
      new_terms[new_ops] += coeff * term.coefficient();
    }

    m_terms = std::move(new_terms);
    normalize();
    return *this;
  }

  Expression& operator*=(const coefficient_type& scalar) {
    for (auto& [ops, coeff] : m_terms) {
      coeff *= scalar;
    }
    normalize();
    return *this;
  }

  friend Expression operator+(Expression lhs, const Expression& rhs) {
    lhs += rhs;
    return lhs;
  }

  friend Expression operator-(Expression lhs, const Expression& rhs) {
    lhs -= rhs;
    return lhs;
  }

  friend Expression operator*(Expression lhs, const Expression& rhs) {
    lhs *= rhs;
    return lhs;
  }

  friend Expression operator*(Expression lhs, const Term& rhs) {
    lhs *= rhs;
    return lhs;
  }

  friend Expression operator*(const Term& lhs, Expression rhs) {
    rhs *= lhs;
    return rhs;
  }

  friend Expression operator*(Expression lhs, const coefficient_type& rhs) {
    lhs *= rhs;
    return lhs;
  }

  friend Expression operator*(const coefficient_type& lhs, Expression rhs) {
    rhs *= lhs;
    return rhs;
  }

  const coefficient_type& operator[](const operators_type& ops) {
    return m_terms[ops];
  }

  size_t size() const { return m_terms.size(); }

  const terms_type& terms() const { return m_terms; }

  terms_type& terms() { return m_terms; }

  bool operator==(const Expression& other) const {
    return m_terms == other.m_terms;
  }

  bool operator!=(const Expression& other) const { return !(*this == other); }

  std::string to_string() const;

  friend std::ostream& operator<<(std::ostream& os, const Expression& expr);

  void normalize();

  Expression adjoint() const;

  Expression flip_spin() const;

  struct Fermion;
  struct Boson;
  struct Spin;

  // Remove
  bool is_purely(const Operator::Statistics& s) const {
    return std::all_of(m_terms.begin(), m_terms.end(), [&](const auto& a) {
      return Term(a.first).is_purely(s);
    });
  }

 private:
  terms_type m_terms;
};

struct Expression::Fermion {
  static Expression hopping(size_t from_orbital, size_t to_orbital,
                            Operator::Spin spin) {
    QMUTILS_ASSERT(from_orbital != to_orbital);
    Expression result;
    result += Term::Fermion::one_body(spin, to_orbital, spin, from_orbital);
    result += Term::Fermion::one_body(spin, from_orbital, spin, to_orbital);
    return result;
  }

  static Expression hopping(coefficient_type coeff, size_t from_orbital,
                            size_t to_orbital, Operator::Spin spin) {
    QMUTILS_ASSERT(from_orbital != to_orbital);
    Expression result;
    result +=
        coeff * Term::Fermion::one_body(spin, to_orbital, spin, from_orbital);
    result += std::conj(coeff) *
              Term::Fermion::one_body(spin, from_orbital, spin, to_orbital);
    return result;
  }
};

struct Expression::Boson {
  static Expression hopping(size_t from_orbital, size_t to_orbital) {
    QMUTILS_ASSERT(from_orbital != to_orbital);
    Expression result;
    result += Term::Boson::one_body(to_orbital, from_orbital);
    result += Term::Boson::one_body(from_orbital, to_orbital);
    return result;
  }

  static Expression hopping(coefficient_type coeff, size_t from_orbital,
                            size_t to_orbital) {
    QMUTILS_ASSERT(from_orbital != to_orbital);
    Expression result;
    result += coeff * Term::Boson::one_body(to_orbital, from_orbital);
    result +=
        std::conj(coeff) * Term::Boson::one_body(from_orbital, to_orbital);
    return result;
  }
};

struct Expression::Spin {
  static Expression spin_x(size_t i) {
    Expression result;
    result += 0.5f * Term::Fermion::one_body(Operator::Spin::Up, i,
                                             Operator::Spin::Down, i);
    result += 0.5f * Term::Fermion::one_body(Operator::Spin::Down, i,
                                             Operator::Spin::Up, i);
    return result;
  }

  static Expression spin_y(size_t i) {
    Expression result;
    result +=
        std::complex<float>(0.0f, 0.5f) *
        Term::Fermion::one_body(Operator::Spin::Up, i, Operator::Spin::Down, i);
    result -=
        std::complex<float>(0.0f, 0.5f) *
        Term::Fermion::one_body(Operator::Spin::Down, i, Operator::Spin::Up, i);
    return result;
  }

  static Expression spin_z(size_t i) {
    Expression result;
    result += 0.5f * Term::Fermion::one_body(Operator::Spin::Up, i,
                                             Operator::Spin::Up, i);
    result -= 0.5f * Term::Fermion::one_body(Operator::Spin::Down, i,
                                             Operator::Spin::Down, i);
    return result;
  }

  static Expression spin_plus(size_t site) {
    return Expression(Term::Fermion::one_body(Operator::Spin::Up, site,
                                              Operator::Spin::Down, site));
  }

  static Expression spin_minus(size_t site) {
    return Expression(Term::Fermion::one_body(Operator::Spin::Down, site,
                                              Operator::Spin::Up, site));
  }

  static Expression dot_product(size_t i, size_t j) {
    Expression result;
    result += spin_x(i) * spin_x(j);
    result += spin_y(i) * spin_y(j);
    result += spin_z(i) * spin_z(j);
    return result;
  }
};

}  // namespace qmutils
