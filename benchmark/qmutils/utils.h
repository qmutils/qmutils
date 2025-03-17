#pragma once

#include <random>
#include <vector>

#include "qmutils/expression.h"
#include "qmutils/operator.h"
#include "qmutils/term.h"

namespace qmutils {

inline std::mt19937& get_random_generator() {
  static std::mt19937 gen(42);
  return gen;
}

inline std::complex<float> random_complex() {
  static std::uniform_real_distribution<float> dis(-1.0, 1.0);
  return {dis(get_random_generator()), dis(get_random_generator())};
}

inline Operator random_operator() {
  static std::uniform_int_distribution<> type_dis(0, 1);
  static std::uniform_int_distribution<> spin_dis(0, 1);
  static std::uniform_int_distribution<> orbital_dis(0, 63);

  return Operator::Fermion::make(
      static_cast<Operator::Type>(type_dis(get_random_generator())),
      static_cast<Operator::Spin>(spin_dis(get_random_generator())),
      orbital_dis(get_random_generator()));
}

inline std::vector<Operator> generate_operator_sequence(size_t size) {
  std::vector<Operator> ops;
  ops.reserve(size);
  for (size_t i = 0; i < size; ++i) {
    ops.push_back(random_operator());
  }
  return ops;
}

inline Term create_random_term(size_t n_operators) {
  return Term(random_complex(), generate_operator_sequence(n_operators));
}

inline Expression generate_large_expression(size_t num_terms,
                                            size_t ops_per_term) {
  Expression expr;
  for (size_t i = 0; i < num_terms; ++i) {
    expr += create_random_term(ops_per_term);
  }
  return expr;
}

}  // namespace qmutils
