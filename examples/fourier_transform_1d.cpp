#include <cmath>
#include <complex>
#include <iostream>
#include <numbers>
#include <vector>

#include "qmutils/expression.h"
#include "qmutils/functional.h"
#include "qmutils/operator.h"

using namespace qmutils;

class TightBindingModel1D {
 public:
  TightBindingModel1D(size_t size, float t) : m_size(size), m_t(t) {
    construct_hamiltonian();
  }

  const Expression& hamiltonian() const { return m_hamiltonian; }

 private:
  size_t m_size;
  float m_t;
  Expression m_hamiltonian;

  void construct_hamiltonian() {
    for (size_t i = 0; i < m_size; ++i) {
      size_t j = (i + 1) % m_size;  // Periodic boundary conditions
      m_hamiltonian +=
          Expression::Fermion::hopping(i, j, Operator::Spin::Up) * m_t;
      m_hamiltonian +=
          Expression::Fermion::hopping(i, j, Operator::Spin::Down) * m_t;
    }
  }
};

static Expression fourier_transform_operator(const Operator& op, size_t size) {
  Expression result;
  const float type_sign =
      (op.type() == Operator::Type::Annihilation) ? -1.0f : 1.0f;
  const float factor =
      2.0f * std::numbers::pi_v<float> / static_cast<float>(size);

  for (size_t k = 0; k < size; ++k) {
    std::complex<float> coefficient(
        0.0f, -type_sign * factor * static_cast<float>(k * op.orbital()));
    coefficient = std::exp(coefficient) / std::sqrt(static_cast<float>(size));

    Operator transformed_op = Operator::Fermion::make(op.type(), op.spin(), k);
    result += Term(coefficient, {transformed_op});
  }

  return result;
}

static void print_hamiltonian(const Expression& hamiltonian,
                              float epsilon = 3e-4f) {
  for (const auto& [ops, coeff] : hamiltonian.terms()) {
    if (std::abs(coeff) > epsilon) {
      std::cout << "  " << Term(coeff, ops).to_string() << std::endl;
    }
  }
  std::cout << std::endl;
}

int main() {
  const size_t lattice_size = 8;
  const float hopping_strength = 1.0f;

  TightBindingModel1D model(lattice_size, hopping_strength);

  std::cout << "Hamiltonian in real space:" << std::endl;
  print_hamiltonian(model.hamiltonian());

  Expression momentum_hamiltonian = transform_expression(
      fourier_transform_operator, model.hamiltonian(), lattice_size);

  std::cout << "Hamiltonian in momentum space:" << std::endl;
  print_hamiltonian(momentum_hamiltonian);

  return 0;
}
