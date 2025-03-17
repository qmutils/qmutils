#include <cstdint>
#include <iostream>
#include <stdexcept>

#include "qmutils/expression.h"
#include "qmutils/functional.h"
#include "qmutils/operator.h"

using namespace qmutils;

class Index2D {
 public:
  Index2D(size_t width, size_t height) : m_width(width), m_height(height) {
    if (width * height > 64) {
      throw std::invalid_argument(
          "Lattice size exceeds maximum number of orbitals (64)");
    }
  }

  size_t to_orbital(size_t x, size_t y) const {
    if (x >= m_width || y >= m_height) {
      throw std::out_of_range("Coordinates out of lattice bounds");
    }
    return y * m_width + x;
  }

  std::pair<size_t, size_t> from_orbital(size_t orbital) const {
    if (orbital >= m_width * m_height) {
      throw std::out_of_range("Orbital index out of lattice bounds");
    }
    return {orbital % m_width, orbital / m_width};
  }

  size_t width() const { return m_width; }
  size_t height() const { return m_height; }

 private:
  size_t m_width;
  size_t m_height;
};

class TightBindingModel2D {
 public:
  TightBindingModel2D(size_t width, size_t height, float t)
      : m_index(width, height), m_t(t) {
    construct_hamiltonian();
  }

  const Expression& hamiltonian() const { return m_hamiltonian; }
  const Index2D& lattice() const { return m_index; }

 private:
  Index2D m_index;
  float m_t;
  Expression m_hamiltonian;

  void construct_hamiltonian() {
    for (size_t y = 0; y < m_index.height(); ++y) {
      for (size_t x = 0; x < m_index.width(); ++x) {
        // Horizontal hopping
        size_t next_x = (x + 1) % m_index.width();
        m_hamiltonian +=
            m_t * Expression::Fermion::hopping(m_index.to_orbital(x, y),
                                               m_index.to_orbital(next_x, y),
                                               Operator::Spin::Up);
        m_hamiltonian +=
            m_t * Expression::Fermion::hopping(m_index.to_orbital(x, y),
                                               m_index.to_orbital(next_x, y),
                                               Operator::Spin::Down);

        // Vertical hopping
        size_t next_y = (y + 1) % m_index.height();
        m_hamiltonian +=
            m_t * Expression::Fermion::hopping(m_index.to_orbital(x, y),
                                               m_index.to_orbital(x, next_y),
                                               Operator::Spin::Up);
        m_hamiltonian +=
            m_t * Expression::Fermion::hopping(m_index.to_orbital(x, y),
                                               m_index.to_orbital(x, next_y),
                                               Operator::Spin::Down);
      }
    }
  }
};

static Expression fourier_transform_operator_2d(const Operator& op,
                                                const Index2D& lattice) {
  Expression result;
  const float type_sign =
      (op.type() == Operator::Type::Annihilation) ? -1.0f : 1.0f;

  constexpr float pi = std::numbers::pi_v<float>;
  const float factor_x = 2.0f * pi / static_cast<float>(lattice.width());
  const float factor_y = 2.0f * pi / static_cast<float>(lattice.height());

  auto [x, y] = lattice.from_orbital(op.orbital());

  for (size_t ky = 0; ky < lattice.height(); ++ky) {
    for (size_t kx = 0; kx < lattice.width(); ++kx) {
      std::complex<float> coefficient(
          0.0f, -type_sign * (factor_x * static_cast<float>(kx * x) +
                              factor_y * static_cast<float>(ky * y)));
      coefficient =
          std::exp(coefficient) /
          std::sqrt(static_cast<float>(lattice.width() * lattice.height()));

      Operator transformed_op = Operator::Fermion::make(
          op.type(), op.spin(), lattice.to_orbital(kx, ky));
      result += Term(coefficient, {transformed_op});
    }
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
  const size_t lattice_width = 4;
  const size_t lattice_height = 3;
  const float hopping_strength = 1.0f;

  TightBindingModel2D model(lattice_width, lattice_height, hopping_strength);

  std::cout << "Hamiltonian in real space:" << std::endl;
  print_hamiltonian(model.hamiltonian());

  Expression momentum_hamiltonian = transform_expression(
      fourier_transform_operator_2d, model.hamiltonian(), model.lattice());

  std::cout << "Hamiltonian in momentum space:" << std::endl;
  print_hamiltonian(momentum_hamiltonian);

  return 0;
}
