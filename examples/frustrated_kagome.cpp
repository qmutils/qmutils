#include <armadillo>
#include <iostream>
#include <vector>

#include "qmutils/basis.h"
#include "qmutils/expression.h"
#include "qmutils/functional.h"
#include "qmutils/index.h"
#include "qmutils/matrix_elements.h"
#include "qmutils/normal_order.h"
#include "qmutils/operator.h"

using namespace qmutils;

template <size_t Lx, size_t Ly>
class KagomeModel {
 public:
  using Index = StaticIndex<Lx, Ly, 3>;  // 3 sites per unit cell

  explicit KagomeModel(float t) : m_t(t), m_index() { construct_hamiltonian(); }

  const Expression& hamiltonian() const { return m_hamiltonian; }
  const Index& lattice() const { return m_index; }

 private:
  float m_t;      // Hopping strength
  Index m_index;  // Maps (x,y,site) to orbital index
  Expression m_hamiltonian;

  void construct_hamiltonian() {
    // Construct hopping terms for each unit cell
    for (size_t x = 0; x < Lx; ++x) {
      for (size_t y = 0; y < Ly; ++y) {
        add_intracell_hopping(x, y);  // Hopping within unit cell
        add_intercell_hopping(x, y);  // Hopping between unit cells
      }
    }
  }

  void add_intracell_hopping(size_t x, size_t y) {
    // Connect sites: 0-1, 1-2, 2-0
    add_hopping_term(x, y, 0, x, y, 1);
    add_hopping_term(x, y, 1, x, y, 2);
    add_hopping_term(x, y, 2, x, y, 0);
  }

  void add_intercell_hopping(size_t x, size_t y) {
    // Horizontal connections (along x)
    add_hopping_term(x, y, 0, (x + 1) % Lx, y, 2);

    // Vertical connections (along y)
    add_hopping_term(x, y, 1, x, (y + 1) % Ly, 0);

    // Diagonal connections
    add_hopping_term(x, y, 2, (x + 1) % Lx, (y + 1) % Ly, 1);
  }

  void add_hopping_term(size_t x1, size_t y1, size_t site1, size_t x2,
                        size_t y2, size_t site2) {
    size_t orbital1 = m_index.to_orbital(x1, y1, site1);
    size_t orbital2 = m_index.to_orbital(x2, y2, site2);

    // Add hopping for both spin up and spin down
    m_hamiltonian += m_t * Expression::Fermion::hopping(orbital1, orbital2,
                                                        Operator::Spin::Up);
    m_hamiltonian += m_t * Expression::Fermion::hopping(orbital1, orbital2,
                                                        Operator::Spin::Down);
  }
};

template <size_t Lx, size_t Ly>
class KagomeCompactState {
 public:
  static constexpr size_t CLS_SIZE = 6;

  static void demonstrate_cls(const KagomeModel<Lx, Ly>& model) {
    // Construct the compact localized state
    Expression cls_state = construct_compact_localized_state(model);

    NormalOrderer orderer;
    Expression norm_expr =
        orderer.normal_order(cls_state.adjoint() * cls_state);
    float norm = norm_expr[{}].real();
    std::cout << "Normalization ⟨ψ|ψ⟩: " << norm << "\n";

    // Print the state components
    std::cout << "\nCompact Localized State components:\n";
    std::cout << "--------------------------------\n";
    for (const auto& [ops, coeff] : cls_state.terms()) {
      auto [i, j, s] = model.lattice().from_orbital(ops[0].orbital());
      std::cout << "Site [" << i << "," << j << "," << s
                << "] Amplitude: " << coeff << "\n";
    }

    // Calculate energy expectation value
    Expression result = orderer.normal_order(cls_state.adjoint() *
                                             model.hamiltonian() * cls_state);

    std::cout << "\nEnergy per cell expectation value ⟨ψ|H|ψ⟩ / 6: "
              << result[{}] << "\n";
    std::cout << "Should be compared with: " << -2.0f / 6.0f << "\n";
  }

 private:
  static Expression construct_compact_localized_state(
      const KagomeModel<Lx, Ly>& model) {
    Expression state;

    // Normalization
    const float norm = 1.0f / std::sqrt(static_cast<float>(CLS_SIZE));

    // Define the hexagonal plaquette sites and their amplitudes
    struct Site {
      size_t x, y, site_index;
      float amplitude;
    };

    // Center coordinates
    size_t cx = Lx / 2;
    size_t cy = Ly / 2;

    const std::array<Site, CLS_SIZE> cls_sites = {
        {{cx, cy, 1, norm},
         {cx, cy, 2, -norm},
         {(cx + 1) % Lx, cy, 1, norm},
         {(cx + 1) % Lx, (cy + 1) % Ly, 2, -norm},
         {cx, (cy + 1) % Ly, 0, norm},
         {cx, cy, 0, -norm}}};

    for (const auto& site : cls_sites) {
      size_t orbital =
          model.lattice().to_orbital(site.x % Lx, site.y % Ly, site.site_index);
      state +=
          site.amplitude * Term::Fermion::creation(Operator::Spin::Up, orbital);
    }

    return state;
  }
};

int main() {
  std::cout << "Kagome Lattice Compact Localized States\n";
  std::cout << "======================================\n";

  // Create a Kagome lattice with unit hopping
  constexpr size_t Lx = 4;
  constexpr size_t Ly = 4;
  KagomeModel<Lx, Ly> model(1.0f);

  // Calculate full spectrum
  qmutils::Basis basis(3 * Lx * Ly, 1);  // Single-particle basis
  auto H_matrix = qmutils::compute_matrix_elements<arma::cx_fmat>(
      basis, model.hamiltonian());

  // Diagonalize Hamiltonian
  arma::fvec eigenvalues;
  arma::cx_fmat eigenvectors;
  arma::eig_sym(eigenvalues, eigenvectors, H_matrix);

  // Print spectrum
  std::cout << "\nEnergy spectrum:\n";
  std::cout << "---------------\n";
  for (size_t i = 0; i < eigenvalues.n_elem; ++i) {
    std::cout << "E[" << i << "] = " << eigenvalues(i) << "\n";
  }

  // Demonstrate CLS
  std::cout << "\nCompact Localized State Analysis:\n";
  std::cout << "--------------------------------\n";
  KagomeCompactState<Lx, Ly>::demonstrate_cls(model);

  return 0;
}
