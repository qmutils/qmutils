#include <armadillo>

#include "dos_utils.h"
#include "qmutils/expression.h"
#include "qmutils/functional.h"
#include "qmutils/index.h"
#include "qmutils/matrix_elements.h"
#include "qmutils/sparse_matrix.h"
#include "qmutils/term.h"

using namespace qmutils;

template <size_t Lx, size_t Ly>
class SquareLatticeLandauLevel {
 public:
  using Index = StaticIndex<Lx, Ly>;

  SquareLatticeLandauLevel(float t, float phi, float U)
      : m_t(t), m_phi(phi), m_U(U), m_index() {
    construct_hamiltonian();
  }

  const Expression& hamiltonian() const { return m_hamiltonian; }

  const Index& index() const { return m_index; }

  float t() const { return m_t; }
  float phi() const { return m_phi; }
  float U() const { return m_U; }

 private:
  float m_t;
  float m_phi;
  float m_U;
  Index m_index;
  Expression m_hamiltonian;

  void construct_hamiltonian() {
    construct_kinetic_part();
    construct_interaction_part();
  }

  void construct_kinetic_part() {
    for (size_t i = 0; i < Lx; ++i) {
      for (size_t j = 0; j < Ly; ++j) {
        size_t current = m_index.to_orbital(i, j);
        // Horizontal hopping (x-direction)
        size_t right = m_index.to_orbital((i + 1) % Lx, j);
        m_hamiltonian += m_t * Expression::Boson::hopping(current, right);

        // Vertical hopping (y-direction) with Peierls phase
        size_t up = m_index.to_orbital(i, (j + 1) % Ly);
        std::complex<float> phase(0.0f, 2.0f * std::numbers::pi_v<float> *
                                            m_phi * static_cast<float>(i) /
                                            static_cast<float>(Lx));
        m_hamiltonian +=
            m_t * Expression::Boson::hopping(std::exp(phase), current, up);
      }
    }
  }

  void construct_interaction_part() {
    for (size_t i = 0; i < Lx; ++i) {
      for (size_t j = 0; j < Ly; ++j) {
        size_t current = m_index.to_orbital(i, j);
        m_hamiltonian += 0.5f * m_U *
                         Term({Operator::Boson::creation(current),
                               Operator::Boson::creation(current),
                               Operator::Boson::annihilation(current),
                               Operator::Boson::annihilation(current)});
      }
    }
  }
};

template <typename Index>
[[maybe_unused]] static void demonstrate_cls(const Expression& hamiltonian,
                                             const Index& index) {
  // Construct the compact localized state
  auto s = Operator::Spin::Up;
  Expression cls_state;
  cls_state += 0.5f * Term::Boson::creation(s, index.to_orbital(0, 0, 0));
  cls_state -= 0.5f * Term::Boson::creation(s, index.to_orbital(0, 0, 1));
  cls_state += 0.5f * Term::Boson::creation(s, index.to_orbital(0, 0, 2));
  cls_state -= 0.5f * Term::Boson::creation(s, index.to_orbital(0, 0, 3));

  NormalOrderer orderer;
  Expression norm_expr = orderer.normal_order(cls_state.adjoint() * cls_state);
  float norm = norm_expr[{}].real();
  std::cout << "Normalization ⟨ψ|ψ⟩: " << norm << "\n";

  // Print the state components
  std::cout << "\nCompact Localized State components:\n";
  std::cout << "--------------------------------\n";
  for (const auto& [ops, coeff] : cls_state.terms()) {
    std::cout << index.to_string(Term(coeff, ops)) << "\n";
  }

  // Calculate energy expectation value
  Expression result =
      orderer.normal_order(cls_state.adjoint() * hamiltonian * cls_state);

  std::cout << "\nEnergy per cell expectation value ⟨ψ|H|ψ⟩: " << result[{}]
            << "\n";
}

int main() {
  const size_t Lx = 16;
  const size_t Ly = 16;
  const size_t particles = 2;

  const float t = 1.0f;
  const float phi = 6.0f;
  const float U = 4.0f;

  SquareLatticeLandauLevel<Lx, Ly> model(t, phi, U);

  // demonstrate_cls(model.hamiltonian(), model.index());

  // for (const auto& [ops, coeff] : model.hamiltonian().terms()) {
  //   std::cout << term_to_string(Term(coeff, ops), model.index()) << "\n";
  // }

  std::cout << "# Finished building the model" << std::endl;

  BosonicBasis basis(Lx * Ly, particles);
  std::cout << basis.size() << std::endl;
  auto H_matrix =
      compute_matrix_elements<arma::sp_cx_fmat>(basis, model.hamiltonian());
  QMUTILS_ASSERT(arma::approx_equal(H_matrix, H_matrix.t(), "absdiff", 1e-4f));
  std::cout << "# Finished computing matrix elements" << std::endl;
  arma::fvec eigenvalues;
  arma::eig_sym(eigenvalues, arma::cx_fmat(H_matrix));
  std::cout << "# Finished computing eigenvalues" << std::endl;

  auto dos_ctx = Dos_Utils(eigenvalues, 0.1f);

  std::cout << "# DOS computed" << std::endl;

  std::ofstream dos_file("dos.dat");

  for (size_t i = 0; i < dos_ctx.dos().size(); ++i) {
    dos_file << dos_ctx.dos()[i].first << " "
             << dos_ctx.dos()[i].second / static_cast<float>(basis.size())
             << " "
             << dos_ctx.integrated_dos()[i].second /
                    static_cast<float>(basis.size())
             << "\n";
  }

  return 0;
}
