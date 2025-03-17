#include <armadillo>
#include <fstream>

#include "dos_utils.h"
#include "qmutils/assert.h"
#include "qmutils/basis.h"
#include "qmutils/expression.h"
#include "qmutils/index.h"
#include "qmutils/matrix_elements.h"

using namespace qmutils;

template <size_t L>
class CreutzLadderModel {
 public:
  using Index = StaticIndex<L, 2>;  // L unit cells, 2 sites per cell

  CreutzLadderModel(float J, float theta, float U)
      : m_J(J), m_theta(theta), m_U(U), m_index() {
    construct_hamiltonian();
  }

  const Expression& hamiltonian() const { return m_hamiltonian; }
  const Index& index() const { return m_index; }

 private:
  float m_J, m_theta, m_U;
  Index m_index;
  Expression m_hamiltonian;

  void construct_hamiltonian() {
    for (size_t i = 0; i < L; ++i) {
      size_t A_site = m_index.to_orbital(i, 0);
      size_t B_site = m_index.to_orbital(i, 1);

      size_t next_A_site = m_index.to_orbital((i + 1) % L, 0);
      size_t next_B_site = m_index.to_orbital((i + 1) % L, 1);

      m_hamiltonian += m_J * Expression::Boson::hopping(A_site, next_B_site);
      m_hamiltonian += m_J * Expression::Boson::hopping(B_site, next_A_site);

      std::complex<float> phase(std::cos(m_theta), std::sin(m_theta));

      Term A_A = m_J * phase * Term::Boson::one_body(A_site, next_A_site);
      m_hamiltonian += A_A;
      m_hamiltonian += A_A.adjoint();

      Term B_B =
          m_J * std::conj(phase) * Term::Boson::one_body(B_site, next_B_site);
      m_hamiltonian += B_B;
      m_hamiltonian += B_B.adjoint();
    }

    for (size_t i = 0; i < L; ++i) {
      size_t A_site = m_index.to_orbital(i, 0);
      size_t B_site = m_index.to_orbital(i, 1);

      m_hamiltonian += 0.5f * m_U *
                       Term({Operator::Boson::creation(A_site),
                             Operator::Boson::creation(A_site),
                             Operator::Boson::annihilation(A_site),
                             Operator::Boson::annihilation(A_site)});

      m_hamiltonian += 0.5f * m_U *
                       Term({Operator::Boson::creation(B_site),
                             Operator::Boson::creation(B_site),
                             Operator::Boson::annihilation(B_site),
                             Operator::Boson::annihilation(B_site)});
    }
  }
};

int main() {
  const size_t L = 24;
  const size_t P = 2;
  const float J = 1.0f;
  const float U = 6.0f * J;

  CreutzLadderModel<L> model(J, 0.5f * std::numbers::pi_v<float>, U);
  BosonicBasis basis(2 * L, P);  // 2 sites per unit cell, 2 particles

  std::cout << "# Basis size: " << basis.size() << std::endl;

  QMUTILS_ASSERT(model.hamiltonian().is_purely(Operator::Statistics::Boson));

  auto H_matrix =
      compute_matrix_elements<arma::cx_fmat>(basis, model.hamiltonian());

  std::cout << "# Matrix elements computed" << std::endl;

  QMUTILS_ASSERT(arma::approx_equal(H_matrix, H_matrix.t(), "absdiff", 1e-4f));

  arma::fvec eigenvalues;
  arma::cx_fmat eigenvectors;
  arma::eig_sym(eigenvalues, eigenvectors, H_matrix);

  std::cout << "# Eigenvalues computed" << std::endl;

  auto dos_ctx = Dos_Utils(eigenvalues, 0.1f * J);

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
