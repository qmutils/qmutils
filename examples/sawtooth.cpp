#include <armadillo>
#include <fstream>

#include "dos_utils.h"
#include "qmutils/assert.h"
#include "qmutils/basis.h"
#include "qmutils/expression.h"
#include "qmutils/functional.h"
#include "qmutils/index.h"
#include "qmutils/matrix_elements.h"
#include "timeintegrator.h"

using namespace qmutils;

// Reference: arXiv:1007.4640

struct Vec2 {
  float x, y;
  Vec2 operator+(const Vec2& other) { return Vec2{x + other.x, y + other.y}; }
  Vec2 operator*(float a) { return Vec2{x * a, y * a}; }
};

template <size_t L>
class SawtoothModel {
 public:
  using Index = StaticIndex<L, 2>;  // L unit cells, 2 sites per cell

  SawtoothModel(float t1, float t2, float U)
      : m_t1(t1), m_t2(t2), m_U(U), m_index() {
    construct_hamiltonian();
  }

  const Expression& hopping_hamiltonian() const {
    return m_hopping_hamiltonian;
  }
  const Expression& interaction_hamiltonian() const {
    return m_interaction_hamiltonian;
  }

  const std::vector<Vec2>& position() const { return m_position; }

  const Index& index() const { return m_index; }
  float t1() const { return m_t1; }
  float t2() const { return m_t2; }
  float U() const { return m_U; }

 private:
  float m_t1, m_t2, m_U;
  Index m_index;
  Expression m_hopping_hamiltonian;
  Expression m_interaction_hamiltonian;
  std::vector<Vec2> m_position;

  void construct_hamiltonian() {
    m_position.resize(L * 2);

    for (size_t i = 0; i < L; ++i) {
      size_t A_site = m_index.to_orbital(i, 0);
      size_t B_site = m_index.to_orbital(i, 1);
      size_t next_A_site = m_index.to_orbital((i + 1) % L, 0);

      m_hopping_hamiltonian +=
          m_t1 * Expression::Boson::hopping(A_site, B_site);
      m_hopping_hamiltonian +=
          m_t1 * Expression::Boson::hopping(B_site, next_A_site);
      m_hopping_hamiltonian +=
          m_t2 * Expression::Boson::hopping(A_site, next_A_site);

      Vec2 displacement = Vec2{1, 0} * static_cast<float>(i);
      m_position[A_site] = Vec2{0, 0} + displacement;
      m_position[B_site] = Vec2{0.5f, std::sqrt(2.0f)} + displacement;
    }

    for (size_t i = 0; i < L; ++i) {
      size_t A_site = m_index.to_orbital(i, 0);
      size_t B_site = m_index.to_orbital(i, 1);

      m_interaction_hamiltonian +=
          0.5f * m_U *
          Term({Operator::Boson::creation(A_site),
                Operator::Boson::creation(A_site),
                Operator::Boson::annihilation(A_site),
                Operator::Boson::annihilation(A_site)});

      m_interaction_hamiltonian +=
          0.5f * m_U *
          Term({Operator::Boson::creation(B_site),
                Operator::Boson::creation(B_site),
                Operator::Boson::annihilation(B_site),
                Operator::Boson::annihilation(B_site)});
    }
  }

 public:
  Expression cls(size_t i) {
    Expression result;
    size_t A_site = m_index.to_orbital(i, 0);
    size_t B_site = m_index.to_orbital(i, 1);
    size_t previous_B_site = m_index.to_orbital((i + L - 1) % L, 1);
    result += 0.5f * std::sqrt(2.0f) * Term::Boson::creation(A_site);
    result -= 0.5f * Term::Boson::creation(B_site);
    result -= 0.5f * Term::Boson::creation(previous_B_site);
    return result;
  }

  Expression density_operator(size_t i, size_t k) const {
    Expression result;
    result += Term::Boson::density(m_index.to_orbital(i, k));
    return result;
  }
};

template <size_t L>
void run_unprojected_simulation(
    const SawtoothModel<L>& model, const BosonicBasis& basis,
    const arma::sp_cx_fmat& H_matrix,
    const std::vector<arma::cx_fmat>& density_matrices,
    arma::cx_fvec state_vector, float initial_time, float final_time,
    size_t num_time_steps) {
  std::cout << "# Time evolution without projection" << std::endl;
  (void)model;
  (void)basis;

  const auto& positions = model.position();

  TimeIntegrator integrator(initial_time, final_time, num_time_steps, H_matrix);

  std::ofstream plot_stream("data.txt");
  std::ofstream animation_stream("data.txt");
  size_t frame = 0;
  do {
    for (const auto& density_matrix : density_matrices) {
      plot_stream
          << arma::cdot(state_vector, density_matrix * state_vector).real();
      plot_stream << " ";
    }
    plot_stream << "\n";

    animation_stream << "\n\n# Frame " << frame << "\n";
    animation_stream << "# x y density\n";

    for (size_t index = 0; index < positions.size(); ++index) {
      float density = arma::dot(arma::conj(state_vector),
                                density_matrices[index] * state_vector)
                          .real();
      animation_stream << positions[index].x << " " << positions[index].y << " "
                       << density << "\n";
    }
    ++frame;
  } while (integrator.step(state_vector));

  plot_stream.close();
  animation_stream.close();
}

template <size_t L>
void run_projected_simulation(
    const SawtoothModel<L>& model, const BosonicBasis& basis,
    const arma::cx_fmat& hopping_matrix,
    const arma::cx_fmat& interaction_matrix,
    const std::vector<arma::cx_fmat>& density_matrices,
    arma::cx_fvec state_vector, float initial_time, float final_time,
    size_t num_time_steps) {
  std::cout << "\n# Time evolution with projection to flat band\n";

  arma::fvec eigenvalues;
  arma::cx_fmat eigenvectors;
  arma::eig_sym(eigenvalues, eigenvectors, hopping_matrix);

  float single_flat_band_energy = -2.0f * model.t2();
  float tol = 0.1f * model.t2() * model.t2() / model.U();
  size_t count_flat_band_state = 0;

  for (size_t i = 0; i < basis.size(); i++) {
    if (eigenvalues[count_flat_band_state] <
        static_cast<float>(basis.particles()) * single_flat_band_energy + tol) {
      ++count_flat_band_state;
    }
  }

  arma::cx_fmat degenerate_submatrix =
      eigenvectors.cols(0, count_flat_band_state - 1);

  arma::cx_fmat projected_H_matrix =
      degenerate_submatrix.t() * interaction_matrix * degenerate_submatrix;

  arma::cx_fvec projected_state_vector =
      arma::normalise(degenerate_submatrix.t() * state_vector);

  std::vector<arma::cx_fmat> projected_density_matrices;
  for (const auto& density_matrix : density_matrices) {
    arma::cx_fmat projected_matrix =
        degenerate_submatrix.t() * density_matrix * degenerate_submatrix;
    projected_density_matrices.push_back(projected_matrix);
  }

  TimeIntegrator projected_integrator(initial_time, final_time, num_time_steps,
                                      projected_H_matrix);

  std::ofstream data_stream("projected_data.txt");
  do {
    for (const auto& density_matrix : projected_density_matrices) {
      data_stream << arma::cdot(projected_state_vector,
                                density_matrix * projected_state_vector)
                         .real();
      data_stream << " ";
    }
    data_stream << std::endl;
  } while (projected_integrator.step(projected_state_vector));

  data_stream.close();
}

int main() {
  const size_t L = 16;
  const size_t P = 2;
  const float t2 = 1.0f;
  const float t1 = t2 * std::sqrt(2.0f);
  const float U = 1.0f;

  float initial_time = 0.0f;
  float final_time = 100.0f / t2;
  size_t num_time_steps = 1000;

  SawtoothModel<L> model(t1, t2, U);
  BosonicBasis basis(2 * L, P);

  auto hopping_matrix = compute_matrix_elements<arma::cx_fmat>(
      basis, model.hopping_hamiltonian());

  auto interaction_matrix = compute_matrix_elements<arma::cx_fmat>(
      basis, model.interaction_hamiltonian());

  auto H_matrix = compute_matrix_elements<arma::sp_cx_fmat>(
      basis, model.hopping_hamiltonian() + model.interaction_hamiltonian());

  std::vector<arma::cx_fmat> density_matrices;
  for (size_t i = 0; i < L; i++) {
    for (size_t k = 0; k < 2; k++) {
      arma::cx_fmat density_matrix = compute_matrix_elements<arma::cx_fmat>(
          basis, model.density_operator(i, k));
      density_matrices.push_back(density_matrix);
    }
  }

  arma::cx_fvec initial_state =
      arma::normalise(compute_vector_elements_serial<arma::cx_fvec>(
          basis, model.cls(L / 2) * model.cls(L / 2)));

  run_unprojected_simulation<L>(model, basis, H_matrix, density_matrices,
                                initial_state, initial_time, final_time,
                                num_time_steps);

  run_projected_simulation<L>(model, basis, hopping_matrix, interaction_matrix,
                              density_matrices, initial_state, initial_time,
                              final_time, num_time_steps);
  return 0;
}
