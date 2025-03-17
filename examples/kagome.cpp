#include <armadillo>
#include <fstream>
#include <iostream>

#include "dos_utils.h"
#include "qmutils/expression.h"
#include "qmutils/index.h"
#include "qmutils/matrix_elements.h"
#include "qmutils/operator.h"
#include "timeintegrator.h"

using namespace qmutils;

// Kagome lattice model

struct Vec2 {
  float x, y;
  Vec2 operator+(const Vec2& other) { return Vec2{x + other.x, y + other.y}; }
  Vec2 operator*(float a) { return Vec2{x * a, y * a}; }
};

struct Bond {
  Vec2 initial_position;
  Vec2 final_position;
  Expression op;
};

template <size_t Lx, size_t Ly>
class KagomeModel {
 public:
  using Index = StaticIndex<Lx, Ly, 3>;

  KagomeModel(float t1, float U) : m_t1(t1), m_U(U), m_index() {
    construct_hamiltonian();
  }

  const Expression& hamiltonian() const { return m_hamiltonian; }
  const std::vector<Vec2>& position() const { return m_position; }
  const std::vector<Bond> bonds() const { return m_bonds; }
  const Index& index() const { return m_index; }

 private:
  float m_t1, m_U;
  Index m_index;
  Expression m_hamiltonian;
  std::vector<Vec2> m_position;
  std::vector<Bond> m_bonds;
  NormalOrderer orderer;

  Term density_density_interaction(size_t site) const {
    return 0.5f * m_U *
           Term({Operator::Boson::creation(site),
                 Operator::Boson::creation(site),
                 Operator::Boson::annihilation(site),
                 Operator::Boson::annihilation(site)});
  }

  void construct_hamiltonian() {
    m_position.resize(Lx * Ly * 3);

    for (size_t i = 0; i < Lx; ++i) {
      for (size_t j = 0; j < Ly; ++j) {
        size_t A_site = m_index.to_orbital(i, j, 0);
        size_t B_site = m_index.to_orbital(i, j, 1);
        size_t C_site = m_index.to_orbital(i, j, 2);

        size_t next_A_y_site = m_index.to_orbital(i, (j + 1) % Ly, 0);
        size_t next_A_x_site = m_index.to_orbital((i + 1) % Lx, j, 0);
        size_t next_C_x_y_site =
            m_index.to_orbital((i + 1) % Lx, (j + Ly - 1) % Ly, 2);

        m_hamiltonian += m_t1 * Expression::Boson::hopping(A_site, B_site);
        m_hamiltonian += m_t1 * Expression::Boson::hopping(B_site, C_site);
        m_hamiltonian += m_t1 * Expression::Boson::hopping(C_site, A_site);

        m_hamiltonian +=
            m_t1 * Expression::Boson::hopping(B_site, next_A_x_site);
        m_hamiltonian +=
            m_t1 * Expression::Boson::hopping(C_site, next_A_y_site);
        m_hamiltonian +=
            m_t1 * Expression::Boson::hopping(B_site, next_C_x_y_site);

        m_hamiltonian += density_density_interaction(A_site);
        m_hamiltonian += density_density_interaction(B_site);
        m_hamiltonian += density_density_interaction(C_site);

        // Build position
        Vec2 displacement =
            Vec2{1, 0} * static_cast<float>(i) +
            Vec2{0.5f, 0.5f * std::sqrt(3.0f)} * static_cast<float>(j);
        m_position[A_site] = Vec2{0, 0} + displacement;
        m_position[B_site] = Vec2{0.5f, 0} + displacement;
        m_position[C_site] =
            Vec2{0.25f, 0.25f * std::sqrt(3.0f)} + displacement;
      }
    }

    // Build bonds
    for (size_t i = 0; i < Lx; ++i) {
      for (size_t j = 0; j < Ly; ++j) {
        size_t A_site = m_index.to_orbital(i, j, 0);
        size_t B_site = m_index.to_orbital(i, j, 1);
        size_t C_site = m_index.to_orbital(i, j, 2);

        size_t next_A_y_site = m_index.to_orbital(i, (j + 1) % Ly, 0);
        size_t next_A_x_site = m_index.to_orbital((i + 1) % Lx, j, 0);
        size_t next_C_x_y_site =
            m_index.to_orbital((i + 1) % Lx, (j + Ly - 1) % Ly, 2);

        m_bonds.push_back({m_position[A_site], m_position[B_site],
                           m_t1 * edge(A_site, B_site)});
        m_bonds.push_back({m_position[B_site], m_position[C_site],
                           m_t1 * edge(B_site, C_site)});
        m_bonds.push_back({m_position[C_site], m_position[A_site],
                           m_t1 * edge(C_site, A_site)});

        m_bonds.push_back({m_position[B_site], m_position[next_A_x_site],
                           m_t1 * edge(B_site, next_A_x_site)});
        m_bonds.push_back({m_position[C_site], m_position[next_A_y_site],
                           m_t1 * edge(C_site, next_A_y_site)});
        m_bonds.push_back({m_position[B_site], m_position[next_C_x_y_site],
                           m_t1 * edge(B_site, next_C_x_y_site)});
      }
    }
  }

  Expression edge(size_t from_orbital, size_t to_orbital) {
    QMUTILS_ASSERT(from_orbital != to_orbital);
    Expression result;
    result += Term::Boson::one_body(from_orbital, to_orbital);
    result -= Term::Boson::one_body(to_orbital, from_orbital);
    return result;
  }

 public:
  Expression v(size_t i, size_t j) const {
    // size_t A_site = m_index.to_orbital(i, j, 0);
    size_t B_site = m_index.to_orbital(i, j, 1);
    size_t C_site = m_index.to_orbital(i, j, 2);

    size_t next_A_site_x = m_index.to_orbital((i + 1) % Lx, j, 0);
    // size_t next_B_site_x = m_index.to_orbital((i + 1) % Lx, j, 1);
    size_t next_C_site_x = m_index.to_orbital((i + 1) % Lx, j, 2);

    size_t next_A_site_y = m_index.to_orbital(i, (j + 1) % Ly, 0);
    size_t next_B_site_y = m_index.to_orbital(i, (j + 1) % Ly, 1);
    // size_t next_C_site_y = m_index.to_orbital(i, (j + 1) % Ly, 2);

    Expression result;

    result += Term::Boson::creation(C_site);
    result -= Term::Boson::creation(B_site);
    result += Term::Boson::creation(next_A_site_x);
    result -= Term::Boson::creation(next_C_site_x);
    result += Term::Boson::creation(next_B_site_y);
    result -= Term::Boson::creation(next_A_site_y);

    return (1.0f / std::sqrt(6.0f)) * result;
  }

  Expression f(size_t j) const {
    Expression result;

    for (size_t i = 0; i < Lx; i++) {
      size_t A_site = m_index.to_orbital(i, j, 0);
      size_t B_site = m_index.to_orbital(i, j, 1);
      // size_t C_site = m_index.to_orbital(i, j, 2);

      result += Term::Boson::creation(A_site);
      result -= Term::Boson::creation(B_site);
    }

    return (1.0f / std::sqrt(Lx)) * result;
  }

  Expression g(size_t i) const {
    Expression result;

    for (size_t j = 0; j < Ly; j++) {
      size_t A_site = m_index.to_orbital(i, j, 0);
      // size_t B_site = m_index.to_orbital(i, j, 1);
      size_t C_site = m_index.to_orbital(i, j, 2);

      result += Term::Boson::creation(A_site);
      result -= Term::Boson::creation(C_site);
    }

    return (1.0f / std::sqrt(Ly)) * result;
  }

  Expression density_operator(size_t i, size_t j, size_t k) {
    Expression result;
    size_t site = m_index.to_orbital(i, j, k);
    result += Term::Boson::density(site);
    return result;
  }
};

int main() {
  const size_t Lx = 6;
  const size_t Ly = 6;
  const size_t P = 2;
  const float t1 = 1.0f;
  const float U = 0.5f;
  KagomeModel<Lx, Ly> model(t1, U);
  BosonicBasis basis(Lx * Ly * 3, P);

  const auto& positions = model.position();

  std::vector<arma::sp_cx_fmat> density_operator_matrices(Lx * Ly * 3);
  for (size_t y = 0; y < Ly; y++) {
    for (size_t x = 0; x < Lx; x++) {
      for (size_t site = 0; site < 3; site++) {
        size_t index = model.index().to_orbital(x, y, site);
        density_operator_matrices[index] =
            compute_matrix_elements<arma::sp_cx_fmat>(
                basis, model.density_operator(x, y, site));
      }
    }
  }

  const auto& bonds = model.bonds();
  std::vector<arma::sp_cx_fmat> bonds_matrices(bonds.size());
  for (size_t i = 0; i < bonds.size(); i++) {
    bonds_matrices[i] =
        compute_matrix_elements<arma::sp_cx_fmat>(basis, bonds[i].op);
  }

  auto H_matrix =
      compute_matrix_elements<arma::sp_cx_fmat>(basis, model.hamiltonian());

  size_t Xoffset = Lx / 2 - 1;
  size_t Yoffset = Ly / 2 - 1;

  auto state =
      model.v(Xoffset + 0, Yoffset + 0) * model.v(Xoffset + 0, Yoffset + 0);

  auto state_vector = compute_vector_elements<arma::cx_fvec>(basis, state);
  state_vector /= arma::norm(state_vector);

  float initial_time = 0.0f;
  float final_time = 100.0f / t1;
  size_t num_time_steps = 1000;
  TimeIntegrator integrator(initial_time, final_time, num_time_steps, H_matrix);

  auto initial_state_vector = state_vector;
  std::ofstream return_file("/tmp/kagome_return_probability.dat");
  std::ofstream evolution_file("/tmp/kagome_evolution.dat");
  std::ofstream current_file("/tmp/kagome_current.dat");

  size_t frame = 0;
  do {
    if (frame % 100 == 0) {
      std::cerr << "Frame: " << frame << "\n";
    }

    return_file << frame << " "
                << std::norm(arma::cdot(initial_state_vector, state_vector))
                << "\n";

    {
      evolution_file << "\n\n# Frame " << frame << "\n";
      evolution_file << "# x y density\n";

      for (size_t index = 0; index < positions.size(); ++index) {
        auto density =
            arma::dot(arma::conj(state_vector),
                      density_operator_matrices[index] * state_vector)
                .real();
        evolution_file << positions[index].x << " " << positions[index].y << " "
                       << density << "\n";
      }
    }

    {
      current_file << "\n\n# Frame " << frame << "\n";
      current_file << "# x y density\n";
      for (size_t i = 0; i < bonds.size(); i++) {
        if (std::abs(bonds[i].final_position.x - bonds[i].initial_position.x) >
                1.0f ||
            std::abs(bonds[i].final_position.y - bonds[i].initial_position.y) >
                1.0f) {
          continue;
        }
        auto current =
            arma::cdot(state_vector, bonds_matrices[i] * state_vector).imag();
        current_file
            << bonds[i].initial_position.x << " " << bonds[i].initial_position.y
            << " " << (bonds[i].final_position.x - bonds[i].initial_position.x)
            << " " << (bonds[i].final_position.y - bonds[i].initial_position.y)
            << " " << current << "\n";
      }
    }

    frame++;
  } while (integrator.step(state_vector));

  return_file.close();
  evolution_file.close();
  current_file.close();
  return 0;
}
