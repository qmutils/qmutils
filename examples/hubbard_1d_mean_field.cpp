#include <armadillo>
#include <cmath>
#include <functional>
#include <iostream>
#include <limits>
#include <vector>

#include "qmutils/expression.h"
#include "qmutils/matrix_elements.h"
#include "qmutils/sparse_matrix.h"

class Hubbard1DModel {
 public:
  Hubbard1DModel(size_t sites, float t, float U)
      : m_sites(sites), m_t(t), m_U(U) {}

  qmutils::Expression construct_hamiltonian(
      const std::vector<float>& up_occupations,
      const std::vector<float>& down_occupations) const {
    qmutils::Expression H;

    // Hopping terms
    for (size_t i = 0; i < m_sites; ++i) {
      size_t j = (i + 1) % m_sites;  // Periodic boundary conditions
      H += -m_t * qmutils::Expression::Fermion::hopping(
                      i, j, qmutils::Operator::Spin::Up);
      H += -m_t * qmutils::Expression::Fermion::hopping(
                      i, j, qmutils::Operator::Spin::Down);
    }

    // Mean-field interaction terms
    for (size_t i = 0; i < m_sites; ++i) {
      H += m_U * down_occupations[i] *
           qmutils::Term::Fermion::density(qmutils::Operator::Spin::Up, i);
      H += m_U * up_occupations[i] *
           qmutils::Term::Fermion::density(qmutils::Operator::Spin::Down, i);
      H += -m_U * up_occupations[i] * down_occupations[i];
    }

    return H;
  }

  size_t sites() const { return m_sites; }
  float t() const { return m_t; }
  float U() const { return m_U; }

 private:
  size_t m_sites;
  float m_t;
  float m_U;
};

class MeanFieldSolver {
 public:
  static std::pair<std::vector<float>, std::vector<float>> solve(
      const Hubbard1DModel& model,
      const std::vector<float>& initial_up_occupations,
      const std::vector<float>& initial_down_occupations, float filling,
      float temperature, float tolerance = 3e-4f, int max_iterations = 1000,
      float mixing = 0.5) {
    std::vector<float> up_occupations = initial_up_occupations;
    std::vector<float> down_occupations = initial_down_occupations;
    int iteration = 0;
    float difference;

    // Construct single-particle basis
    auto basis = qmutils::Basis(model.sites(), 1);

    for (const auto& elm : basis) {
      std::cout << qmutils::Term(elm).to_string() << std::endl;
    }

    do {
      std::cout << "Iteration: " << iteration << std::endl;
      std::cout << "Magnetization: "
                << compute_total_magnetization(up_occupations, down_occupations)
                << std::endl;

      // Construct mean-field Hamiltonian
      auto H = model.construct_hamiltonian(up_occupations, down_occupations);

      // Compute eigenvectors and eigenvalues
      auto mat = qmutils::compute_matrix_elements<arma::cx_fmat>(basis, H);
      arma::fvec eigenvalues;
      arma::cx_fmat eigenvectors;
      arma::eig_sym(eigenvalues, eigenvectors, mat);

      // Calculate new occupations
      std::vector<float> new_up_occupations(model.sites());
      std::vector<float> new_down_occupations(model.sites());
      calculate_occupations(eigenvalues, eigenvectors, model.sites(), filling,
                            temperature, new_up_occupations,
                            new_down_occupations);

      // Check for convergence
      difference = 0.0;
      for (size_t i = 0; i < new_up_occupations.size(); ++i) {
        difference += std::abs(new_up_occupations[i] - up_occupations[i]);
        difference += std::abs(new_down_occupations[i] - down_occupations[i]);
      }
      std::cout << "Difference: " << difference << std::endl;

      // Update occupations
      for (size_t i = 0; i < model.sites(); ++i) {
        up_occupations[i] =
            mixing * new_up_occupations[i] + (1 - mixing) * up_occupations[i];
        down_occupations[i] = mixing * new_down_occupations[i] +
                              (1 - mixing) * down_occupations[i];
      }

      iteration++;
    } while (difference > tolerance && iteration < max_iterations);

    if (iteration == max_iterations) {
      std::cout << "Warning: Maximum iterations reached without convergence."
                << std::endl;
    } else {
      std::cout << "Converged after " << iteration << " iterations."
                << std::endl;
    }

    return {up_occupations, down_occupations};
  }

 private:
  static void calculate_occupations(const arma::fvec& eigenvalues,
                                    const arma::cx_fmat& eigenvectors,
                                    size_t sites, float filling,
                                    float temperature,
                                    std::vector<float>& new_up_occupations,
                                    std::vector<float>& new_down_occupations) {
    size_t total_states = eigenvalues.n_elem;
    float chemical_potential =
        find_chemical_potential(eigenvalues, filling, temperature);

    std::fill(new_up_occupations.begin(), new_up_occupations.end(), 0.0f);
    std::fill(new_down_occupations.begin(), new_down_occupations.end(), 0.0f);

    for (size_t j = 0; j < total_states; ++j) {
      float occupation =
          fermi_distribution(eigenvalues(j), chemical_potential, temperature);
      for (size_t i = 0; i < sites; ++i) {
        size_t up_index = i;
        size_t down_index = i + sites;
        new_up_occupations[i] +=
            occupation * std::norm(eigenvectors(up_index, j));
        new_down_occupations[i] +=
            occupation * std::norm(eigenvectors(down_index, j));
      }
    }
  }

  static float find_chemical_potential(const arma::fvec& eigenvalues,
                                       float target_filling, float temperature,
                                       float tolerance = 3e-4f) {
    float low = eigenvalues.min();
    float high = eigenvalues.max();

    while (high - low > tolerance) {
      float mid = (low + high) / 2;
      float current_filling = 0.0f;
      for (float e : eigenvalues) {
        current_filling += fermi_distribution(e, mid, temperature);
      }
      current_filling /= static_cast<float>(eigenvalues.n_elem);

      if (current_filling < target_filling) {
        low = mid;
      } else {
        high = mid;
      }
    }

    return (low + high) / 2;
  }

  static float fermi_distribution(float energy, float chemical_potential,
                                  float temperature) {
    return 1.0f /
           (1.0f + std::exp((energy - chemical_potential) / temperature));
  }

  static float compute_total_magnetization(
      const std::vector<float>& up_occupations,
      const std::vector<float>& down_occupations) {
    float total_magnetization = 0.0;
    for (size_t i = 0; i < up_occupations.size(); ++i) {
      total_magnetization += (up_occupations[i] - down_occupations[i]) /
                             (up_occupations[i] + down_occupations[i]);
    }
    return total_magnetization;
  }
};

int main() {
  size_t sites = 16;
  float t = 1.0f;
  float U = 4.0f;
  float filling = 0.25f;
  float temperature = 0.03f;

  Hubbard1DModel model(sites, t, U);

  // Initial guess for occupations
  std::vector<float> initial_up(sites, filling / 2);
  std::vector<float> initial_down(sites, filling / 2);

  auto [final_up, final_down] = MeanFieldSolver::solve(
      model, initial_up, initial_down, filling, temperature);

  std::cout << "Up occs" << std::endl;
  for (float up_occ : final_up) {
    std::cout << up_occ << std::endl;
  }

  std::cout << "Low occs" << std::endl;
  for (float down_occ : final_down) {
    std::cout << down_occ << std::endl;
  }

  return 0;
}
