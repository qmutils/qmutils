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

// Kagome lattice model with DynamicIndex

class KagomeModel {
 public:
  using Index = DynamicIndex;

  KagomeModel(size_t lx, size_t ly, float t1, float U)
      : m_t1(t1), m_U(U), m_index({lx, ly, 3}) {
    construct_hamiltonian();
  }

  const Expression& hamiltonian() const { return m_hamiltonian; }
  const Index& index() const { return m_index; }

 private:
  float m_t1, m_U;
  Index m_index;
  Expression m_hamiltonian;

  Term density_density_interaction(size_t site) const {
    auto s = Operator::Spin::Up;
    return 0.5f * m_U *
           Term({Operator::Boson::creation(s, site),
                 Operator::Boson::creation(s, site),
                 Operator::Boson::annihilation(s, site),
                 Operator::Boson::annihilation(s, site)});
  }

  void construct_hamiltonian() {
    auto s = Operator::Spin::Up;
    const size_t lx = m_index.dimension(0);
    const size_t ly = m_index.dimension(1);

    for (size_t i = 0; i < lx; ++i) {
      for (size_t j = 0; j < ly; ++j) {
        size_t A_site = m_index.to_orbital({i, j, 0});
        size_t B_site = m_index.to_orbital({i, j, 1});
        size_t C_site = m_index.to_orbital({i, j, 2});

        size_t next_A_y_site = m_index.to_orbital({i, (j + 1) % ly, 0});
        size_t next_A_x_site = m_index.to_orbital({(i + 1) % lx, j, 0});
        size_t next_C_x_y_site =
            m_index.to_orbital({(i + 1) % lx, (j + ly - 1) % ly, 2});

        m_hamiltonian += m_t1 * Expression::Boson::hopping(A_site, B_site, s);
        m_hamiltonian += m_t1 * Expression::Boson::hopping(B_site, C_site, s);
        m_hamiltonian += m_t1 * Expression::Boson::hopping(C_site, A_site, s);

        m_hamiltonian +=
            m_t1 * Expression::Boson::hopping(B_site, next_A_x_site, s);
        m_hamiltonian +=
            m_t1 * Expression::Boson::hopping(C_site, next_A_y_site, s);
        m_hamiltonian +=
            m_t1 * Expression::Boson::hopping(B_site, next_C_x_y_site, s);

        m_hamiltonian += density_density_interaction(A_site);
        m_hamiltonian += density_density_interaction(B_site);
        m_hamiltonian += density_density_interaction(C_site);
      }
    }
  }
};

class Cluster {
 private:
  float m_threshold;
  std::vector<std::vector<float>> m_clusters;

 public:
  Cluster(const std::vector<float>& points, float distance_threshold)
      : m_threshold(distance_threshold) {
    if (points.empty()) {
      return;
    }

    std::vector<float> sorted_points = points;
    std::sort(sorted_points.begin(), sorted_points.end());

    std::vector<float> current_cluster = {sorted_points[0]};

    for (size_t i = 1; i < sorted_points.size(); i++) {
      if (sorted_points[i] - sorted_points[i - 1] <= m_threshold) {
        current_cluster.push_back(sorted_points[i]);
      } else {
        m_clusters.push_back(current_cluster);
        current_cluster = {sorted_points[i]};
      }
    }

    m_clusters.push_back(current_cluster);
  }

  size_t size() const { return m_clusters.size(); }

  std::vector<size_t> cluster_sizes() const {
    std::vector<size_t> sizes;
    sizes.reserve(m_clusters.size());
    for (const auto& cluster : m_clusters) {
      sizes.push_back(cluster.size());
    }
    return sizes;
  }

  const std::vector<std::vector<float>>& clusters() const { return m_clusters; }
};

[[maybe_unused]] static float compute_bandwidth(
    const std::vector<float>& cluster) {
  if (cluster.empty()) {
    return 0;
  }

  float max_val = std::numeric_limits<float>::lowest();
  float min_val = std::numeric_limits<float>::max();

  for (float x : cluster) {
    max_val = std::max(max_val, x);
    min_val = std::min(min_val, x);
  }

  return max_val - min_val;
}

static float ener(float t, size_t i, size_t j, size_t lx, size_t ly) {
  float kx = 2.0f * std::numbers::pi_v<float> * static_cast<float>(i) /
             static_cast<float>(lx);
  float ky = 2.0f * std::numbers::pi_v<float> * static_cast<float>(j) /
             static_cast<float>(ly);
  return t - t * std::sqrt(3.0f + 2.0f * std::cos(kx) + 2.0f * std::cos(ky) +
                           2.0f * std::cos(kx - ky));
}

int main() {
  const size_t P = 2;
  const float t1 = 1.0f;
  const float U = 4.0f;

  std::ofstream interval_file("scaling/intervals.txt");

  for (size_t lx = 3; lx < 7; lx++) {
    size_t ly = lx;
    KagomeModel model(lx, ly, t1, U);
    BosonicBasis basis(lx * ly * 3, P);

    auto H_matrix =
        compute_matrix_elements<arma::cx_fmat>(basis, model.hamiltonian());

    arma::fvec eigenvalues;
    arma::eig_sym(eigenvalues, H_matrix);

    auto vals = arma::conv_to<std::vector<float>>::from(eigenvalues);

    std::string vals_filename =
        "scaling/eigenvalues_lx_" + std::to_string(lx) + ".txt";
    std::ofstream vals_file(vals_filename);

    float max_val = -2.0f + ener(t1, 1, 0, lx, ly);
    float min_val = -4.0f * t1;
    float offset = (max_val - min_val) / 10.0f;

    float max_ = std::numeric_limits<float>::lowest();
    float min_ = std::numeric_limits<float>::max();

    for (float x : eigenvalues) {
      if (x > min_val + offset && x < max_val - offset) {
        max_ = std::max(max_, x);
        min_ = std::min(min_, x);
      }
      if (x > min_val && x < max_val) {
        vals_file << x << "\n";
      }
      if (x > max_val - offset) break;
    }

    vals_file.close();
    interval_file << lx << " " << (max_ + min_) / 2.f << " " << min_ << " "
                  << max_ << "\n";
  }
  interval_file.close();

  return 0;
}
