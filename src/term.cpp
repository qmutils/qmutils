#include "qmutils/term.h"

#include <numeric>
#include <sstream>

namespace qmutils {

std::ostream& operator<<(std::ostream& os, const Term& term) {
  os << term.to_string();
  return os;
}

Term Term::adjoint() const {
  container_type adjoint_operators;
  adjoint_operators.reserve(m_operators.size());

  for (auto it = m_operators.rbegin(); it != m_operators.rend(); ++it) {
    adjoint_operators.push_back(it->adjoint());
  }

  return Term(std::conj(m_coefficient), std::move(adjoint_operators));
}

Term Term::flip_spin() const {
  container_type flipped_operators;
  flipped_operators.reserve(m_operators.size());
  for (const auto& op : m_operators) {
    flipped_operators.push_back(op.flip_spin());
  }
  return Term(m_coefficient, std::move(flipped_operators));
}

std::string Term::to_string() const {
  std::ostringstream oss;
  oss << m_coefficient << " ";
  for (size_t i = 0; i < m_operators.size(); ++i) {
    oss << m_operators[i].to_string();
  }
  return oss.str();
}

}  // namespace qmutils
