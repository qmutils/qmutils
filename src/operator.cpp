#include "qmutils/operator.h"

#include <sstream>

namespace qmutils {

[[nodiscard]] std::string Operator::to_string() const {
  if (statistics() == Statistics::Fermion) {
    std::string spin_str = (spin() == Spin::Up) ? "↑" : "↓";
    std::string type_str = (type() == Type::Creation) ? "+" : "";
    return "f" + type_str + "(" + spin_str + "," + std::to_string(orbital()) +
           ")";
  } else {
    std::string type_str = (type() == Type::Creation) ? "+" : "";
    return "b" + type_str + "(" + std::to_string(orbital()) + ")";
  }
}

std::ostream& operator<<(std::ostream& os, const Operator& op) {
  os << op.to_string();
  return os;
}

}  // namespace qmutils
