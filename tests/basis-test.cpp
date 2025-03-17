#include "qmutils/basis.h"

#include <gtest/gtest.h>

namespace qmutils {
namespace {

class BasisTest : public ::testing::Test {
 protected:
  static Basis::operators_type create_operators(
      std::initializer_list<std::pair<Operator::Spin, Operator::int_type>>
          ops) {
    Basis::operators_type result;
    for (const auto& [spin, orbital] : ops) {
      result.push_back(Operator::Fermion::creation(spin, orbital));
    }
    return result;
  }
};

TEST_F(BasisTest, ConstructorAndBasicProperties) {
  Basis basis(3, 2);
  EXPECT_EQ(basis.orbitals(), 3);
  EXPECT_EQ(basis.particles(), 2);
  EXPECT_EQ(basis.size(), qmutils_choose(6, 2));
}

TEST_F(BasisTest, EmptyBasis) {
  Basis basis(3, 0);
  EXPECT_EQ(basis.size(), 1);
  EXPECT_TRUE(
      basis.contains(Basis::operators_type{}));  // Only the vacuum state
}

TEST_F(BasisTest, FullBasis) {
  Basis basis(2, 4);
  EXPECT_EQ(basis.size(), 1);  // Only one fully occupied state
}

TEST_F(BasisTest, ContainsAndIndex) {
  Basis basis(2, 2);

  auto state1 =
      create_operators({{Operator::Spin::Up, 0}, {Operator::Spin::Down, 0}});
  auto state2 =
      create_operators({{Operator::Spin::Up, 0}, {Operator::Spin::Up, 1}});
  auto state3 =
      create_operators({{Operator::Spin::Up, 0}, {Operator::Spin::Down, 1}});

  EXPECT_TRUE(basis.contains(state1));
  EXPECT_TRUE(basis.contains(state2));
  EXPECT_TRUE(basis.contains(state3));
}

TEST_F(BasisTest, DoesNotContain) {
  Basis basis(2, 2);

  auto invalid_state1 =
      create_operators({{Operator::Spin::Up, 0},
                        {Operator::Spin::Up, 0}});  // Same orbital, same spin
  auto invalid_state2 =
      create_operators({{Operator::Spin::Up, 0},
                        {Operator::Spin::Up, 1},
                        {Operator::Spin::Down, 1}});  // Too many particles

  EXPECT_FALSE(basis.contains(invalid_state1));
  EXPECT_FALSE(basis.contains(invalid_state2));
}

TEST_F(BasisTest, OrderDependence) {
  Basis basis(2, 2);

  auto state1 =
      create_operators({{Operator::Spin::Up, 0}, {Operator::Spin::Down, 1}});
  auto state2 =
      create_operators({{Operator::Spin::Down, 1}, {Operator::Spin::Up, 0}});

  EXPECT_TRUE(basis.contains(state1));
  EXPECT_FALSE(basis.contains(state2));
}

TEST_F(BasisTest, CorrectSizeForVariousConfigurations) {
  const size_t max_orbital_count = 8;
  for (size_t orbitals = 0; orbitals < max_orbital_count; ++orbitals) {
    for (size_t particles = 0; particles < 2 * orbitals; ++particles) {
      Basis basis(orbitals, particles);
      EXPECT_EQ(basis.size(), qmutils_choose(2 * orbitals, particles));
    }
  }
}

TEST_F(BasisTest, EqualityOperator) {
  Basis basis1(2, 2);
  Basis basis2(2, 2);
  Basis basis3(3, 2);

  EXPECT_EQ(basis1, basis2);
  EXPECT_NE(basis1, basis3);
}

class FermionicBasisSzTest : public ::testing::Test {
 protected:
  static Basis::operators_type create_operators(
      std::initializer_list<std::pair<Operator::Spin, Operator::int_type>>
          ops) {
    Basis::operators_type result;
    for (const auto& [spin, orbital] : ops) {
      result.push_back(Operator::Fermion::creation(spin, orbital));
    }
    return result;
  }

  static int calculate_sz(const Term::container_type& ops) {
    int sz = 0;
    for (const auto& op : ops) {
      if (op.type() == Operator::Type::Creation) {
        sz += (op.spin() == Operator::Spin::Up) ? 1 : -1;
      }
    }
    return sz;
  }
};

TEST_F(FermionicBasisSzTest, ConstructorAndBasicProperties) {
  FermionicBasis basis(3, 2, 0);  // 3 orbitals, 2 particles, Sz = 0
  EXPECT_EQ(basis.orbitals(), 3);
  EXPECT_EQ(basis.particles(), 2);
  EXPECT_EQ(basis.size(), 9);
}

TEST_F(FermionicBasisSzTest, UnconstrainedBasis) {
  FermionicBasis basis(2, 2, std::nullopt);
  EXPECT_EQ(basis.size(), 6);
}

TEST_F(FermionicBasisSzTest, MaximumSz) {
  FermionicBasis basis(2, 2, 2);
  EXPECT_EQ(basis.size(), 1);
  auto state =
      create_operators({{Operator::Spin::Up, 0}, {Operator::Spin::Up, 1}});
  EXPECT_TRUE(basis.contains(state));
}

TEST_F(FermionicBasisSzTest, MinimumSz) {
  FermionicBasis basis(2, 2, -2);
  EXPECT_EQ(basis.size(), 1);
  auto state =
      create_operators({{Operator::Spin::Down, 0}, {Operator::Spin::Down, 1}});
  EXPECT_TRUE(basis.contains(state));
}

TEST_F(FermionicBasisSzTest, VerifySzValueForAllStates) {
  FermionicBasis basis(3, 2, 0);
  for (const auto& term : basis) {
    EXPECT_EQ(calculate_sz(term.operators()), 0);
  }
}

TEST_F(FermionicBasisSzTest, NonZeroSz) {
  FermionicBasis basis(3, 2, 2);
  for (const auto& term : basis) {
    EXPECT_EQ(calculate_sz(term.operators()), 2);
  }
}

TEST_F(FermionicBasisSzTest, VerifyStateOrdering) {
  FermionicBasis basis(2, 2, 0);
  std::vector<Term> states(basis.begin(), basis.end());

  // Verify states are ordered
  for (size_t i = 1; i < states.size(); ++i) {
    EXPECT_TRUE(states[i - 1].operators() < states[i].operators());
  }
}

TEST_F(FermionicBasisSzTest, SpecificStateContent) {
  FermionicBasis basis(2, 2, 0);

  // These states should be present
  EXPECT_TRUE(basis.contains(
      create_operators({{Operator::Spin::Up, 0}, {Operator::Spin::Down, 0}})));

  EXPECT_TRUE(basis.contains(
      create_operators({{Operator::Spin::Up, 0}, {Operator::Spin::Down, 1}})));

  // These states should not be present
  EXPECT_FALSE(basis.contains(
      create_operators({{Operator::Spin::Up, 0}, {Operator::Spin::Up, 1}})));

  EXPECT_FALSE(basis.contains(create_operators(
      {{Operator::Spin::Down, 0}, {Operator::Spin::Down, 1}})));
}

TEST_F(FermionicBasisSzTest, CompareConstrainedVsUnconstrained) {
  FermionicBasis unconstrained(2, 2, std::nullopt);
  FermionicBasis sz_zero(2, 2, 0);
  FermionicBasis sz_plus_two(2, 2, 2);
  FermionicBasis sz_minus_two(2, 2, -2);

  // Total states should equal sum of all Sz sectors
  EXPECT_EQ(unconstrained.size(),
            sz_zero.size() + sz_plus_two.size() + sz_minus_two.size());
}

TEST_F(FermionicBasisSzTest, EmptyBasisEdgeCases) {
  // Empty basis with Sz constraint
  FermionicBasis basis1(2, 0, 0);
  EXPECT_EQ(basis1.size(), 1);  // Only vacuum state
}

TEST_F(FermionicBasisSzTest, FullBasisEdgeCases) {
  // Fully occupied with Sz = 0
  FermionicBasis basis1(2, 4, 0);
  EXPECT_EQ(basis1.size(), 1);
}

}  // namespace
}  // namespace qmutils
