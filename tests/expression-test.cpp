#include "qmutils/expression.h"

#include <gtest/gtest.h>

namespace qmutils {
namespace {

class ExpressionTest : public ::testing::Test {
 protected:
  Operator op1 = Operator::Fermion::creation(Operator::Spin::Up, 0);
  Operator op2 = Operator::Fermion::annihilation(Operator::Spin::Down, 1);
  Operator op3 = Operator::Fermion::creation(Operator::Spin::Up, 2);
  Expression::coefficient_type coeff1{1.0f, 0.0f};
  Expression::coefficient_type coeff2{0.0f, 2.0f};

  static Operator c(Operator::Spin spin, Operator::int_type orbital) {
    return Operator::Fermion::creation(spin, orbital);
  }

  static Operator a(Operator::Spin spin, Operator::int_type orbital) {
    return Operator::Fermion::annihilation(spin, orbital);
  }
};

TEST_F(ExpressionTest, DefaultConstructor) {
  Expression expr;
  EXPECT_EQ(expr.size(), 0);
}

TEST_F(ExpressionTest, ConstructorWithTerm) {
  Term term(coeff1, {op1, op2});
  Expression expr(term);
  EXPECT_EQ(expr.size(), 1);
  EXPECT_EQ(expr.terms().begin()->second, coeff1);
}

TEST_F(ExpressionTest, CopyConstructor) {
  Expression expr1(Term(coeff1, {op1, op2}));
  Expression expr2(expr1);
  EXPECT_EQ(expr2.size(), 1);
  EXPECT_EQ(expr2.terms(), expr1.terms());
}

TEST_F(ExpressionTest, MoveConstructor) {
  Expression expr1(Term(coeff1, {op1, op2}));
  Expression expr2(std::move(expr1));
  EXPECT_EQ(expr2.size(), 1);
  EXPECT_EQ(expr1.size(), 0);  // expr1 should be empty after move
}

TEST_F(ExpressionTest, AdditionOperator) {
  Expression expr1(Term(coeff1, {op1, op2}));
  Expression expr2(Term(coeff2, {op2, op3}));
  Expression result = expr1 + expr2;
  EXPECT_EQ(result.size(), 2);
}

TEST_F(ExpressionTest, SubtractionOperator) {
  Expression expr1(Term(coeff1, {op1, op2}));
  Expression expr2(Term(coeff1, {op1, op2}));
  Expression result = expr1 - expr2;
  EXPECT_EQ(result.size(), 0);  // Should cancel out
}

TEST_F(ExpressionTest, MultiplicationOperator) {
  Expression expr1(Term(coeff1, {op1}));
  Expression expr2(Term(coeff2, {op2}));
  Expression result = expr1 * expr2;
  EXPECT_EQ(result.size(), 1);
  EXPECT_EQ(result.terms().begin()->second, coeff1 * coeff2);
}

TEST_F(ExpressionTest, ScalarMultiplication) {
  Expression expr(Term(coeff1, {op1, op2}));
  Expression::coefficient_type scalar(2.0f, 1.0f);
  Expression result = expr * scalar;
  EXPECT_EQ(result.size(), 1);
  EXPECT_EQ(result.terms().begin()->second, coeff1 * scalar);
}

TEST_F(ExpressionTest, Normalization) {
  Expression expr;
  expr += Term(coeff1, {op1, op1});  // Should be removed (a^2 = 0)
  expr += Term(Expression::coefficient_type(0.0f, 0.0f),
               {op2, op3});  // Should be removed (zero coefficient)
  expr += Term(coeff2, {op1, op2});
  expr.normalize();
  EXPECT_EQ(expr.size(), 1);
  EXPECT_EQ(expr.terms().begin()->second, coeff2);
}

TEST_F(ExpressionTest, ToString) {
  Expression expr;
  expr += Term(coeff1, {op1, op2});
  expr += Term(coeff2, {op2, op3});
  // Not necessarily ordered
  std::string expected = "(0,2) f(↓,1)f+(↑,2) + (1,0) f+(↑,0)f(↓,1)";
  EXPECT_EQ(expr.to_string(), expected);
}

TEST_F(ExpressionTest, EqualityOperator) {
  Expression expr1(Term(coeff1, {op1, op2}));
  Expression expr2(Term(coeff1, {op1, op2}));
  Expression expr3(Term(coeff2, {op1, op2}));
  EXPECT_EQ(expr1, expr2);
  EXPECT_NE(expr1, expr3);
}

TEST_F(ExpressionTest, ComplexExpression) {
  Expression expr1(Term(coeff1, {op1, op2}));
  Expression expr2(Term(coeff2, {op2, op3}));
  Expression result = expr1 * expr2 + expr1 - expr2;
  EXPECT_EQ(result.size(), 2);
}

TEST_F(ExpressionTest, ZeroExpression) {
  Expression expr1(Term(coeff1, {op1, op2}));
  Expression expr2(Term(-coeff1, {op1, op2}));
  Expression result = expr1 + expr2;
  EXPECT_EQ(result.size(), 0);
}

TEST_F(ExpressionTest, AdjointEmptyExpression) {
  Expression expr;
  Expression adj = expr.adjoint();
  EXPECT_EQ(adj.size(), 0);
}

TEST_F(ExpressionTest, AdjointSingleTerm) {
  Expression expr(Term(Expression::coefficient_type(1.0f, 2.0f),
                       {c(Operator::Spin::Up, 0), a(Operator::Spin::Down, 1)}));
  Expression adj = expr.adjoint();

  ASSERT_EQ(adj.size(), 1);
  auto it = adj.terms().begin();
  EXPECT_EQ(it->second, Expression::coefficient_type(1.0f, -2.0f));
  ASSERT_EQ(it->first.size(), 2);
  EXPECT_EQ(it->first[0], c(Operator::Spin::Down, 1));
  EXPECT_EQ(it->first[1], a(Operator::Spin::Up, 0));
}

TEST_F(ExpressionTest, AdjointMultipleTerms) {
  Expression expr;
  expr += Term(Expression::coefficient_type(1.0f, 1.0f),
               {c(Operator::Spin::Up, 0), a(Operator::Spin::Down, 1)});
  expr += Term(Expression::coefficient_type(2.0f, -1.0f),
               {c(Operator::Spin::Down, 2)});

  Expression adj = expr.adjoint();

  ASSERT_EQ(adj.size(), 2);
  for (const auto& [ops, coeff] : adj.terms()) {
    if (ops.size() == 2) {
      EXPECT_EQ(coeff, Expression::coefficient_type(1.0f, -1.0f));
      EXPECT_EQ(ops[0], c(Operator::Spin::Down, 1));
      EXPECT_EQ(ops[1], a(Operator::Spin::Up, 0));
    } else {
      EXPECT_EQ(coeff, Expression::coefficient_type(2.0f, 1.0f));
      ASSERT_EQ(ops.size(), 1);
      EXPECT_EQ(ops[0], a(Operator::Spin::Down, 2));
    }
  }
}

TEST_F(ExpressionTest, AdjointOfAdjointIsOriginal) {
  Expression expr;
  expr += Term(Expression::coefficient_type(1.0f, 1.0f),
               {c(Operator::Spin::Up, 0), a(Operator::Spin::Down, 1)});
  expr += Term(Expression::coefficient_type(2.0f, -1.0f),
               {c(Operator::Spin::Down, 2)});

  Expression adj_adj = expr.adjoint().adjoint();

  EXPECT_EQ(expr, adj_adj);
}

TEST_F(ExpressionTest, AdjointRealCoefficients) {
  Expression expr;
  expr += Term(3.0f, {c(Operator::Spin::Up, 0), a(Operator::Spin::Down, 1)});

  Expression adj = expr.adjoint();

  ASSERT_EQ(adj.size(), 1);
  auto it = adj.terms().begin();
  EXPECT_EQ(it->second, Expression::coefficient_type(3.0f, 0.0f));
  ASSERT_EQ(it->first.size(), 2);
  EXPECT_EQ(it->first[0], c(Operator::Spin::Down, 1));
  EXPECT_EQ(it->first[1], a(Operator::Spin::Up, 0));
}

TEST_F(ExpressionTest, AdjointHermitianOperator) {
  Expression expr;
  expr += Term(Expression::coefficient_type(1.0f, 0.0f),
               {c(Operator::Spin::Up, 0), a(Operator::Spin::Up, 0)});
  expr += Term(Expression::coefficient_type(1.0f, 0.0f),
               {c(Operator::Spin::Down, 0), a(Operator::Spin::Down, 0)});

  Expression adj = expr.adjoint();

  EXPECT_EQ(expr, adj);
}

TEST_F(ExpressionTest, ExpressionFlipSpin) {
  Expression expr;
  expr += Term({1.0f, 0.0f},
               {c(Operator::Spin::Up, 0), a(Operator::Spin::Down, 1)});
  expr += Term({0.0f, 1.0f}, {c(Operator::Spin::Down, 2)});

  Expression flipped = expr.flip_spin();

  ASSERT_EQ(flipped.size(), 2);
  for (const auto& [ops, coeff] : flipped.terms()) {
    if (ops.size() == 2) {
      EXPECT_EQ(coeff, Expression::coefficient_type(1.0f, 0.0f));
      EXPECT_EQ(ops[0], c(Operator::Spin::Down, 0));
      EXPECT_EQ(ops[1], a(Operator::Spin::Up, 1));
    } else {
      EXPECT_EQ(coeff, Expression::coefficient_type(0.0f, 1.0f));
      ASSERT_EQ(ops.size(), 1);
      EXPECT_EQ(ops[0], c(Operator::Spin::Up, 2));
    }
  }
}

TEST_F(ExpressionTest, DoubleFlipIsIdentity) {
  Operator op = c(Operator::Spin::Up, 0);
  EXPECT_EQ(op.flip_spin().flip_spin(), op);

  Term term({1.0f, 0.0f},
            {c(Operator::Spin::Up, 0), a(Operator::Spin::Down, 1)});
  EXPECT_EQ(term.flip_spin().flip_spin(), term);

  Expression expr;
  expr += Term({1.0f, 0.0f},
               {c(Operator::Spin::Up, 0), a(Operator::Spin::Down, 1)});
  expr += Term({0.0f, 1.0f}, {c(Operator::Spin::Down, 2)});
  EXPECT_EQ(expr.flip_spin().flip_spin(), expr);
}

TEST_F(ExpressionTest, ExpressionIsPurely) {
  Expression fermion_expr;
  fermion_expr += Term(
      {1.0f, 0.0f}, {Operator::Fermion::creation(Operator::Spin::Up, 0),
                     Operator::Fermion::annihilation(Operator::Spin::Down, 1)});
  fermion_expr += Term({0.0f, 1.0f},
                       {Operator::Fermion::creation(Operator::Spin::Down, 2)});

  EXPECT_TRUE(fermion_expr.is_purely(Operator::Statistics::Fermion));

  Expression boson_expr;
  boson_expr += Term({1.0f, 0.0f}, {Operator::Boson::creation(0),
                                    Operator::Boson::annihilation(1)});
  boson_expr += Term({0.0f, 1.0f}, {Operator::Boson::creation(2)});

  EXPECT_TRUE(boson_expr.is_purely(Operator::Statistics::Boson));

  Expression mixed_expr;
  mixed_expr +=
      Term({1.0f, 0.0f}, {Operator::Fermion::creation(Operator::Spin::Up, 0),
                          Operator::Boson::annihilation(1)});
  mixed_expr += Term({0.0f, 1.0f},
                     {Operator::Fermion::creation(Operator::Spin::Down, 2)});

  EXPECT_FALSE(mixed_expr.is_purely(Operator::Statistics::Fermion));
  EXPECT_FALSE(mixed_expr.is_purely(Operator::Statistics::Boson));
}

}  // namespace
}  // namespace qmutils
