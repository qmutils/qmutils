#include "qmutils/matrix_elements.h"

#include <gtest/gtest.h>

#include <complex>
#include <vector>

#include "qmutils/sparse_matrix.h"

namespace qmutils {
namespace {

class FermionicOperatorMatrixElementsTest : public ::testing::Test {
 protected:
  static Operator c(Operator::Spin spin, Operator::int_type orbital) {
    return Operator::Fermion::creation(spin, orbital);
  }

  static Operator a(Operator::Spin spin, Operator::int_type orbital) {
    return Operator::Fermion::annihilation(spin, orbital);
  }
};

TEST_F(FermionicOperatorMatrixElementsTest,
       SparseMatrixComputationOffUpperDiagonal) {
  FermionicBasis basis(2, 1);

  Expression H = Expression(
      Term(1.0f, {c(Operator::Spin::Up, 0), a(Operator::Spin::Up, 1)}));

  auto matrix = compute_matrix_elements_serial<
      SparseMatrix<Expression::coefficient_type>>(basis, H);

  EXPECT_EQ(matrix.rows(), 4);
  EXPECT_EQ(matrix.cols(), 4);
  EXPECT_EQ(matrix(0, 0), Expression::coefficient_type(0, 0));
  EXPECT_EQ(matrix(0, 1), Expression::coefficient_type(1.0f, 0));
  EXPECT_EQ(matrix(1, 0), Expression::coefficient_type(0, 0));
  EXPECT_EQ(matrix(1, 1), Expression::coefficient_type(0, 0));
}

TEST_F(FermionicOperatorMatrixElementsTest,
       SparseMatrixComputationOffLowerDiagonal) {
  FermionicBasis basis(2, 1);

  Expression H = Expression(
      Term(1.0f, {c(Operator::Spin::Up, 1), a(Operator::Spin::Up, 0)}));

  auto matrix = compute_matrix_elements_serial<
      SparseMatrix<Expression::coefficient_type>>(basis, H);

  EXPECT_EQ(matrix.rows(), 4);
  EXPECT_EQ(matrix.cols(), 4);
  EXPECT_EQ(matrix(0, 0), Expression::coefficient_type(0, 0));
  EXPECT_EQ(matrix(0, 1), Expression::coefficient_type(0, 0));
  EXPECT_EQ(matrix(1, 0), Expression::coefficient_type(1.0f, 0));
  EXPECT_EQ(matrix(1, 1), Expression::coefficient_type(0, 0));
}

TEST_F(FermionicOperatorMatrixElementsTest,
       SparseMatrixComputationOffDiagonal) {
  FermionicBasis basis(2, 1);

  Expression H;
  H += Expression(
      Term(1.0f, {c(Operator::Spin::Up, 1), a(Operator::Spin::Up, 0)}));
  H += Expression(
      Term(1.0f, {c(Operator::Spin::Up, 0), a(Operator::Spin::Up, 1)}));

  auto matrix = compute_matrix_elements_serial<
      SparseMatrix<Expression::coefficient_type>>(basis, H);

  EXPECT_EQ(matrix.rows(), 4);
  EXPECT_EQ(matrix.cols(), 4);
  EXPECT_EQ(matrix(0, 0), Expression::coefficient_type(0, 0));
  EXPECT_EQ(matrix(0, 1), Expression::coefficient_type(1.0f, 0));
  EXPECT_EQ(matrix(1, 0), Expression::coefficient_type(1.0f, 0));
  EXPECT_EQ(matrix(1, 1), Expression::coefficient_type(0, 0));
}

TEST_F(FermionicOperatorMatrixElementsTest,
       SparseMatrixComputationOffUpperDiagonalTwoBody) {
  FermionicBasis basis(2, 2);

  Expression H = Expression(
      Term(2.0f, {c(Operator::Spin::Up, 0), c(Operator::Spin::Down, 0),
                  a(Operator::Spin::Up, 0), a(Operator::Spin::Down, 1)}));

  auto matrix = compute_matrix_elements_serial<
      SparseMatrix<Expression::coefficient_type>>(basis, H);

  EXPECT_EQ(matrix.rows(), 6);
  EXPECT_EQ(matrix.cols(), 6);
  EXPECT_EQ(matrix(0, 0), Expression::coefficient_type(0, 0));
  EXPECT_EQ(matrix(0, 1), Expression::coefficient_type(0, 0));
  EXPECT_EQ(matrix(0, 2), Expression::coefficient_type(0, 0));
  EXPECT_EQ(matrix(1, 0), Expression::coefficient_type(0, 0));
  EXPECT_EQ(matrix(1, 1), Expression::coefficient_type(0, 0));
  EXPECT_EQ(matrix(1, 2), Expression::coefficient_type(-2.0f, 0));
  EXPECT_EQ(matrix(2, 0), Expression::coefficient_type(0, 0));
  EXPECT_EQ(matrix(2, 1), Expression::coefficient_type(0, 0));
  EXPECT_EQ(matrix(2, 2), Expression::coefficient_type(0, 0));
}

class BosonicOperatorMatrixElementsTest : public ::testing::Test {
 protected:
  static Operator c(Operator::int_type orbital) {
    return Operator::Boson::creation(orbital);
  }

  static Operator a(Operator::int_type orbital) {
    return Operator::Boson::annihilation(orbital);
  }
};

TEST_F(BosonicOperatorMatrixElementsTest,
       SparseMatrixComputationOffUpperDiagonal) {
  BosonicBasis basis(2, 1);

  Expression H = Expression(Term(1.0f, {c(0), a(1)}));

  auto matrix = compute_matrix_elements_serial<
      SparseMatrix<Expression::coefficient_type>>(basis, H);

  EXPECT_EQ(matrix.rows(), 2);
  EXPECT_EQ(matrix.cols(), 2);
  EXPECT_EQ(matrix(0, 0), Expression::coefficient_type(0, 0));
  EXPECT_EQ(matrix(0, 1), Expression::coefficient_type(1.0f, 0));
  EXPECT_EQ(matrix(1, 0), Expression::coefficient_type(0, 0));
  EXPECT_EQ(matrix(1, 1), Expression::coefficient_type(0, 0));
}

TEST_F(BosonicOperatorMatrixElementsTest,
       SparseMatrixComputationOffLowerDiagonal) {
  BosonicBasis basis(2, 1);

  Expression H = Expression(Term(1.0f, {c(1), a(0)}));

  auto matrix = compute_matrix_elements_serial<
      SparseMatrix<Expression::coefficient_type>>(basis, H);

  EXPECT_EQ(matrix.rows(), 2);
  EXPECT_EQ(matrix.cols(), 2);
  EXPECT_EQ(matrix(0, 0), Expression::coefficient_type(0, 0));
  EXPECT_EQ(matrix(0, 1), Expression::coefficient_type(0, 0));
  EXPECT_EQ(matrix(1, 0), Expression::coefficient_type(1.0f, 0));
  EXPECT_EQ(matrix(1, 1), Expression::coefficient_type(0, 0));
}

TEST_F(BosonicOperatorMatrixElementsTest, SparseMatrixComputationOffDiagonal) {
  BosonicBasis basis(2, 1);

  Expression H;
  H += Expression(Term(1.0f, {c(1), a(0)}));
  H += Expression(Term(1.0f, {c(0), a(1)}));

  auto matrix = compute_matrix_elements_serial<
      SparseMatrix<Expression::coefficient_type>>(basis, H);

  EXPECT_EQ(matrix.rows(), 2);
  EXPECT_EQ(matrix.cols(), 2);
  EXPECT_EQ(matrix(0, 0), Expression::coefficient_type(0, 0));
  EXPECT_EQ(matrix(0, 1), Expression::coefficient_type(1.0f, 0));
  EXPECT_EQ(matrix(1, 0), Expression::coefficient_type(1.0f, 0));
  EXPECT_EQ(matrix(1, 1), Expression::coefficient_type(0, 0));
}

TEST_F(BosonicOperatorMatrixElementsTest,
       SparseMatrixComputationOffUpperDiagonalTwoBody) {
  BosonicBasis basis(2, 2);

  Expression H = Expression(Term(2.0f, {c(0), c(0), a(0), a(1)}));

  auto matrix = compute_matrix_elements_serial<
      SparseMatrix<Expression::coefficient_type>>(basis, H);

  EXPECT_EQ(matrix.rows(), 3);
  EXPECT_EQ(matrix.cols(), 3);
  EXPECT_EQ(matrix(0, 0), Expression::coefficient_type(0, 0));

  EXPECT_FLOAT_EQ(matrix(0, 1).real(), 4.0f / std::sqrt(2.0f));
  EXPECT_FLOAT_EQ(matrix(0, 1).imag(), 0);

  EXPECT_EQ(matrix(0, 2), Expression::coefficient_type(0, 0));
  EXPECT_EQ(matrix(1, 0), Expression::coefficient_type(0, 0));
  EXPECT_EQ(matrix(1, 1), Expression::coefficient_type(0, 0));
  EXPECT_EQ(matrix(1, 2), Expression::coefficient_type(0, 0));
  EXPECT_EQ(matrix(2, 0), Expression::coefficient_type(0, 0));
  EXPECT_EQ(matrix(2, 1), Expression::coefficient_type(0, 0));
  EXPECT_EQ(matrix(2, 2), Expression::coefficient_type(0, 0));
}

}  // namespace
}  // namespace qmutils
