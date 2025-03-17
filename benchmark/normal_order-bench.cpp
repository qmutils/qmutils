#include <benchmark/benchmark.h>

#include <algorithm>
#include <random>

#include "qmutils/normal_order.h"
#include "qmutils/utils.h"

namespace qmutils {
namespace {

// Helper function to create a random operator
Operator random_operator(std::mt19937& gen) {
  std::uniform_int_distribution<> type_dist(0, 1);
  std::uniform_int_distribution<> spin_dist(0, 1);
  std::uniform_int_distribution<> orbital_dist(0,
                                               63);  // Assuming 6-bit orbital

  Operator::Type type = static_cast<Operator::Type>(type_dist(gen));
  Operator::Spin spin = static_cast<Operator::Spin>(spin_dist(gen));
  uint8_t orbital = static_cast<uint8_t>(orbital_dist(gen));

  return Operator::Fermion::make(type, spin, orbital);
}

// Helper function to create a term with random operators
Term create_random_term(size_t n_operators) {
  static std::mt19937 gen = get_random_generator();

  Term::container_type operators;
  operators.reserve(n_operators);

  for (size_t i = 0; i < n_operators; ++i) {
    operators.push_back(random_operator(gen));
  }

  return Term(Term::coefficient_type(1.0f, 0.0f), operators);
}

Term create_noncommuting_term(size_t n_operators) {
  Term::container_type operators;
  operators.reserve(n_operators);

  for (size_t i = 0; i < n_operators / 2; ++i) {
    operators.push_back(Operator::Fermion::annihilation(Operator::Spin::Up, 0));
  }

  for (size_t i = 0; i < n_operators / 2; ++i) {
    operators.push_back(Operator::Fermion::creation(Operator::Spin::Up, 0));
  }

  return Term(Term::coefficient_type(1.0f, 0.0f), std::move(operators));
}

Term create_commuting_term(size_t n_operators) {
  Term::container_type operators;
  operators.reserve(n_operators);

  for (size_t i = 0; i < n_operators; ++i) {
    operators.push_back(
        Operator::Fermion::creation(Operator::Spin::Up, i % 64));
  }

  std::reverse(operators.begin(), operators.end());

  return Term(Term::coefficient_type(1.0f, 0.0f), std::move(operators));
}

// Benchmark for normal ordering a single random term
static void BM_NormalOrderRandomTerm(benchmark::State& state) {
  NormalOrderer orderer;
  Term term = create_random_term(state.range(0));

  for (auto _ : state) {
    benchmark::DoNotOptimize(orderer.normal_order(term));
  }

  state.SetComplexityN(state.range(0));
}

// Benchmark for normal ordering a term with many non-commuting operators (worst
// case)
static void BM_NormalOrderNonCommutingTerm(benchmark::State& state) {
  NormalOrderer orderer;
  Term term = create_noncommuting_term(state.range(0));

  for (auto _ : state) {
    benchmark::DoNotOptimize(orderer.normal_order(term));
  }

  state.SetComplexityN(state.range(0));
}

static void BM_NormalOrderCommutingTerm(benchmark::State& state) {
  NormalOrderer orderer;
  Term term = create_commuting_term(state.range(0));

  for (auto _ : state) {
    benchmark::DoNotOptimize(orderer.normal_order(term));
  }

  state.SetComplexityN(state.range(0));
}

// Benchmark for normal ordering an expression with multiple terms
static void BM_NormalOrderExpression(benchmark::State& state) {
  NormalOrderer orderer;
  Expression expr;

  // Create an expression with 10 random terms
  for (int i = 0; i < 10; ++i) {
    expr += create_random_term(state.range(0));
  }

  for (auto _ : state) {
    benchmark::DoNotOptimize(orderer.normal_order(expr));
  }

  state.SetComplexityN(state.range(0));
}

Term generate_momentum_conserving_term(uint8_t nk) {
  static std::mt19937 gen = get_random_generator();
  std::uniform_int_distribution<> dist(0, nk - 1);

  uint8_t k1 = dist(gen);
  uint8_t k2 = dist(gen);
  uint8_t k3 = dist(gen);
  uint8_t k4 = (k1 + k2 - k3 + nk) % nk;  // Ensure momentum conservation

  return Term(Term::coefficient_type(1.0f, 0.0f),
              {Operator::Fermion::creation(Operator::Spin::Up, k1),
               Operator::Fermion::creation(Operator::Spin::Down, k2),
               Operator::Fermion::annihilation(Operator::Spin::Up, k3),
               Operator::Fermion::annihilation(Operator::Spin::Down, k4)});
}

static void BM_NormalOrderMomentumConserving(benchmark::State& state) {
  NormalOrderer orderer;
  uint8_t nk = state.range(0);  // Number of momentum states

  for (auto _ : state) {
    state.PauseTiming();
    Term term = generate_momentum_conserving_term(nk);
    state.ResumeTiming();

    benchmark::DoNotOptimize(orderer.normal_order(term));
  }

  state.SetComplexityN(nk);
}

Term generate_momentum_spin_conserving_term(uint8_t n, uint8_t nk) {
  static std::mt19937 gen = get_random_generator();
  std::uniform_int_distribution<> momentum_dist(0, nk - 1);
  std::uniform_int_distribution<> spin_dist(0, 1);

  std::vector<Operator> creation_ops;
  std::vector<Operator> annihilation_ops;

  for (uint8_t i = 0; i < n; ++i) {
    uint8_t momentum = momentum_dist(gen);
    Operator::Spin spin = static_cast<Operator::Spin>(spin_dist(gen));
    creation_ops.push_back(Operator::Fermion::creation(spin, momentum));
  }

  for (auto& op : creation_ops) {
    annihilation_ops.push_back(Operator::Fermion::make(
        Operator::Type::Annihilation, op.spin(), op.orbital()));
  }

  std::shuffle(annihilation_ops.begin(), annihilation_ops.end(), gen);

  std::vector<Operator> operators = creation_ops;
  operators.insert(operators.end(), annihilation_ops.begin(),
                   annihilation_ops.end());

  std::shuffle(operators.begin(), operators.end(), gen);
  return Term(Term::coefficient_type(1.0f, 0.0f), std::move(operators));
}

static void BM_NormalOrderMomentumSpinConservingN(benchmark::State& state) {
  NormalOrderer orderer;
  uint8_t nk = state.range(1);  // Number of momentum states
  uint8_t n = state.range(0);   // Number of creation/annihilation operators

  for (auto _ : state) {
    state.PauseTiming();
    Term term = generate_momentum_spin_conserving_term(n, nk);
    state.ResumeTiming();

    benchmark::DoNotOptimize(orderer.normal_order(term));
  }

  state.SetComplexityN(n);
}

BENCHMARK(BM_NormalOrderRandomTerm)
    ->RangeMultiplier(2)
    ->Range(1, 1 << 7)
    ->Complexity();

BENCHMARK(BM_NormalOrderNonCommutingTerm)
    ->RangeMultiplier(2)
    ->Range(1, 1 << 7)
    ->Complexity();

BENCHMARK(BM_NormalOrderCommutingTerm)
    ->RangeMultiplier(2)
    ->Range(1, 1 << 10)
    ->Complexity();

BENCHMARK(BM_NormalOrderExpression)
    ->RangeMultiplier(2)
    ->Range(1, 1 << 6)
    ->Complexity();

BENCHMARK(BM_NormalOrderMomentumConserving)
    ->RangeMultiplier(2)
    ->Ranges({{8, 64}, {2, 8}})
    ->Complexity();

BENCHMARK(BM_NormalOrderMomentumSpinConservingN)
    ->RangeMultiplier(2)
    ->Ranges({{2, 16}, {8, 64}})
    ->Complexity();
}  // namespace
}  // namespace qmutils
