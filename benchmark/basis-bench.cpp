#include <benchmark/benchmark.h>

#include <algorithm>

#include "qmutils/basis.h"
#include "qmutils/utils.h"

namespace qmutils {
namespace {

static void BM_BasisConstruction(benchmark::State& state) {
  const size_t orbitals = state.range(0);
  const size_t unchecked_particles = state.range(1);
  const size_t particles = std::min(unchecked_particles, 2 * orbitals);

  for (auto _ : state) {
    Basis basis(orbitals, particles);
    benchmark::DoNotOptimize(basis);
  }

  state.SetComplexityN(orbitals * particles);
}

static void BM_BasisContains(benchmark::State& state) {
  const size_t orbitals = state.range(0);
  const size_t unchecked_particles = state.range(1);
  const size_t particles = std::min(unchecked_particles, 2 * orbitals);

  Basis basis(orbitals, particles);

  // Create a state that's guaranteed to be in the basis
  Basis::operators_type test_state;
  for (size_t i = 0; i < particles; ++i) {
    test_state.push_back(
        Operator::Fermion::creation(Operator::Spin::Up, i % orbitals));
  }

  for (auto _ : state) {
    bool result = basis.contains(test_state);
    benchmark::DoNotOptimize(result);
  }

  state.SetComplexityN(orbitals * particles);
}

static void BM_BasisIteration(benchmark::State& state) {
  const size_t orbitals = state.range(0);
  const size_t unchecked_particles = state.range(1);
  const size_t particles = std::min(unchecked_particles, 2 * orbitals);

  Basis basis(orbitals, particles);

  for (auto _ : state) {
    size_t count = 0;
    for (const auto& basis_state : basis) {
      count += basis_state.size();
    }
    benchmark::DoNotOptimize(count);
  }

  state.SetComplexityN(basis.size());
}

Term::container_type generate_valid_operators(size_t orbitals,
                                              size_t particles) {
  static std::mt19937 gen = get_random_generator();
  std::uniform_int_distribution<> orbital_dist(0, orbitals - 1);
  std::uniform_int_distribution<> spin_dist(0, 1);

  Term::container_type operators;
  operators.reserve(particles);

  while (operators.size() < particles) {
    Operator::Spin spin = static_cast<Operator::Spin>(spin_dist(gen));
    uint8_t orbital = static_cast<uint8_t>(orbital_dist(gen));
    Operator op = Operator::Fermion::creation(spin, orbital);

    if (std::find(operators.begin(), operators.end(), op) == operators.end()) {
      operators.push_back(op);
    }
  }

  std::sort(operators.begin(), operators.end());
  return operators;
}

static void BM_BasisInsert(benchmark::State& state) {
  const size_t orbitals = state.range(0);
  const size_t particles =
      std::min(static_cast<size_t>(state.range(1)), 2 * orbitals);

  Basis basis(0, 0);

  for (auto _ : state) {
    state.PauseTiming();
    auto operators = generate_valid_operators(orbitals, particles);
    state.ResumeTiming();

    if (!basis.contains(operators)) {
      basis.insert(operators);
    }
  }

  state.SetComplexityN(orbitals * particles);
}

BENCHMARK(BM_BasisConstruction)
    ->ArgsProduct({benchmark::CreateRange(1, 8, /*multi=*/2),
                   benchmark::CreateDenseRange(1, 16, /*step=*/2)})
    ->Complexity();

BENCHMARK(BM_BasisContains)
    ->ArgsProduct({benchmark::CreateRange(1, 8, /*multi=*/2),
                   benchmark::CreateDenseRange(1, 16, /*step=*/2)})
    ->Complexity();

BENCHMARK(BM_BasisIteration)
    ->ArgsProduct({benchmark::CreateRange(1, 8, /*multi=*/2),
                   benchmark::CreateDenseRange(1, 16, /*step=*/2)})
    ->Complexity();

BENCHMARK(BM_BasisInsert)
    ->ArgsProduct({benchmark::CreateRange(1, 8, /*multi=*/2),
                   benchmark::CreateDenseRange(1, 16, /*step=*/2)})
    ->Complexity();

}  // namespace
}  // namespace qmutils

BENCHMARK_MAIN();
