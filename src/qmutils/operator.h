#pragma once

#include <bit>
#include <cstdint>
#include <ostream>
#include <string>
#include <type_traits>

#include "assert.h"

namespace qmutils {

class Operator {
 public:
  using int_type = uint16_t;
  enum class Type : int_type { Creation = 0, Annihilation = 1 };
  enum class Spin : int_type { Up = 0, Down = 1 };
  enum class Statistics : int_type { Fermion = 0, Boson = 1 };

  struct Fermion {
    static constexpr size_t ORBITAL_BITFIELD_WIDTH = sizeof(int_type) * 8 - 3;

    [[nodiscard]] static constexpr Operator make(Type type, Spin spin,
                                                 size_t orbital) noexcept {
      QMUTILS_ASSERT(orbital <
                     max_orbital_size());  // Max orbital is 2^13 - 1 = 8191
      uint16_t data = 0;                   // Bit 15 = 0 (fermion)
      data |= static_cast<uint16_t>(type) << 14;  // Bit 14 = type
      data |= static_cast<uint16_t>(spin) << 13;  // Bit 13 = spin
      data |= static_cast<uint16_t>(orbital);     // Bits 0-12 = orbital
      return Operator(data);
    }

    [[nodiscard]] static constexpr size_t max_orbital_size() noexcept {
      return 1 << ORBITAL_BITFIELD_WIDTH;
    }

    [[nodiscard]] static constexpr Operator creation(Spin spin,
                                                     size_t orbital) noexcept {
      QMUTILS_ASSERT(orbital < max_orbital_size());
      return make(Type::Creation, spin, orbital);
    }

    [[nodiscard]] static constexpr Operator annihilation(
        Spin spin, size_t orbital) noexcept {
      QMUTILS_ASSERT(orbital < max_orbital_size());
      return make(Type::Annihilation, spin, orbital);
    }

    [[nodiscard]] static constexpr Spin spin(Operator this_) noexcept {
      return static_cast<Spin>((this_.data_ >> 13) & 1);
    }

    [[nodiscard]] static constexpr size_t orbital(Operator this_) noexcept {
      return this_.data_ & 0x1FFF;  // Bits 0-12 (13 bits)
    }

    [[nodiscard]] static constexpr Operator flip_spin(Operator this_) noexcept {
      return Operator(this_.data_ ^ (1 << 13));
    }
  };

  struct Boson {
    static constexpr size_t ORBITAL_BITFIELD_WIDTH = sizeof(int_type) * 8 - 2;

    [[nodiscard]] static constexpr Operator make(Type type,
                                                 size_t orbital) noexcept {
      QMUTILS_ASSERT(orbital <
                     max_orbital_size());  // Max orbital is 2^14 - 1 = 16383
      uint16_t data = 1 << 15;             // Bit 15 = 1 (boson)
      data |= static_cast<uint16_t>(type) << 14;  // Bit 14 = type
      data |= static_cast<uint16_t>(orbital);     // Bits 0-13 = orbital
      return Operator(data);
    }

    [[nodiscard]] static constexpr size_t max_orbital_size() noexcept {
      return 1 << ORBITAL_BITFIELD_WIDTH;
    }

    [[nodiscard]] static constexpr Operator creation(size_t orbital) noexcept {
      return make(Type::Creation, orbital);
    }

    [[nodiscard]] static constexpr Operator annihilation(
        size_t orbital) noexcept {
      return make(Type::Annihilation, orbital);
    }

    [[nodiscard]] static constexpr size_t orbital(Operator o) noexcept {
      return o.data_ & 0x1FFF;  // Bits 0-12 (13 bits)
    }
  };

  Operator() = default;

  [[nodiscard]] constexpr Statistics statistics() const noexcept {
    return static_cast<Statistics>((data_ >> 15) & 1);
  }

  [[nodiscard]] constexpr Type type() const noexcept {
    return static_cast<Type>((data_ >> 14) & 1);
  }

  [[nodiscard]] constexpr size_t orbital() const noexcept {
    switch (statistics()) {
      case Statistics::Fermion:
        return Fermion::orbital(*this);
      case Statistics::Boson:
        return Boson::orbital(*this);
    }
    QMUTILS_UNREACHABLE();
    return 0;
  }

  [[nodiscard]] constexpr Spin spin() const noexcept {
    QMUTILS_ASSERT(statistics() == Statistics::Fermion);
    return Fermion::spin(*this);
  }

  [[nodiscard]] Operator adjoint() const noexcept {
    return Operator(data_ ^ (1 << 14));
  }

  [[nodiscard]] Operator flip_spin() const noexcept {
    QMUTILS_ASSERT(statistics() == Statistics::Fermion);
    return Fermion::flip_spin(*this);
  }

  [[nodiscard]] constexpr bool commutes(Operator other) const noexcept {
    return (this->data_ ^ other.data_) != (1 << 14);
  }

  constexpr bool operator<(Operator other) const noexcept {
    return data() < other.data();
  }

  constexpr bool operator>(Operator other) const noexcept {
    return data() > other.data();
  }

  constexpr bool operator==(Operator other) const noexcept {
    return data() == other.data();
  }

  constexpr bool operator>=(Operator other) const noexcept {
    return data() >= other.data();
  }

  constexpr bool operator<=(Operator other) const noexcept {
    return data() <= other.data();
  }

  constexpr bool is_fermion() const {
    return statistics() == Statistics::Fermion;
  }

  constexpr bool is_boson() const { return statistics() == Statistics::Boson; }

  static constexpr int_type parity(Operator op1, Operator op2) {
    if (op1.statistics() != op2.statistics()) {
      return 0;
    } else {
      switch (op1.statistics()) {
        case Statistics::Fermion:
          return 1;
        case Statistics::Boson:
          return 0;
      }
    }
    QMUTILS_UNREACHABLE();
    return 0;
  }

  [[nodiscard]] std::string to_string() const;

  friend std::ostream &operator<<(std::ostream &os, const Operator &op);

  [[nodiscard]] constexpr int_type data() const noexcept { return data_; }

 private:
  constexpr explicit Operator(uint16_t data) noexcept : data_(data) {}
  int_type data_;
};
static_assert(sizeof(Operator) == 2, "Operator must be 2 byte in size");
}  // namespace qmutils

template <>
struct std::hash<qmutils::Operator> {
  size_t operator()(const qmutils::Operator &op) const noexcept {
    return static_cast<size_t>(op.data());
  }
};
