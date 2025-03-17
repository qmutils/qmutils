#pragma once

#include <cstdlib>
#include <iostream>

namespace qmutils {

[[noreturn]] inline void qmutils_assert_fail(const char* expression,
                                             const char* file, int line,
                                             const char* function) {
  std::cerr << file << ":" << line << ": " << function << ": Assertion `"
            << expression << "' failed." << std::endl;
  __builtin_trap();
}

#ifdef NDEBUG
#define QMUTILS_ASSERT(condition) ((void)0)
#else
#define QMUTILS_ASSERT(condition)                                     \
  ((condition) ? ((void)0)                                            \
               : ::qmutils::qmutils_assert_fail(#condition, __FILE__, \
                                                __LINE__, __func__))

#define QMUTILS_UNREACHABLE() __builtin_trap()
#endif

}  // namespace qmutils
