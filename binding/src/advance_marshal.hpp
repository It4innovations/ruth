#pragma once

// Marshalling helpers for flat POD vectors. Header-only because they are
// templates. Layout: [size_t count][count * T], matching the byte-copy style
// used by the alternatives workloads in lib.cpp.

#include <cstring>
#include <vector>

namespace ruthlib {

template <typename T>
inline void put_pod_vec(char*& p, const std::vector<T>& v) {
  size_t n = v.size();
  std::memcpy(p, &n, sizeof(size_t)); p += sizeof(size_t);
  if (n) { std::memcpy(p, v.data(), n * sizeof(T)); p += n * sizeof(T); }
}

template <typename T>
inline std::vector<T> get_pod_vec(const char*& p) {
  size_t n;
  std::memcpy(&n, p, sizeof(size_t)); p += sizeof(size_t);
  std::vector<T> v(n);
  if (n) { std::memcpy(v.data(), p, n * sizeof(T)); p += n * sizeof(T); }
  return v;
}

template <typename T>
inline size_t pod_vec_size(const std::vector<T>& v) {
  return sizeof(size_t) + v.size() * sizeof(T);
}

}  // namespace ruthlib
