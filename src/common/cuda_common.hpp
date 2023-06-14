#pragma once

#include <cstddef>
#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>

#define CUDA_CHECK(stmt)                                                       \
  do {                                                                         \
    cudaError_t result = (stmt);                                               \
    if (cudaSuccess != result) {                                               \
      fprintf(stderr, "[%s] CUDA failed with %s\n", __FUNCTION__,              \
              cudaGetErrorString(result));                                     \
      exit(-1);                                                                \
    }                                                                          \
  } while (0)

namespace YuriPerf {

/* Class to ease memory management on the GPU */
template <typename T> class CudaMemory {
public:
  CudaMemory() : m_ptr(nullptr), valid(false) {}
  CudaMemory(size_t size) : m_ptr(nullptr), valid(false) {
    CUDA_CHECK(cudaMalloc((void **)&m_ptr, size * sizeof(T)));
    valid = true;
  }
  ~CudaMemory() {
    if (valid)
      CUDA_CHECK(cudaFree(m_ptr));
  }

  // Get the pointer to the device memory
  inline T *get() { return m_ptr; }

  // Allocate size * sizeof(T) bytes of memory on the GPU
  inline void newMem(size_t size) {
    if (valid)
      CUDA_CHECK(cudaFree(m_ptr));

    CUDA_CHECK(cudaMalloc((void **)&m_ptr, size * sizeof(T)));
    valid = true;
  }

  // Copy size * sizeof(T) bytes from hostPtr to m_ptr
  inline void copyToDevice(const T *hostPtr, size_t size) {
    CUDA_CHECK(
        cudaMemcpy(m_ptr, hostPtr, size * sizeof(T), cudaMemcpyHostToDevice));
  }

  // Copy size * sizeof(T) bytes from hostPtr to m_ptr, using the given stream
  inline void copyToDevice(const T *hostPtr, size_t size, cudaStream_t stream) {
    CUDA_CHECK(cudaMemcpyAsync(m_ptr, hostPtr, size * sizeof(T),
                               cudaMemcpyHostToDevice, stream));
  }

  // Copy size * sizeof(T) bytes from m_ptr to hostPtr
  inline void copyToHost(T *hostPtr, size_t size) {
    CUDA_CHECK(
        cudaMemcpy(hostPtr, m_ptr, size * sizeof(T), cudaMemcpyDeviceToHost));
  }

  // Copy size * sizeof(T) bytes from m_ptr to hostPtr, using the given stream
  inline void copyToHost(T *hostPtr, size_t size, cudaStream_t stream) {
    CUDA_CHECK(cudaMemcpyAsync(hostPtr, m_ptr, size * sizeof(T),
                               cudaMemcpyDeviceToHost, stream));
  }

private:
  T *m_ptr;
  bool valid;
};

class CudaStream {
public:
  CudaStream() { CUDA_CHECK(cudaStreamCreate(&m_stream)); }
  ~CudaStream() { CUDA_CHECK(cudaStreamDestroy(m_stream)); }
  inline cudaStream_t get() { return m_stream; }

private:
  cudaStream_t m_stream;
};

} // namespace YuriPerf
