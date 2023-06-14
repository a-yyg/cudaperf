#include <chrono>
#include <cstdio>

#define CUDA_CHECK(stmt)                                                       \
  do {                                                                         \
    cudaError_t result = (stmt);                                               \
    if (cudaSuccess != result) {                                               \
      fprintf(stderr, "[%s] CUDA failed with %s\n", __FUNCTION__,              \
              cudaGetErrorString(result));                                     \
      exit(-1);                                                                \
    }                                                                          \
  } while (0)

typedef std::chrono::high_resolution_clock::time_point TimeVar;
#define duration(a)                                                            \
  (std::chrono::duration_cast<std::chrono::microseconds>(a).count())
#define timeNow() std::chrono::high_resolution_clock::now()

#define ARRAY_LEN(arr) (sizeof(arr) / sizeof(arr[0]))

#define BENCHMARK 1

/* Class to ease memory management on the GPU */
template <typename T> class CudaMemory {
public:
  CudaMemory() : valid(false) {}
  CudaMemory(size_t size) : valid(true) {
    CUDA_CHECK(cudaMalloc(&m_ptr, size * sizeof(T)));
  }
  ~CudaMemory() {
    if (valid)
      CUDA_CHECK(cudaFree(m_ptr));
  }
  inline T *get() { return m_ptr; }
  inline void newMem(size_t size) {
    if (valid) {
      CUDA_CHECK(cudaFree(m_ptr));
    }
    CUDA_CHECK(cudaMalloc(&m_ptr, size * sizeof(T)));
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

/* Kernels */

// B = A * scale
__global__ void MatMul(float *A, float *B, int N, int scale) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int j = blockDim.y * blockIdx.y + threadIdx.y;

  if (i < N && j < N) {
    int idx = i * N + j;
    B[idx] = A[idx] * scale;
  }
}

// C = A + B
__global__ void MatAdd(float *A, float *B, float *C, int N) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int j = blockDim.y * blockIdx.y + threadIdx.y;

  if (i < N && j < N) {
    int idx = i * N + j;
    C[idx] = A[idx] + B[idx];
  }
}

// This simulates the following graph execution:
// Input: Matrix A, scalars s1, ..., sN
// Multiply A by each scalar, saving the results in
// intermediate matrices B1, ..., BN
// Add all intermediate matrices together two by two,
// example: C1 = B1 + B2, C2 = C1 + B3, ..., C(N-1) = C(N-2) + BN
// in this case, the final result will be stored in CN
class GraphA {
public:
  GraphA(int N, int numScalars) : m_N(N), m_numScalars(numScalars) {
    // Allocate memory for the input matrix and scalars
    host_A = new float[m_N * m_N];
    scalars = new float[m_numScalars];

    m_A.newMem(m_N * m_N);

    // Allocate memory for the intermediate matrices (only on the device)
    m_B = new CudaMemory<float>[m_numScalars];
    for (int i = 0; i < m_numScalars; i++) {
      m_B[i].newMem(m_N * m_N);
    }

    // Allocate memory for the output matrix
    host_C = new float[m_N * m_N];
    m_C = new CudaMemory<float>[m_numScalars - 1];
    for (int i = 0; i < m_numScalars - 1; i++) {
      m_C[i].newMem(m_N * m_N);
    }

    // Initialize the input matrix and scalars
    for (int i = 0; i < m_N * m_N; i++) {
      host_A[i] = i;
    }
    for (int i = 0; i < m_numScalars; i++) {
      scalars[i] = i;
    }
  }
  ~GraphA() {
    delete[] host_A;
    delete[] host_C;
    delete[] scalars;
    delete[] m_B;
  }

  void exec() {
#if (BENCHMARK == 1)
    TimeVar start, end;
    start = timeNow();
#endif
    // Copy the input matrix and scalars to the GPU
    m_A.copyToDevice(host_A, m_N * m_N);
#if (BENCHMARK == 1)
    end = timeNow();
    printf("Copy input to GPU: %ld us\n", duration(end - start));
#endif

    int numStreams = m_numScalars + 1;
    CudaStream *streams = new CudaStream[numStreams];

    // Dim3 sizes. Is this ok?
    dim3 dimBlock(32, 32);
    dim3 dimGrid((m_N + 31) / 32, (m_N + 31) / 32);

#if (BENCHMARK == 1)
    start = timeNow();
#endif
    // Multily the input matrix by each scalar
    for (int i = 0; i < m_numScalars; i++) {
      MatMul<<<dimGrid, dimBlock, 0, streams[i].get()>>>(
          m_A.get(), m_B[i].get(), m_N, scalars[i]);
    }

    cudaDeviceSynchronize(); // Make sure all streams are done
#if (BENCHMARK == 1)
    end = timeNow();
    printf("Multiply by scalars: %ld us\n", duration(end - start));
    start = timeNow();
#endif

    // Add all intermediate matrices together two by two
    // First B1 + B2 = C1
    // Doing this synchronously, as we need the result of this operation
    MatAdd<<<dimGrid, dimBlock>>>(
        m_B[0].get(), m_B[1].get(), m_C[0].get(), m_N);

    // Then C1 + B3 = C2, and so on
    for (int i = 1; i < m_numScalars - 1; i++) {
      MatAdd<<<dimGrid, dimBlock>>>(
          m_C[i - 1].get(), m_B[i + 1].get(), m_C[i].get(), m_N);
    }
    cudaDeviceSynchronize(); // Make sure all streams are done

#if (BENCHMARK == 1)
    end = timeNow();
    printf("Add intermediate matrices: %ld us\n", duration(end - start));
    start = timeNow();
#endif

    // Copy the output matrix back to the host
    m_C[m_numScalars - 2].copyToHost(host_C, m_N * m_N);

#if (BENCHMARK == 1)
    end = timeNow();
    printf("Copy output to CPU: %ld us\n", duration(end - start));
#endif
  }

  void print() {
    for (int i = 0; i < m_N; i++) {
      for (int j = 0; j < m_N; j++) {
        printf("%.2f\t", host_C[i * m_N + j]);
      }
      printf("\n");
    }
    printf("\n");
  }

private:
  int m_N;
  int m_numScalars;
  float *host_A;
  float *host_C;
  float *scalars;
  CudaMemory<float> m_A;
  CudaMemory<float> *m_B;
  CudaMemory<float> *m_C;
};

int main(int argc, char *argv[]) {
  int N = argc > 1 ? atoi(argv[1]) : 8;
  int numScalars = argc > 2 ? atoi(argv[2]) : 2;

  // Create the graph
  GraphA graph(N, numScalars);
  graph.exec();
  // graph.print();

  return 0;
}
