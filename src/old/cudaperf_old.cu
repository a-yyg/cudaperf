// Includes

#include <stdio.h>
#include <time.h>

#include <chrono>

typedef std::chrono::high_resolution_clock::time_point TimeVar;
#define duration(a) std::chrono::duration_cast<std::chrono::seconds>(a).count()
#define duration2(a)                                                           \
  ((double)std::chrono::duration_cast<std::chrono::microseconds>(a).count() /  \
   1e6)
#define timeNow() std::chrono::high_resolution_clock::now()

#define ARRAY_LEN(arr) (sizeof(arr) / sizeof(arr[0]))

// Variables

int iNumElements = 100; // Length of float arrays to process

bool DEBUG = true;

// Functions

void fillFloatArray(float *arr, int length);
void printArray(float *arr, const char *name, int length);

// Device code

__global__ void VecAdd(const float *A, const float *B, float *C, int N)
{
  int i = blockDim.x * blockIdx.x + threadIdx.x;

  if (i < N)
    C[i] = A[i] + B[i];
}

__global__ void MatMul(float *A, float *B, int N, int scale) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int j = blockDim.y * blockIdx.y + threadIdx.y;

  if (i < N && j < N) {
    int idx = i * N + j;
    B[idx] = A[idx] * scale;
  }
}

__global__ void MatAdd(float *A, float *B, float *C, int N) {
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int j = blockDim.y * blockIdx.y + threadIdx.y;

  if (i < N && j < N) {
    int idx = i * N + j;
    C[idx] = A[idx] + B[idx];
  }
}

// Host code

int main(int argc, char **argv)
{
  if (argc > 1) {
    iNumElements = atoi(argv[1]);
    printf("Setting numbers to %d\n", iNumElements);
    if (argc == 3) {
      DEBUG = false;
    }
  }

  TimeVar start, end;
  double t_hd = 0.0, t_exe = 0.0, t_dh = 0.0;

  srand(time(NULL));

  size_t size = iNumElements * iNumElements * sizeof(float);
  // int values[] = {
  //     1, 2, 3, 4, 5, 6, 7, 8, 9,
  // };

  int values[32];
  for (int i = 0; i < ARRAY_LEN(values); i++) {
    values[i] = 1;
  }
  
  int num = ARRAY_LEN(values);

  float *d_A;
  float *d_B[num];
  float *d_C[num - 2];
  float *d_D;
  float *h_A;
  float *h_D;
  // for (int i = 0; i < iNumElements; i++) {
  //   h_B[i] = malloc(size * size);
  // }

  // Allocate input vectors h_A and h_B in host memory
  h_A = (float *)malloc(size);
  h_D = (float *)malloc(size);

  // C = (float *)malloc(size * size);

  // Initialize input vectors
  fillFloatArray(h_A, iNumElements);
  // printArray(h_A, "Array A", iNumElements);
  // fillFloatArray(h_B, iNumElements);

  // for (int i = 0; i < iNumElements; i++) {
  //   C[i] = h_A[i] + h_B[i];
  // }

  // if (DEBUG) {
  //   printFloatArray(h_A, "Array A", iNumElements);
  //   printFloatArray(h_B, "Array B", iNumElements);
  //   printFloatArray(C, "Array C", iNumElements);
  // }

  // Allocate vectors in device memory
  cudaMalloc((void **)&d_A, size);
  for (int i = 0; i < num; i++) {
    cudaMalloc((void **)&d_B[i], size);
  }

  for (int i = 0; i < num - 2; i++) {
    cudaMalloc((void **)&d_C[i], size);
  }

  cudaMalloc((void **)&d_D, size);

  // Copy vectors from host memory to device memory

  int tries = 100;
  for (int i = 0; i < tries; ++i)
  {
    start = timeNow();

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);

    end = timeNow();

    // printf("Time taken for copy H->D: %f\n", duration2(end - start));
    t_hd += duration2(end - start);

    // Invoke kernel

    // int threadsPerBlock = 256;
    // int blocksPerGrid = (iNumElements + threadsPerBlock - 1) / threadsPerBlock;
    dim3 dimBlock(32, 32);
    dim3 dimGrid((iNumElements + dimBlock.x - 1) / dimBlock.x,
                 (iNumElements + dimBlock.y - 1) / dimBlock.y);

    start = timeNow(); 

    // VecAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, iNumElements);
    for (int i = 0; i < num; i++) {
      MatMul<<<dimGrid, dimBlock>>>(d_A, d_B[i], iNumElements, values[i]);
    }

    cudaDeviceSynchronize();

    MatAdd<<<dimGrid, dimBlock>>>(d_B[0], d_B[1], d_C[0], iNumElements);
    cudaDeviceSynchronize();
    for (int i = 2; i < num - 1; i++) {
      MatAdd<<<dimGrid, dimBlock>>>(d_B[i], d_C[i - 2], d_C[i - 1], iNumElements);
      cudaDeviceSynchronize();
    }
    MatAdd<<<dimGrid, dimBlock>>>(d_C[num - 3], d_B[num - 1], d_D, iNumElements);
    cudaDeviceSynchronize();
    // cudaMemcpy(d_D, d_C[6], size, cudaMemcpyDeviceToDevice);

    end = timeNow();

    // printf("Time taken for execution: %f\n", duration2(end - start));
    t_exe += duration2(end - start);

    start = timeNow();

    // Copy result from device memory to host memory
    // h_C contains the result in host memory
    cudaMemcpy(h_D, d_D, size, cudaMemcpyDeviceToHost);

    end = timeNow();
    t_dh += duration2(end - start);

    // printf("Time taken for copy D->H: %f\n", duration2(end - start));
  }

  printf("Average copy H->D: %.6lf\n", t_hd / tries);
  printf("Average execution: %.6lf\n", t_exe / tries);
  printf("Average copy D->H: %.6lf\n", t_dh / tries);

  // if (DEBUG) {
  //   printArray(h_D, "Result", iNumElements);
  // }
  printArray(h_D, "Result", iNumElements);

  // for (int i = 0; i < iNumElements; i++) {
  //   if (h_D[i] != d_D[i]) {
  //     printf("Error at %d\n", i);

  //     break;
  //   }
  // }

  // Free device memory

  if (d_A)
    cudaFree(d_A);

  for (int i = 0; i < num; i++) {
    if (d_B[i])
      cudaFree(d_B[i]);
  }

  for (int i = 0; i < num - 2; i++) {
    if (d_C[i])
      cudaFree(d_C[i]);
  }

  if (d_D)
    cudaFree(d_D);

  // Free host memory

  if (h_A)
    free(h_A);

  if (h_D)
    free(h_D);
}

// void fillFloatArray(float *arr, int length) {
//   for (int i = 0; i < length; i++) {
//     arr[i] = (float)rand() / (float)RAND_MAX;
//   }
// }

void fillFloatArray(float *arr, int length) {
  for (int i = 0; i < length * length; i++) {
    arr[i] = 1;
  }
}

void printArray(float *arr, const char *name, int length) {
  printf("%s:\n", name);

  for (int i = 0; i < length; i++) {
    for (int j = 0; j < length; ++j)
    {
      printf("%f ", arr[i * length + j]);
    }
    // printf("%.1f ", arr[i]);
    printf("\n");
  }

  printf("\n\n");
}
