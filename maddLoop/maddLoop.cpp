
/*
  KMDUMPISA=1 /opt/rocm/hip/bin/hipcc maddLoop.cpp
  or
  nvcc maddLoop.cu --ptx
  nvcc maddLoop.cu

  ./a.out


*/

#ifdef __HIPCC__

const char *gpuName = "AMD   ";

#include <hip/hip_runtime.h>

#define cudaStream_t hipStream_t
#define cudaEvent_t hipEvent_t
#define cudaMalloc hipMalloc
#define cudaEventCreate hipEventCreate
#define cudaEventRecord hipEventRecord
#define cudaEventSynchronize hipEventSynchronize
#define cudaEventElapsedTime hipEventElapsedTime
#define cudaStreamCreate hipStreamCreate
#define cudaSuccess hipSuccess
#define cudaGetErrorString hipGetErrorString
#define cudaError_t hipError_t
#define cudaMemcpy hipMemcpy
#define cudaMemset hipMemset
#define cudaMemcpyHostToDevice hipMemcpyHostToDevice
#define cudaMemcpyDeviceToHost hipMemcpyDeviceToHost

#else

const char *gpuName = "NVIDIA";

#endif

#include <cstdio>
#include <clocale>

constexpr int nUnroll = 64;

__global__ void test(int *gN, float b, float c, float d, int *out) {

  // if (threadIdx.x) return;
  const int n = *gN;
  b += threadIdx.x;
#pragma unroll(nUnroll)
  for (int i = 0; i < n; i++) {
    b *= c;
    b += d;
  }
  out[threadIdx.x] = b;
}

/*
__global__ void test(int *gN, float b, float c, float d, int *out) {
  const int n = *gN;
  b += threadIdx.x;
  #pragma unroll(1)
  for (int i = 0; i < n; i++) {
    if (threadIdx.x == 0) {
      b *= c;
      b += d;
    }
  }
  //if (threadIdx.x == 0)
    out[threadIdx.x] = b;
}
*/

int main(int argc, char **argv) {

  const int GlobalArraySIZE = 1 * 1024 * 1024 * 1024 / sizeof(int); // 1 GB

  const int nIter = 10;

  float b = 10, c = 1, d = 0;

  int *devPtr1 = 0;
  int *devPtr2 = 0;
  int *hostPtr1 = new int[1];
  int *hostPtr2 = new int[GlobalArraySIZE];

  cudaMalloc(&devPtr1, sizeof(int));
  cudaMalloc(&devPtr2, sizeof(int) * GlobalArraySIZE);

  for (int i = 0; i < GlobalArraySIZE; i++) {
    hostPtr2[i] = 0;
  }

  cudaMemcpy(devPtr2, hostPtr2, sizeof(int) * GlobalArraySIZE,
             cudaMemcpyHostToDevice);

  cudaStream_t stream;
  cudaEvent_t start[nIter], stop[nIter];
  cudaStreamCreate(&stream);

  for (int i = 0; i < nIter; i++) {
    cudaEventCreate(&start[i]);
    cudaEventCreate(&stop[i]);
  }

  for (int nLoops = 1; nLoops <= 1000000000; nLoops *= 10) {

    hostPtr1[0] = nLoops;
    cudaMemcpy(devPtr1, hostPtr1, sizeof(int), cudaMemcpyHostToDevice);

    int nBlocks = 1;

    for (int nThreads = 64; nThreads <= 1024; nThreads *= 4) {
      // for (int nThreads = 64; nThreads <= 64; nThreads *= 2) {

      printf("\n");

      for (int iter = 0; iter < nIter; iter++) {
        cudaEventRecord(start[iter], stream);
#ifdef __HIPCC__
        hipLaunchKernelGGL(test, dim3(nBlocks), dim3(nThreads), 0, stream,
                           devPtr1, b, c, d, devPtr2);
#else
        test<<<nBlocks, nThreads, 0, stream>>>(devPtr1, b, c, d, devPtr2);
#endif
        cudaEventRecord(stop[iter], stream);
      }

      for (int iter = 0; iter < nIter; iter++) {
        cudaEventSynchronize(stop[iter]);
        float timeMS;
        cudaEventElapsedTime(&timeMS, start[iter], stop[iter]);
        cudaMemcpy(hostPtr2, devPtr2, sizeof(int) * nThreads,
                   cudaMemcpyDeviceToHost);
        int sum = 0;
        for (int i = 0; i < nThreads; i++) {
          sum += hostPtr2[i];
        }
        setlocale(LC_NUMERIC, "");
        setlocale(LC_ALL, "");
        printf("%s  %'d loops. Unroll %d, NThreads %d. LoopTest Time %'d usec, "
               "output: %d\n",
               gpuName, nLoops, nUnroll, nThreads, (int)rint(timeMS * 1000),
               sum);
      } // iter
    }
  }
  printf("\n");

  delete[] hostPtr1;
  delete[] hostPtr2;

  return 0;
}
