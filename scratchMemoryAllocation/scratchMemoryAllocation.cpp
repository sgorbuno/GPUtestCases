
/*
  KMDUMPISA=1 /opt/rocm/hip/bin/hipcc scratchMemoryAllocation.cpp
  or
  nvcc scratchMemoryAllocation.cu --ptx
  nvcc scratchMemoryAllocation.cu

  CUDA_VISIBLE_DEVICES=0 ./a.out

 The problem: apparently, too much time is spent in run-time initialization of 
 the scratch memory.

 The kernel is not doing any actual job, it only declares a scratch array which 
 does not fit to registers.

 Runtime on NVIDIA is 0.1 ms, on AMD it is 700 ms.

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

#include <clocale>
#include <cstdio>

#define nUnroll -1

constexpr int nnUnroll = nUnroll;

__global__ void test(int *gN, int *gOut) {

  if (*gN >= 0) { // always happens
    return;
  }
 
  // the code below is not executed

  constexpr int N = 1000;
  int A[N];

  for (int i = 0; i < N; i+=2000) {
    A[i] = i;
  }
  
  gOut[threadIdx.x] = A[threadIdx.x];
}

int main(int argc, char **argv) {

  const int GlobalArraySIZE = 1024; //

  const int nMeasurements = 10;

  int *devPtr1 = 0;
  int *devPtr2 = 0;
  int *hostPtr1 = new int[1];
  int *hostPtr2 = new int[GlobalArraySIZE];

  cudaMalloc(&devPtr1, sizeof(int));
  cudaMalloc(&devPtr2, sizeof(int) * GlobalArraySIZE);

  hostPtr1[0] = 1;
  for (int i = 0; i < GlobalArraySIZE; i++) {
    hostPtr2[i] = 0;
  }

  cudaMemcpy(devPtr1, hostPtr1, sizeof(int), cudaMemcpyHostToDevice);
  cudaMemcpy(devPtr2, hostPtr2, sizeof(int) * GlobalArraySIZE,
             cudaMemcpyHostToDevice);

  cudaStream_t stream;
  cudaEvent_t start[nMeasurements], stop[nMeasurements];
  cudaStreamCreate(&stream);

  for (int i = 0; i < nMeasurements; i++) {
    cudaEventCreate(&start[i]);
    cudaEventCreate(&stop[i]);
  }

  int nBlocks = 160;
  int nThreads = 1024;

  for (int iter = 0; iter < nMeasurements; iter++) {
    cudaEventRecord(start[iter], stream);
    for (int iSector = 0; iSector < 36; iSector++) {
#ifdef __HIPCC__
      hipLaunchKernelGGL(test, dim3(nBlocks), dim3(nThreads), 0, stream,
                         devPtr1, devPtr2);
#else
      test<<<nBlocks, nThreads, 0, stream>>>(devPtr1, devPtr2);
#endif
    }
    cudaEventRecord(stop[iter], stream);
  }

  for (int iter = 0; iter < nMeasurements; iter++) {
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

    printf("%s Scratch Memory Allocation test Time %'d usec, "
           "output: %d\n",
           gpuName, (int)rint(timeMS * 1000), sum);
  } // iter

  printf("\n");

  delete[] hostPtr1;
  delete[] hostPtr2;

  return 0;
}
