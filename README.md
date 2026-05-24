# GEMM Benchmark with Tensor Cores

## Who
Sandeep | Age 17 | Prakasam AP

## What
High performance Matrix Multiplication using CUDA Tensor Cores
and wmma API. Benchmarked against theoretical T4 GPU peak.

## Skills Shown
- CUDA Tensor Cores
- wmma API (fragment, load_matrix_sync, mma_sync)
- FP16 input FP32 output mixed precision
- Shared memory optimization
- CUDA Events for benchmarking
- Pinned memory (cudaMallocHost)

## Result
- Matrix size: 4096 x 4096
- Achieved: X TFLOPS on T4 GPU
- T4 Peak: 65 TFLOPS
- cuBLAS gap reason: years of NVIDIA tuning,
  float4 vectorization, perfect memory coalescing

## How to run
nvcc -o gemm kernel.cu -arch=sm_75
./gemm

## Output
TIME:   xx.xxxx ms
GFLOPS: xxxx.xx
TFLOPS: x.xxxx
Result[0]: 8192.00
