#include <cuda.h>
#include <cuda_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define CUDA_CHECK(ans)                        \
    {                                          \
        cudaAssert((ans), __FILE__, __LINE__); \
    }
inline void cudaAssert(cudaError_t code, const char* file, int line) {
    if (code != cudaSuccess) {
        fprintf(stderr, "CUDA error %s: %s at %s: %d\n",
                cudaGetErrorName(code), cudaGetErrorString(code),
                file, line);
        exit(code);
    }
}
#define CEIL_DIV(x, y) ((x) >= 0 ? (((x) + (y) - 1) / (y)) : ((x) / (y)))
#define M_PI 3.14159265358979323846f

/*
This kernel implements a naive softmax operation on a matrix of size (M, N).
The softmax operation is performed on the last dimension of the matrix.

How this works:
One thread processes one entire row, and thus this kernel will be the slowest
since we aren't exploiting parallelism capabilities of GPUs that much.
We are only parallelizing over the rows.
*/
__global__ void softmax_kernel_0(float* xd, float* resd, int M, int N) {
    /*
    xd (float*):
        The data over which we compute the softmax.
        This is a 1D array of length M * N * sizeof(float).
        We add the d to x, i.e. xd to indicate this variable was allocated on the device.
    resd (float*)
        The array where we store the result.
        This is a 1D array of length M * N * sizeof(float).
        We add the d to res, i.e. resd to indicate this variable was allocated on the device.
    M (int):
        The number of rows.
    N (int):
        The number of columns.
    */

    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = blockDim.x * gridDim.x;
    printf("index=%d -> blockid=%d blockdim=%d threadidx%d\n", index, blockIdx.x, blockDim.x, threadIdx.x);

    // float normalizer = 0;
    for (int i = index; i < M; i += stride){
        printf("%d\n", xd[i]);
        // normalizer += expf(xd[i]);
    }
    // printf("normalizer=%d\n", normalizer);
}

/*
This kernel implements an online softmax operation on a matrix of size (M, N).
The softmax operation is performed on the last dimension of the matrix.

How this works:
One thread processes one entire row, but instead of 3 passes we do only 2 passes.
This is possible due to the property of exponentials.
We are parallelizing over the rows.
*/
__global__ void softmax_kernel_1(float* xd, float* resd, int M, int N) {
}

/*
This kernel implements an online softmax operation on a matrix of size (M, N).
The softmax operation is performed on the last dimension of the matrix.

How this works:
In this, we handle each row with a block where the threads within one block work together
to process one row (max and norm factor). Each thread will process some elements
and will contains its local max and local norm in shared memory. Then, we perform reduction
operations to compute the final max and norm factor. Also, we compute maxes and norms
in one pass itself.
*/
__global__ void softmax_kernel_2(float* xd, float* resd, int M, int N) {
}

/*
This kernel implements an online softmax operation on a matrix of size (M, N).
The softmax operation is performed on the last dimension of the matrix.

How this works:
This one is largely similar to the above kernel. The difference is instead of accessing
shared memory and having sync barrier overhead, we will use warp-level primitives (then
block-level) for performing max and sum reductions. The benefit is: it is faster than shared
memory access and also does not need syncing since each warp (group of 32 threads) execute
an instuction parallely on GPU so no chance of race conditions.
*/
__global__ void softmax_kernel_3(float* xd, float* resd, int M, int N) {
}

/*
Helper function to generate a clamped random number sampled from a
normal distribution with mean 0 and std 1
*/
float random_normal_clamped(float min, float max) {
    float u1 = (float)rand() / RAND_MAX;
    float u2 = (float)rand() / RAND_MAX;
    float num = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * M_PI * u2);
    if (num < min)
        return min;
    if (num > max)
        return max;
    return num;
}

int main() {
    // int M = 1024;
    // int N = 32768;
    int M = 4;
    int N = 8;
    int matsize = M * N;
    int totalsize = matsize * sizeof(float);

    // allocate and initialize host matrix
    float* mat = (float*)malloc(totalsize);
    float* res = (float*)malloc(totalsize);
    for (int i = 0; i < matsize; i++) {
        // mat[i] = random_normal_clamped(-10, 10);
        mat[i] = i;
        printf("mat[i]=%f\n", mat[i]);
    }

    // arrays to allocate on device ends with 'd'
    float *xd, *resd;
    dim3 block_size(2);
    dim3 grid_size(CEIL_DIV(M, block_size.x));

    // below code calculates the time elapsed for
    // each cuda operation performed such as GPU allocation,
    // copying from host to device, kernel execution time etc...

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    float ms = 0.0f;

    cudaEventRecord(start);
    CUDA_CHECK(cudaMalloc(&xd, totalsize));
    CUDA_CHECK(cudaMalloc(&resd, totalsize));
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    printf(">> GPU allocation time: %f ms\n", ms);

    cudaEventRecord(start);
    CUDA_CHECK(cudaMemcpy(xd, mat, totalsize, cudaMemcpyHostToDevice));
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    printf(">> Host to device transfer time: %f ms\n", ms);

    cudaEventRecord(start);
    softmax_kernel_0<<<grid_size, block_size>>>(xd, resd, M, N);
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    printf(">> Kernel execution time: %f ms\n", ms);

    cudaEventRecord(start);
    CUDA_CHECK(cudaMemcpy(res, resd, totalsize, cudaMemcpyDeviceToHost));
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&ms, start, stop);
    printf(">> Device to host transfer time: %f ms\n", ms);

    // correctness check on the first row
    // the output should be 1.0 (or a number very close to it)
    // TODO: add full correctness check
    float sum = 0.f;
    for (int i = 0; i < N; i++) {
        sum += res[i];
    }
    printf("\nSum of the 1st row of softmax result: %f\n", sum);

    free(mat);
    free(res);
    cudaFree(xd);
    cudaFree(resd);
}
