#include "find_repeats.h"

// Now you can call `find_repeats` within `cudaRenderer.cu`
#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <driver_functions.h>

#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include <thrust/device_malloc.h>
#include <thrust/device_free.h>


#define THREADS_PER_BLOCK 256


// helper function to round an integer up to the next power of 2
static inline int nextPow2(int n) {
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n++;
    return n;
}

__global__ void upsweep(int* result, int N, int two_d, int two_dplus1) {
    long tid = blockIdx.x * blockDim.x + threadIdx.x;
    long i = tid * two_dplus1;  // Indexing based on two_dplus1 to avoid race conditions

    if (i >= N) return;  // Out-of-bounds check
    result[i + two_dplus1 - 1] += result[i + two_d - 1];
}

__global__ void downsweep(int* result, int N, int two_d, int two_dplus1) {
    long tid = blockIdx.x * blockDim.x + threadIdx.x;
    long i = tid * two_dplus1;  // Indexing based on two_dplus1 to avoid race conditions

    if (i >= N) return;  // Out-of-bounds check
    long t = result[i + two_d - 1];
    result[i + two_d - 1] = result[i + two_dplus1 - 1];
    result[i + two_dplus1 - 1] += t;
}



__global__ void construct_adj_flags(int* device_input, int length, int* flags_output) {
    long tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= length - 1) return;
    if (device_input[tid] == device_input[tid+1])
        flags_output[tid] = 1;
}
__global__ void map_flags(int* flags_input, int* indices_input, int length, int* device_output) {
    long tid = blockIdx.x * blockDim.x + threadIdx.x;
    // only the first N-1 elements are candidates for repeats
    if (tid >= length - 1) return;
    // we only want to write results for repeats (since ES produces multiple copies of indices)
    if (flags_input[tid] == 0) return;
    device_output[indices_input[tid]] = tid;
}





// find_repeats --
//
// Given an array of integers `device_input`, returns an array of all
// indices `i` for which `device_input[i] == device_input[i+1]`.
//
// Returns the total number of pairs found
int find_repeats(int* device_input, int length, int* device_output) {
  
    // Note: As in the scan code, the calling code ensures that
    // allocated arrays are a power of 2 in size, so you can use your
    // exclusive_scan function with them. However, your implementation
    // must ensure that the results of find_repeats are correct given
    // the actual array length.
    // first get the flags representing adjacencies
    int* adj_flags;
    int rounded_length = nextPow2(length);
    int num_blocks = (length + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    // cudaMalloc((void **)&adj_flags, length*sizeof(int));
    cudaMalloc((void **)&adj_flags, rounded_length*sizeof(int));
    construct_adj_flags<<<num_blocks, THREADS_PER_BLOCK>>>(device_input, length, adj_flags);
    int* indices_into_output;
    // cudaMalloc((void **)& indices_into_output, length*sizeof(int));
    cudaMalloc((void **)& indices_into_output, rounded_length*sizeof(int));
    // run ES to get indices to place answers into output
    exclusive_scan(adj_flags, length, indices_into_output);
    // get desired 'solution indices'
    map_flags<<<num_blocks, THREADS_PER_BLOCK>>>(adj_flags, indices_into_output, length, device_output);

    // // since this function returns the # of repeats, we must query last item of `indices_into_output`
    int num_repeats[1];
    cudaMemcpy(num_repeats, indices_into_output + (length - 1), sizeof(int), cudaMemcpyDeviceToHost);
    // // clean up CUDA arrays allocated during this function call
    cudaFree(adj_flags);
    cudaFree(indices_into_output);
    return num_repeats[0];
}

