# GPU Programming Assignment 2

Arjun Menon Vadakkeveedu, EE18B104

2 March 2022

### Implementation Versions:
This code contains the following implementations of the matrix operation $(A + B^T)CD^T$:

1. Implementation on CPU 
2. Baseline GPU Implementation using separate kernels for matrix transpose ($D^T$ operation), Matrix-Matrix-transpose add ($A + B^T$ operation) and Matrix-Matrix Multiply ($XY$; used to perform the $CD^T$ and $(A+B^T)CD^T$ operations). The Matrix-Matrix Multiply is performed by parallelizing the nested for loop implementation of sequential matrix multiplication.
3. GPU Implementation with separate kernels for Matrix-Matrix-transpose add ($A + B^T$ operation) and Tiled Matrix-Matrix-Transpose Multiply (directly perform the $CD^T$ operation). The second kernel is also used for the final matmul operation ($(A+B^T)CD^T$)

    this is achieved by switching the ordering of the $CD^T$ operation:

    $Y$ = $(A + B^T)CD^T$ 		
        = $(A + B^T)(DC^T)^T$
    
    $Y$ = `MMTransposeKernel`(`MMAddKernel`($A$, $B$), `MMTransposeKernel`($D$, $C$))
	
Implementations 2 and 3 utilise shared memory wherever applicable, use coalesced accesses and avoid shared memory bank conflicts

### Further Optimizations:

1. Using **Pinned Memory** for Host Matrices reduced the overhead associated with pageable memcpy from host. 

    The macro `USE_PINNED_MEMORY` is used to determine if the host matrices are to be allocated using `malloc()` (pageable) or using `cudaMallocHost()` (pinned).

2. Using **Multiple Streams** to perform the independent kernel operations: In Implementations 2 and 3, the $A+B^T$ operation 
can be interleaved/ performed concurrently with the $CD^T$ operation.

    `COMPUTE_MODE` 2 and 4 use two cuda streams to launch
the corresponding kernels. 

    However, the dominant operations in terms of compute time are the matrix multiplies, which cannot be performed asynchronously. Hence, the execution time does not improve dramatically with multiple streams.

### Results:

| Compute Mode | Pinned Memory <br>Setting | Execution Time (ms): <br>input1 (4, 5, 6, 7) | Execution Time (ms):<br>input2 (8, 9, 10, 11) | Execution Time (ms):<br>input3 (1000, 1023, 544, 542) |
|---|---|---|---|---|
| CPU | Pageable | 98.219 | 72.682 | 3272.158 |
| Baseline null stream | Pageable | 101.812 | 77.946 | 111.898 |
| Baseline w/ 2 streams | --- | --- | --- | --- |
| Tiled Multiply <br>null stream | Pageable | 103.833 | 99.695 | 111.33 |
| Tiled Multiply <br>w/ 2 streams | --- | --- | --- | --- |
| CPU | Pinned   | 0.049 | 0.085 | 3152.902 |
| Baseline null stream | Pinned   | 0.298 | 0.312 | 57.808 |
| Baseline w/ 2 streams | Pinned   | 0.303 | 0.316 | 54.018 |
| Tiled Multiply <br>null stream | Pinned   | 0.43 | 0.315 | 42.64 |
| Tiled Multiply <br>w/ 2 streams | Pinned   | 0.316 | 0.31 | 37.889 |

### Inferences:
1. The `memcpy()` overhead in case of pageable host memory allocation is significant, as a result of which the execution time on the GPU for the three input test cases are similar.
2. In case of pinned host memory allocation, the CPU implementation is faster for input test cases 1 and 2; however, the GPU implementation is much faster for larger matrix sizes.
3. The performance of the GPU implementations with and without asynchronous memcpy and kernel execution are similar. This is because of the nature of the program- the dominant matmul operations must proceed sequentially.
4. Tiled matrix-matrix-transpose multiplication has some performance improvements over the baseline matmul operation.

### References:
1. Tiled Matrix Transpose: [An Efficient Matrix Transpose in CUDA C/C++](https://developer.nvidia.com/blog/efficient-matrix-transpose-cuda-cc/)
2. Pinned Memory Allocation: [How to Optimize Data Transfers in CUDA C/C++](https://developer.nvidia.com/blog/how-optimize-data-transfers-cuda-cc/)
3. CUDA Streams and Asynchronous Memcpy: [CUDA C/C++: Streams and Concurrency, Steve Rennich (NVIDIA)](https://developer.download.nvidia.com/CUDA/training/StreamsAndConcurrencyWebinar.pdf)