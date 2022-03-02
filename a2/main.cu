/*
	GPU Programming Assignment 2
	Arjun Menon V, EE18B104
	2 March 2022

	This code contains the following implementations of the matrix operation (A + B.T)CD.T
	(.T indicates matrix transform):
		1. Implementation on CPU 
		2. Baseline GPU Implementation using separate kernels for matrix transpose (D.T operation), matrix add 
		   (A + B.T operation) and Matrix-Matrix Multiply (C X D.T and (A+B.T) X CD.T operations). The Matrix-
		   Matrix Multiply is performed by parallelizing the nested for loop implementation of sequential matmul
		3. GPU Implementation with separate kernels for matrix add (A + B.T operation) and Tiled Matrix-Matrix-Transpose
		   Multiply (C x D.T operation). The second kernel is also used for the final matmul operation ((A+B.T) X CD.T);
		   this is achieved by switching the ordering of the CD.T operation:
		   		Y = (A + B.T) x C x D.T 
				  = (A + B.T) x (DC.T).T
				  = MMTransposeKernel(MMAddKernel(A, B), MMTransposeKernel(D, C))
	
	Implementations 2 and 3 utilise shared memory wherever applicable, use coalesced accesses and avoid shared memory bank conflicts

	Further Optimizations:
	1. Using Pinned Memory for Host Matrices reduced the overhead associated with pageable memcpy from host. The macro USE_PINNED_MEMORY
	   is used to determine if the host matrices are to be allocated using malloc() (pageable) or using cudaMallocHost() (pinned)
	2. Using Multiple Streams to perform the independent kernel operations: In Implementations 2 and 3, the A+B.T operation 
	   can be interleaved/ performed concurrently with the CD.T operation. COMPUTE_MODE 2 and 4 use 2 cuda streams to launch
	   the corresponding kernels. However, the dominant operations in terms of compute time are the matrix multiplies, 
	   which cannot be performed asynchronously; hence, the execution time does not improve dramatically with multiple streams.
*/

#include<iostream>
#include<sys/time.h>
#include<cuda.h>
#include<stdlib.h>
using namespace std;

#define TILE_SIZE 32
#define COMPUTE_MODE 4
#define USE_PINNED_MEMORY 1
/*
	COMPUTE_MODE:
		0 => cpu_impl
		1 => baseline GPU implementation
		2 => baseline with async memcpy and async execution, using pinned memory
		3 => Tiled Matrix-Matrix Transpose Multiply on null stream
		4 => Asynchronous Tiled Matrix-Matrix Transpose Multiply on 2 streams
*/

// write kernels here...
__global__ void dTileTranspose(int rows, int cols, int *matrix, int *matrixT){
	__shared__ int mat_tile[TILE_SIZE][TILE_SIZE];
	unsigned int id_x = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int id_y = blockIdx.y*blockDim.y + threadIdx.y; 
	
	if ((id_x < cols) && (id_y < rows)){
		mat_tile[threadIdx.y][threadIdx.x] = matrix[id_y*cols + id_x];
	}
	__syncthreads();
	unsigned int id_xT = blockIdx.y*blockDim.y + threadIdx.x;
	unsigned int id_yT = blockIdx.x*blockDim.x + threadIdx.y;
	if ((id_xT < rows) && (id_yT < cols)){
		matrixT[id_yT*rows + id_xT] = mat_tile[threadIdx.x][threadIdx.y];
	}	
}

__global__ void dApBT(int rows, int cols, int *matrixA, int *matrixB, int *matrixApBT){
	// Kernel to perform the operation A + B.T
	__shared__ int mat_tile[TILE_SIZE][TILE_SIZE];
	unsigned int id_x = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int id_y = blockIdx.y*blockDim.y + threadIdx.y; 
	
	if ((id_x < cols) && (id_y < rows)){
		mat_tile[threadIdx.y][threadIdx.x] = matrixB[id_y*cols + id_x];
	}
	__syncthreads();
	unsigned int id_xT = blockIdx.y*blockDim.y + threadIdx.x;
	unsigned int id_yT = blockIdx.x*blockDim.x + threadIdx.y;
	if ((id_xT < rows) && (id_yT < cols)){
		matrixApBT[id_yT*rows + id_xT] = mat_tile[threadIdx.x][threadIdx.y] + matrixA[id_yT*rows + id_xT];
	}		
}

__global__ void dmatmul(int left_cols, int *leftMatrix, int *rightMatrix, int *outMatrix){
	unsigned int id = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int ii = id/blockDim.x;
	unsigned int jj = id % blockDim.x;
	outMatrix[ii*blockDim.x + jj] = 0;
	for (unsigned int kk = 0; kk < left_cols; kk++){
		outMatrix[ii*blockDim.x + jj] += leftMatrix[ii*left_cols + kk]*rightMatrix[kk*blockDim.x + jj];
	}
}

__global__ void dTileAxBT(int left_rows, int right_rows, int inp_cols, int *leftMatrix, int *rightMatrix, int *outMatrix){
	/* 
        Tiled version of AB.T 
        leftMatrix: left_rows x inp_cols; rightMatrix: right_rows x inp_cols
        outMatrix: left_rows x right_rows
    */
	__shared__ int aTile[TILE_SIZE][TILE_SIZE];
	__shared__ int bTile[TILE_SIZE][TILE_SIZE];
	__shared__ int prodTile[TILE_SIZE][TILE_SIZE];
	
	unsigned int left_id = (blockIdx.y*TILE_SIZE + threadIdx.y)*inp_cols + threadIdx.x;
	unsigned int right_id = (blockIdx.x*TILE_SIZE + threadIdx.y)*inp_cols + threadIdx.x;
    
	prodTile[threadIdx.y][threadIdx.x] = 0;		//initialise to 0
	for (int j = 0; j < inp_cols; j += TILE_SIZE){
        if (blockIdx.y*TILE_SIZE + threadIdx.y < left_rows && threadIdx.x + j < inp_cols){
			aTile[threadIdx.y][threadIdx.x] = leftMatrix[left_id + j];
        }
        else{
            aTile[threadIdx.y][threadIdx.x] = 0;
        }
        
        if (blockIdx.x*TILE_SIZE + threadIdx.y < right_rows && threadIdx.x + j < inp_cols){
			bTile[threadIdx.y][threadIdx.x] = rightMatrix[right_id + j];
		}
        else{
            bTile[threadIdx.y][threadIdx.x] = 0;
        }
        __syncthreads();
        
        for (int k = 0; k < TILE_SIZE; k++){
			prodTile[threadIdx.y][threadIdx.x] += aTile[threadIdx.y][k]*bTile[threadIdx.x][k];
		}
		__syncthreads();
	}    
	unsigned int outid_x = blockIdx.x*TILE_SIZE + threadIdx.x;
	unsigned int outid_y = blockIdx.y*TILE_SIZE + threadIdx.y;
	
	if (outid_x < right_rows && outid_y < left_rows){
		outMatrix[outid_y*right_rows + outid_x] = prodTile[threadIdx.y][threadIdx.x];
	}
}

// CPU implementation:
void cpu_impl(int p, int q, int r, int s, int *h_matrixA, int *h_matrixB, 
	         int *h_matrixC, int *h_matrixD, int *h_matrixX){
	int *h_temp = (int *)calloc(p*r, sizeof(int));
	memset(h_matrixX, 0, p * s * sizeof(int));

	for (int i = 0; i < p; i++){
		for (int j = 0; j < r; j++){
			for  (int k = 0; k < q; k++){
				h_temp[i*r+ j] += (h_matrixA[i*q + k] + h_matrixB[k*p + i])*(h_matrixC[k*r+j]);
			}
		}
	}

	for (int i = 0; i < p; i++){
		for (int j = 0; j < s; j++){
			for (int k = 0; k < r; k++){
				h_matrixX[i*s + j] += h_temp[i*r+k]*h_matrixD[j*r + k];
			}
		}
	}
	free(h_temp);
}


// function to compute the output matrix
void compute(int p, int q, int r, int s, int *h_matrixA, int *h_matrixB, 
	         int *h_matrixC, int *h_matrixD, int *h_matrixX) {
	cout<<"Pinned Memory Setting: "<<USE_PINNED_MEMORY<<endl;
	if (COMPUTE_MODE == 0){
		cout<<"CPU Implementation..."<<endl;
		cpu_impl(p, q, r, s, h_matrixA, h_matrixB, h_matrixC, h_matrixD, h_matrixX);
	}
	else{
		int *d_matrixA, *d_matrixB, *d_matrixC, *d_matrixD, *d_matrixX;
		cudaMalloc(&d_matrixA, p * q * sizeof(int));
		cudaMalloc(&d_matrixB, q * p * sizeof(int));
		cudaMalloc(&d_matrixC, q * r * sizeof(int));
		cudaMalloc(&d_matrixD, s * r * sizeof(int));
		cudaMalloc(&d_matrixX, p * s * sizeof(int));
		if (COMPUTE_MODE == 1 || (COMPUTE_MODE == 2 && USE_PINNED_MEMORY == 0)){
			cout<<"Baseline GPU Implementation..."<<endl;
			int *d_Dtransp, *d_ApBT, *d_CDT;
			// variable declarations...
			dim3 grid_transpD(ceil((float)r/TILE_SIZE), ceil((float)s/TILE_SIZE), 1);
			dim3 block_transpD(TILE_SIZE, TILE_SIZE, 1);
			dim3 grid_ApBT(ceil((float)p/TILE_SIZE), ceil((float)q/TILE_SIZE), 1);
			// allocate memory...
			cudaMalloc(&d_Dtransp, s * r * sizeof(int));
			cudaMalloc(&d_ApBT, p * q * sizeof(int));
			cudaMalloc(&d_CDT, q * s * sizeof(int));
			// copy the values...
			cudaMemcpy(d_matrixD, h_matrixD, s * r * sizeof(int), cudaMemcpyHostToDevice);
			cudaMemcpy(d_matrixA, h_matrixA, p * q * sizeof(int), cudaMemcpyHostToDevice);
			cudaMemcpy(d_matrixB, h_matrixB, q * p * sizeof(int), cudaMemcpyHostToDevice);
			cudaMemcpy(d_matrixC, h_matrixC, q * r * sizeof(int), cudaMemcpyHostToDevice);	
			// call the kernels for doing required computations...
			dTileTranspose<<<grid_transpD, block_transpD>>>(s, r, d_matrixD, d_Dtransp);
			dApBT<<<grid_ApBT, block_transpD>>>(q, p, d_matrixA, d_matrixB, d_ApBT);
			dmatmul<<<q, s>>>(r, d_matrixC, d_Dtransp, d_CDT);
			dmatmul<<<p, s>>>(q, d_ApBT, d_CDT, d_matrixX);
			// copy the result back...
			cudaMemcpy(h_matrixX, d_matrixX, p * s * sizeof(int), cudaMemcpyDeviceToHost);
			cudaFree(d_Dtransp);
			cudaFree(d_ApBT);
			cudaFree(d_CDT);
		}		
		else if (COMPUTE_MODE == 2 && USE_PINNED_MEMORY == 1){
			//Async Memcpy requires pinned memory on host
			cout<<"GPU Baseline with Asynch Memcpy and Kernel Execution"<<endl;
			
			int *d_Dtransp, *d_ApBT, *d_CDT;
			// variable declarations...
			dim3 grid_transpD(ceil((float)r/TILE_SIZE), ceil((float)s/TILE_SIZE), 1);
			dim3 block_transpD(TILE_SIZE, TILE_SIZE, 1);
			dim3 grid_ApBT(ceil((float)p/TILE_SIZE), ceil((float)q/TILE_SIZE), 1);
			// allocate memory...
			cudaMalloc(&d_Dtransp, s * r * sizeof(int));
			cudaMalloc(&d_ApBT, p * q * sizeof(int));
			cudaMalloc(&d_CDT, q * s * sizeof(int));
			// create streams
			cudaStream_t stream1, stream2;
			cudaStreamCreate(&stream1);
			cudaStreamCreate(&stream2);
			// async memcpy + kernel calls
			cudaMemcpyAsync(d_matrixD, h_matrixD, s * r * sizeof(int), cudaMemcpyHostToDevice, stream1);
			dTileTranspose<<<grid_transpD, block_transpD, 0, stream1>>>(s, r, d_matrixD, d_Dtransp);
			cudaMemcpyAsync(d_matrixA, h_matrixA, p * q * sizeof(int), cudaMemcpyHostToDevice, stream2);
			cudaMemcpyAsync(d_matrixB, h_matrixB, q * p * sizeof(int), cudaMemcpyHostToDevice, stream2);
			dApBT<<<grid_ApBT, block_transpD, 0, stream2>>>(q, p, d_matrixA, d_matrixB, d_ApBT);
			cudaMemcpyAsync(d_matrixC, h_matrixC, q * r * sizeof(int), cudaMemcpyHostToDevice, stream1);	
			dmatmul<<<q, s, 0, stream1>>>(r, d_matrixC, d_Dtransp, d_CDT);
			dmatmul<<<p, s>>>(q, d_ApBT, d_CDT, d_matrixX);	// in null stream, kernel launched only after previous kernels are completed
			cudaMemcpy(h_matrixX, d_matrixX, p * s * sizeof(int), cudaMemcpyDeviceToHost);
			//
			cudaFree(d_Dtransp);
			cudaFree(d_ApBT);
			cudaFree(d_CDT);
			cudaStreamDestroy(stream1);
			cudaStreamDestroy(stream2);
		}
		else if (COMPUTE_MODE == 3 || (COMPUTE_MODE == 4 && USE_PINNED_MEMORY == 0)){
			//Synchronous Kernel Execution with Tiled Multiply
			cout<<"Tiled Matrix-Matrix Transpose Multiply"<<endl;
			//
			int *d_ApBT, *d_DCT;
			// variable declarations...
			dim3 grid_DCT(ceil((float)q/TILE_SIZE), ceil((float)s/TILE_SIZE), 1);
			dim3 grid_ApBT(ceil((float)p/TILE_SIZE), ceil((float)q/TILE_SIZE), 1);
			dim3 grid_lastmatmul(ceil((float)s/TILE_SIZE), ceil((float)p/TILE_SIZE), 1);
			dim3 block_tiled(TILE_SIZE, TILE_SIZE, 1);
			// allocate memory...
			cudaMalloc(&d_ApBT, p * q * sizeof(int));
			cudaMalloc(&d_DCT, q * s * sizeof(int));
			// copy the values...
			cudaMemcpy(d_matrixD, h_matrixD, s * r * sizeof(int), cudaMemcpyHostToDevice);
			cudaMemcpy(d_matrixA, h_matrixA, p * q * sizeof(int), cudaMemcpyHostToDevice);
			cudaMemcpy(d_matrixB, h_matrixB, q * p * sizeof(int), cudaMemcpyHostToDevice);
			cudaMemcpy(d_matrixC, h_matrixC, q * r * sizeof(int), cudaMemcpyHostToDevice);	
			// call the kernels for doing required computations...
			dTileAxBT<<<grid_DCT, block_tiled>>>(s, q, r, d_matrixD, d_matrixC, d_DCT);
			dApBT<<<grid_ApBT, block_tiled>>>(q, p, d_matrixA, d_matrixB, d_ApBT);
			dTileAxBT<<<grid_lastmatmul, block_tiled>>>(p, s, q, d_ApBT, d_DCT, d_matrixX);
			// copy the result back...
			cudaMemcpy(h_matrixX, d_matrixX, p * s * sizeof(int), cudaMemcpyDeviceToHost);
			cudaFree(d_ApBT);
			cudaFree(d_DCT);
		}
		else if (COMPUTE_MODE == 4 && USE_PINNED_MEMORY == 1){
			//Asynchronous Kernel Execution with Tiled Multiply
			cout<<"Async Tiled Matrix-Matrix Transpose Multiply"<<endl;
			//
			int *d_ApBT, *d_DCT;
			// variable declarations...
			dim3 grid_DCT(ceil((float)q/TILE_SIZE), ceil((float)s/TILE_SIZE), 1);
			dim3 grid_ApBT(ceil((float)p/TILE_SIZE), ceil((float)q/TILE_SIZE), 1);
			dim3 grid_lastmatmul(ceil((float)s/TILE_SIZE), ceil((float)p/TILE_SIZE), 1);
			dim3 block_tiled(TILE_SIZE, TILE_SIZE, 1);
			// allocate memory...
			cudaMalloc(&d_ApBT, p * q * sizeof(int));
			cudaMalloc(&d_DCT, q * s * sizeof(int));
			// create streams
			cudaStream_t stream1, stream2;
			cudaStreamCreate(&stream1);
			cudaStreamCreate(&stream2);
			//Async Memcpy and Kernel Execution
			cudaMemcpyAsync(d_matrixD, h_matrixD, s * r * sizeof(int), cudaMemcpyHostToDevice, stream1);
			cudaMemcpyAsync(d_matrixC, h_matrixC, q * r * sizeof(int), cudaMemcpyHostToDevice, stream1);
			dTileAxBT<<<grid_DCT, block_tiled, 0, stream1>>>(s, q, r, d_matrixD, d_matrixC, d_DCT);
			cudaMemcpyAsync(d_matrixA, h_matrixA, p * q * sizeof(int), cudaMemcpyHostToDevice, stream2);
			cudaMemcpyAsync(d_matrixB, h_matrixB, q * p * sizeof(int), cudaMemcpyHostToDevice, stream2);
			dApBT<<<grid_ApBT, block_tiled, 0, stream2>>>(q, p, d_matrixA, d_matrixB, d_ApBT);
			dTileAxBT<<<grid_lastmatmul, block_tiled>>>(p, s, q, d_ApBT, d_DCT, d_matrixX);	// null stream: implicit kernel sync 
			// copy the result back...
			cudaMemcpy(h_matrixX, d_matrixX, p * s * sizeof(int), cudaMemcpyDeviceToHost);
			cudaFree(d_ApBT);
			cudaFree(d_DCT);
			cudaStreamDestroy(stream1);
			cudaStreamDestroy(stream2);
		}
		cudaFree(d_matrixA);
		cudaFree(d_matrixB);
		cudaFree(d_matrixC);
		cudaFree(d_matrixD);
		cudaFree(d_matrixX);
	}
}


// function to read the input matrices from the input file
void readMatrix(FILE *inputFilePtr, int *matrix, int rows, int cols) {
	for(int i=0; i<rows; i++) {
		for(int j=0; j<cols; j++) {
			fscanf(inputFilePtr, "%d", &matrix[i*cols+j]);
		}
	}
}

// function to write the output matrix into the output file
void writeMatrix(FILE *outputFilePtr, int *matrix, int rows, int cols) {
	for(int i=0; i<rows; i++) {
		for(int j=0; j<cols; j++) {
			fprintf(outputFilePtr, "%d ", matrix[i*cols+j]);
		}
		fprintf(outputFilePtr, "\n");
	}
}

int main(int argc, char **argv) {
	// variable declarations
	int p, q, r, s;
	int *matrixA, *matrixB, *matrixC, *matrixD, *matrixX;
	struct timeval t1, t2;
	double seconds, microSeconds;

	// get file names from command line
	char *inputFileName = argv[1];
	char *outputFileName = argv[2];

	// file pointers
	FILE *inputFilePtr, *outputFilePtr;
    
    inputFilePtr = fopen(inputFileName, "r");
	if(inputFilePtr == NULL) {
	    printf("Failed to open the input file.!!\n"); 
		return 0;
	}

	// read input values
	fscanf(inputFilePtr, "%d %d %d %d", &p, &q, &r, &s);

	// allocate memory and read input matrices
	if (USE_PINNED_MEMORY == 0){
		matrixA = (int*) malloc(p * q * sizeof(int));
		matrixB = (int*) malloc(q * p * sizeof(int));
		matrixC = (int*) malloc(q * r * sizeof(int));
		matrixD = (int*) malloc(s * r * sizeof(int));
	}
	else{
		cudaMallocHost(&matrixA, p * q * sizeof(int));
		cudaMallocHost(&matrixB, q * p * sizeof(int));
		cudaMallocHost(&matrixC, q * r * sizeof(int));
		cudaMallocHost(&matrixD, s * r * sizeof(int));
	}
	readMatrix(inputFilePtr, matrixA, p, q);
	readMatrix(inputFilePtr, matrixB, q, p);
	readMatrix(inputFilePtr, matrixC, q, r);
	readMatrix(inputFilePtr, matrixD, s, r);

	// allocate memory for output matrix
	if (USE_PINNED_MEMORY == 0){
		matrixX = (int*) malloc(p * s * sizeof(int));
	}
	else{
		cudaMallocHost(&matrixX, p * s * sizeof(int));
	}

	// call compute function to get the output matrix. it is expected that 
	// the compute function will store the result in matrixX.
	gettimeofday(&t1, NULL);
	compute(p, q, r, s, matrixA, matrixB, matrixC, matrixD, matrixX);
	cudaDeviceSynchronize();
	gettimeofday(&t2, NULL);

	// print the time taken by the compute function
	seconds = t2.tv_sec - t1.tv_sec;
	microSeconds = t2.tv_usec - t1.tv_usec;
	printf("Time taken (ms): %.3f\n", 1000*seconds + microSeconds/1000);

	// store the result into the output file
	outputFilePtr = fopen(outputFileName, "w");
	writeMatrix(outputFilePtr, matrixX, p, s);

	// close files
    fclose(inputFilePtr);
    fclose(outputFilePtr);

	// deallocate memory
	if (USE_PINNED_MEMORY == 0){
		free(matrixA);
		free(matrixB);
		free(matrixC);
		free(matrixD);
		free(matrixX);
	}
	else{
		cudaFreeHost(matrixA);
		cudaFreeHost(matrixB);
		cudaFreeHost(matrixC);
		cudaFreeHost(matrixD);
		cudaFreeHost(matrixX);
	}
	return 0;
}