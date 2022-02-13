/*
	GPU Programming Assignment 1
	Arjun Menon V, ee18b104
	11 Feb, 2022

	Timing Comparison Results: It is clear from the following numbers that CDS dominates over computation for the 3 test
	cases. The best case speedup of ~15x is obtained for the most compute-intensive test case ('large.txt'), which is much
	smaller than the number of threads launched. 

	- tc1.txt
	|**************************TIMING COMPARISON*****************************|
        Per Row Column Kernel: 3.8e-05s
                Speedup: 0.315789
        Per Column Row Kernel: 2.3e-05s
                Speedup: 0.521739
        Per Element Kernel: 1.9e-05s
                Speedup: 0.631579
        Sequential Loop on CPU: 1.2e-05s
                Speedup: 1
	|************************************************************************|
	
	- small.txt
	|**************************TIMING COMPARISON*****************************|
        Per Row Column Kernel: 3.7e-05s
                Speedup: 0.243243
        Per Column Row Kernel: 2.3e-05s
                Speedup: 0.391304
        Per Element Kernel: 1e-05s
                Speedup: 0.9
        Sequential Loop on CPU: 9e-06s
                Speedup: 1
	|************************************************************************|

	
	- large.txt
	|**************************TIMING COMPARISON*****************************|
        Per Row Column Kernel: 0.021741s
                Speedup: 3.32377
        Per Column Row Kernel: 0.034714s
                Speedup: 2.08164
        Per Element Kernel: 0.00332s
                Speedup: 21.7657
        Sequential Loop on CPU: 0.072262s
                Speedup: 1
	|************************************************************************|
*/

#include <cuda.h>
#include <bits/stdc++.h>
#include <fstream>
#include <time.h>
using namespace std;

ofstream outfile; //the handle for printing the output

// complete the following kernel...
__global__ void per_row_column_kernel(long int *A, long int *B, long int *C,long int m, long int n){
	unsigned int row_num = blockIdx.x*blockDim.x + threadIdx.x;
	if(row_num < m){
		for(unsigned int ii = 0; ii< n; ii++){
			C[row_num*n + ii] = (A[row_num*n + ii] + B[ii*m + row_num])*(B[ii*m + row_num] - A[row_num*n + ii]);
		}
	}
}

// complete the following kernel...
__global__ void per_column_row_kernel(long int *A, long int *B, long int *C,long int m, long int n){
	unsigned int col_num = blockIdx.x*blockDim.x*blockDim.y + threadIdx.x*blockDim.y + threadIdx.y;
	if (col_num < n){
		for(unsigned int ii = 0; ii < m; ii++){
			C[ii*n + col_num] = (A[ii*n + col_num] + B[col_num*m + ii])*(B[col_num*m + ii] - A[ii*n + col_num]);
		}
	}
}

// complete the following kernel...
__global__ void per_element_kernel(long int *A, long int *B, long int *C,long int m, long int n){
	unsigned int i = blockIdx.x*blockDim.x + threadIdx.x;
	unsigned int j = blockIdx.y*blockDim.y + threadIdx.y;
	if ((i < m) && (j < n)){
		C[i*n + j] = (A[i*n + j] + B[j*m + i])*(B[j*m + i] - A[i*n + j]);
	}
}

/**
 * Prints any 1D array in the form of a matrix 
 * */
void printMatrix(long int *arr, long int rows, long int cols, char* filename) {

	outfile.open(filename);
	for(long int i = 0; i < rows; i++) {
		for(long int j = 0; j < cols; j++) {
			outfile<<arr[i * cols + j]<<" ";
		}
		outfile<<"\n";
	}
	outfile.close();
}

int main(int argc,char **argv){

	//variable declarations
	long int m,n;	
	cin>>m>>n;	

	//host_arrays 
	long int *h_a,*h_b,*h_c;

	//device arrays 
	long int *d_a,*d_b,*d_c;

	// clock variables
	clock_t start_rowwise, start_colwise, start_elemwise, start_cpu; 
	double time_rowwise, time_colwise, time_elemwise, time_cpu;

	//Allocating space for the host_arrays 
	h_a = (long int *) malloc(m * n * sizeof(long int));
	h_b = (long int *) malloc(m * n * sizeof(long int));	
	h_c = (long int *) malloc(m * n * sizeof(long int));	

	//Allocating memory for the device arrays 
	cudaMalloc(&d_a, m * n * sizeof(long int));
	cudaMalloc(&d_b, m * n * sizeof(long int));
	cudaMalloc(&d_c, m * n * sizeof(long int));

	//Read the input matrix A 
	for(long int i = 0; i < m * n; i++) {
		cin>>h_a[i];
	}

	//Read the input matrix B 
	for(long int i = 0; i < m * n; i++) {
		cin>>h_b[i];
	}

	//Transfer the input host arrays to the device 
	cudaMemcpy(d_a, h_a, m * n * sizeof(long int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, h_b, m * n * sizeof(long int), cudaMemcpyHostToDevice);

	long int gridDimx, gridDimy;
	//Launch the kernels 
	/**
	 * Kernel 1 - per_row_column_kernel
	 * To be launched with 1D grid, 1D block
	 * */
	gridDimx = ceil(float(m) / 1024);
	dim3 grid1(gridDimx,1,1);
	dim3 block1(1024,1,1);
	start_rowwise = clock();
	per_row_column_kernel<<<grid1,block1>>>(d_a,d_b,d_c,m,n);
	cudaDeviceSynchronize();
	time_rowwise = ((double)(clock() - start_rowwise))/CLOCKS_PER_SEC;
	cudaMemcpy(h_c, d_c, m * n * sizeof(long int), cudaMemcpyDeviceToHost);
	printMatrix(h_c, m, n,"kernel1.txt");
	
	/**
	 * Kernel 2 - per_column_row_kernel
	 * To be launched with 1D grid, 2D block
	 * */
	gridDimx = ceil(float(n) / 1024);
	dim3 grid2(gridDimx,1,1);
	dim3 block2(32,32,1);
	start_colwise = clock();
	per_column_row_kernel<<<grid2,block2>>>(d_a,d_b,d_c,m,n);
	cudaDeviceSynchronize();
	time_colwise = ((double)(clock() - start_colwise))/CLOCKS_PER_SEC;
	cudaMemcpy(h_c, d_c, m * n * sizeof(long int), cudaMemcpyDeviceToHost);
	printMatrix(h_c, m, n,"kernel2.txt");

	/**
	 * Kernel 3 - per_element_kernel
	 * To be launched with 2D grid, 2D block
	 * */
	gridDimx = ceil(float(m) / 64);
	gridDimy = ceil(float(n) / 16);
	dim3 grid3(gridDimx,gridDimy,1);
	dim3 block3(64,16,1);
	start_elemwise = clock();
	per_element_kernel<<<grid3,block3>>>(d_a,d_b,d_c,m,n);
	cudaDeviceSynchronize();
	time_elemwise = ((double)(clock() - start_elemwise))/CLOCKS_PER_SEC;
	cudaMemcpy(h_c, d_c, m * n * sizeof(long int), cudaMemcpyDeviceToHost);
	printMatrix(h_c, m, n,"kernel3.txt");

	start_cpu = clock();
	for(int ii= 0; ii < m; ii++){
		for(int jj= 0; jj< n; jj++){
			h_c[ii*n + jj] = (h_a[ii*n + jj] + h_b[jj*m + ii])*(h_b[jj*m + ii] - h_a[ii*n + jj]);
		}
	}
	time_cpu = ((double)(clock() - start_cpu))/CLOCKS_PER_SEC;
	printMatrix(h_c, m, n,"cpu_seqloop.txt");

	cout<<"|**************************TIMING COMPARISON*****************************|\n";
	cout<<"\tPer Row Column Kernel: "<<time_rowwise<<"s\n\t\tSpeedup: "<<time_cpu/time_rowwise<<endl;
	cout<<"\tPer Column Row Kernel: "<<time_colwise<<"s\n\t\tSpeedup: "<<time_cpu/time_colwise<<endl;
	cout<<"\tPer Element Kernel: "<<time_elemwise<<"s\n\t\tSpeedup: "<<time_cpu/time_elemwise<<endl;
	cout<<"\tSequential Loop on CPU: "<<time_cpu<<"s\n\t\tSpeedup: "<<time_cpu/time_cpu<<endl;
	cout<<"|************************************************************************|\n";
	return 0;
}
