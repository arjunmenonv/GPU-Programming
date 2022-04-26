/*
	GPU Programming
	Assignment 4: Train-ticket booking application
	Arjun Menon Vadakkeveedu, EE18B104

	To ensure synchronization across thread-blocks, I am using multiple kernels that separate the synchronization boundaries.
	I chose not to use cooperative_groups::this_grid().sync() as this requires using the API call for kernel launch, a 
	consequence of which is that all the kernel inputs must be pointers (I found this a bit cumbersome for the application)
*/

#include <stdio.h>
#include <cuda.h>
#include <stdlib.h>
#include <iostream>

using namespace std;

#define MAX_CLASSES 25
#define MAX_STOPS 50 		// = max(abs(src-dest))
#define MAX_THREADS_PER_BLOCK 1024

__global__ void initGlobalVars(int *max_timeslot, int *sold_seats){
	*max_timeslot = 0;
	*sold_seats = 0;
}

__global__ void scan_and_scheduleReqs(int *reqTrain, int *reqClass, int numReqs, int *timeslots, int *max_timeslot){
	unsigned int tid = blockIdx.x*blockDim.x + threadIdx.x;
	if (tid < numReqs){
		timeslots[tid] = 0;
		for (int off = 0; off < numReqs; off++){
			if (tid > off){
				if((reqTrain[tid] == reqTrain[off]) && (reqClass[tid] == reqClass[off])){
					timeslots[tid]++;
				}
			}
		}
		atomicMax(max_timeslot, timeslots[tid]);	
	}
}

__global__ void processReqGPU(int *trainStatus, int *trainSrc, int *trainDest, int *reqTrain, int *reqClass, int *reqSrc, 
					int *reqDest, int *reqNumSeats, int numReqs, int *timeslots, int *req_success, int time_step, int *sold_seats){
	//
	unsigned int tid = blockIdx.x*blockDim.x + threadIdx.x;
	if (tid < numReqs){
	int trainID = reqTrain[tid];
	int classNum = reqClass[tid];
	int srcNum = reqSrc[tid];
	int destNum = reqDest[tid];
	int numSeats = reqNumSeats[tid];
	int legit_req = 0;
	int vacant = 1;
	int array_idx_src;
	int array_idx_dest;
	//
	if (timeslots[tid] == time_step){
		if (trainSrc[trainID] < trainDest[trainID]){
			legit_req = ((srcNum >= trainSrc[trainID]) && (destNum <= trainDest[trainID])) ? 1 : 0;
			array_idx_src = srcNum - trainSrc[trainID];
			array_idx_dest = destNum - trainSrc[trainID];
			for (int j = array_idx_src; j < array_idx_dest; j++){
				if (trainStatus[trainID*MAX_CLASSES*MAX_STOPS + classNum*MAX_STOPS + j] < numSeats){
					vacant = 0;
					break;
				}
			}
		}
		else{
			legit_req = ((srcNum <= trainSrc[trainID]) && (destNum >= trainDest[trainID])) ? 1 : 0;
			array_idx_src = srcNum - trainDest[trainID];
			array_idx_dest = destNum - trainDest[trainID];
			for (int j = array_idx_dest; j < array_idx_src; j++){
				if (trainStatus[trainID*MAX_CLASSES*MAX_STOPS + classNum*MAX_STOPS + j] < numSeats){
					vacant = 0;
					break;
				}
			}
		}
		if (legit_req && vacant){
			if (trainSrc[trainID] < trainDest[trainID]){
				for (int j = array_idx_src; j < array_idx_dest; j++){
					trainStatus[trainID*MAX_CLASSES*MAX_STOPS + classNum*MAX_STOPS + j] -= numSeats;
				}
			}
			else{
				for (int j = array_idx_dest; j < array_idx_src; j++){
					trainStatus[trainID*MAX_CLASSES*MAX_STOPS + classNum*MAX_STOPS + j] -= numSeats;
				}
			}
			req_success[tid] = 1;
			atomicAdd(sold_seats, abs(srcNum - destNum)*numSeats);
		}
	}
	}
}

int main() {
	unsigned int numTrains;
	/*
		Read Config of Trains and Classes:
			- "static", one-time effort and not performed during "runtime" (processing of batches)
			- ok to do sequentially
	*/
	cin >> numTrains;
	int *h_trainStatus = (int *)malloc(numTrains*MAX_CLASSES*MAX_STOPS*sizeof(int));
	int *d_trainStatus;
	cudaMalloc(&d_trainStatus, numTrains*MAX_CLASSES*MAX_STOPS*sizeof(int));
	memset(h_trainStatus, -1, numTrains*MAX_CLASSES*MAX_STOPS*sizeof(int));			// init to -1 => invalid status

	int *h_src, *h_dest;
	int *d_src, *d_dest;
	h_src = (int *)malloc(numTrains*sizeof(int));
	h_dest = (int *)malloc(numTrains*sizeof(int));
	//
	cudaMalloc(&d_src, numTrains*sizeof(int));
	cudaMalloc(&d_dest, numTrains*sizeof(int));

	for (int i = 0; i < numTrains; i++){
		int train_id, num_classes;
		cin >> train_id >> num_classes >> h_src[i] >> h_dest[i];
		int trainNumStops = abs(h_src[i] - h_dest[i]);
		for (int j = 0; j < num_classes; j++){
			int class_id, class_capacity;
			cin >> class_id >> class_capacity;
			for (int k = 0; k < trainNumStops; k++){
				h_trainStatus[i*MAX_CLASSES*MAX_STOPS + j*MAX_STOPS + k] = class_capacity;
			}
		}
	}
	
	cudaMemcpy(d_trainStatus, h_trainStatus, numTrains*MAX_CLASSES*MAX_STOPS*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_src, h_src, numTrains*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_dest, h_dest, numTrains*sizeof(int), cudaMemcpyHostToDevice);
	/*	
		Process Batches one by one
	*/
	unsigned int numBatches;
	int reqsInBatch;
	int *h_reqTrain, *h_reqClass, *h_reqSrc, *h_reqDest, *h_reqNumSeats;
	int *d_reqTrain, *d_reqClass, *d_reqSrc, *d_reqDest, *d_reqNumSeats;
	int *d_timeslots, *d_success;
	int *h_timeslots, *h_success;

	cin >> numBatches;
	for (int i = 0; i < numBatches; i++){
		cin >> reqsInBatch;
		int *h_max_timeslot = (int *)malloc(sizeof(int));
		int *h_sold_seats = (int *)malloc(sizeof(int));
		int *d_max_timeslot, *d_sold_seats;
		//
		h_reqTrain = (int *)malloc(reqsInBatch*sizeof(int));
		h_reqClass = (int *)malloc(reqsInBatch*sizeof(int));
		h_reqSrc = (int *)malloc(reqsInBatch*sizeof(int));
		h_reqDest = (int *)malloc(reqsInBatch*sizeof(int));
		h_reqNumSeats = (int *)malloc(reqsInBatch*sizeof(int));
		//
		h_success = (int *)malloc(reqsInBatch*sizeof(int));
		h_timeslots = (int *)malloc(reqsInBatch*sizeof(int));
		//
		cudaMalloc(&d_reqTrain, reqsInBatch*sizeof(int));
		cudaMalloc(&d_reqClass, reqsInBatch*sizeof(int));
		cudaMalloc(&d_reqSrc, reqsInBatch*sizeof(int));
		cudaMalloc(&d_reqDest, reqsInBatch*sizeof(int));		
		cudaMalloc(&d_reqNumSeats, reqsInBatch*sizeof(int));		
		//
		cudaMalloc(&d_timeslots, reqsInBatch*sizeof(int));
		cudaMalloc(&d_success, reqsInBatch*sizeof(int));
		cudaMalloc(&d_max_timeslot, sizeof(int));
		cudaMalloc(&d_sold_seats, sizeof(int));

		for (int j = 0; j < reqsInBatch; j++){
			int req_id;
			cin >> req_id >> h_reqTrain[j] >> h_reqClass[j] >> h_reqSrc[j] >> h_reqDest[j] >> h_reqNumSeats[j];
		}
		// calc launch config based on number of requests
		int numBlocks, threadsPerBlock;
		if (reqsInBatch < MAX_THREADS_PER_BLOCK){
			numBlocks = 1;
			threadsPerBlock = reqsInBatch;
		}
		else{
			numBlocks = ceil((float)reqsInBatch/MAX_THREADS_PER_BLOCK);
			threadsPerBlock = MAX_THREADS_PER_BLOCK;
		}
		/*
		 launch kernels:
		 	- init global variables used to track Processing State 
		  	- scan+sort of reqsInBatch to schedule them appropriately
			- process requests + update trainStatus buffer
		*/
		cudaMemcpy(d_reqTrain, h_reqTrain, reqsInBatch*sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(d_reqClass, h_reqClass, reqsInBatch*sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(d_reqSrc, h_reqSrc, reqsInBatch*sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(d_reqDest, h_reqDest, reqsInBatch*sizeof(int), cudaMemcpyHostToDevice);
		cudaMemcpy(d_reqNumSeats, h_reqNumSeats, reqsInBatch*sizeof(int), cudaMemcpyHostToDevice);
		//
		cudaMemset(d_success, 0, reqsInBatch*sizeof(int));

		initGlobalVars<<<1,1>>>(d_max_timeslot, d_sold_seats);
		scan_and_scheduleReqs<<<numBlocks, threadsPerBlock>>>(d_reqTrain, d_reqClass, reqsInBatch, d_timeslots, d_max_timeslot);
		cudaMemcpy(h_max_timeslot, d_max_timeslot, sizeof(int), cudaMemcpyDeviceToHost);
		cudaMemcpy(h_timeslots, d_timeslots, reqsInBatch*sizeof(int), cudaMemcpyDeviceToHost);
		for (int j = 0; j <= *h_max_timeslot; j++){
			processReqGPU<<<numBlocks, threadsPerBlock, threadsPerBlock*sizeof(unsigned)>>>(d_trainStatus, d_src, d_dest, 
										d_reqTrain, d_reqClass, d_reqSrc, d_reqDest, d_reqNumSeats, reqsInBatch,
										d_timeslots, d_success, j, d_sold_seats);
		
		}
		cudaMemcpy(h_sold_seats, d_sold_seats, sizeof(int), cudaMemcpyDeviceToHost);
		cudaMemcpy(h_success, d_success, reqsInBatch*sizeof(int), cudaMemcpyDeviceToHost);
		//
		int num_success = 0;
		for (int j = 0; j < reqsInBatch; j++){
			if (h_success[j] == 1){
				num_success++;
				cout << "success" << endl;
			}
			else{
				cout << "failure" << endl;
			}
		}
		cout << num_success << " " << reqsInBatch - num_success << endl;
		cout << *h_sold_seats << endl;
		//
		free(h_reqTrain);
		free(h_reqSrc);
		free(h_reqDest);
		free(h_reqNumSeats);
		//
		free(h_max_timeslot);
		free(h_sold_seats);
		free(h_success);
		free(h_timeslots);
		//
		cudaFree(d_reqTrain);
		cudaFree(d_reqClass);
		cudaFree(d_reqSrc);
		cudaFree(d_reqDest);
		cudaFree(d_reqNumSeats);
		//
		cudaFree(d_timeslots);
		cudaFree(d_success);
		cudaFree(d_max_timeslot);
		cudaFree(d_sold_seats);
	}
	free(h_trainStatus);
	free(h_src);
	free(h_dest);
	//
	cudaFree(d_trainStatus);
	cudaFree(d_src);
	cudaFree(d_dest);
	return 0;
}