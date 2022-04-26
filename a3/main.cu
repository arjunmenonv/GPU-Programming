/*
GPU Programming Assignment 3
Arjun Menon V, ee18b104
April 5, 2022
*/
#include <stdio.h>
#include <cuda.h>

using namespace std;
#define TRUE 1
#define FALSE 0

__device__ int d_nextTaskIdx;
__device__ int d_clk_cycle;
__device__ int d_done;

__host__ __device__ int h_findFreeCore(int *core_status, int m){
    for (int i = 0; i < m; i++){
        if(core_status[i] == 0){
            return i;
        }
    }
    return -1;
}

void seq_scheduler(int m, int n, int *executionTime, int *priority, int *result){
    // sequential impl of the scheduling algorithm
    int nextTaskIdx = 0;    // points to next Task ID to be allocated, init to 0
    int *coreStatus = (int *)malloc(m*sizeof(int)); //track status of core => free/unallocated: val = 0
    memset(coreStatus, 0, m*sizeof(int));
    // track time for a task on a core to complete; free and unallocted cores hold junk (possibly negative) values here
    signed int *time_to_complete = (int *)malloc(m*sizeof(int)); 
    int *priorityCoreMap = (int *)malloc(m*sizeof(int));
    memset(priorityCoreMap, -1, m*sizeof(int));
    //
    int clk_cycle = 0;
    int done = FALSE;
    
    do{
        int assign_next_task = TRUE;
        while(assign_next_task){
            int task_priority = priority[nextTaskIdx];
            int desig_core = priorityCoreMap[task_priority];
            if (desig_core == -1){ // task with new priority code
                int freeCoreIdx = h_findFreeCore(coreStatus, m);
                if(freeCoreIdx != -1){
                    priorityCoreMap[task_priority] = freeCoreIdx;
                    time_to_complete[freeCoreIdx] = executionTime[nextTaskIdx];
                    coreStatus[freeCoreIdx] = 1;
                }
                result[nextTaskIdx] = clk_cycle + executionTime[nextTaskIdx];
                nextTaskIdx++;
            }
            else if (coreStatus[desig_core] == 0){    // designated core is free 
                time_to_complete[desig_core] = executionTime[nextTaskIdx];
                coreStatus[desig_core] = 1;
                result[nextTaskIdx] = clk_cycle + executionTime[nextTaskIdx];
                nextTaskIdx++;          
            }
            else{
                assign_next_task = FALSE;
            }
            if (nextTaskIdx == n){
                assign_next_task = FALSE;
                done = TRUE;
            }
        }
        clk_cycle++;
        for (int i = 0; i<m; i++){
            time_to_complete[i]--;
        }
        for (int i = 0; i < m; i++){
            if(time_to_complete[i] == 0 && coreStatus[i] != 0){
                coreStatus[i] = 0;
            }
        }
    
    }while(done == FALSE);    
}

__global__ void d_schedulerv1(int n, int *executionTime, int *priority, int *result){
    /*
        Marginal Improvement over the sequential impl where the array traversals during task-freeing is 
        parallelised.
        - The amount of parallelism in this problem is severely limited by the data-blocking constraint on the queue
        - In cases where multiple tasks may be issued in the same cycle, I am using a while-loop to sequentially allo-
          cate them one-by-one. 
        - Tasks can be issued in the same cycle (in parallel) only if all of them have different priorities and map to 
          different cores. The process of determining if a chunk of tasks can be scheduled in parallel is inherently
          sequential as a result- the actual scheduling step isn't costly in comparison- hence, it does not pay off to
          try and parallelise the process

    */
    unsigned int m = blockDim.x;
    unsigned int tid = threadIdx.x;
    extern __shared__ int s[];
    int *coreStatus = s;
    int *time_to_complete = s + m;
    int *priorityCoreMap = s + 2*m;
    coreStatus[tid] = 0;
    priorityCoreMap[tid] = -1;
    if (tid == 0){
        d_nextTaskIdx = 0;
        d_clk_cycle = 0;
        d_done = FALSE;
    }
    __syncthreads();
    
    do{
        //allocate + clk_cycle update in tid0
        if (tid == 0){
            int assign_next_task = TRUE;
            while(assign_next_task){
                int task_priority = priority[d_nextTaskIdx];
                int desig_core = priorityCoreMap[task_priority];
                if (desig_core == -1){
                    int newCore;
                    for (int i = 0; i < m; i++){
                        if(coreStatus[i] == 0){
                            newCore = i;
                            break;
                        }
                    }
                    priorityCoreMap[task_priority] = newCore;
                    time_to_complete[newCore] = executionTime[d_nextTaskIdx];
                    coreStatus[newCore] = 1;
                    result[d_nextTaskIdx] = d_clk_cycle + executionTime[d_nextTaskIdx];
                    d_nextTaskIdx++;
                    
                }
                else if (coreStatus[desig_core] == 0){
                    time_to_complete[desig_core] = executionTime[d_nextTaskIdx];
                    coreStatus[desig_core] = 1;
                    result[d_nextTaskIdx] = d_clk_cycle + executionTime[d_nextTaskIdx];
                    d_nextTaskIdx++;
                }
                else{
                    assign_next_task =  FALSE;
                }
                if (d_nextTaskIdx == n){
                    assign_next_task = FALSE;
                    d_done = TRUE;
                }
            }
            d_clk_cycle++;
        }
        __syncthreads();
        time_to_complete[tid]--;
        if ((time_to_complete[tid] == 0) && (coreStatus[tid] != 0)){
            coreStatus[tid] = 0; 
        }
        __syncthreads();
    }while(d_done == FALSE);
}

void version1(int m, int n, int *executionTime, int *priority, int *result){
    int *d_executionTime, *d_priority, *d_result;
    cudaMalloc(&d_executionTime, n*sizeof(int));
    cudaMalloc(&d_priority, n*sizeof(int));
    cudaMalloc(&d_result, n*sizeof(int));
    cudaMemcpy(d_executionTime, executionTime, n*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_priority, priority, n*sizeof(int), cudaMemcpyHostToDevice);
    //
    d_schedulerv1<<<1, m, (m*3)*sizeof(int)>>>(n, d_executionTime, d_priority, d_result);
    //
    cudaMemcpy(result, d_result, n*sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_executionTime);
    cudaFree(d_priority);
    cudaFree(d_result);
}

void operations ( int m, int n, int *executionTime, int *priority, int *result )  {
    //seq_scheduler(m, n, executionTime, priority, result);
    version1(m, n, executionTime, priority, result);
}

int main(int argc,char **argv)
{
    int m,n;
    //Input file pointer declaration
    FILE *inputfilepointer;
    
    //File Opening for read
    char *inputfilename = argv[1];
    inputfilepointer    = fopen( inputfilename , "r");
    
    //Checking if file ptr is NULL
    if ( inputfilepointer == NULL )  {
        printf( "input.txt file failed to open." );
        return 0; 
    }

    fscanf( inputfilepointer, "%d", &m );      //scaning for number of cores
    fscanf( inputfilepointer, "%d", &n );      //scaning for number of tasks
   
   //Taking execution time and priorities as input	
    int *executionTime = (int *) malloc ( n * sizeof (int) );
    int *priority = (int *) malloc ( n * sizeof (int) );
    for ( int i=0; i< n; i++ )  {
            fscanf( inputfilepointer, "%d", &executionTime[i] );
    }

    for ( int i=0; i< n; i++ )  {
            fscanf( inputfilepointer, "%d", &priority[i] );
    }

    //Allocate memory for final result output 
    int *result = (int *) malloc ( (n) * sizeof (int) );
    for ( int i=0; i<n; i++ )  {
        result[i] = 0;
    }
    
     cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float milliseconds = 0;
    cudaEventRecord(start,0);

    //==========================================================================================================
	

	operations ( m, n, executionTime, priority, result ); 
	
    //===========================================================================================================
    
    
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Time taken by function to execute is: %.6f ms\n", milliseconds);
    
    // Output file pointer declaration
    char *outputfilename = argv[2]; 
    FILE *outputfilepointer;
    outputfilepointer = fopen(outputfilename,"w");

    //Total time of each task: Final Result
    for ( int i=0; i<n; i++ )  {
        fprintf( outputfilepointer, "%d ", result[i]);
    }

    fclose( outputfilepointer );
    fclose( inputfilepointer );
    
    free(executionTime);
    free(priority);
    free(result);
    
    
    
}