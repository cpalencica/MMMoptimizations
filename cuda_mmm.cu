/* INCLUDE STATEMENTS */
#include <cstdio>
#include <cstdlib>
#include <math.h>
#include <time.h>

/* DEFINE STATEMENTS */
#define PRINT_TIME         1
#define TOL            .1
#define OMEGA          1.00
#define ITER           2000
#define TILE_WIDTH 16

/* TIMER FUNCTION */
double interval(struct timespec start, struct timespec end)
{
    struct timespec temp;
    temp.tv_sec = end.tv_sec - start.tv_sec;
    temp.tv_nsec = end.tv_nsec - start.tv_nsec;
    if (temp.tv_nsec < 0)
    {
        temp.tv_sec = temp.tv_sec - 1;
        temp.tv_nsec = temp.tv_nsec + 1000000000;
    }
    return (((double)temp.tv_sec) + ((double)temp.tv_nsec) * 3.0e-9);
}

/* Assertion to check for errors */
#define CUDA_SAFE_CALL(ans) { gpuAssert((ans), (char *)__FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, char *file, int line, bool abort=true)
{
  if (code != cudaSuccess)
  {
    fprintf(stderr, "CUDA_SAFE_CALL: %s %s %d\n",
                                       cudaGetErrorString(code), file, line);
    if (abort) exit(code);
  }
}

/* INITIALIZE 2D ARRAY FUNCTION */
void initializeArray2D(float *arr, int arrLen, int seed) {
    srand(seed);
    for (int i = 0; i < arrLen; i++) {
      for (int j = 0; j < arrLen; j++) {
        arr[i * arrLen + j] = rand() / (float)RAND_MAX;
      }
    }
}

/* FUNCTION TO PERFORM BLOCKED ijk MMM */
void mmm_cpu_blocked(const float *A, const float *B, float *C, int N, int block_size) {
    int i, j, k, jj, kk;
    int en = block_size * (N / block_size);  

    for (kk = 0; kk < en; kk += block_size) {
        for (jj = 0; jj < en; jj += block_size) {
            for (i = 0; i < N; ++i) {
                for (j = jj; j < jj + block_size; ++j) {
                    float sum = C[i * N + j];
                    for (k = kk; k < kk + block_size; ++k) {
                        sum += A[i * N + k] * B[k * N + j];
                    }
                    C[i * N + j] = sum;
                }
            }
        }
    }
}

/* ( PART 1 ) GLOBAL MEMORY KERNEL IMPLEMENTATION */
__global__ void matrixMul(float* C, float* A, float* B, int row_a, int col_a, int col_b)
{
    // index var
    int i, j, k;

    // Block index
    const int bId_x = blockIdx.x;
    const int bId_y = blockIdx.y;

    // Local thread index
    const int local_tid_x = threadIdx.x;
    const int local_tid_y = threadIdx.y;

    // Number of rows and columns of the result matrix to be evaluated by each block
    const int rows_per_block = row_a / gridDim.x;
    const int cols_per_block = col_b / gridDim.y;

    const int rows_per_thread = rows_per_block / blockDim.x;
    const int cols_per_thread = cols_per_block / blockDim.y;

    // Row and column indices of the result matrix that the current block has to compute
    const int blockStartId_row = bId_x * rows_per_block;
    const int blockEndId_row = (bId_x + 1) * rows_per_block - 1;

    const int blockStartId_col = bId_y * cols_per_block;
    const int blockEndId_col = (bId_y + 1) * cols_per_block - 1;

    // Row and column indices for the current thread within the block
    const int threadStartId_row = blockStartId_row + local_tid_x * rows_per_thread;
    const int threadEndId_row = blockStartId_row + (local_tid_x + 1) * rows_per_thread - 1;

    const int threadStartId_col = blockStartId_col + local_tid_y * cols_per_thread;
    const int threadEndId_col = blockStartId_col + (local_tid_y + 1) * cols_per_thread - 1;

    int resultId;
    float sum = 0;

    for (i = threadStartId_row; i <= threadEndId_row; i++) {
        for (j = threadStartId_col; j <= threadEndId_col; j++) {
            sum = 0;
            resultId = i * col_b + j;
            for (k = 0; k < col_a; k++) {
                sum += A[i * col_a + k] * B[k * col_b + j];
            }
            C[resultId] = sum;
        }
    }
}


/* ( PART 2 ) SHARED MEMORY KERNEL IMPLEMENTATION */
__global__ void MatrixMulKernel(float* Md, float* Nd, float* Pd, int Width)
{
    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH]; // Per-block shared memory
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;

    float Pvalue = 0;

    for (int m = 0; m < Width / TILE_WIDTH; ++m) {
        Mds[ty][tx] = Md[Row * Width + (m * TILE_WIDTH + tx)];
        Nds[ty][tx] = Nd[Col + (m * TILE_WIDTH + ty) * Width];
        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; ++k)
            Pvalue += Mds[ty][k] * Nds[k][tx];

        __syncthreads();
    }

    Pd[Row * Width + Col] = Pvalue;
}

int main(int argc, char **argv){
    // size variables
    int blockSizeX = TILE_WIDTH;
    int blockSizeY = TILE_WIDTH;
    int gridSizeX = 128;
    int gridSizeY = 128;
    int N = blockSizeX * gridSizeX; // Total grid size

    // CPU timing variables
    struct timespec time_start, time_stop;
    double time_dur;

    // GPU Timing variables
    cudaEvent_t start, start_mmm, stop, stop_mmm;
    float elapsed_gpu, elapsed_mmm;

    // error variables
    int errCount = 0, zeroCount = 0;

    // Grid on GPU global memory
    float *a_gpu;
    float *b_gpu;
    float *c_gpu;
  
    // Grid on the host memory
    float *a_host;
    float *b_host;
    float *c_host;
    float *c_verify;

    // Select GPU
    CUDA_SAFE_CALL(cudaSetDevice(0));

    // Allocate GPU memory
    size_t allocSize = N * N * sizeof(float);
    CUDA_SAFE_CALL(cudaMalloc((void **)&a_gpu, allocSize));
    CUDA_SAFE_CALL(cudaMalloc((void **)&b_gpu, allocSize));
    CUDA_SAFE_CALL(cudaMalloc((void **)&c_gpu, allocSize));

    // Allocate arrays on host memory
    a_host = (float *) malloc(allocSize);
    b_host = (float *) malloc(allocSize);
    c_host = (float *) malloc(allocSize);
    c_verify = (float *) malloc(allocSize);

    // Initialize the host arrays
    printf("Length of the array = %d\n", N);
    printf("\nInitializing the host grid ...");

    initializeArray2D(a_host, N, 2453);
    initializeArray2D(b_host, N, 2454);

    printf("\t... done\n\n");

    // GPU timers
    #if PRINT_TIME
    // Create the cuda events
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventCreate(&start_mmm);
    cudaEventCreate(&stop_mmm);
    // Record event on the default stream
    cudaEventRecord(start, 0);

    #endif

    // Transfer the arrays to the GPU memory
    CUDA_SAFE_CALL(cudaMemcpy(a_gpu, a_host, allocSize, cudaMemcpyHostToDevice));
    CUDA_SAFE_CALL(cudaMemcpy(b_gpu, b_host, allocSize, cudaMemcpyHostToDevice));

    // setup kernel variables 
    dim3 blockDim(blockSizeX, blockSizeY);
    dim3 gridDim(gridSizeX, gridSizeY);

    // time MMM execution
    cudaEventRecord(start_mmm, 0);

    // Launch ( Part 1 ) Global Memory Kernel
    // matrixMul<<<gridDim, blockDim>>>(c_gpu, a_gpu, b_gpu, N, N, N);

    // Launch ( Part 2 ) Shared Memory Kernel
    MatrixMulKernel<<<gridDim, blockDim>>>(a_gpu, b_gpu, c_gpu, N);

    // stop MMM execution timing
    cudaEventRecord(stop_mmm,0);
    cudaEventSynchronize(stop_mmm);

    // Check for errors during launch
    CUDA_SAFE_CALL(cudaPeekAtLastError());

    // Transfer the results back to the host
    CUDA_SAFE_CALL(cudaMemcpy(c_host, c_gpu, allocSize, cudaMemcpyDeviceToHost));

    #if PRINT_TIME
        // Stop and destroy the timer
        cudaEventRecord(stop,0);
        cudaEventSynchronize(stop);

        cudaEventElapsedTime(&elapsed_gpu, start, stop);
        cudaEventElapsedTime(&elapsed_mmm, start_mmm, stop_mmm);

        printf("\nGPU MMM time: %f (msec)\n", elapsed_mmm);
        printf("\nGPU time: %f (msec)\n", elapsed_gpu);
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        cudaEventDestroy(start_mmm);
        cudaEventDestroy(stop_mmm);
    #endif

    // compute the results on the host machine
    clock_gettime(CLOCK_REALTIME, &time_start);
    mmm_cpu_blocked(a_host, b_host, c_verify, N, 8);
    clock_gettime(CLOCK_REALTIME, &time_stop);

    time_dur = interval(time_start, time_stop);
    printf("CPU duration: %10.4g (ms)\n", 1000*time_dur);

    // Compare CPU and GPU results
    for (int i = 0; i < N; ++i) {
        for (int j = 0; j < N; ++j) {
          int idx = i * N + j;
          if (fabs(c_host[idx] - c_verify[idx]) > TOL) {
            errCount++;
          }
        }
    }

    if (errCount > 0) {
        printf("\n@ERROR: TEST FAILED: %d results did not match\n", errCount);
    } else if (zeroCount > 0) {
        printf("\n@ERROR: TEST FAILED: %d results (from GPU) are zero\n", zeroCount);
    } else {
        printf("\nTEST PASSED: All results matched\n");
    }


    // Free-up device and host memory
    CUDA_SAFE_CALL(cudaFree(a_gpu));
    CUDA_SAFE_CALL(cudaFree(b_gpu));
    CUDA_SAFE_CALL(cudaFree(c_gpu));

    free(a_host);
    free(b_host);
    free(c_host);
    free(c_verify);
    
    return 0;
}