#include <cstdio>
#include <chrono>

__global__ void plain_matrix_mul(float *M, float *N, float *P, int width)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    float pValue = 0;

    if (col < width && row < width)
    {
        for (int k = 0; k < width; k++)
        {
            pValue += M[row * width + k] * N[k * width + col];
        }
        P[row * width + col] = pValue;
    }
}

#define TILE_WIDTH 16
__global__ void tiled_matrix_mul(float *M, float *N, float *P, int width)
{
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int col = blockIdx.x * blockDim.x + tx;
    int row = blockIdx.y * blockDim.y + ty;

    __shared__ float ds_M[TILE_WIDTH][TILE_WIDTH];
    __shared__ float ds_N[TILE_WIDTH][TILE_WIDTH];

    float pValue = 0;

    for (int k = 0; k < width / TILE_WIDTH; k++)
    {
        // loading tile
        // https://imgur.com/a/pMWtqPK
        ds_M[ty][tx] = M[row * width + k * TILE_WIDTH + tx];
        ds_N[ty][tx] = N[(k * TILE_WIDTH + ty) * width + col];
        __syncthreads();

        // in-tile multiplication, partial
        for (int i = 0; i < TILE_WIDTH; i++)
        {
            pValue += ds_M[ty][i] * ds_N[i][tx];
        }
    }
    P[row * width + col] = pValue;
}

// handles arbitrary matrix sizes by adding boundary checks
__global__ void practical_tile_matrix_mul(float *M, float *N, float *P, int width)
{
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    int col = blockIdx.x * blockDim.x + tx;
    int row = blockIdx.y * blockDim.y + ty;

    __shared__ float ds_M[TILE_WIDTH][TILE_WIDTH];
    __shared__ float ds_N[TILE_WIDTH][TILE_WIDTH];

    float pValue = 0;

    for (int k = 0; k < width / TILE_WIDTH; k++)
    {
        ds_M[ty][tx] = row < width && k * TILE_WIDTH + tx < width ? 
                        M[row * width + k * TILE_WIDTH + tx] : 0;

        ds_N[ty][tx] = col < width && k * TILE_WIDTH + ty < width ?
                        N[(k * TILE_WIDTH + ty) * width + col] : 0;

        __syncthreads();

        if (row < width && col < width){
            for (int i = 0; i < TILE_WIDTH; i++)
            {
                pValue += ds_M[ty][i] * ds_N[i][tx];
            }
        }
    }
    if(row < width && col < width){
        P[row * width + col] = pValue;
    }
}

enum KernelType
{
    SIMPLE,
    TILED,
    PRACTICAL
};

void test_matrix_mul(float *M, float *N, float *P, int width, KernelType type){
    float *d_M, *d_N, *d_P;

    cudaMalloc(&d_M, width * width * sizeof(float));
    cudaMalloc(&d_N, width * width * sizeof(float));
    cudaMalloc(&d_P, width * width * sizeof(float));

    cudaMemcpy(d_M, M, width * width * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_N, N, width * width * sizeof(float), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(16.0, 16.0);
    dim3 numBlocks(ceil(width / 16.0), ceil(width / 16.0));
    
    switch (type)
    {
    case SIMPLE:
        plain_matrix_mul<<<threadsPerBlock, numBlocks>>>(d_M, d_N, d_P, width);
        break;
    case TILED:
        tiled_matrix_mul<<<threadsPerBlock, numBlocks>>>(d_M, d_N, d_P, width);
        break;
    case PRACTICAL:
        practical_tile_matrix_mul<<<threadsPerBlock, numBlocks>>>(d_M, d_N, d_P, width);
        break;
    }

    cudaMemcpy(P, d_P, width * width * sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_M);
    cudaFree(d_N);
    cudaFree(d_P);
}

int main()
{
    printf("Matrix Multiplication\n");
    int width = 1024;
    float *M = new float[width * width]();
    float *N = new float[width * width]();
    float *P = new float[width * width]();
    for (int i = 0; i < width * width; i++)
    {
        M[i] = rand() % 100;
        N[i] = rand() % 100;
    }

    auto start = std::chrono::high_resolution_clock::now();
    test_matrix_mul(M, N, P, width, SIMPLE);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    printf("simple_matrix_mul: %f\n", diff.count());

    float *ground_truth = new float[width * width];
    memcpy(ground_truth, P, width * width * sizeof(float));
    
    start = std::chrono::high_resolution_clock::now();
    test_matrix_mul(M, N, P, width, TILED);
    end = std::chrono::high_resolution_clock::now();
    if(memcmp(ground_truth, P, width * width * sizeof(float)) == 0){
        printf("tiled_matrix_mul: Correct\n");
    } else {
        printf("tiled_matrix_mul: Incorrect\n");
    }
    diff = end - start;
    printf("tiled_matrix_mul: %f\n", diff.count());

    start = std::chrono::high_resolution_clock::now();
    test_matrix_mul(M, N, P, width, PRACTICAL);
    end = std::chrono::high_resolution_clock::now();
    if(memcmp(ground_truth, P, width * width * sizeof(float)) == 0){
        printf("practical_matrix_mul: Correct\n");
    } else {
        printf("practical_matrix_mul: Incorrect\n");
    }
    diff = end - start;
    printf("practical_matrix_mul: %f\n", diff.count());

    return 0;
}