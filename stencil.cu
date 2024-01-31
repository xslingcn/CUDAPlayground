#include <cstdio>
#include <chrono>

__global__ void simple_conv_1d(float *N, float *M, float *P, int size, int kernelSize)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float pValue = 0;
    int iStart = i - kernelSize / 2;

    if (i < size)
    {
        for (int j = 0; j < kernelSize; j++)
        {
            if (iStart + j < 0 || iStart + j >= size)
                continue;
            pValue += N[iStart + j] * M[j];
        }
    }

    P[i] = pValue;
}

__global__ void simple_conv_2d(float *N, float *M, float *P, int width, int height, int kernelSize)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    float pValue = 0;
    int colStart = col - kernelSize / 2;
    int rowStart = row - kernelSize / 2;

    if (col < width && row < height)
    {
        for (int j = 0; j < kernelSize; j++)
        {
            for (int k = 0; k < kernelSize; k++)
            {
                if (colStart + k < 0 || colStart + k >= width)
                    continue;
                if (rowStart + j < 0 || rowStart + j >= height)
                    continue;
                pValue += N[(rowStart + j) * width + colStart + k] * M[j * kernelSize + k];
            }
        }
    }

    P[row * width + col] = pValue;
}

#define TILE_SIZE 1020
__global__ void tiled_conv_1d(float *N, float *M, float *P, int size, int kernelSize)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int tx = threadIdx.x;

    __shared__ float Ns[TILE_SIZE + 4];

    int nStart = blockIdx.x * blockDim.x - kernelSize / 2; // index_o - n, n=kernelSize/2
    if (nStart + tx >= 0 && nStart + tx < size)
    {
        Ns[tx] = N[nStart + tx];
    }
    else
    {
        Ns[tx] = 0.0f;
    }
    __syncthreads();

    float pValue = 0;
    if (i < size)
    {
        for (int j = 0; j < kernelSize; j++)
        {
            pValue += Ns[tx + j] * M[j];
        }
        P[i] = pValue;
    }
}

void test_simple_conv_1d()
{
    // int size = 7;
    int size = 5000;
    int kernelSize = 5;

    // float h_N[size] = {1, 2, 3, 4, 5, 6, 7};
    float h_N[size];
    for (int i = 0; i < size; i++)
    {
        h_N[i] = static_cast<float>(i);
    }
    float h_M[kernelSize] = {3, 4, 5, 4, 3};
    float h_P[size];

    float *d_N, *d_M, *d_P;

    cudaMalloc((void **)&d_N, size * sizeof(float));
    cudaMalloc((void **)&d_M, kernelSize * sizeof(float));
    cudaMalloc((void **)&d_P, size * sizeof(float));

    cudaMemcpy(d_N, h_N, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_M, h_M, kernelSize * sizeof(float), cudaMemcpyHostToDevice);

    simple_conv_1d<<<ceil(size / 256.0), 256.0>>>(d_N, d_M, d_P, size, kernelSize);

    cudaMemcpy(h_P, d_P, size * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < size; i++)
    {
        printf("%f ", h_P[i]);
    }
    printf("\n");

    cudaFree(d_N);
    cudaFree(d_M);
    cudaFree(d_P);
}

void test_simple_conv_2d()
{
    int width = 5, height = 5;
    int kernelSize = 3;
    float h_N[width * height];
    float h_M[kernelSize * kernelSize] = {3, 4, 5, 6, 7, 6, 5, 4, 3};
    float h_P[width * height];

    for (int i = 0; i < width * height; i++)
    {
        h_N[i] = i;
    }

    float *d_N, *d_M, *d_P;

    cudaMalloc((void **)&d_N, width * height * sizeof(float));
    cudaMalloc((void **)&d_M, kernelSize * kernelSize * sizeof(float));
    cudaMalloc((void **)&d_P, width * height * sizeof(float));

    cudaMemcpy(d_N, h_N, width * height * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_M, h_M, kernelSize * kernelSize * sizeof(float), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(32, 32);
    dim3 numBlocks(ceil(width / 32.0), ceil(height / 32.0));
    simple_conv_2d<<<numBlocks, threadsPerBlock>>>(d_N, d_M, d_P, width, height, kernelSize);

    cudaMemcpy(h_P, d_P, width * height * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < width * height; i++)
    {
        printf("%f ", h_P[i]);
    }
    printf("\n");

    cudaFree(d_N);
    cudaFree(d_M);
    cudaFree(d_P);
}

void test_tiled_conv_1d (){
    int size = 5000;
    int kernelSize = 5;
    float h_N[size];
    float h_M[kernelSize] = {3, 4, 5, 4, 3};
    float h_P[size];

    for (int i = 0; i < size; i++)
    {
        h_N[i] = static_cast<float>(i);
    }

    float *d_N, *d_M, *d_P;

    cudaMalloc((void **)&d_N, size * sizeof(float));
    cudaMalloc((void **)&d_M, kernelSize * sizeof(float));
    cudaMalloc((void **)&d_P, size * sizeof(float));

    cudaMemcpy(d_N, h_N, size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_M, h_M, kernelSize * sizeof(float), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(TILE_SIZE + 4, 1, 1);
    dim3 numBlocks(ceil(size / TILE_SIZE), 1, 1);

    tiled_conv_1d<<<threadsPerBlock, numBlocks>>>(d_N, d_M, d_P, size, kernelSize);

    cudaMemcpy(h_P, d_P, size * sizeof(float), cudaMemcpyDeviceToHost);

    for (int i = 0; i < size; i++)
    {
        printf("%f ", h_P[i]);
    }
    printf("\n");

    cudaFree(d_N);
    cudaFree(d_M);
    cudaFree(d_P);
}

int main()
{
    auto start = std::chrono::high_resolution_clock::now();
    test_simple_conv_1d();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    printf("simple_conv_1d: %f\n", diff.count());

    start = std::chrono::high_resolution_clock::now();
    test_simple_conv_2d();
    end = std::chrono::high_resolution_clock::now();
    diff = end - start;
    printf("simple_conv_2d: %f\n", diff.count());

    start = std::chrono::high_resolution_clock::now();
    test_tiled_conv_1d();
    end = std::chrono::high_resolution_clock::now();
    diff = end - start;
    printf("tiled_conv_1d: %f\n", diff.count());
}
