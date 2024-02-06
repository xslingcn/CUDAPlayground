#include <cstdio>
#include <chrono>

// inclusive scan add
__global__ void naive_scan_add(int *IN, int *OUT, int size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    int result = 0;
    if (i < size)
    {
        for (int j = 0; j <= i; j++)
        {
            result += IN[j];
        }
        OUT[i] = result;
    }
}

#define BLOCK_SIZE 1024
__global__ void HnS_scan_add(int *IN, int *OUT, int size) // cannot compute more than 1 block
{
    int tx = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ int temp[BLOCK_SIZE];

    if (i < size)
    {
        temp[tx] = IN[i];

        for (int stride = 1; stride < BLOCK_SIZE; stride *= 2)
        {
            __syncthreads();

            int inVal = 0;
            if (tx >= stride)
            {
                inVal = temp[tx - stride];
            }
            __syncthreads();
            if (tx >= stride)
            {
                temp[tx] += inVal;
            }
        }
        OUT[i] = temp[tx];
    }
}

__global__ void blelloch_scan_add(int *IN, int *OUT, int size)
{
    int tx = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ int temp[2*BLOCK_SIZE];

    if(i < size)
    {
        temp[tx] = IN[i];
    }
    else
    {
        temp[tx] = 0;
    }
    __syncthreads();

    // reduction
    for (int stride = 1; stride <= BLOCK_SIZE; stride *= 2)
    {
        __syncthreads();

        int index = (tx + 1) * 2 * stride - 1;
        if(index < 2*BLOCK_SIZE)
        {
            temp[index] += temp[index - stride];
        }
    }
    
    //down-sweep
    for (int stride = BLOCK_SIZE/2; stride > 0; stride /= 2)
    {
        __syncthreads();

        int index = (tx + 1) * 2 * stride - 1;
        if(index + stride < 2*BLOCK_SIZE)
        {
            temp[index + stride] += temp[index];
        }
    }

    __syncthreads();
    if(i < size)
    {
        OUT[i] = temp[tx];
    }
}

__global__ void complete_blelloch_scan_add(int *IN, int *OUT, int *SUM, int size)
{
    int tx = threadIdx.x;
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    __shared__ int temp[2 * BLOCK_SIZE];

    if (i < size)
    {
        temp[tx] = IN[i];
    }
    else
    {
        temp[tx] = 0;
    }
    __syncthreads();

    // reduction
    for (int stride = 1; stride <= BLOCK_SIZE; stride *= 2)
    {
        __syncthreads();

        int index = (tx + 1) * 2 * stride - 1;
        if (index < 2 * BLOCK_SIZE)
        {
            temp[index] += temp[index - stride];
        }
    }

    // down-sweep
    for (int stride = BLOCK_SIZE / 2; stride > 0; stride /= 2)
    {
        __syncthreads();

        int index = (tx + 1) * 2 * stride - 1;
        if (index + stride < 2 * BLOCK_SIZE)
        {
            temp[index + stride] += temp[index];
        }
    }

    __syncthreads();
    if (i < size)
    {
        OUT[i] = temp[tx];
    }

    if (tx == 0)
        SUM[blockIdx.x] = temp[blockDim.x - 1];
}

__global__ void add_sum(int *OUT, int *SUM, int size)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (blockIdx.x > 0 && i < size)
    {
        OUT[i] += SUM[blockIdx.x - 1];  // use inclusive sum
    }
}

enum KernelType
{
    NAIVE,
    H_S,
    BLELLOCH,
    COMPLETE
};

void test_scan_add(int *IN, int *OUT, int size, KernelType type)
{
    int *d_IN, *d_OUT;

    cudaMalloc((void **)&d_IN, size * sizeof(int));
    cudaMalloc((void **)&d_OUT, size * sizeof(int));

    cudaMemcpy(d_IN, IN, size * sizeof(int), cudaMemcpyHostToDevice);

    dim3 threadsPerBlock(BLOCK_SIZE);
    dim3 numBlocks((size - 1) / BLOCK_SIZE + 1);

    switch (type)
    {
    case NAIVE:
        naive_scan_add<<<numBlocks, threadsPerBlock>>>(d_IN, d_OUT, size);
        break;
    case H_S:
        HnS_scan_add<<<1, BLOCK_SIZE>>>(d_IN, d_OUT, size); // not ceil, int size
        break;
    case BLELLOCH:
        blelloch_scan_add<<<1, BLOCK_SIZE>>>(d_IN, d_OUT, size);
        break;
    case COMPLETE:
        int *d_SUM;
        cudaMalloc((void **)&d_SUM, numBlocks.x * sizeof(int));
        complete_blelloch_scan_add<<<numBlocks, threadsPerBlock>>>(d_IN, d_OUT, d_SUM, size);
        blelloch_scan_add<<<1, BLOCK_SIZE>>>(d_SUM, d_SUM, numBlocks.x);
        add_sum<<<numBlocks, threadsPerBlock>>>(d_OUT, d_SUM, size);
        break;
    }

    cudaMemcpy(OUT, d_OUT, size * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(d_IN);
    cudaFree(d_OUT);
}

int main()
{
    // int size = 1024;
    int size = 2048;
    int IN[size], OUT[size];
    for (int i = 0; i < size; i++)
    {
        IN[i] = rand() % 1000;
    }

    auto start = std::chrono::high_resolution_clock::now();
    test_scan_add(IN, OUT, size, NAIVE);
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    printf("naive_scan_add: %f\n", diff.count());

    int *ground_truth = new int[size];
    memcpy(ground_truth, OUT, size * sizeof(int));

    start = std::chrono::high_resolution_clock::now();
    test_scan_add(IN, OUT, size, H_S);
    end = std::chrono::high_resolution_clock::now();
    if (memcmp(ground_truth, OUT, size * sizeof(int)) == 0)
    {
        printf("HnS_scan_add: Correct\n");
    }
    else
    {
        printf("HnS_scan_add: Incorrect\n");
        for (int i = 0; i < size; i++)
        {
            printf("%d ", OUT[i]);
        }
        printf("\nExpected: \n");
        for (int i = 0; i < size; i++)
        {
            printf("%d ", ground_truth[i]);
        }
        printf("\n");
    }
    diff = end - start;
    printf("HnS_scan_add: %f\n", diff.count());

    start = std::chrono::high_resolution_clock::now();
    test_scan_add(IN, OUT, size, BLELLOCH);
    end = std::chrono::high_resolution_clock::now();
    if (memcmp(ground_truth, OUT, size * sizeof(int)) == 0)
    {
        printf("blelloch_scan_add: Correct\n");
    }
    else
    {
        printf("blelloch_scan_add: Incorrect\n");
        for (int i = 0; i < size; i++)
        {
            printf("%d ", OUT[i]);
        }
        printf("\nExpected: \n");
        for (int i = 0; i < size; i++)
        {
            printf("%d ", ground_truth[i]);
        }
        printf("\n");
    }
    diff = end - start;
    printf("blelloch_scan_add: %f\n", diff.count());

    start = std::chrono::high_resolution_clock::now();
    test_scan_add(IN, OUT, size, COMPLETE);
    end = std::chrono::high_resolution_clock::now();
    if (memcmp(ground_truth, OUT, size * sizeof(int)) == 0)
    {
        printf("complete_blelloch_scan_add: Correct\n");
    }
    else
    {
        printf("complete_blelloch_scan_add: Incorrect\n");
        for (int i = 0; i < size; i++)
        {
            printf("%d ", OUT[i]);
        }
        printf("\nExpected: \n");
        for (int i = 0; i < size; i++)
        {
            printf("%d ", ground_truth[i]);
        }
        printf("\n");
    }
    diff = end - start;
    printf("complete_blelloch_scan_add: %f\n", diff.count());

    return 0;
}