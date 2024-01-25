#include <cstdio>
#include <chrono>

int n = 0x7fffffff >> 2;

__global__ void mykernal(int *d_a, int *d_b, int *d_c, int n)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n)
    {
        d_c[i] = d_a[i] + d_b[i];
    }
}

void addWithCuda (int *d_a, int *d_b, int *d_c){
    // mykernal<<<ceil(n / 512.0), 512>>>(d_a, d_b, d_c, n);
    mykernal<<<ceil(n/256.0), 256>>>(d_a, d_b, d_c, n);
    // mykernal<<<1, 1>>>(d_a, d_b, d_c, n);

    int *h_c;
    h_c = (int *)malloc(n*sizeof(int));
    cudaMemcpy(h_c, d_c, n*sizeof(int), cudaMemcpyDeviceToHost);
}


void pageableMemory (void){
    int *h_a, *h_b, *h_c;
    h_a = (int *)malloc(n*sizeof(int));
    h_b = (int *)malloc(n*sizeof(int));
    h_c = (int *)malloc(n*sizeof(int));

    for (int i = 0; i < n; i++){
        h_a[i] = i;
        h_b[i] = i;
    }
    
    memset(h_c, 0, n*sizeof(int));

    int *d_a, *d_b, *d_c;
    cudaMalloc((void **)&d_a, n*sizeof(int));
    cudaMalloc((void **)&d_b, n*sizeof(int));    
    cudaMalloc((void **)&d_c, n*sizeof(int));

    cudaMemcpy(d_a, h_a, n*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, n*sizeof(int), cudaMemcpyHostToDevice);
    
    addWithCuda(d_a, d_b, d_c);
    
    free(h_a); free(h_b); free(h_c);
    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
}

void pinnedMemory (void){
    int *h_a, *h_b, *h_c;

    cudaMallocHost((void **) &h_a, n*sizeof(int));
    cudaMallocHost((void **) &h_b, n*sizeof(int));
    cudaMallocHost((void **) &h_c, n*sizeof(int));

    for (int i = 0; i < n; i++){
        h_a[i] = i;
        h_b[i] = i;
    }

    memset(h_c, 0, n*sizeof(int));

    int *d_a, *d_b, *d_c;
    cudaMalloc((void **)&d_a, n*sizeof(int));
    cudaMalloc((void **)&d_b, n*sizeof(int));
    cudaMalloc((void **)&d_c, n*sizeof(int));

    cudaMemcpy(d_a, h_a, n*sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, n*sizeof(int), cudaMemcpyHostToDevice);

    addWithCuda(d_a, d_b, d_c);

    cudaFreeHost(h_a); cudaFreeHost(h_b); cudaFreeHost(h_c);
}

void mappedMemory (void){
    int *h_a, *h_b, *h_c;
    cudaHostAlloc((void **)&h_a, n * sizeof(int), cudaHostAllocMapped);
    cudaHostAlloc((void **)&h_b, n * sizeof(int), cudaHostAllocMapped);
    cudaHostAlloc((void **)&h_c, n * sizeof(int), cudaHostAllocMapped);

    for (int i = 0; i < n; i++){
        h_a[i] = i;
        h_b[i] = i;
    }

    int *d_a, *d_b, *d_c;
    cudaHostGetDevicePointer((void **)&d_a, (void *)h_a, 0);
    cudaHostGetDevicePointer((void **)&d_b, (void *)h_b, 0);
    cudaHostGetDevicePointer((void **)&d_c, (void *)h_c, 0);

    addWithCuda(d_a, d_b, d_c);

    cudaFreeHost(h_a); cudaFreeHost(h_b); cudaFreeHost(h_c);
}

void unifiedMemory (void){
    int *a, *b, *c;

    cudaMallocManaged((void **)&a, n*sizeof(int));
    cudaMallocManaged((void **)&b, n*sizeof(int));
    cudaMallocManaged((void **)&c, n*sizeof(int));

    for (int i = 0; i < n; i++){
        a[i] = i;
        b[i] = i;
    }

    memset(c, 0, n*sizeof(int));

    addWithCuda(a, b, c);

    cudaFree(a); cudaFree(b); cudaFree(c);
}

int main (void){
    printf("Testing with n=%d\n", n);

    auto start = std::chrono::high_resolution_clock::now();
    pageableMemory();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> diff = end - start;
    printf("Pageable memory: %f\n", diff.count());

    start = std::chrono::high_resolution_clock::now();
    pinnedMemory();
    end = std::chrono::high_resolution_clock::now();
    diff = end - start;
    printf("Pinned memory: %f\n", diff.count());

    start = std::chrono::high_resolution_clock::now();
    mappedMemory();
    end = std::chrono::high_resolution_clock::now();
    diff = end - start;
    printf("Mapped memory: %f\n", diff.count());

    start = std::chrono::high_resolution_clock::now();
    unifiedMemory();
    end = std::chrono::high_resolution_clock::now();
    diff = end - start;
    printf("Unified memory: %f\n", diff.count());

    return 0;
}