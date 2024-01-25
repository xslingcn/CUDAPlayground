#include <cstdio>

__global__ void mykernal (int *d_a, int *d_b, int *d_c, int n) {
    uint3 threidId = threadIdx;
    uint3 blockId = blockIdx;
    dim3 blockDimId = blockDim;

    int i = threadIdx.x;
    if (i < n)
    {
        d_c[i] = d_a[i] + d_b[i];
    }
}

int main (void) {
    int n = 100;
    int h_a[n], h_b[n], h_c[n] = {0};

    for (int i = 0; i < n; i++){
        h_a[i] = i * rand() % 100;
        h_b[i] = i * rand() % 100;
    }

    int *d_a = 0, *d_b = 0, *d_c = 0;

    cudaError_t err = cudaSetDevice(0);
    if (err != cudaSuccess)
    {
        printf("Error: %s\n", cudaGetErrorString(err));
    }
    
    err = cudaMalloc((void **)&d_a, n * sizeof(int));
    if (err != cudaSuccess){
        printf("Error: %s\n", cudaGetErrorString(err));
    }

    err = cudaMalloc((void **)&d_b, n * sizeof(int));
    if (err != cudaSuccess){
        printf("Error: %s\n", cudaGetErrorString(err));
    }

    err = cudaMalloc((void **)&d_c, n * sizeof(int));
    if (err != cudaSuccess){
        printf("Error: %s\n", cudaGetErrorString(err));
    }

    err = cudaMemcpy(d_a, h_a, n * sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess){
        printf("Error: %s\n", cudaGetErrorString(err));
    }

    err = cudaMemcpy(d_b, h_b, n * sizeof(int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess){
        printf("Error: %s\n", cudaGetErrorString(err));
    }

    mykernal<<<1, n>>>(d_a, d_b, d_c, n);

    err = cudaGetLastError();
    if (err != cudaSuccess){
        printf("Error: %s\n", cudaGetErrorString(err));
    }

    err = cudaMemcpy(h_c, d_c, n * sizeof(int), cudaMemcpyDeviceToHost);
    if (err != cudaSuccess){
        printf("Error: %s\n", cudaGetErrorString(err));
    }

    for (int i = 0; i < n; i++){
        printf("%d + %d = %d\n", h_a[i], h_b[i], h_c[i]);
    }

    cudaFree(d_a);  cudaFree(d_b);  cudaFree(d_c);
    return 0;
}