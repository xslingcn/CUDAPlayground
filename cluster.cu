#include <cstdio>

__global__ void __cluster_dims__(2,1,1) attributedClusterKernel (float *input, float *output) {

}

__global__ void clusterKernel(float *input, float *output) {
    
}

int n = 1024;

void launchCompileTimeClusteredKernel () {
    float *input, *output;

    dim3 threadsPerBlock (16, 16);
    dim3 numBlolcks(n / threadsPerBlock.x, n / threadsPerBlock.y);

    attributedClusterKernel<<<numBlolcks, threadsPerBlock>>>(input, output);
}

void launchRunTimeClusteredKernel () {
    float *input, *output;

    dim3 threadsPerBlock (16, 16);
    dim3 numBlolcks(n / threadsPerBlock.x, n / threadsPerBlock.y);

    {
        cudaLaunchConfig_t config = {0};
        config.gridDim = numBlolcks;
        config.blockDim = threadsPerBlock;

        cudaLaunchAttribute attr[1];
        attr[0].id = cudaLaunchAttributeClusterDimension;
        attr[0].val.clusterDim.x = 2;
        attr[0].val.clusterDim.y = 1;
        attr[0].val.clusterDim.z = 1;

        config.attrs = attr;
        config.numAttrs = 1;

        cudaLaunchKernelEx(&config, clusterKernel, input, output);
    }
}

int main () {
    launchCompileTimeClusteredKernel();
    launchRunTimeClusteredKernel();
}