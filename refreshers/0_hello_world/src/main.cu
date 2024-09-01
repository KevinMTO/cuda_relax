#include <stdio.h>
#include <cuda_runtime.h>

int main() {
    cudaError_t error;
    int deviceCount;
    
    error = cudaGetDeviceCount(&deviceCount);
    if (error != cudaSuccess) {
        printf("Error: %s\n", cudaGetErrorString(error));
        return -1;
    }
    
    printf("Number of CUDA devices: %d\n", deviceCount);
    
    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp deviceProp;
        error = cudaGetDeviceProperties(&deviceProp, i);
        
        if (error != cudaSuccess) {
            printf("Error: %s\n", cudaGetErrorString(error));
            return -1;
        }
        
        printf("\nDevice %d: %s\n", i, deviceProp.name);
        printf("  Compute Capability: %d.%d\n", deviceProp.major, deviceProp.minor);
        printf("  Max Grid Size: (%d, %d, %d)\n", 
               deviceProp.maxGridSize[0], deviceProp.maxGridSize[1], deviceProp.maxGridSize[2]);
        printf("  Max Block Size: (%d, %d, %d)\n", 
               deviceProp.maxThreadsDim[0], deviceProp.maxThreadsDim[1], deviceProp.maxThreadsDim[2]);
        printf("  Max Threads per Block: %d\n", deviceProp.maxThreadsPerBlock);
        printf("  Max Threads per Multiprocessor: %d\n", deviceProp.maxThreadsPerMultiProcessor);
        printf("  Number of Multiprocessors: %d\n", deviceProp.multiProcessorCount);
        printf("  Warp Size: %d\n", deviceProp.warpSize);
    }
    
    return 0;
}
