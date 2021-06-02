#define TEST
#include "bits_aes.cu"


__device__ int __test_bitorder_transform(){
    printf("Todo __test_bitorder_transform\n");
    return 1;
}

__device__ int __test_bitorder_retransform(){
    printf("Todo __test_bitorder_transform\n");
    return 1;
}

__global__ void test_functions() {
    printf("Todo function tests\n");
    if(__test_bitorder_transform() != 0){
        printf("Error: __test_bitorder_transform\n");
        return;
    }
    if(__test_bitorder_retransform() != 0){
        printf("Error: __test_bitorder_retransform\n");
        return;
    }
}


int main(void) {
    char *d_cypher;
    char cypher;

    cudaSetDevice(DEVICE);    
    cudaMalloc((void**)&d_cypher, 1);

    
    test_functions<<<1,1>>>();
    
    cudaMemcpy(&cypher, d_cypher, 1, cudaMemcpyDeviceToHost);
    
    cudaFree(d_cypher);
    return 0;
}