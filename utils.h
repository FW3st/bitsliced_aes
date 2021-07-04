#include "cuda_uint128.h"
#include "device.h"

void set_byte(unsigned char a[8][16], int n, unsigned char v){
    int c = n/8;
    int b = 7-n%8;
    for(int i = 7; i>=0; i--){
        if(v&(1<<i)){
            a[i][c] = a[i][c] | (1<<b);
        } else {
            a[i][c] = a[i][c] & (~(1<<b));
        }
    } 
}

void set_byte128(uint128_t* a, int n, unsigned char v){
    set_byte((unsigned char (*)[16])(void*)a, n, v);
}

__device__ unsigned char get_byte128(uint128_t a[8], int n){
    n = n/16+8*(n%16);
    unsigned char ret = 0;
    
    for(int i = 7; i>=0; i--){
        if(n>=64){
           ret = (ret << 1) | ((unsigned int)(a[i].hi>>(n%64))&1);
        }else{
           ret = (ret << 1) | ((unsigned int)(a[i].lo>>(n%64))&1); 
        }
    }
    return ret;    
}
__device__ void cu_printHex(unsigned char* ptr, int len){
    for(int i=0; i<len; i++){
        printf("%02x ", ptr[i]&0xff);
        if(i%4 == 3) printf("| ");
    }
    printf("\n");
}

__device__ void cu_printHex128(uint128_t* a, int l){
    cu_printHex((unsigned char*)(void*)a,l);
}

__device__ void cu_print_state(unsigned char* a){
    cu_printHex(a,16);
    cu_printHex(a+1*16,16);
    cu_printHex(a+2*16,16);
    cu_printHex(a+3*16,16);
    cu_printHex(a+4*16,16);
    cu_printHex(a+5*16,16);
    cu_printHex(a+6*16,16);
    cu_printHex(a+7*16,16);
    printf("____________________\n");
}


__device__ void cu_print_state128(uint128_t* a){
    cu_printHex128(a,16);
    cu_printHex128(a+1,16);
    cu_printHex128(a+2,16);
    cu_printHex128(a+3,16);
    cu_printHex128(a+4,16);
    cu_printHex128(a+5,16);
    cu_printHex128(a+6,16);
    cu_printHex128(a+7,16);
    printf("____________________\n");
}


void printHex(unsigned char* ptr, int len){
    for(int i=0; i<len; i++){
        printf("%02x ", ptr[i]&0xff);
        if(i%4 == 3) printf("| ");
    }
    puts("");
}

void print_state(unsigned char* a){
    printHex(a,16);
    printHex(a+1*16,16);
    printHex(a+2*16,16);
    printHex(a+3*16,16);
    printHex(a+4*16,16);
    printHex(a+5*16,16);
    printHex(a+6*16,16);
    printHex(a+7*16,16);
    printf("____________________\n");
}

void printHex128(uint128_t* a, int l){
    printHex((unsigned char*)(void*)a,l);
}

void print_state128(uint128_t* a){
    printHex128(a,16);
    printHex128(a+1,16);
    printHex128(a+2,16);
    printHex128(a+3,16);
    printHex128(a+4,16);
    printHex128(a+5,16);
    printHex128(a+6,16);
    printHex128(a+7,16);
    printf("____________________\n");
}

void fill_random(int* ar, unsigned long len){
    for(unsigned long i=0; i<len; i++){
        ar[i] = rand();
    }
}

void printError(){
    cudaError_t error = cudaGetLastError ();
    printf("error: %s\n",cudaGetErrorName(error));
    printf("error: %s\n",cudaGetErrorString(error));
}

void print_device_info(){
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, DEVICE);
    printf("name: %s\n", prop.name);
    printf("GlobalMem: %lu\n", prop.totalGlobalMem);
    printf("sharedMemPerBlock: %lu\n", prop.sharedMemPerBlock);
    printf("regsPerBlock: %i\n", prop.regsPerBlock);
    printf("maxThreadsPerBlock: %i\n", prop.maxThreadsPerBlock);
    printf("maxThreadsDim[0]: %i\n", prop.maxThreadsDim[0]);
    printf("maxThreadsDim[1]: %i\n", prop.maxThreadsDim[1]);
    printf("maxThreadsDim[2]: %i\n", prop.maxThreadsDim[2]);
    printf("clockRate: %i\n", prop.clockRate);
}

int get_num_threads(){
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, DEVICE);
    return 0;
}
