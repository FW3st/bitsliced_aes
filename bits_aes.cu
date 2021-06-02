#include <stdio.h>
#include <stdint.h>
#include "cuda_uint128.h"
#include "device.h"

#define WORDSIZE 32
#define KEY_SIZE 128
//#define DEVICE   0 moved to device.h

#define NK KEY_SIZE/WORDSIZE
#define NR NK + 6
#define ROUND_KEY_COUNT NR + 1
#define ROUND_KEY_SIZE ROUND_KEY_COUNT * KEY_SIZE


#define NUM_BLOCKS 1 //128
#define BLOCK_SIZE 128
#define PLAIN_SIZE NUM_BLOCKS*BLOCK_SIZE

__device__ static void swapByte(uint128_t* a , uint128_t* b, uint128_t m, int n){
    uint128_t t = ((((*a)>>n)^(*b)))&m;
    *b = (*b) ^ t;
    *a = (*a) ^ (t << n);
}

__device__ void bitorder_retransform(char* plain, uint128_t* a){
    const uint128_t m1 = (uint128_t) 0x5555555555555555 << 64 | 0x5555555555555555;
    const uint128_t m2 = (uint128_t) 0x3333333333333333 << 64 | 0x3333333333333333;
    const uint128_t m3 = (uint128_t) 0x0f0f0f0f0f0f0f0f << 64 | 0x0f0f0f0f0f0f0f0f;

    swapByte(a+7, a+3, m3, 4);
    swapByte(a+6, a+2, m3, 4);
    swapByte(a+5, a+1, m3, 4);
    swapByte(a+4,   a, m3, 4);
    
    swapByte(a+7, a+5, m2, 2);
    swapByte(a+6, a+4, m2, 2);
    swapByte(a+3, a+1, m2, 2);
    swapByte(a+2,   a, m2, 2);
    
    swapByte(a+7, a+6, m1, 1);
    swapByte(a+5, a+4, m1, 1);
    swapByte(a+3, a+2, m1, 1);
    swapByte(a+1,   a, m1, 1);
    
    for(int i=0; i<8; i++){
            ((uint128_t*)plain)[i] = a[i];
    }
}

__device__ void bitorder_transform(char* plain, uint128_t* a){
    const uint128_t m1 = (uint128_t) 0x5555555555555555 << 64 | 0x5555555555555555;
    const uint128_t m2 = (uint128_t) 0x3333333333333333 << 64 | 0x3333333333333333;
    const uint128_t m3 = (uint128_t) 0x0f0f0f0f0f0f0f0f << 64 | 0x0f0f0f0f0f0f0f0f;

    for(int i=0; i<8; i++){
            a[i] = ((uint128_t*)plain)[i];
    }
    
    swapByte(a+1,   a, m1, 1);
    swapByte(a+3, a+2, m1, 1);
    swapByte(a+5, a+4, m1, 1);
    swapByte(a+7, a+6, m1, 1);
    
    swapByte(a+2,   a, m2, 2);
    swapByte(a+3, a+1, m2, 2);
    swapByte(a+6, a+4, m2, 2);
    swapByte(a+7, a+5, m2, 2);
    
    swapByte(a+4,   a, m3, 4);
    swapByte(a+5, a+1, m3, 4);
    swapByte(a+6, a+2, m3, 4);
    swapByte(a+7, a+3, m3, 4);
}


__global__ void encrypt(char*plain, char*keys, char*cypher) {
    uint4 a[8];
    plain = plain + (128*8) * (blockIdx.x*blockDim.x + threadIdx.x);
    
    bitorder_transform(plain, (uint128_t*)a);
    
    //uint64_t x = 0;
    /*
    self.addRoundKey(self.key[0:NB])
    for rnd in range(1, NR):
        self.s.subBytes()
        self.s.shiftRows()
        self.s.mixColumns()
        self.addRoundKey(self.key[rnd * NB: (rnd + 1) * NB])
    self.s.subBytes()
    self.s.shiftRows()
    self.addRoundKey(self.key[NR * NB: (NR + 1) * NB])
    return self.s.getCypher()
    */
}

void printHex(char* ptr, int len){
    for(int i=0; i<len; i++){
        printf("%x ", ptr[i]);
    }
    puts("");
}

void create_round_key(char* key, char* roundkey){}

/*
def keyExpansion(self, k):

    def rotWord(a: bytes):
        return bytes((a[1], a[2], a[3], a[0]))

    def rcon(idx):
        return modg(1 << (idx - 1)) + b'\x00\x00\x00'

    def subWord(a: bytes) -> bytes:
        return b''.join(sboxi(b).to_bytes(1, "little") for b in a)

    for i in range(NK):
        self.key[i] = k[4 * i:4 * i + 4]

    for i in range(NK, NB * (NR + 1)):
        tmp: bytes = self.key[i - 1]
        if i % NK == 0:
            tmp = xor(subWord(rotWord(tmp)), rcon(i // NK))
        elif NK > 6 and i % NK == 4:
            tmp = subWord(tmp)
        self.key[i] = xor(self.key[i - NK], tmp)
*/


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


#ifndef TEST // Q'n'D ToDo
int main(void) {
    //print_device_info();
    char *d_plain;
    char *d_roundkey;
    char *d_cypher;
        
    char* plain = (char*) malloc(PLAIN_SIZE);
    char* cypher = (char*) malloc(PLAIN_SIZE);
    char* key = (char*) malloc(KEY_SIZE);
    char* roundkey = (char*) malloc(ROUND_KEY_SIZE);
    
    cudaSetDevice(DEVICE);
    int num_threads = get_num_threads();
    
        
    for(int i=0; i<KEY_SIZE; i++){
        key[i] = (char)i;
    }
    create_round_key(key, roundkey);
    
    for(int i=0; i<BLOCK_SIZE; i++){
        plain[i] = (char)i;
    }
    for(int i=1; i<NUM_BLOCKS; i++){
        memcpy(plain+i*BLOCK_SIZE, plain, BLOCK_SIZE);
    }
    
    cudaMalloc((void**)&d_plain, PLAIN_SIZE);
    cudaMalloc((void**)&d_cypher, PLAIN_SIZE);
    cudaMalloc((void**)&d_roundkey, ROUND_KEY_SIZE);
    
    cudaMemcpy(d_plain, plain, PLAIN_SIZE, cudaMemcpyHostToDevice);
    cudaMemcpy(d_roundkey, roundkey, ROUND_KEY_SIZE, cudaMemcpyHostToDevice);
    
    encrypt<<<NUM_BLOCKS,1>>>(d_plain, d_roundkey, d_cypher);
    
    cudaMemcpy(cypher, d_cypher, PLAIN_SIZE, cudaMemcpyDeviceToHost);
    
    cudaFree(d_roundkey); 
    cudaFree(d_cypher);
    cudaFree(d_plain); 
    return 0;
}
#endif