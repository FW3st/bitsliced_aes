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

// from [13] A Fast and Cache-Timing Resistant Implementation of the AES
__device__ void mixColumns(uint128_t a[8]){
    uint128_t t0 = a[0] ^ (a[0] << 32);
    uint128_t t1 = (a[1] << 32) ^ (a[1] << 64);
    uint128_t t2 = a[2] ^ (a[2] << 32);
    uint128_t t3 = (a[3] << 32) ^ (a[3] << 64);
    uint128_t t4 = a[4] ^ (a[4] << 32);
    uint128_t t5 = (a[5] << 32) ^ (a[5] << 64);
    uint128_t t6 = a[6] ^ (a[6] << 32);
    uint128_t t7 = (a[7] << 32) ^ (a[7] << 64);
    
    a[2] ^= t1;
    t1 ^= t0;
    t1 = t1<<32;
    a[1] ^= t1;
    t0 = t0 << 64;
    a[0] ^= t0;
    a[4] ^= t3;
    t3 ^= t2;
    t3 = t3 << 32;
    a[3] ^= t3;
    t2 = t2 <<64;
    a[2] ^= t2;
    a[6] ^= t5;
    t5 ^= t4;
    t5 = t5 << 32;
    a[5] ^= t5;
    t4 = t4 <<64;
    a[4] ^= t4;
    a[0] ^= t7;
    a[1] ^= t7;
    a[3] ^= t7;
    a[4] ^= t7;
    t7 ^= t6;
    t7 = t7 << 32;
    a[7] ^= t7;
    t6 = t6 << 64;
    a[6] ^= t6;    
}



// from [14] A Small Depth-16 Circuit for the AES S-Box
__device__ void subBytes(uint128_t a[8]){
    uint128_t T1,T2,T3,T4,T5,T6,T7,T8,T9,T10,T11,T12,T13,T14,T15,T16,T17,T18,T19,T20,T21,T22,T23,T24,T25,T26,T27,M1,M2,M3,M4,M5,M6,M7,M8,M9,M10,M11,M12,M13,M14,M15,M16,M17,M18,M19,M20,M21,M22,M23,M24,M25,M26,M27,M28,M29,M30,M31,M32,M33,M34,M35,M36,M37,M38,M39,M40,M41,M42,M43,M44,M45,M46,M47,M48,M49,M50,M51,M52,M53,M54,M55,M56,M57,M58,M59,M60,M61,M62,M63,L0,L1,L2,L3,L4,L5,L6,L7,L8,L9,L10,L11,L12,L13,L14,L15,L16,L17,L18,L19,L20,L21,L22,L23,L24,L25,L26,L27,L28,L29;
    T1 = a[7] ^ a[4];
    T2 = a[7] ^ a[2];
    T3 = a[7] ^ a[1];
    T4 = a[4] ^ a[2];
    T5 = a[3] ^ a[1];
    T6 = T1 ^ T5;
    T7 = a[6] ^ a[5];
    T8 = a[0] ^ T6;
    T9 = a[0] ^ T7;
    T10 = T6 ^ T7;
    T11 = a[6] ^ a[2];
    T12 = a[5] ^ a[2];
    T13 = T3 ^ T4;
    T14 = T6 ^ T11;
    T15 = T5 ^ T11;
    T16 = T5 ^ T12;
    T17 = T9 ^ T16;
    T18 = a[4] ^ a[0];
    T19 = T7 ^ T18;
    T20 = T1 ^ T19;
    T21 = a[1] ^ a[0];
    T22 = T7 ^ T21;
    T23 = T2 ^ T22;
    T24 = T2 ^ T10;
    T25 = T20 ^ T17;
    T26 = T3 ^ T16;
    T27 = T1 ^ T12;

    M1 = T13 & T6;
    M2 = T23 & T8;
    M3 = T14 ^ M1;
    M4 = T19 & a[0];
    M5 = M4 ^ M1;
    M6 = T3 & T16;
    M7 = T22 & T9;
    M8 = T26 ^ M6;
    M9 = T20 & T17;
    M10 = M9 ^ M6;
    M11 = T1 & T15;
    M12 = T4 & T27;
    M13 = M12 ^ M11;
    M14 = T2 & T10;
    M15 = M14 ^ M11;
    M16 = M3 ^ M2;
    M17 = M5 ^ T24;
    M18 = M8 ^ M7;
    M19 = M10 ^ M15;
    M20 = M16 ^ M13;
    M21 = M17 ^ M15;
    M22 = M18 ^ M13;
    M23 = M19 ^ T25;
    M24 = M22 ^ M23;
    M25 = M22 & M20;
    M26 = M21 ^ M25;
    M27 = M20 ^ M21;
    M28 = M23 ^ M25;
    M29 = M28 & M27;
    M30 = M26 & M24;
    M31 = M20 & M23;
    M32 = M27 & M31;
    M33 = M27 ^ M25;
    M34 = M21 & M22;
    M35 = M24 & M34;
    M36 = M24 ^ M25;
    M37 = M21 ^ M29;
    M38 = M32 ^ M33;
    M39 = M23 ^ M30;
    M40 = M35 ^ M36;
    M41 = M38 ^ M40;
    M42 = M37 ^ M39;
    M43 = M37 ^ M38;
    M44 = M39 ^ M40;
    M45 = M42 ^ M41;
    M46 = M44 & T6;
    M47 = M40 & T8;
    M48 = M39 & a[0];
    M49 = M43 & T16;
    M50 = M38 & T9;
    M51 = M37 & T17;
    M52 = M42 & T15;
    M53 = M45 & T27;
    M54 = M41 & T10;
    M55 = M44 & T13;
    M56 = M40 & T23;
    M57 = M39 & T19;
    M58 = M43 & T3;
    M59 = M38 & T22;
    M60 = M37 & T20;
    M61 = M42 & T1;
    M62 = M45 & T4;
    M63 = M41 & T2;

    L0 = M61 ^ M62;
    L1 = M50 ^ M56;
    L2 = M46 ^ M48;
    L3 = M47 ^ M55;
    L4 = M54 ^ M58;
    L5 = M49 ^ M61;
    L6 = M62 ^ L5;
    L7 = M46 ^ L3;
    L8 = M51 ^ M59;
    L9 = M52 ^ M53;
    L10 = M53 ^ L4;
    L11 = M60 ^ L2;
    L12 = M48 ^ M51;
    L13 = M50 ^ L0;
    L14 = M52 ^ M61;
    L15 = M55 ^ L1;
    L16 = M56 ^ L0;
    L17 = M57 ^ L1;
    L18 = M58 ^ L8;
    L19 = M63 ^ L4;
    L20 = L0 ^ L1;
    L21 = L1 ^ L7;
    L22 = L3 ^ L12;
    L23 = L18 ^ L2;
    L24 = L15 ^ L9;
    L25 = L6 ^ L10;
    L26 = L7 ^ L9;
    L27 = L8 ^ L10;
    L28 = L11 ^ L14;
    L29 = L11 ^ L17;
    a[7] = L6 ^ L24;
    a[6] = (~(L16^L26))&1;
    a[5] = (~(L19^L28))&1;
    a[4] = L6 ^ L21;
    a[3] = L20 ^ L22;
    a[2] = L25 ^ L29;
    a[1] = (~(L13^L27))&1;
    a[0] = (~(L6^L23))&1;
}

__device__ void shiftRows(uint128_t a[8]){
    for(int i=0; i<8; i++){
        a[i].hi = __byte_perm((uint64_t)(a[i].hi),0, 0b0010000100000011);
        a[i].lo = ((uint64_t)__byte_perm(a[i].lo>>32,0, 0b0001000000110010)<<32) | __byte_perm((uint64_t)(a[i].lo),0, 0b0000001100100001);
    }
}

__device__ void addRoundKey(uint128_t a[8], uint128_t key[8]){
    for(int i=0; i<8; i++){
        a[i] ^= key[i];
    }
}

__global__ void encrypt(char*plain, uint128_t keys[ROUND_KEY_COUNT][8], char*cypher) {
    uint128_t a[8];
    plain = plain + (128*8) * (blockIdx.x*blockDim.x + threadIdx.x);
    
    bitorder_transform(plain, a);
    
    addRoundKey(a, keys[0]);
    for(int i=0; i< NR; i++){
        subBytes(a);
        shiftRows(a);
        mixColumns(a);
        addRoundKey(a, keys[i]);
    }

    subBytes(a);
    shiftRows(a);
    addRoundKey(a, keys[NR]);

    bitorder_retransform(cypher, (uint128_t*)a);
}

void printHex(char* ptr, int len){
    for(int i=0; i<len; i++){
        printf("%x ", ptr[i]);
    }
    puts("");
}

// TODO: consider, keys need to be bit sliced
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
    uint128_t d_roundkey[ROUND_KEY_COUNT][8];
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
        plain[i] = (char)0;
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