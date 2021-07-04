#include <stdio.h>
#include <stdint.h>
#include <unistd.h>
#include "cuda_uint128.h"
#include <time.h>
#include "device.h"
#include "aes_cbc.h"
#include "utils.h"

#define WORDSIZE 32
#define KEY_SIZE 128

#define NK KEY_SIZE/WORDSIZE
#define NR (NK + 6)
#define ROUND_KEY_COUNT (NR + 1)
#define ROUND_KEY_SIZE ROUND_KEY_COUNT * KEY_SIZE

#define SERIAL 1       // let a Thread process n aes blocks
#define THREADPARA 64 // number of threads in a working group
#define INPLACE 1      // comment out to disable in place encryption
#define NENCRYPT 1     // encrypt n-times, only for test purpose
#define DATAINGIGS 40  // desired plaintext size in GB

#define NUM_BLOCKS 7812480lu*DATAINGIGS/SERIAL*SERIAL/THREADPARA*THREADPARA
#define BLOCK_SIZE 128lu
#define PLAIN_SIZE NUM_BLOCKS*BLOCK_SIZE


// from [13] A Fast and Cache-Timing Resistant Implementation of the AES
__device__ static void swapMove(uint128_t* __restrict__  a , uint128_t* __restrict__  b, uint128_t m, int n){
    uint128_t t = ((((*a)>>n)^(*b)))&m;
    *b = (*b) ^ t;
    *a = (*a) ^ (t << n);
}


__device__ void swapMove(unsigned char* a, int i, int j){
    unsigned char tmp = a[i];
    a[i] = a[j];
    a[j]=tmp;
}

__device__ void reverseBundle(unsigned char* __restrict__  plain, uint128_t* __restrict__  a){
    const uint128_t m1 = (uint128_t) 0x5555555555555555 << 64 | 0x5555555555555555;
    const uint128_t m2 = (uint128_t) 0x3333333333333333 << 64 | 0x3333333333333333;
    const uint128_t m3 = (uint128_t) 0x0f0f0f0f0f0f0f0f << 64 | 0x0f0f0f0f0f0f0f0f;

    swapMove(a,   a+4, m3, 4);
    swapMove(a+1, a+5, m3, 4);
    swapMove(a+2, a+6, m3, 4);
    swapMove(a+3, a+7, m3, 4);

    swapMove(a,   a+2, m2, 2);
    swapMove(a+1, a+3, m2, 2);
    swapMove(a+4, a+6, m2, 2);
    swapMove(a+5, a+7, m2, 2);

    swapMove(a,   a+1, m1, 1);
    swapMove(a+2, a+3, m1, 1);
    swapMove(a+4, a+5, m1, 1);
    swapMove(a+6, a+7, m1, 1);

    for(int i=0; i<8; i++){
        swapMove((unsigned char*)(void*)&a[i], 1, 4);
        swapMove((unsigned char*)(void*)&a[i], 2, 8);
        swapMove((unsigned char*)(void*)&a[i], 3,12);
        swapMove((unsigned char*)(void*)&a[i], 6, 9);
        swapMove((unsigned char*)(void*)&a[i],13, 7);
        swapMove((unsigned char*)(void*)&a[i],14,11);
    }

    for(int i=0; i<8; i++){
        ((uint128_t*)plain)[i] = a[i];
    }
}


__device__ void createBundle(unsigned char* __restrict__  plain, uint128_t* __restrict__  a){
    const uint128_t m1 = (uint128_t) 0x5555555555555555 << 64 | 0x5555555555555555;
    const uint128_t m2 = (uint128_t) 0x3333333333333333 << 64 | 0x3333333333333333;
    const uint128_t m3 = (uint128_t) 0x0f0f0f0f0f0f0f0f << 64 | 0x0f0f0f0f0f0f0f0f;

    for(int i=0; i<8; i++){
        a[i] = ((uint128_t*)plain)[i];
    }

    for(int i=0; i<8; i++){ //TODO possible to improove?
        swapMove((unsigned char*)(void*)&a[i], 1, 4);
        swapMove((unsigned char*)(void*)&a[i], 2, 8);
        swapMove((unsigned char*)(void*)&a[i], 3,12);
        swapMove((unsigned char*)(void*)&a[i], 6, 9);
        swapMove((unsigned char*)(void*)&a[i],13, 7);
        swapMove((unsigned char*)(void*)&a[i],14,11);
    }
    swapMove(a,   a+1, m1, 1);
    swapMove(a+2, a+3, m1, 1);
    swapMove(a+4, a+5, m1, 1);
    swapMove(a+6, a+7, m1, 1);

    swapMove(a,   a+2, m2, 2);
    swapMove(a+1, a+3, m2, 2);
    swapMove(a+4, a+6, m2, 2);
    swapMove(a+5, a+7, m2, 2);

    swapMove(a,   a+4, m3, 4);
    swapMove(a+1, a+5, m3, 4);
    swapMove(a+2, a+6, m3, 4);
    swapMove(a+3, a+7, m3, 4);
}

__device__ static uint128_t rot(uint128_t a, const int N){
    return a>>N | a<<(128-N);
}

__device__ void mixColumns(uint128_t a[8]){
    static const int N1 = 32;
    static const int N2 = 32*2;
    uint128_t t0 = (a[7]^rot(a[7],N1))^rot(a[0],N1)^rot((a[0]^rot(a[0],N1)),N2);
    uint128_t t1 = (a[0]^rot(a[0],N1))^(a[7]^rot(a[7],N1))^rot(a[1],N1)^rot((a[1]^rot(a[1],N1)),N2);
    uint128_t t2 = (a[1]^rot(a[1],N1))^rot(a[2],N1)^rot((a[2]^rot(a[2],N1)),N2);
    uint128_t t3 = (a[2]^rot(a[2],N1))^(a[7]^rot(a[7],N1))^rot(a[3],N1)^rot((a[3]^rot(a[3],N1)),N2);
    uint128_t t4 = (a[3]^rot(a[3],N1))^(a[7]^rot(a[7],N1))^rot(a[4],N1)^rot((a[4]^rot(a[4],N1)),N2);
    uint128_t t5 = (a[4]^rot(a[4],N1))^rot(a[5],N1)^rot((a[5]^rot(a[5],N1)),N2);
    uint128_t t6 = (a[5]^rot(a[5],N1))^rot(a[6],N1)^rot((a[6]^rot(a[6],N1)),N2);
    uint128_t t7 = (a[6]^rot(a[6],N1))^rot(a[7],N1)^rot((a[7]^rot(a[7],N1)),N2);

   a[0]=t0;
   a[1]=t1;
   a[2]=t2;
   a[3]=t3;
   a[4]=t4;
   a[5]=t5;
   a[6]=t6;
   a[7]=t7;
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
    a[6] = (~(L16^L26));
    a[5] = (~(L19^L28));
    a[4] = L6 ^ L21;
    a[3] = L20 ^ L22;
    a[2] = L25 ^ L29;
    a[1] = (~(L13^L27));
    a[0] = (~(L6^L23));
}

__device__ void shiftRows(uint128_t a[8]){
    for(int i=0; i<8; i++){
        a[i].lo = (uint64_t)__byte_perm((uint64_t)(a[i].lo)>>32,0,0b0000001100100001)<<32 | a[i].lo&0xffffffff;
        a[i].hi = ((uint64_t)__byte_perm(a[i].hi>>32,0,
        0b0010000100000011)<<32) | __byte_perm((uint64_t)(a[i].hi),0, 0b0001000000110010);
    }
}

__device__ void addRoundKey(uint128_t* __restrict__ a, uint128_t* __restrict__  key){
    for(int i=0; i<8; i++){
        a[i] ^= key[i];
    }
}
#ifdef INPLACE
__global__ void encrypt(unsigned char*  plain, uint128_t* keys){
#else
__global__ void encrypt(unsigned char*  __restrict__ plain, unsigned char*  __restrict__ cypher, uint128_t* __restrict__ keys){
    cypher = cypher + BLOCK_SIZE * (blockIdx.x*SERIAL*THREADPARA+threadIdx.x);
#endif
    uint128_t a[8];
    plain = plain + BLOCK_SIZE * (blockIdx.x*SERIAL*THREADPARA+threadIdx.x);
    for(int j=0; j<SERIAL;j++){
        createBundle(plain, a);
        addRoundKey(a, keys);
        for(int i=1; i< NR; i++){
            subBytes(a);
            shiftRows(a);
            mixColumns(a);
            addRoundKey(a, keys+i*8);
        }
        subBytes(a);
        shiftRows(a);
        addRoundKey(a, keys+8*10);

#ifdef INPLACE
        reverseBundle(plain, (uint128_t*)a);
#else
        reverseBundle(cypher, (uint128_t*)a);
        cypher = cypher + BLOCK_SIZE*THREADPARA;
#endif
        plain = plain + BLOCK_SIZE*THREADPARA;
    }
}

void subWord(unsigned char word[4], unsigned char result[4]){

  static const unsigned char sbox[16][16] = {{0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76},
                                {0xca, 0x82, 0xc9, 0x7d, 0xfa, 0x59, 0x47, 0xf0, 0xad, 0xd4, 0xa2, 0xaf, 0x9c, 0xa4, 0x72, 0xc0},
                                {0xb7, 0xfd, 0x93, 0x26, 0x36, 0x3f, 0xf7, 0xcc, 0x34, 0xa5, 0xe5, 0xf1, 0x71, 0xd8, 0x31, 0x15},
                                {0x04, 0xc7, 0x23, 0xc3, 0x18, 0x96, 0x05, 0x9a, 0x07, 0x12, 0x80, 0xe2, 0xeb, 0x27, 0xb2, 0x75},
                                {0x09, 0x83, 0x2c, 0x1a, 0x1b, 0x6e, 0x5a, 0xa0, 0x52, 0x3b, 0xd6, 0xb3, 0x29, 0xe3, 0x2f, 0x84},
                                {0x53, 0xd1, 0x00, 0xed, 0x20, 0xfc, 0xb1, 0x5b, 0x6a, 0xcb, 0xbe, 0x39, 0x4a, 0x4c, 0x58, 0xcf},
                                {0xd0, 0xef, 0xaa, 0xfb, 0x43, 0x4d, 0x33, 0x85, 0x45, 0xf9, 0x02, 0x7f, 0x50, 0x3c, 0x9f, 0xa8},
                                {0x51, 0xa3, 0x40, 0x8f, 0x92, 0x9d, 0x38, 0xf5, 0xbc, 0xb6, 0xda, 0x21, 0x10, 0xff, 0xf3, 0xd2},
                                {0xcd, 0x0c, 0x13, 0xec, 0x5f, 0x97, 0x44, 0x17, 0xc4, 0xa7, 0x7e, 0x3d, 0x64, 0x5d, 0x19, 0x73},
                                {0x60, 0x81, 0x4f, 0xdc, 0x22, 0x2a, 0x90, 0x88, 0x46, 0xee, 0xb8, 0x14, 0xde, 0x5e, 0x0b, 0xdb},
                                {0xe0, 0x32, 0x3a, 0x0a, 0x49, 0x06, 0x24, 0x5c, 0xc2, 0xd3, 0xac, 0x62, 0x91, 0x95, 0xe4, 0x79},
                                {0xe7, 0xc8, 0x37, 0x6d, 0x8d, 0xd5, 0x4e, 0xa9, 0x6c, 0x56, 0xf4, 0xea, 0x65, 0x7a, 0xae, 0x08},
                                {0xba, 0x78, 0x25, 0x2e, 0x1c, 0xa6, 0xb4, 0xc6, 0xe8, 0xdd, 0x74, 0x1f, 0x4b, 0xbd, 0x8b, 0x8a},
                                {0x70, 0x3e, 0xb5, 0x66, 0x48, 0x03, 0xf6, 0x0e, 0x61, 0x35, 0x57, 0xb9, 0x86, 0xc1, 0x1d, 0x9e},
                                {0xe1, 0xf8, 0x98, 0x11, 0x69, 0xd9, 0x8e, 0x94, 0x9b, 0x1e, 0x87, 0xe9, 0xce, 0x55, 0x28, 0xdf},
                                {0x8c, 0xa1, 0x89, 0x0d, 0xbf, 0xe6, 0x42, 0x68, 0x41, 0x99, 0x2d, 0x0f, 0xb0, 0x54, 0xbb, 0x16}};

  for(int i=0; i<4; i++){
    result[i] = sbox[(int)(word[i]>>4)][(int)(word[i]&0x0F)];
  }
}

void bitslice_key(unsigned char exkey[176], unsigned char slicedkey[11][8][16]){
  for(int i=0; i<176; i++){
    if((exkey[i] & 0x80) != 0)
      slicedkey[i/16][7][(i%4)*4+(i%16)/4] = 0xff;
    else
      slicedkey[i/16][7][(i%4)*4+(i%16)/4] = 0x00;
    if((exkey[i] & 0x40) != 0)
      slicedkey[i/16][6][(i%4)*4+(i%16)/4] = 0xff;
    else
      slicedkey[i/16][6][(i%4)*4+(i%16)/4] = 0x00;
    if((exkey[i] & 0x20) != 0)
      slicedkey[i/16][5][(i%4)*4+(i%16)/4] = 0xff;
    else
      slicedkey[i/16][5][(i%4)*4+(i%16)/4] = 0x00;
    if((exkey[i] & 0x10) != 0)
      slicedkey[i/16][4][(i%4)*4+(i%16)/4] = 0xff;
    else
      slicedkey[i/16][4][(i%4)*4+(i%16)/4] = 0x00;
    if((exkey[i] & 0x08) != 0)
      slicedkey[i/16][3][(i%4)*4+(i%16)/4] = 0xff;
    else
      slicedkey[i/16][3][(i%4)*4+(i%16)/4] = 0x00;
    if((exkey[i] & 0x04) != 0)
      slicedkey[i/16][2][(i%4)*4+(i%16)/4] = 0xff;
    else
      slicedkey[i/16][2][(i%4)*4+(i%16)/4] = 0x00;
    if((exkey[i] & 0x02) != 0)
      slicedkey[i/16][1][(i%4)*4+(i%16)/4] = 0xff;
    else
      slicedkey[i/16][1][(i%4)*4+(i%16)/4] = 0x00;
    if((exkey[i] & 0x01) != 0)
      slicedkey[i/16][0][(i%4)*4+(i%16)/4] = 0xff;
    else
      slicedkey[i/16][0][(i%4)*4+(i%16)/4] = 0x00;
  }
}

static inline uint32_t bit_length(const uint32_t x) {
  uint32_t y;
  asm ("\tbsr %1, %0\n"
      : "=r"(y)
      : "r" (x)
  );
  return y+1;
}

unsigned char modg(unsigned int a){
    unsigned int m = 0b100011011;
    unsigned int end = m;
    if(a < m){
        if (a > 255){
            a = a ^ m;
        }
        return (unsigned char)a;
    }
    m = m << (bit_length(a) - bit_length(m));
    unsigned int mask = 1 << (bit_length(m) - 1);
    while(1){
        if((a & mask) != 0){
            a ^= m;
        }
        if(m == end){
            return (unsigned char)a;
        }
        m >>= 1;
        mask >>= 1;
    }
}

void create_round_key(unsigned char* key, unsigned char* roundkey){
  //Die Runden anhand des Schlüssels bestimmen
  int rounds = 10;
  if(KEY_SIZE == 192)
    rounds = 12;
  else if(KEY_SIZE == 256)
    rounds = 14;
  //Anzahl der 4 Byte words ermitteln:
  int number_key_words = KEY_SIZE/(8*4);
  //Die ersten Schlüsselwords werden einfach übertragen
  for(int i = 0; i<number_key_words*4; i++)
    roundkey[i] = key[i];

  //Berechnen der weiteren Keys
  unsigned char tmpL[4];
  unsigned char rC[4];
  unsigned char subL[4] = {0x00, 0x00, 0x00, 0x00};
  for(int i = number_key_words; i<=(number_key_words*(rounds+1)); i++){
    tmpL[0] = roundkey[i*4-4];
    tmpL[1] = roundkey[i*4+1-4];
    tmpL[2] = roundkey[i*4+2-4];
    tmpL[3] = roundkey[i*4+3-4];
    if(i%number_key_words==0){
      //Rundenkonstante ermitteln 2^(i/number_key_words-1)
      //rC[0] = pow(2, (floor(i/number_key_words)-1));
      rC[0] = modg(1<<(i/number_key_words-1));
      rC[1] = 0x00;
      rC[2] = 0x00;
      rC[3] = 0x00;
      //Rotiere nach links und wende SBOX an und xor mit Rundenkonstante
      subWord(tmpL, subL);
      tmpL[0] = subL[1]^rC[0];
      tmpL[1] = subL[2]^rC[1];
      tmpL[2] = subL[3]^rC[2];
      tmpL[3] = subL[0]^rC[3];
    }
    else if(number_key_words == 8 && i%8 == 4){
      subWord(tmpL, subL);
      tmpL[0] = subL[0];
      tmpL[1] = subL[1];
      tmpL[2] = subL[2];
      tmpL[3] = subL[3];
    }
    roundkey[i*4] = roundkey[i*4-number_key_words*4] ^ tmpL[0];
    roundkey[i*4+1] = roundkey[i*4+1-number_key_words*4] ^ tmpL[1];
    roundkey[i*4+2] = roundkey[i*4+2-number_key_words*4] ^ tmpL[2];
    roundkey[i*4+3] = roundkey[i*4+3-number_key_words*4] ^ tmpL[3];
  }
}


#ifndef TEST // Q'n'D ToDo
int main(void) {
    time_t t;
    srand((unsigned) time(&t));
    //print_device_info();
    unsigned char* d_plain;
    unsigned char* d_cypher;
    uint128_t* d_roundkey;
    cudaEvent_t start, stop;
    float time;

    cudaSetDevice(DEVICE);

    cudaMalloc((void**)&d_plain, PLAIN_SIZE);
#ifdef INPLACE
    d_cypher = d_plain;
#else
    cudaMalloc((void**)&d_cypher, PLAIN_SIZE);
#endif
    cudaMalloc((void**)&d_roundkey, ROUND_KEY_SIZE);

    unsigned char* plain = (unsigned char*) malloc(PLAIN_SIZE);
    unsigned char* cypher = (unsigned char*) malloc(PLAIN_SIZE);
    unsigned char* key = (unsigned char*) malloc(KEY_SIZE);
    unsigned char* roundkey = (unsigned char*) malloc(ROUND_KEY_SIZE);
    unsigned char* bs_roundkey = (unsigned char*) malloc(ROUND_KEY_SIZE);


    fill_random((int*)key, KEY_SIZE/4);
    create_round_key(key, roundkey);
    bitslice_key(roundkey, (unsigned char (*)[8][16])bs_roundkey);

    cudaMemcpy(d_roundkey, bs_roundkey, ROUND_KEY_SIZE, cudaMemcpyHostToDevice);
#ifdef INPLACE
        encrypt<<<NUM_BLOCKS/SERIAL/THREADPARA,THREADPARA>>>(d_plain, d_roundkey); // ^= fill random
#else
        encrypt<<<NUM_BLOCKS/SERIAL/THREADPARA,THREADPARA>>>(d_plain, d_cypher, d_roundkey);
#endif
    cudaMemcpy(plain, d_plain, PLAIN_SIZE, cudaMemcpyDeviceToHost); // write back for verification

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);
    for(int i=0; i<NENCRYPT; i++){
#ifdef INPLACE
        encrypt<<<NUM_BLOCKS/SERIAL/THREADPARA,THREADPARA>>>(d_plain, d_roundkey);
#else
        encrypt<<<NUM_BLOCKS/SERIAL/THREADPARA,THREADPARA>>>(d_plain, d_cypher, d_roundkey);
#endif
    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);

    cudaMemcpy(cypher, d_cypher, PLAIN_SIZE, cudaMemcpyDeviceToHost);

    printf("GPU: %lu Mbytes in %f ms\n", PLAIN_SIZE/1000/1000*NENCRYPT, time);
    printf("Makes %f Gbps\n", 1.0*PLAIN_SIZE*1000/1000/time/1000/1000*8*NENCRYPT);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_roundkey);
#ifndef INPLACE
    cudaFree(d_cypher);
#endif
    cudaFree(d_plain);

    //VERIFY AVX AES
    clock_t startav, endav;
    double cpu_time_usedav;
    struct cbc_key_data avx_roundkeys;

    uint8_t iv[16];
    unsigned char* out = (unsigned char*)malloc(BLOCK_SIZE*NUM_BLOCKS);
    memset(iv, 0, 16);

    aes_cbc_precomp((uint8_t*)key,CBC_128_BITS,&avx_roundkeys);
    startav = clock();
    for(unsigned long i=0; i<8*NUM_BLOCKS; i++){
        aes_cbc_enc_128(plain+16*i, iv, avx_roundkeys.enc_keys,out+16*i, 16);
    }
    endav = clock();
    cpu_time_usedav = ((double) (endav - startav)) / CLOCKS_PER_SEC;
    printf("AVX: %lu Mbytes in %f s\n", PLAIN_SIZE/1000/1000*NENCRYPT, cpu_time_usedav);
    printf("Makes %f Gbps\n", 1.0*PLAIN_SIZE/1000/time/1000/1000*8*NENCRYPT);

    if(memcmp(out, cypher,  PLAIN_SIZE)!= 0){
        printf("failed\n");
        printError();
    } else {
        printf("passed\n");
    }

    int ran_line = rand()%(PLAIN_SIZE-25);
    printHex(cypher+ran_line,20);
    printHex(out+ran_line,20);

    free(out);

    free(bs_roundkey);
    free(roundkey);
    free(key);
    free(cypher);
    return 0;
}
#endif
