#define TEST
#include "bits_aes.cu"
#include "aes_cbc.h"


const uint8_t SBOX[16][16] = {{0x63, 0x7c, 0x77, 0x7b, 0xf2, 0x6b, 0x6f, 0xc5, 0x30, 0x01, 0x67, 0x2b, 0xfe, 0xd7, 0xab, 0x76},
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

unsigned char substitute(unsigned char c){
    return SBOX[c >> 4][c & 0xf];
}

char get_byte(char a[8][16], int n){
    int c = n/8;
    int b = 7-n%8;
    char ret = 0;
    for(int i = 7; i>=0; i--){
        ret = (ret << 1) | ((a[i][c]>>b)&1);
    }
    return ret;    
}

void printError(){
    cudaError_t error = cudaGetLastError ();
    printf("error: %s\n",cudaGetErrorName(error) );
}

void fill_random(int* ar, int len){
    for(int i=0; i<len; i++){
        ar[i] = rand();
    }
}

__global__ void __test_bitorder_transform(char*plain, uint128_t transformed[8]){
    bitorder_transform(plain, transformed);
}

void check_bitorder(char (*raw)[16], char (*ord)[16]){
    char x,y;
    for(int by=0; by<16; by++){
        for(int bi=0; bi<8; bi++){
            for(int d=0; d<8; d++){
                x = (raw[d][by]>>bi)&1;
                y = (ord[bi][by]>>(7-d))&1;
                if(x!=y){
                    printf("check_bitorder failed\n");
                    printf("x:%i, y:%i\n", x, y);
                    printf("by:%i, by:%i, d:%i\n", by, bi,d);
                    return;
                }
            }
        }
    }
    printf("check_bitorder passed\n");
}

__global__ void __test_bitorder_retransform(char*plain, uint128_t transformed[8]){
     bitorder_retransform(plain, transformed);
}

void check_bitreorder(char* raw, char* reo){
    if(memcmp(raw, reo, 16*8) != 0){
        printf("check_bitreorder failed\n");
        return;
    }
    printf("check_bitreorder passed\n");
}

uint128_t touint128(void* ar){
    uint64_t* inp = (uint64_t*) ar;
    uint128_t ret;
    ret.lo = inp[0];
    ret.hi = inp[1];
    return ret;
}


void check_encrypt(char* plain, char* key){
    //intel avx cbc aes, n times without IV ~> ecb
    struct cbc_key_data round_keys;
    uint8_t iv[16];
    uint8_t out[16*8];
    memset(iv, 0, 128);
    aes_cbc_precomp((uint8_t*)key,CBC_128_BITS,&round_keys);
    for(int i=0; i<8; i++){
        aes_cbc_enc_128(plain+16*i, iv, round_keys.enc_keys,out+16*i, 16);
    }
    
    //bitsliced aes
    uint8_t* bs_out[16*8];
    char *d_plain;
    uint128_t d_roundkey[11][8];
    char *d_cypher;
    char* bs_roundkey = (char*) malloc(1408);
    create_round_key(key, bs_roundkey);

    cudaMalloc((void**)&d_plain, 16*8);
    cudaMalloc((void**)&d_cypher, 16*8);
    cudaMalloc((void**)&d_roundkey, 1408);

    cudaMemcpy(d_plain, plain, 16*8, cudaMemcpyHostToDevice);
    cudaMemcpy(d_roundkey, bs_roundkey, 1408, cudaMemcpyHostToDevice);

    encrypt<<<NUM_BLOCKS,1>>>(d_plain, d_roundkey, d_cypher);

    cudaMemcpy(bs_out, d_cypher, 16*8, cudaMemcpyDeviceToHost);

    if(memcpy(bs_out,out,16*8)==0){
        printf("Encrypt passed\n");
    } else {
        printf("Encrypt failed\n");
    }
}

void check_addRoundKey(){
    printf("check_addRoundKey failed\n");
}

__global__ void __test_subBytes(uint128_t a[8]){
    subBytes(a);
}

void check_subBytes(char a[8][16], char s[8][16]){
    for(int i=0; i<128; i++){
        unsigned char ac = get_byte(a,i);
        unsigned char sc = get_byte(s,i);
        unsigned char sub = substitute(ac);
        if(sub != sc){
            printf("check_subBytes failed\n");
            return;
        }
    }
    printf("check_subBytes passed\n");
}

__global__ void __test_shiftRows(uint128_t a[8]){
    shiftRows(a);
}

void check_shiftRows(char a[8][16], char s[8][16]){
    int fail = 0;
    for(int i=0; i<8; i++){
        // no shift
        fail += abs(memcmp(a[i],s[i],4));
        // one byte
        fail += abs(memcmp(a[i]+5,s[i]+4,3));
        fail += a[i][4]!=s[i][7];
        // zwo bytes
        fail += abs(memcmp(a[i]+8,s[i]+10,2));
        fail += abs(memcmp(a[i]+10,s[i]+8,2));
        // three bytes
        fail += abs(memcmp(a[i]+12,s[i]+13,3));
        fail += a[i][15]!=s[i][12];
        
        if(fail != 0){
            printf("check_shiftRows failed\n");
            return;
        }
    }
    printf("check_shiftRows passed\n");
}

int main(void) {
    time_t t;
    srand((unsigned) time(&t));
    uint128_t inp128[8];
    uint128_t inp1282[8];
    uint128_t out128[8];
    uint128_t out1282[8];
    uint128_t* inp128_cuda;
    uint128_t* out128_cuda;
    uint128_t* out128_cuda2;
    
    char (*raw)[16] = (char(*)[16]) malloc(16*8);
    char (*raw2)[16] = (char(*)[16]) malloc(16*8);
    int* ran_buf = (int*) raw;
    int* ran_buf2 = (int*) raw2;
    //memset(ran_buf,0,16*8);
    //raw[0][0] = 0x80;
    for(int i = 0; i<8; i++){
        fill_random(ran_buf, 32);
        fill_random(ran_buf2, 32);
        inp128[i] = touint128(ran_buf+4*i);
        inp1282[i] = touint128(ran_buf2+4*i);
    }        
    
    cudaMalloc((void**)&inp128_cuda, 16*8);
    cudaMalloc((void**)&out128_cuda, 16*8);
    cudaMalloc((void**)&out128_cuda2, 16*8);

    // CHECK bitorder_transform
    cudaMemcpy(inp128_cuda, inp128, 16*8, cudaMemcpyHostToDevice);
    __test_bitorder_transform<<<1,1>>>((char*)((void*)inp128_cuda),out128_cuda);
    cudaMemcpy(out128, out128_cuda, 16*8, cudaMemcpyDeviceToHost);
    check_bitorder( (char(*)[16]) ((void*)inp128), (char(*)[16]) ((void*)out128));

    // CHECK bitorder_retransform
    __test_bitorder_retransform<<<1,1>>>((char*)((void*)out128_cuda2),out128_cuda);
    cudaMemcpy(out1282, out128_cuda2, 16*8, cudaMemcpyDeviceToHost);
    check_bitreorder((char(*)) ((void*)inp128), (char(*)) ((void*)out1282));

    // CHECK addRoundKey
    check_addRoundKey();

    // CHECK subBytes
    cudaMemcpy(inp128_cuda, inp128, 16*8, cudaMemcpyHostToDevice);
    __test_subBytes<<<1,1>>>(inp128_cuda);
    cudaMemcpy(out128, inp128_cuda, 16*8, cudaMemcpyDeviceToHost);
    check_subBytes((char(*)[16]) ((void*)inp128), (char(*)[16]) ((void*)out128));

    // CHECK shiftRows
    cudaMemcpy(inp128_cuda, inp128, 16*8, cudaMemcpyHostToDevice);
    __test_shiftRows<<<1,1>>>(inp128_cuda);
    cudaMemcpy(out128, inp128_cuda, 16*8, cudaMemcpyDeviceToHost);
    check_shiftRows((char(*)[16]) ((void*)inp128), (char(*)[16]) ((void*)out128));

    // CHECK encrypt
    check_encrypt((char*)raw, (char*)raw2);
    //cudaDeviceSynchronize();

    cudaFree(out128_cuda2);
    cudaFree(out128_cuda);
    cudaFree(inp128_cuda);
    return 0;
}