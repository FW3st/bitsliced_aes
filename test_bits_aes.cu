#define TEST
#include "bits_aes.cu"
#include "aes_cbc.h"

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
    
    
    // CHECK encrypt
    check_encrypt((char*)raw, (char*)raw2);
    //cudaDeviceSynchronize();
    
    cudaFree(out128_cuda2);
    cudaFree(out128_cuda);
    cudaFree(inp128_cuda);
    return 0;
}