# Computer Security 2021:  Bitsliced AES

## Based on:
- "Implementation of Bitsliced AES Encryption
on CUDA-Enabled GPU", Naoki Nishikawa, Hideharu Amano and Keisuke Iwai
- "A Fast and Cache-Timing Resistant Implementation
of the AES*", Robert Könighofer
- "A Small Depth-16 Circuit for the AES S-Box", Joan Boyar and René Peralta

## Using following repos:
 - "CUDA-uint128-master", see https://github.com/curtisseizert/CUDA-uint128
 - "isa-l_crypto", see https://github.com/intel/isa-l_crypto (for verification purpose)

Outputs on a Nvidia A100:
```
GPU: 3999 Mbytes in 36.316158 ms
Makes 881.148216 Gbps
AVX: 3999 Mbytes in 2.503652 s
Makes 0.881148 Gbps
passed
51 dc 1b 3f | 8f 13 08 3c | 7a f8 c7 af | c6 48 19 2d | 08 d1 37 09 |
51 dc 1b 3f | 8f 13 08 3c | 7a f8 c7 af | c6 48 19 2d | 08 d1 37 09 |
```
