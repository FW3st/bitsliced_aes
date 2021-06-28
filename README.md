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
GPU: 3999 Mbytes in 28.263424 ms
Makes 1132.202460 Gbps
AVX: 3999 Mbytes in 2.836259 s
Makes 1.132202 Gbps
passed
7d b4 0b 23 | 65 a0 51 04 | 56 46 ea 93 | 43 48 66 71 | 52 ca c0 93 |
7d b4 0b 23 | 65 a0 51 04 | 56 46 ea 93 | 43 48 66 71 | 52 ca c0 93 |
```
