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
GPU: 1999 Mbytes in 29.345793 ms
Makes 545.221564 Gbps
AVX: 1999 Mbytes in 1.414386 s
Makes 0.545222 Gbps
passed
```
