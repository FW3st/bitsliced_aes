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

## Usage
 - run the "init.sh" script to pull and build th sub repositorys
 - "compile&run.sh" runs the benchmark or a verification test if a parameter is given

Outputs on a Nvidia A100:
```
GPU: 39999 Mbytes in 251.169220 ms
Makes 1274.038200 Gbps
AVX: 39999 Mbytes in 41.234511 s
Makes 1.274038 Gbps
passed
9e f8 b1 ad | 3a 17 b4 b1 | 8e f2 dc 6c | 03 0e d3 60 | 9e f8 b1 ad |
9e f8 b1 ad | 3a 17 b4 b1 | 8e f2 dc 6c | 03 0e d3 60 | 9e f8 b1 ad |
```
