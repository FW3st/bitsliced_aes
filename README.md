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
GPU: 39999 Mbytes in 234.436417 ms
Makes 1364.972155 Gbps
AVX: 39999 Mbytes in 41.961212 s
Makes 1.350191 Gbps
passed
ac 3b 8b 50 | 05 15 67 78 | 5b 18 e1 24 | ad b9 e0 b4 | ac 3b 8b 50 |
ac 3b 8b 50 | 05 15 67 78 | 5b 18 e1 24 | ad b9 e0 b4 | ac 3b 8b 50 |
```
