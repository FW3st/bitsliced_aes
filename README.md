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
GPU: 39999 Mbytes in 268.310364 ms
Makes 1192.645622 Gbps
AVX: 39999 Mbytes in 41.007761 s
Makes 1.192646 Gbps
passed
89 1c 10 1b | c2 72 6e bb | e3 bf c6 dc | 53 f3 69 c8 | 89 1c 10 1b |
89 1c 10 1b | c2 72 6e bb | e3 bf c6 dc | 53 f3 69 c8 | 89 1c 10 1b |
```
