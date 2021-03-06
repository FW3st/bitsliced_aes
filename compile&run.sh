export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:./isa-l_crypto/bin
export PATH=$PATH:/usr/local/cuda/bin
if [ "$#" != "0" ]
then
    echo "Compile Test:"
    nvcc -G -o test test_bits_aes.cu -I CUDA-uint128 -I isa-l_crypto/include -L isa-l_crypto/bin -lisal_crypto
    ./test
else
    echo "Compile AES:"
    nvcc -o aes bits_aes.cu -I CUDA-uint128 -I isa-l_crypto/include  -L isa-l_crypto/bin -lisal_crypto
    ./aes
fi

