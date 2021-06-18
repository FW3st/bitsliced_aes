if [ "$#" != "0" ] 
then
    echo "Compile Test:"
    nvcc -o test test_bits_aes.cu -I CUDA-uint128 -I isa-l_crypto/include -L isa-l_crypto/bin -lisal_crypto
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:./isa-l_crypto/bin
    ./test
else
    echo "Compile AES:"
    nvcc -o aes bits_aes.cu -I CUDA-uint128 -I isa-l_crypto/include -L isa-l_crypto/bin -lisal_crypto
    ./aes
fi
