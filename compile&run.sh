if [ "$#" != "0" ] 
then
    echo "Compile Test:"
    nvcc -o test test_bits_aes.cu -I CUDA-uint128-master
    ./test
else
    echo "Compile AES:"
    nvcc -o aes bits_aes.cu -I CUDA-uint128-master
    ./aes
fi
