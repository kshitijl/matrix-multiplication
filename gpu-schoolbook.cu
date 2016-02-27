#include "cuda-common.hxx"

template<typename TT>
__global__ void mm_kernel(const TT * AA, const TT * BB, TT * CC, size_t nn) {
    unsigned int ii = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int jj = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int index = ii*nn + jj;
    if(ii < nn and jj < nn) {
        CC[index] = 0;
        for(int kk = 0; kk < nn; ++kk)
            CC[index] += AA[ii*nn + kk] * BB[kk*nn + jj];
    }
}

template<typename TT>
void matrix_multiply(const TT * AA, const TT * BB, TT * CC, size_t nn) {
    TT *da, *db, *dc;

    CUDA_CALL(cudaMalloc((void **) &da, sizeof(TT)*nn*nn));
    CUDA_CALL(cudaMalloc((void **) &db, sizeof(TT)*nn*nn));
    CUDA_CALL(cudaMalloc((void **) &dc, sizeof(TT)*nn*nn));

    CUDA_CALL(cudaMemcpy(da, AA, sizeof(TT)*nn*nn, cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(db, BB, sizeof(TT)*nn*nn, cudaMemcpyHostToDevice));

    dim3 dimGrid(ceil(nn/32.0), ceil(nn/32.0));
    dim3 dimBlock(32,32);
    mm_kernel<<< dimGrid, dimBlock >>>(da, db, dc, nn);
    CUDA_CALL(cudaPeekAtLastError());
    CUDA_CALL(cudaDeviceSynchronize());

    CUDA_CALL(cudaMemcpy(CC, dc, sizeof(TT)*nn*nn, cudaMemcpyDeviceToHost));
    
    CUDA_CALL(cudaFree(da));
    CUDA_CALL(cudaFree(db));
    CUDA_CALL(cudaFree(dc));
}

template void matrix_multiply<float>(const float*, const float*, float*, size_t);
template void matrix_multiply<double>(const double*, const double*, double*, size_t);
