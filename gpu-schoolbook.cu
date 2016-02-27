#include <cuda.h>
#include <cuda_runtime.h>

template<typename TT>
__global__ void mm_kernel(const TT * AA, const TT * BB, TT * CC, size_t nn) {
    unsigned int ii = blockIdx.x; // * blockDim.x + threadIdx.x;
    unsigned int jj = threadIdx.x; // kddi
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

    cudaMalloc((void **) &da, sizeof(TT)*nn*nn);
    cudaMalloc((void **) &db, sizeof(TT)*nn*nn);
    cudaMalloc((void **) &dc, sizeof(TT)*nn*nn);        

    cudaMemcpy(da, AA, sizeof(TT)*nn*nn, cudaMemcpyHostToDevice);
    cudaMemcpy(db, BB, sizeof(TT)*nn*nn, cudaMemcpyHostToDevice);    

    dim3 dimGrid(nn, 1);
    dim3 dimBlock(nn,1);
    mm_kernel<<< dimGrid, dimBlock >>>(da, db, dc, nn);

    cudaMemcpy(CC, dc, sizeof(TT)*nn*nn, cudaMemcpyDeviceToHost);

    cudaFree(da);
    cudaFree(db);
    cudaFree(dc);
}

template void matrix_multiply<float>(const float*, const float*, float*, size_t);
template void matrix_multiply<double>(const double*, const double*, double*, size_t);
