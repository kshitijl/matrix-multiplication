#include "cuda-common.hxx"

const int TILESIZE = 16;

template<typename TT>
__global__ void mm_kernel(const TT * AA, const TT * BB, TT * CC, size_t nn) {
    __shared__ TT sA[TILESIZE][TILESIZE];
    __shared__ TT sB[TILESIZE][TILESIZE];

    // each thread computes 1 element of the answer
    unsigned int row_C = blockIdx.y * TILESIZE + threadIdx.y;
    unsigned int col_C = blockIdx.x * TILESIZE + threadIdx.x;

    // loop over all the tiles necessary to compute the answer
    TT answer = 0;
    for(int tile = 0; tile < nn/TILESIZE; ++tile) {
        // all threads cooperatively load a tile of A and a tile of B
        // into shared memory
        sA[threadIdx.y][threadIdx.x] = AA[row_C*nn + tile*TILESIZE + threadIdx.x];
        sB[threadIdx.y][threadIdx.x] = BB[(tile*TILESIZE + threadIdx.y)*nn + col_C];
        __syncthreads();

        // now add up the partial answer from this tile. Each thread
        // only uses a single row of AA and a single column of BB
        for(int kk = 0; kk < TILESIZE; ++kk) {
            answer += sA[threadIdx.y][kk] * sB[kk][threadIdx.x];
        }
        __syncthreads();
    }
    CC[row_C*nn + col_C] = answer;
}

template<typename TT>
void matrix_multiply(const TT * AA, const TT * BB, TT * CC, size_t nn) {
    TT *da, *db, *dc;

    CUDA_CALL(cudaMalloc((void **) &da, sizeof(TT)*nn*nn));
    CUDA_CALL(cudaMalloc((void **) &db, sizeof(TT)*nn*nn));
    CUDA_CALL(cudaMalloc((void **) &dc, sizeof(TT)*nn*nn));

    CUDA_CALL(cudaMemcpy(da, AA, sizeof(TT)*nn*nn, cudaMemcpyHostToDevice));
    CUDA_CALL(cudaMemcpy(db, BB, sizeof(TT)*nn*nn, cudaMemcpyHostToDevice));

    dim3 dimGrid(ceil(1.0*nn/TILESIZE), ceil(1.0*nn/TILESIZE));
    dim3 dimBlock(TILESIZE,TILESIZE);
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
