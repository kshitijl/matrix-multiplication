#include "cuda-common.hxx"

const int TILESIZE = 32;

template<typename TT>
__global__ void mm_kernel(const TT * AA, const TT * BB, TT * CC, size_t nn) {
    __shared__ TT sA[TILESIZE][TILESIZE];
    __shared__ TT sB[TILESIZE][TILESIZE];

    unsigned int tx = threadIdx.x, ty = threadIdx.y;

    // each thread computes 1 element of the answer
    unsigned int row_C = blockIdx.y * TILESIZE + ty;
    unsigned int col_C = blockIdx.x * TILESIZE + tx;

    // loop over all the tiles necessary to compute the answer
    TT answer = 0;
    for(int tile = 0; tile < (nn + TILESIZE - 1)/TILESIZE; ++tile) {
        // all threads cooperatively load a tile of A and a tile of B
        // into shared memory
        if(row_C < nn and tile*TILESIZE + tx < nn) {
            // grab AA[row_C][tile*TILESIZE + tx]
            sA[ty][tx] = AA[row_C*nn + tile*TILESIZE + tx];
        } else {
            sA[ty][tx] = 0;
        }
        if(col_C < nn and tile*TILESIZE + ty < nn) {
            // grab BB[tile*TILESIZE + ty][col_C]
            sB[ty][tx] = BB[(tile*TILESIZE + ty)*nn + col_C];
        }
        else {
            sB[ty][tx] = 0;
        }

        __syncthreads();

        // now add up the partial answer from this tile. Each thread
        // only uses a single row of AA and a single column of BB
        for(int kk = 0; kk < TILESIZE; ++kk) {
            answer += sA[ty][kk] * sB[kk][tx];
        }
        __syncthreads();
    }
    if(row_C < nn and col_C < nn)
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
