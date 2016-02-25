#include <stdlib.h> // malloc, srand, rand
#include <stddef.h> // size_t
#include <string.h> // stoi

#include <assert.h>

#include <iostream> // std::cout
#include <chrono> // std::chrono::high_resolution_clock

#include "print-matrix.hxx"

template<typename TT>
void matrix_multiply_simple(const TT * AA, const TT * BB, TT * CC, size_t nn) {
    for (size_t ii = 0; ii < nn; ++ii) {
        for (size_t jj = 0; jj < nn; ++jj) {
            size_t index = ii*nn + jj;
            CC[index] = 0;
            for (size_t kk = 0; kk < nn; ++kk) {
                CC[index] += AA[ii*nn + kk]*BB[kk*nn + jj];
            }
        }
    }
}

void timetrial(size_t nn, unsigned long ntrials) {
    
    typedef float real_t;
    real_t *aa, *bb, *cc;

    aa = (real_t *)malloc(sizeof(real_t)*nn*nn);
    bb = (real_t *)malloc(sizeof(real_t)*nn*nn);
    cc = (real_t *)malloc(sizeof(real_t)*nn*nn);

    srand(1);
    for(size_t ii = 0; ii < nn*nn; ++ii) {
        aa[ii] = (real_t)rand() / RAND_MAX;
        bb[ii] = (real_t)rand() / RAND_MAX;
    }

    auto begin = std::chrono::high_resolution_clock::now();

    for(unsigned ii = 0; ii < ntrials; ++ii)
        matrix_multiply_simple(aa, bb, cc, nn);

    auto end = std::chrono::high_resolution_clock::now();

    // 1e-6 seconds
    double micro_seconds = std::chrono::duration<double, std::micro>(end-begin).count();

    // ntrials * nn^3 mults, same number of adds
    double megaflops = 2*nn*nn*nn*ntrials/micro_seconds;

    
    std::cout << "N = " << nn << ", megaflops = " <<  megaflops << ", size = " << 3*nn*nn*sizeof(real_t) << ", ntrials = " << ntrials << "\n";

    free(aa); free(bb); free(cc);
}

int main(int argc, char **argv) {
    for(int ii = 0; ii < 20; ++ii) {
        size_t nn = ii*5+5;
        timetrial(nn, std::max(1e9/(nn*nn*nn), 5.0));
    }

    for(int ii = 2; ii < 40; ++ii) {
        size_t nn = ii*100;
        timetrial(nn, std::max(1e9/(nn*nn*nn), 5.0));
    }
    
        
    return 0;
}
