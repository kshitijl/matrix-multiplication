#include <stdlib.h> // malloc, srand, rand
#include <stddef.h> // size_t
#include <string.h> // stoi

#include <assert.h>

#include <iostream> // std::cout
#include <chrono> // std::chrono::high_resolution_clock

#include "print-matrix.hxx"

template<typename TT>
void matrix_multiply(const TT * AA, const TT * BB, TT * CC, size_t nn);

void timetrial(size_t nn, unsigned long ntrials) {
    real_t *aa, *bb, *cc;

    aa = (real_t *)malloc(sizeof(real_t)*nn*nn);
    bb = (real_t *)malloc(sizeof(real_t)*nn*nn);
    cc = (real_t *)malloc(sizeof(real_t)*nn*nn);

    for(size_t ii = 0; ii < nn*nn; ++ii) {
        aa[ii] = (real_t)rand() / RAND_MAX;
        bb[ii] = (real_t)rand() / RAND_MAX;
    }

    auto begin = std::chrono::high_resolution_clock::now();

    for(unsigned ii = 0; ii < ntrials; ++ii)
        matrix_multiply(aa, bb, cc, nn);

    auto end = std::chrono::high_resolution_clock::now();

    // 1e-9 seconds
    double nano_seconds = std::chrono::duration<double, std::nano>(end-begin).count();

    // ntrials * nn^3 mults, same number of adds
    double gigaflops = 2*nn*nn*nn*ntrials/nano_seconds;

    real_t answer_hash = 0;
    for(unsigned ii = 0; ii < nn*nn; ++ii)
        answer_hash += cc[ii]*(real_t)ii;
    
    std::cout << "N = " << nn << ", gigaflops = " <<  gigaflops << ", size = " << 3*nn*nn*sizeof(real_t) << ", ntrials = " << ntrials << ", hash = " << answer_hash << "\n";

    free(aa); free(bb); free(cc);
}

int main(int argc, char **argv) {
    assert(argc <= 2);
    srand(2);

    int stepsize = 1;
    if (argc == 2)
        stepsize = std::atoi(argv[1]);

    timetrial(5, 1e6);
    for(int ii = stepsize; ii < 20000; ii += stepsize) {
        size_t nn = ii;
        timetrial(nn, std::max(1e9/(nn*nn*nn), 5.0));
        std::cout.flush();
    }    
        
    return 0;
}
