#include <stddef.h> // size_t
#include <string.h> // memset

template<typename TT>
void matrix_multiply(const TT * AA, const TT * BB, TT * CC, size_t nn) {
    memset(CC, 0, nn*nn*sizeof(TT));
    
    for (size_t ii = 0; ii < nn; ++ii) {
        TT * ci = CC + (ii*nn);
        for(size_t kk = 0; kk < nn; ++kk) {
            for (size_t jj = 0; jj < nn; ++jj) {
                *(ci+jj) += AA[ii*nn + kk]*BB[kk*nn + jj];
            }
        }
    }
}

template void matrix_multiply<float>(const float*, const float*, float*, size_t);
template void matrix_multiply<double>(const double*, const double*, double*, size_t);
