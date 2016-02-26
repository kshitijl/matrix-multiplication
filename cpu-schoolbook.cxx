#include <stddef.h> // size_t

template<typename TT>
void matrix_multiply(const TT * AA, const TT * BB, TT * CC, size_t nn) {
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

template void matrix_multiply<float>(const float*, const float*, float*, size_t);
template void matrix_multiply<double>(const double*, const double*, double*, size_t);
