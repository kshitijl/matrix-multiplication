#include <stddef.h> // size_t
#include <cblas.h>

template<typename TT>
void matrix_multiply(const TT * AA, const TT * BB, TT * CC, size_t nn);

template<>
void matrix_multiply(const float * AA, const float * BB, float * CC, size_t nn) {
    // C = alpha*A*B + beta*C
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                nn, nn, nn, // dimensions
                1.0,       // alpha
                AA, nn,
                BB, nn,
                0,
                CC, nn);                
}

template<>
void matrix_multiply(const double * AA, const double * BB, double * CC, size_t nn) {
    // C = alpha*A*B + beta*C
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                nn, nn, nn, // dimensions
                1.0,       // alpha
                AA, nn,
                BB, nn,
                0,
                CC, nn);                
}
