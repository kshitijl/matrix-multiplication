#include <iostream>
#include "print-matrix.hxx"

template<typename TT>
void print_matrix(const TT *cc, size_t nn) {
    for(size_t ii = 0; ii < nn; ++ii) {
        for(size_t jj = 0; jj < nn; ++jj) {
            std::cout << cc[ii*nn + jj] << " ";
        }
        std::cout << "\n";
    }
}

template void print_matrix<float>(const float *, size_t);

