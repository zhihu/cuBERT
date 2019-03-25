#include <cstddef>
#include <cmath>

#include "cuBERT/common.h"
#include "./GELU.h"

namespace cuBERT {
#ifdef HAVE_MKL
    template <>
    void gelu<float>(size_t N, float *inout, void *stream) {
#pragma omp parallel for
        for (int i = 0; i < N; ++i) {
            inout[i] = inout[i] * 0.5f * (1.0f + erff(inout[i] * sqrtf(0.5f)));
        }
    }
#endif

    template <typename T>
    void GELU<T>::compute_(size_t N, T *inout, void* stream) {
        gelu<T>(N, inout, stream);
    }

    template class GELU<float>;
#ifdef HAVE_CUDA
    template class GELU<half>;
#endif
}
