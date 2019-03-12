#include <cstddef>
#include <cmath>

#include "cuBERT/common.h"
#include "./GELU.h"

namespace cuBERT {
    template <>
    void gelu<true>(size_t N, float *inout, void *stream) {
#pragma omp parallel for
        for (int i = 0; i < N; ++i) {
            inout[i] = inout[i] * 0.5f * (1.0f + erff(inout[i] * sqrtf(0.5)));
        }
    }

    void GELU::compute_(size_t N, float *inout, void* stream) {
        gelu<!cuBERT::gpu()>(N, inout, stream);
    }
}
