#include <cstddef>
#include <cmath>
#include <omp.h>

#include "cuBERT/common.h"
#include "./GELU.h"

namespace cuBERT {
    void GELU::compute_(size_t N, float *inout, void* stream) {
#ifdef HAVE_CUDA
        if (cuBERT::gpu()) {
            gelu(N, inout, stream);
            return;
        }
#endif

#pragma omp parallel for
        for (int i = 0; i < N; ++i) {
            inout[i] = inout[i] * 0.5f * (1.0f + erff(inout[i] * sqrtf(0.5)));
        }
    }
}
