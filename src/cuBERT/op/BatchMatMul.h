//
// Created by 田露 on 2019/1/18.
//

#ifndef CUBERT_BATCHMATMUL_H
#define CUBERT_BATCHMATMUL_H


#include <cublas_v2.h>

namespace cuBERT {

    /**
     * op(in_A): ? * M * K
     * op(in_B): ? * K * N
     * out: ? * M * N
     */
    class BatchMatMul {
    public:
        explicit BatchMatMul(cublasHandle_t handle,
                             bool transpose_a, bool transpose_b,
                             size_t M, size_t N, size_t K,
                             size_t max_batch_size,
                             float alpha = 1, float beta = 0);

        virtual ~BatchMatMul() = default;

        void compute(size_t batch_size, const float *in_A_gpu, const float *in_B_gpu, float *out_gpu);

    private:
        cublasHandle_t handle;

        bool transpose_a;
        bool transpose_b;

        size_t M;
        size_t N;
        size_t K;

        const float alpha;
        const float beta;
    };

}


#endif //CUBERT_BATCHMATMUL_H
