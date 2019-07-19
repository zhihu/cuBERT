#include <iostream>

#include "cuBERT/common.h"
#include "cuBERT/op/Dense.h"
#include "cuBERT/op_att/BatchMatMul.h"

// define half/float precision
// typedef half_float::half Dtype;
// const int algo_begin = 99;
// const int algo_end = 115;
typedef float Dtype;
const int algo_begin = -1;
const int algo_end = 23;

const int max_batch_size = 128;
const int batch_size = 128;
const int seq_length = 32;
const int hidden_size = 768;
const int intermediate_size = 3072;
const int num_attention_heads = 12;
const int attention_head_size = hidden_size / num_attention_heads;

const int iter = 100;

void benchmark_gemm(void* handle, int inputs, int units, int batch_size) {
    Dtype *kernel = new Dtype[inputs * units];
    Dtype *bias = new Dtype[units];

    Dtype *input = new Dtype[batch_size * inputs];
    Dtype *input_gpu = (Dtype*) cuBERT::malloc(batch_size * inputs * sizeof(Dtype));
    Dtype *output = new Dtype[batch_size * units];
    Dtype *output_gpu = (Dtype*) cuBERT::malloc(batch_size * units * sizeof(Dtype));

    for (int algo = algo_begin; algo <= algo_end; algo++) {
        cuBERT::Dense<Dtype> dense(handle, hidden_size, hidden_size, kernel, bias, max_batch_size * seq_length, algo);

        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < iter; i++) {
            // copy input to GPU
            cuBERT::memcpy(input_gpu, input, batch_size * inputs * sizeof(Dtype), 1);
            dense.compute(max_batch_size * seq_length, input_gpu, output_gpu);
            // copy output to CPU
            cuBERT::memcpy(output, output_gpu, batch_size * units * sizeof(Dtype), 2);
        }
        auto finish = std::chrono::high_resolution_clock::now();
        long long milli = std::chrono::duration_cast<std::chrono::milliseconds>(finish - start).count();
        std::cout << algo << ": " << milli << "ms" << std::endl;
    }

    cuBERT::free(output_gpu);
    cuBERT::free(input_gpu);
    delete []output;
    delete []input;
    delete []bias;
    delete []kernel;
}

// GEMM_ALGO_ATTENTION
// inputs = hidden_size
// units = hidden_size
// max_batch_size = max_batch_size * seq_length
void benchmark_attention(void* handle) {
    std::cout << "GEMM_ALGO_ATTENTION:" << std::endl;
    benchmark_gemm(handle, hidden_size, hidden_size, max_batch_size * seq_length);
    std::cout << std::endl;
}

// GEMM_ALGO_INTERMEDIATE
// inputs = hidden_size
// units = intermediate_size
// max_batch_size = max_batch_size * seq_length
void benchmark_intermediate(void* handle) {
    std::cout << "GEMM_ALGO_INTERMEDIATE:" << std::endl;
    benchmark_gemm(handle, hidden_size, intermediate_size, max_batch_size * seq_length);
    std::cout << std::endl;
}

// GEMM_ALGO_OUTPUT
// inputs = intermediate_size
// units = hidden_size
// max_batch_size = max_batch_size * seq_length
void benchmark_output(void* handle) {
    std::cout << "GEMM_ALGO_OUTPUT:" << std::endl;
    benchmark_gemm(handle, intermediate_size, hidden_size, max_batch_size * seq_length);
    std::cout << std::endl;
}

void benchmark_qk(void* handle) {
    Dtype *out = new Dtype[max_batch_size * num_attention_heads * seq_length * seq_length];

    Dtype *query_gpu = (Dtype*) cuBERT::malloc(sizeof(Dtype) * max_batch_size * seq_length * hidden_size);
    Dtype *key_gpu = (Dtype*) cuBERT::malloc(sizeof(Dtype) * max_batch_size * seq_length * hidden_size);
    Dtype *out_gpu = (Dtype*) cuBERT::malloc(sizeof(Dtype) * max_batch_size * num_attention_heads * seq_length * seq_length);

    cuBERT::Att_Q_K<Dtype> bqk(handle, max_batch_size, seq_length, num_attention_heads, attention_head_size,
                               query_gpu, key_gpu, out_gpu,
                               1.0 / std::sqrt(attention_head_size), -10000.0f);

    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iter; i++) {
        bqk.compute(max_batch_size);
        cuBERT::memcpy(out, out_gpu, sizeof(Dtype) * max_batch_size * num_attention_heads * seq_length * seq_length, 2);
    }
    auto finish = std::chrono::high_resolution_clock::now();
    long long milli = std::chrono::duration_cast<std::chrono::milliseconds>(finish - start).count();
    std::cout << "qk: " << milli << "ms" << std::endl;

    cuBERT::free(out_gpu);
    cuBERT::free(key_gpu);
    cuBERT::free(query_gpu);
    delete []out;
}

void benchmark_qkv(void* handle) {
    Dtype *out = new Dtype[max_batch_size * seq_length * hidden_size];

    Dtype *qk_gpu = (Dtype*) cuBERT::malloc(sizeof(Dtype) * max_batch_size * num_attention_heads * seq_length * seq_length);
    Dtype *value_gpu = (Dtype*) cuBERT::malloc(sizeof(Dtype) * max_batch_size * seq_length * hidden_size);
    Dtype *out_gpu = (Dtype*) cuBERT::malloc(sizeof(Dtype) * max_batch_size * seq_length * hidden_size);

    cuBERT::Att_QK_V<Dtype> bqkv(handle, max_batch_size, seq_length, num_attention_heads, attention_head_size,
                                 qk_gpu, value_gpu, out_gpu);
    
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iter; i++) {
        bqkv.compute(max_batch_size);
        cuBERT::memcpy(out, out_gpu, sizeof(Dtype) * max_batch_size * seq_length * hidden_size, 2);   
    }
    auto finish = std::chrono::high_resolution_clock::now();
    long long milli = std::chrono::duration_cast<std::chrono::milliseconds>(finish - start).count();
    std::cout << "qkv: " << milli << "ms" << std::endl;

    cuBERT::free(out_gpu);
    cuBERT::free(value_gpu);
    cuBERT::free(qk_gpu);
    delete []out;
}

int main() {
    cuBERT::initialize();
    void *handle = cuBERT::blas_create();

    benchmark_attention(handle);
    benchmark_intermediate(handle);
    benchmark_output(handle);

    cuBERT::blas_destroy(handle);
    cuBERT::finalize();
}
