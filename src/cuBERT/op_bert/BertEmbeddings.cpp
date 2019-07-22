#include "cuBERT/common.h"
#include "BertEmbeddings.h"

namespace cuBERT {

    const static float ONE = 1.f;

    template <typename T>
    BertEmbeddings<T>::BertEmbeddings(void* handle,
                                   const std::unordered_map<std::string, T *> &var,
                                   size_t max_batch_size,
                                   size_t vocab_size, size_t type_vocab_size, size_t hidden_size, size_t seq_length) {
        this->handle = handle;

        this->seq_length = seq_length;
        this->hidden_size = hidden_size;
        this->algo = gemm_algo<T>("GEMM_ALGO_EMBEDDINGS");

        this->word_embeddings = new Embedding<int, T>(vocab_size, hidden_size,
                                              var.at("bert/embeddings/word_embeddings"));
        this->token_type_embeddings = new Embedding<int8_t, T>(type_vocab_size, hidden_size,
                                                               var.at("bert/embeddings/token_type_embeddings"));
        this->layer_norm = new LayerNorm<T>(max_batch_size * seq_length, hidden_size,
                                                 var.at("bert/embeddings/LayerNorm/beta"),
                                                 var.at("bert/embeddings/LayerNorm/gamma"));

        T *full_position_embeddings = var.at("bert/embeddings/position_embeddings");
        this->position_embeddings = static_cast<T *>(cuBERT::malloc(sizeof(T) * seq_length * hidden_size));
        cuBERT::memcpy(position_embeddings, full_position_embeddings, sizeof(T) * seq_length * hidden_size, 1);

        this->ones = static_cast<T *>(cuBERT::malloc(sizeof(T) * max_batch_size));
        T one; T2T(&ONE, &one, 1);
        cuBERT::fill_n<T>(ones, max_batch_size, one);

        this->token_type_embeddings_out = static_cast<T *>(cuBERT::malloc(sizeof(T) * max_batch_size * seq_length * hidden_size));
    }

    template <typename T>
    BertEmbeddings<T>::~BertEmbeddings() {
        cuBERT::free(this->token_type_embeddings_out);
        cuBERT::free(this->ones);
        cuBERT::free(this->position_embeddings);

        delete layer_norm;
        delete token_type_embeddings;
        delete word_embeddings;
    }

    template <typename T>
    void BertEmbeddings<T>::compute(size_t batch_size, int *input_ids_gpu, int8_t *token_type_ids_gpu, T *out_gpu) {
        void *stream = cuBERT::blas_get_stream(handle);

        word_embeddings->compute(input_ids_gpu, batch_size * seq_length, out_gpu, stream);
        token_type_embeddings->compute(token_type_ids_gpu, batch_size * seq_length, token_type_embeddings_out, stream);

        cuBERT::blas_gemm(handle, false, false,
                           seq_length * hidden_size, batch_size, 1,
                           1.f,
                           position_embeddings, seq_length * hidden_size,
                           ones, 1,
                           1.f,
                           out_gpu, seq_length * hidden_size,
                           algo);

        layer_norm->compute_(batch_size * seq_length, token_type_embeddings_out, out_gpu, stream);
    }

    template class BertEmbeddings<float>;
#ifdef HAVE_CUDA
    template class BertEmbeddings<half>;
#endif
}
