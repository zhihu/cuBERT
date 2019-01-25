#ifndef CUBERT_BERTMGPU_H
#define CUBERT_BERTMGPU_H

#include <vector>
#include <string>
#include <atomic>
#include <mutex>

#include "cuBERT/bert/Bert.h"
#include "cuBERT/tf/Graph.h"

namespace cuBERT {
    class BertMGPU {
    public:
        explicit BertMGPU(const char *model_file,
                          size_t max_batch_size,
                          size_t seq_length,
                          size_t num_hidden_layers = 12,
                          size_t num_attention_heads = 12);

        virtual ~BertMGPU();

        void compute_cpu(size_t batch_size, int *input_ids, char *input_mask, char *segment_ids, float *logits);

    private:
        Graph graph;
        std::vector<Bert *> bert_instances;
        std::vector<std::mutex *> mutex_instances;

        std::atomic<uint8_t> rr;
    };
}

#endif //CUBERT_BERTMGPU_H
