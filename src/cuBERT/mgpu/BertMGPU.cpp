#include <cuda_runtime.h>
#include <fstream>

#include "tensorflow/core/framework/graph.pb.h"
#include "BertMGPU.h"

namespace cuBERT {
    BertMGPU::BertMGPU(const char *model_file,
                       size_t max_batch_size,
                       size_t seq_length,
                       size_t num_hidden_layers,
                       size_t num_attention_heads)
            : rr(0), graph(model_file) {
        int count;
        cudaGetDeviceCount(&count);
        for (int device = 0; device < count; ++device) {
            cudaSetDevice(device);

            auto *bert = new Bert(graph.var, max_batch_size, seq_length,
                                  graph.vocab_size,
                                  graph.type_vocab_size,
                                  graph.hidden_size,
                                  num_hidden_layers,
                                  num_attention_heads,
                                  graph.intermediate_size);
            bert_instances.push_back(bert);

            mutex_instances.push_back(new std::mutex());
        }
    }

    BertMGPU::~BertMGPU() {
        for (auto &bert_instance : bert_instances) {
            delete bert_instance;
        }
        for (auto &mutex_instance : mutex_instances) {
            delete mutex_instance;
        }
    }

    unsigned int BertMGPU::compute_cpu(size_t batch_size, int *input_ids, char *input_mask, char *segment_ids, float *logits) {
        uint8_t count = rr++;
        unsigned int choice = count % bert_instances.size();

        Bert *bert_instance = bert_instances[choice];
        std::mutex *mutex_instance = mutex_instances[choice];

        std::lock_guard<std::mutex> lg(*mutex_instance);
        bert_instance->compute_cpu(batch_size, input_ids, input_mask, segment_ids);
        bert_instance->logits(batch_size, logits);

        return choice;
    }
}
