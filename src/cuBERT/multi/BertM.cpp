#include <fstream>
#include <iostream>
#include <stdexcept>
#include <chrono>

#include "tensorflow/core/framework/graph.pb.h"
#include "cuBERT/common.h"
#include "BertM.h"

namespace cuBERT {
    BertM::BertM(const char *model_file,
                       size_t max_batch_size,
                       size_t seq_length,
                       size_t num_hidden_layers,
                       size_t num_attention_heads)
            : rr(0), graph(model_file) {
        int count = cuBERT::get_gpu_count();
        std::cerr << "Found GPU count: " << count << std::endl;

        if (count == 0) {
            if (cuBERT::gpu()) {
                throw std::invalid_argument("No GPU device detected, but caller choose to use gpu.");
            } else {
                std::cerr << "Use CPU instead" << std::endl;
                count++;
            }
        }

        for (int device = 0; device < count; ++device) {
            auto start = std::chrono::high_resolution_clock::now();
            cuBERT::set_gpu(device);

            auto *bert = new Bert(graph.var, max_batch_size, seq_length,
                                  graph.vocab_size,
                                  graph.type_vocab_size,
                                  graph.hidden_size,
                                  num_hidden_layers,
                                  num_attention_heads,
                                  graph.intermediate_size);
            bert_instances.push_back(bert);

            mutex_instances.push_back(new std::mutex());

            auto finish = std::chrono::high_resolution_clock::now();
            std::cerr << "device setup: " << device << ". Took "
                      << std::chrono::duration_cast<std::chrono::milliseconds>(finish - start).count()
                      << " milliseconds." << std::endl;
        }
    }

    BertM::~BertM() {
        for (auto &bert_instance : bert_instances) {
            delete bert_instance;
        }
        for (auto &mutex_instance : mutex_instances) {
            delete mutex_instance;
        }
    }

    unsigned int BertM::compute(size_t batch_size,
                                int *input_ids, char *input_mask, char *segment_ids,
                                float *output,
                                cuBERT_OutputType output_type) {
        uint8_t count = rr++;
        unsigned int choice = count % bert_instances.size();

        cuBERT::set_gpu(choice);
        Bert *bert_instance = bert_instances[choice];
        std::mutex *mutex_instance = mutex_instances[choice];

        std::lock_guard<std::mutex> lg(*mutex_instance);
        bert_instance->compute(batch_size, input_ids, input_mask, segment_ids);
        switch (output_type) {
            case cuBERT_LOGITS:
                bert_instance->logits(batch_size, output);
                break;
            case cuBERT_POOLED_OUTPUT:
                bert_instance->pooled_output(batch_size, output);
                break;
            case cuBERT_EMBEDDING_OUTPUT:
                bert_instance->embedding_output(batch_size, output);
                break;
            case cuBERT_SEQUENCE_OUTPUT:
                bert_instance->sequence_output(batch_size, output);
                break;
            default:
                throw std::invalid_argument("invalid output type");
        }

        return choice;
    }
}
