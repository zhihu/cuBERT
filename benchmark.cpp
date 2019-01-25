#include <random>
#include <chrono>
#include <iostream>
#include <cmath>

#include "tensorflow/c/c_api.h"
#include "cuBERT.h"

TF_Output t_inputs[3];
TF_Output t_output;

TF_Session* open_tf(const char* export_dir) {
    const char* tag = "serve";
    TF_Status* status = TF_NewStatus();

    TF_SessionOptions* opts = TF_NewSessionOptions();
    unsigned char config[16] = {0x10, 8, 0x28, 8, 0x32, 0x02, 0x20, 1, 0x38, 0x01, 0x52, 0x04, 0x1a, 0x02, 0x28, 0};
    TF_SetConfig(opts, config, 16, status);

    TF_Graph* graph = TF_NewGraph();
    TF_Session* session = TF_LoadSessionFromSavedModel(opts, nullptr, export_dir, &tag, 1, graph, nullptr, status);

    t_inputs[0] = {TF_GraphOperationByName(graph, "input_ids"), 0};
    t_inputs[1] = {TF_GraphOperationByName(graph, "input_mask"), 0};
    t_inputs[2] = {TF_GraphOperationByName(graph, "segment_ids"), 0};
    t_output = {TF_GraphOperationByName(graph, "loss/output"), 0};

    return session;
}

void compute_tf(TF_Session* session,
                TF_Tensor* input_ids, TF_Tensor* input_mask, TF_Tensor* segment_ids, TF_Tensor** output) {
    TF_Tensor* input_values[3] = {input_ids, input_mask, segment_ids};

    TF_Status* status = TF_NewStatus();
    TF_SessionRun(session, nullptr,
                  t_inputs, input_values, 3,
                  &t_output, output, 1,
                  nullptr, 0,
                  nullptr, status);
    TF_DeleteStatus(status);
}

void close_tf(TF_Session* session) {
    TF_DeleteSession(session, nullptr);
}

void random_input(int* input_ids, char* input_mask, char* segment_ids, size_t length,
                  TF_Tensor* tf_input_ids, TF_Tensor* tf_input_mask, TF_Tensor* tf_segment_ids) {
    std::random_device r;
    std::default_random_engine e(r());

    std::uniform_int_distribution<int> id_dist(0, 21120);
    std::uniform_int_distribution<int> zo_dist(0, 1);
    for (int i = 0; i < length; ++i) {
        input_ids[i] = id_dist(e);
        input_mask[i] = zo_dist(e);
        segment_ids[i] = zo_dist(e);

        ((int64_t*) TF_TensorData(tf_input_ids))[i] = input_ids[i];
        ((int64_t*) TF_TensorData(tf_input_mask))[i] = input_mask[i];
        ((int64_t*) TF_TensorData(tf_segment_ids))[i] = segment_ids[i];
    }
}

void compare_output(float* logits, size_t length, TF_Tensor* output) {
    for (int i = 0; i < length; ++i) {
        float tf = ((float*) TF_TensorData(output))[i];
        if (fabsf(tf - logits[i]) >= 1e-5) {
            std::cout << "err: i=" << i << ", cuBERT=" << logits[i] << ", TF=" << tf << std::endl;
        }
    }
}

int main() {
    int max_batch_size = 512;
    int batch_size = 400;
    int seq_length = 32;

    // cuBERT
    int input_ids[batch_size * seq_length];
    char input_mask[batch_size * seq_length];
    char segment_ids[batch_size * seq_length];
    float logits[batch_size];

    // tensorflow
    int64_t dims[2] = {batch_size, seq_length};
    TF_Tensor* tf_input_ids = TF_AllocateTensor(TF_INT64, dims, 2, sizeof(int64_t) * dims[0] * dims[1]);
    TF_Tensor* tf_input_mask = TF_AllocateTensor(TF_INT64, dims, 2, sizeof(int64_t) * dims[0] * dims[1]);
    TF_Tensor* tf_segment_ids = TF_AllocateTensor(TF_INT64, dims, 2, sizeof(int64_t) * dims[0] * dims[1]);

    void* model = cuBERT_open("bert_frozen_seq32.pb", max_batch_size, seq_length, 12, 12);
    TF_Session* tf_model = open_tf("bert");

    std::cout << "=== warm_up ===" << std::endl;
    for (int i = 0; i < 10; ++i) {
        random_input(input_ids, input_mask, segment_ids, batch_size * seq_length,
                     tf_input_ids, tf_input_mask, tf_segment_ids);

        auto start = std::chrono::high_resolution_clock::now();
        cuBERT_compute(model, batch_size, input_ids, input_mask, segment_ids, logits);
        auto finish = std::chrono::high_resolution_clock::now();
        long long milli = std::chrono::duration_cast<std::chrono::milliseconds>(finish - start).count();
        std::cout << "cuBERT: " << milli << "ms" << std::endl;

        TF_Tensor* output;
        start = std::chrono::high_resolution_clock::now();
        compute_tf(tf_model, tf_input_ids, tf_input_mask, tf_segment_ids, &output);
        finish = std::chrono::high_resolution_clock::now();
        milli = std::chrono::duration_cast<std::chrono::milliseconds>(finish - start).count();
        std::cout << "TF: " << milli << "ms" << std::endl;

        compare_output(logits, batch_size, output);
        TF_DeleteTensor(output);
    }

    std::cout << "=== benchmark ===" << std::endl;
    for (int i = 0; i < 10; ++i) {
        random_input(input_ids, input_mask, segment_ids, batch_size * seq_length,
                     tf_input_ids, tf_input_mask, tf_segment_ids);

        auto start = std::chrono::high_resolution_clock::now();
        cuBERT_compute(model, batch_size, input_ids, input_mask, segment_ids, logits);
        auto finish = std::chrono::high_resolution_clock::now();
        long long milli = std::chrono::duration_cast<std::chrono::milliseconds>(finish - start).count();
        std::cout << "cuBERT: " << milli << "ms" << std::endl;

        TF_Tensor* output;
        start = std::chrono::high_resolution_clock::now();
        compute_tf(tf_model, tf_input_ids, tf_input_mask, tf_segment_ids, &output);
        finish = std::chrono::high_resolution_clock::now();
        milli = std::chrono::duration_cast<std::chrono::milliseconds>(finish - start).count();
        std::cout << "TF: " << milli << "ms" << std::endl;

        compare_output(logits, batch_size, output);
        TF_DeleteTensor(output);
    }

    TF_DeleteTensor(tf_segment_ids);
    TF_DeleteTensor(tf_input_mask);
    TF_DeleteTensor(tf_input_ids);
    close_tf(tf_model);

    cuBERT_close(model);
}
