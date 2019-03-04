#include <random>
#include <chrono>
#include <iostream>
#include <cmath>
#include <fstream>

#include "tensorflow/c/c_api.h"

TF_Output t_inputs[3];
TF_Output t_output;

TF_Graph* graph;
TF_Session* session;

void tf_open(const char *filename) {
    std::ifstream input(filename, std::ios::binary);
    std::string str(std::istreambuf_iterator<char>{input}, {});
    input.close();

    TF_Status* status = TF_NewStatus();
    TF_SessionOptions* opts = TF_NewSessionOptions();
    TF_ImportGraphDefOptions* gopts = TF_NewImportGraphDefOptions();
    TF_Buffer* buffer = TF_NewBufferFromString(str.c_str(), str.length());

    graph = TF_NewGraph();
    TF_GraphImportGraphDef(graph, buffer, gopts, status);
    session = TF_NewSession(graph, opts, status);

    t_inputs[0] = {TF_GraphOperationByName(graph, "input_ids"), 0};
    t_inputs[1] = {TF_GraphOperationByName(graph, "input_mask"), 0};
    t_inputs[2] = {TF_GraphOperationByName(graph, "segment_ids"), 0};
    t_output = {TF_GraphOperationByName(graph, "loss/output"), 0};

    TF_DeleteBuffer(buffer);
    TF_DeleteImportGraphDefOptions(gopts);
    TF_DeleteSessionOptions(opts);
    TF_DeleteStatus(status);
}

void tf_compute(TF_Tensor *input_ids, TF_Tensor *input_mask, TF_Tensor *segment_ids, TF_Tensor **output) {
    TF_Tensor* input_values[3] = {input_ids, input_mask, segment_ids};

    TF_Status* status = TF_NewStatus();
    TF_SessionRun(session, nullptr,
                  t_inputs, input_values, 3,
                  &t_output, output, 1,
                  nullptr, 0,
                  nullptr, status);
    TF_DeleteStatus(status);
}

void tf_close() {
    TF_Status* status = TF_NewStatus();
    TF_DeleteSession(session, status);
    TF_DeleteGraph(graph);
    TF_DeleteStatus(status);
}

void random_input(std::default_random_engine& e,
                  TF_Tensor* tf_input_ids, TF_Tensor* tf_input_mask, TF_Tensor* tf_segment_ids, size_t length) {
    std::uniform_int_distribution<int> id_dist(0, 21120);
    std::uniform_int_distribution<int> zo_dist(0, 1);
    for (int i = 0; i < length; ++i) {
        ((int64_t*) TF_TensorData(tf_input_ids))[i] = id_dist(e);
        ((int64_t*) TF_TensorData(tf_input_mask))[i] = zo_dist(e);
        ((int64_t*) TF_TensorData(tf_segment_ids))[i] = zo_dist(e);
    }
}

int main() {
    std::default_random_engine e(0);

    int max_batch_size = 128;
    int batch_size = 128;
    int seq_length = 32;

    // tensorflow
    int64_t dims[2] = {batch_size, seq_length};
    TF_Tensor* tf_input_ids = TF_AllocateTensor(TF_INT64, dims, 2, sizeof(int64_t) * dims[0] * dims[1]);
    TF_Tensor* tf_input_mask = TF_AllocateTensor(TF_INT64, dims, 2, sizeof(int64_t) * dims[0] * dims[1]);
    TF_Tensor* tf_segment_ids = TF_AllocateTensor(TF_INT64, dims, 2, sizeof(int64_t) * dims[0] * dims[1]);
    std::cout << TF_Version() << std::endl;

    tf_open("bert_frozen_seq32.pb");
    std::ofstream result("tfBERT.txt");

    std::cout << "=== warm_up ===" << std::endl;
    for (int i = 0; i < 10; ++i) {
        TF_Tensor* output;
        random_input(e, tf_input_ids, tf_input_mask, tf_segment_ids, batch_size * seq_length);

        auto start = std::chrono::high_resolution_clock::now();
        tf_compute(tf_input_ids, tf_input_mask, tf_segment_ids, &output);
        auto finish = std::chrono::high_resolution_clock::now();
        long long milli = std::chrono::duration_cast<std::chrono::milliseconds>(finish - start).count();
        std::cout << "TF: " << milli << "ms" << std::endl;

        for (int j = 0; j < batch_size; ++j) {
            result << ((float*) TF_TensorData(output))[j] << std::endl;
        }
        TF_DeleteTensor(output);
    }

    std::cout << "=== benchmark ===" << std::endl;
    for (int i = 0; i < 10; ++i) {
        TF_Tensor* output;
        random_input(e, tf_input_ids, tf_input_mask, tf_segment_ids, batch_size * seq_length);

        auto start = std::chrono::high_resolution_clock::now();
        tf_compute(tf_input_ids, tf_input_mask, tf_segment_ids, &output);
        auto finish = std::chrono::high_resolution_clock::now();
        long long milli = std::chrono::duration_cast<std::chrono::milliseconds>(finish - start).count();
        std::cout << "TF: " << milli << "ms" << std::endl;

        for (int j = 0; j < batch_size; ++j) {
            result << ((float*) TF_TensorData(output))[j] << std::endl;
        }
        TF_DeleteTensor(output);
    }

    TF_DeleteTensor(tf_segment_ids);
    TF_DeleteTensor(tf_input_mask);
    TF_DeleteTensor(tf_input_ids);

    result.close();
    tf_close();
}
