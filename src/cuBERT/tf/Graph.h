//
// Created by 田露 on 2019/1/25.
//

#ifndef CUBERT_GRAPH_H
#define CUBERT_GRAPH_H

#include <string>
#include <unordered_map>

#include "tensorflow/core/framework/graph.pb.h"

namespace cuBERT {
    class Graph {
    public:
        explicit Graph(const char *filename);

        virtual ~Graph() = default;

        std::unordered_map<std::string, float *> var;
        size_t vocab_size;
        size_t type_vocab_size;
        size_t hidden_size;
        size_t intermediate_size;

    private:
        tensorflow::GraphDef graphDef;
    };
}

#endif //CUBERT_GRAPH_H
