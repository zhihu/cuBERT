#include "Graph.h"

#include <stdexcept>
#include <fstream>
#include <iostream>
#include <sstream>

#include "tensorflow/core/framework/graph.pb-c.h"

namespace cuBERT {

    template <typename T>
    Graph<T>::Graph(const char *filename) {
        std::ifstream input(filename);
        if (!input) {
            // try to check file exist
            // https://stackoverflow.com/questions/12774207/fastest-way-to-check-if-a-file-exist-using-standard-c-c11-c
            throw std::invalid_argument("model file not found");
        }
        std::stringstream stream;
        stream << input.rdbuf();
        input.close();

        std::string buffer = stream.str();
        Tensorflow__GraphDef *graphDef = tensorflow__graph_def__unpack(nullptr, buffer.size(), (const uint8_t*) buffer.c_str());
        std::cerr << "model loaded from: " << filename << std::endl;

        for (int i = 0; i < graphDef->n_node; i++) {
            Tensorflow__NodeDef *nodeDef = graphDef->node[i];

            Tensorflow__AttrValue *attrValue = nullptr;
            for (int j = 0; j < nodeDef->n_attr; j++) {
                Tensorflow__NodeDef__AttrEntry *attr = nodeDef->attr[j];
                if (std::strcmp(attr->key, "value") == 0) {
                    attrValue = attr->value;
                }
            }
            if (attrValue == nullptr) {
                continue;
            }
            if (attrValue->value_case != TENSORFLOW__ATTR_VALUE__VALUE_TENSOR) {
                continue;
            }

            Tensorflow__TensorProto *tensorProto = attrValue->tensor;
            if (tensorProto == nullptr) {
                continue;
            }

            size_t len;
            if (tensorProto->dtype == TENSORFLOW__DATA_TYPE__DT_FLOAT) {
                len = tensorProto->tensor_content.len / sizeof(float);
            } else if (tensorProto->dtype == TENSORFLOW__DATA_TYPE__DT_HALF) {
                len = tensorProto->tensor_content.len / sizeof(half);
            } else {
                continue;
            }
            auto *data_t = new T[len];
            if (tensorProto->dtype == TENSORFLOW__DATA_TYPE__DT_FLOAT) {
                T2T((const float*) tensorProto->tensor_content.data, data_t, len);
            } else if (tensorProto->dtype == TENSORFLOW__DATA_TYPE__DT_HALF) {
                T2T((const half*) tensorProto->tensor_content.data, data_t, len);
            } else {
                continue;
            }
            var[std::string(nodeDef->name)] = data_t;

            if (std::strcmp(nodeDef->name, "bert/embeddings/word_embeddings") == 0) {
                vocab_size = tensorProto->tensor_shape->dim[0]->size;
                hidden_size = tensorProto->tensor_shape->dim[1]->size;
            } else if (std::strcmp(nodeDef->name, "bert/embeddings/token_type_embeddings") == 0) {
                type_vocab_size = tensorProto->tensor_shape->dim[0]->size;
                hidden_size = tensorProto->tensor_shape->dim[1]->size;
            } else if (std::strcmp(nodeDef->name, "bert/encoder/layer_0/intermediate/dense/bias") == 0) {
                intermediate_size = tensorProto->tensor_shape->dim[0]->size;
            } else if (std::strcmp(nodeDef->name, "output_weights") == 0) {
                num_labels = tensorProto->tensor_shape->dim[0]->size;
            }
        }

        tensorflow__graph_def__free_unpacked(graphDef, nullptr);

        if (var.empty()) {
            throw std::invalid_argument("model file invalid");
        }
        std::cerr << "model param: {vocab_size=" << vocab_size
                  << ";type_vocab_size=" << type_vocab_size
                  << ";hidden_size=" << hidden_size
                  << ";intermediate_size=" << intermediate_size
                  << ";num_labels=}" << num_labels << std::endl;
    }

    template <typename T>
    Graph<T>::~Graph() {
        for (auto& iter : var) {
            delete[] iter.second;
        }
    }

    template class Graph<float>;
    template class Graph<half>;
}
