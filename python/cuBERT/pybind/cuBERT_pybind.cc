/**
 * Copyright (c) 2019-present, NTC, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <vector>

#include <cuBERT.h>
#include <half.hpp>


using float16 = half_float::half;

namespace pybind11 { namespace detail {
template <> struct type_caster<float16> {
public:
    PYBIND11_TYPE_CASTER(float16, _("float16"));
    using float_caster = type_caster<float>;

    // convert Python -> C++
    bool load(handle src, bool convert) {
        float_caster caster;
        if (caster.load(src, convert)) {
            this->value = float16(float(caster));  // Implicit cast defined by `type_caster`.
            return true;
        }
        return false;
    }

    // Convert C++ -> Python
    static handle cast(float16 src, return_value_policy policy, handle parent) {
        return float_caster::cast(float(src), policy, parent);
    }
};
}} // namespace pybind11::detail


char **str2char(const std::vector<std::string> input) {
  unsigned int sz = input.size();

  char **result = new char *[sz + 1];
  result[sz] = nullptr;

  for (unsigned int i = 0; i < sz; ++i) {
    unsigned int len = input[i].size();
    result[i] = new char[len + 1];
    std::strcpy(result[i], input[i].c_str());
    result[len] = '\0';
  }

  return result;
}


namespace py = pybind11;
PYBIND11_MODULE(cuBERT_pybind, m) {
  py::enum_<cuBERT_ComputeType>(m, "cuBERT_ComputeType")
    .value("cuBERT_COMPUTE_FLOAT", cuBERT_ComputeType::cuBERT_COMPUTE_FLOAT)
    .value("cuBERT_COMPUTE_HALF", cuBERT_ComputeType::cuBERT_COMPUTE_HALF)
    .export_values();

  py::enum_<cuBERT_OutputType>(m, "cuBERT_OutputType")
    .value("cuBERT_LOGITS", cuBERT_OutputType::cuBERT_LOGITS)
    .value("cuBERT_POOLED_OUTPUT", cuBERT_OutputType::cuBERT_POOLED_OUTPUT)
    .value("cuBERT_SEQUENCE_OUTPUT", cuBERT_OutputType::cuBERT_SEQUENCE_OUTPUT)
    .value("cuBERT_EMBEDDING_OUTPUT", cuBERT_OutputType::cuBERT_EMBEDDING_OUTPUT)
    .export_values();

  m.def(
    "cuBERT_initialize",
    []() { cuBERT_initialize(); }
  );

  m.def(
    "cuBERT_finalize",
    []() { cuBERT_finalize(); }
  );

  m.def(
    "cuBERT_open",
    [](
      const char *model_file,
      int max_batch_size,
      int seq_length,
      int num_hidden_layers,
      int num_attention_heads,
      cuBERT_ComputeType compute_type
    ) {
      return cuBERT_open(model_file, max_batch_size, seq_length, num_hidden_layers, num_attention_heads);
    },
    py::arg("model_file"),
    py::arg("max_batch_size"),
    py::arg("seq_length"),
    py::arg("num_hidden_layers"),
    py::arg("num_attention_heads"),
    py::arg("compute_type") = cuBERT_ComputeType::cuBERT_COMPUTE_FLOAT
  );

  m.def(
    "cuBERT_close",
    [](void *model, cuBERT_ComputeType compute_type) {
      cuBERT_close(model, compute_type);
    },
    py::arg("model"),
    py::arg("compute_type") = cuBERT_ComputeType::cuBERT_COMPUTE_FLOAT
   );

  m.def(
    "_cuBERT_compute",
    [](
      void *model,
      int batch_size,
      std::vector<int> input_ids,
      std::vector<std::int8_t> input_mask,
      std::vector<std::int8_t> segment_ids,
      std::vector<float> output,
      cuBERT_OutputType output_type
    ) {
      cuBERT_compute(
        model,
        batch_size,
        input_ids.data(),
        input_mask.data(),
        segment_ids.data(),
        (void *)output.data(),
        output_type,
        cuBERT_ComputeType::cuBERT_COMPUTE_FLOAT
      );

      return output;
    },
    py::arg("model"),
    py::arg("batch_size"),
    py::arg("input_ids"),
    py::arg("input_mask"),
    py::arg("segment_ids"),
    py::arg("output"),
    py::arg("output_type") = cuBERT_OutputType::cuBERT_LOGITS
  );

  m.def(
    "_cuBERT_compute",
    [](
      void *model,
      int batch_size,
      std::vector<int> input_ids,
      std::vector<std::int8_t> input_mask,
      std::vector<std::int8_t> segment_ids,
      std::vector<float16> output,
      cuBERT_OutputType output_type
    ) {
      cuBERT_compute(
        model,
        batch_size,
        input_ids.data(),
        input_mask.data(),
        segment_ids.data(),
        (void *)output.data(),
        output_type,
        cuBERT_ComputeType::cuBERT_COMPUTE_HALF
      );

      return output;
    },
    py::arg("model"),
    py::arg("batch_size"),
    py::arg("input_ids"),
    py::arg("input_mask"),
    py::arg("segment_ids"),
    py::arg("output"),
    py::arg("output_type") = cuBERT_OutputType::cuBERT_LOGITS
  );

  /** high level API including tokenization **/
  m.def(
    "cuBERT_open_tokenizer",
    [](const char *vocab_file, int do_lower_case = 1) {
      return cuBERT_open_tokenizer(vocab_file, do_lower_case);
    },
    py::arg("vocab_file"),
    py::arg("do_lower_case") = 1
  );

  m.def(
    "cuBERT_close_tokenizer",
    [](void *tokenizer) { cuBERT_close_tokenizer(tokenizer); },
    py::arg("tokenizer")
  );

  m.def(
    "_cuBERT_tokenize_compute",
    [](
      void *model,
      void *tokenizer,
      int batch_size,
      std::vector<std::string> text_a,
      std::vector<std::string> text_b,
      std::vector<float> output,
      cuBERT_OutputType output_type
    ) {
      char **arr_text_a = str2char(text_a);
      char **arr_text_b = str2char(text_b);

      cuBERT_tokenize_compute(
        model,
        tokenizer,
        batch_size,
        (const char **)arr_text_a,
        (const char **)arr_text_b,
        (void *)output.data(),
        output_type,
        cuBERT_ComputeType::cuBERT_COMPUTE_FLOAT
      );

      delete[] arr_text_a;
      delete[] arr_text_b;

      return output;
    },
    py::arg("model"),
    py::arg("tokenizer"),
    py::arg("batch_size"),
    py::arg("text_a"),
    py::arg("text_b"),
    py::arg("output"),
    py::arg("output_type") = cuBERT_OutputType::cuBERT_LOGITS
  );

  m.def(
    "_cuBERT_tokenize_compute",
    [](
      void *model,
      void *tokenizer,
      int batch_size,
      std::vector<std::string> text_a,
      std::vector<std::string> text_b,
      std::vector<float16> output,
      cuBERT_OutputType output_type
    ) {
      char **arr_text_a = str2char(text_a);
      char **arr_text_b = str2char(text_b);

      cuBERT_tokenize_compute(
        model,
        tokenizer,
        batch_size,
        (const char **)arr_text_a,
        (const char **)arr_text_b,
        (void *)output.data(),
        output_type,
        cuBERT_ComputeType::cuBERT_COMPUTE_HALF
      );

      delete[] arr_text_a;
      delete[] arr_text_b;

      return output;
    },
    py::arg("model"),
    py::arg("tokenizer"),
    py::arg("batch_size"),
    py::arg("text_a"),
    py::arg("text_b"),
    py::arg("output"),
    py::arg("output_type") = cuBERT_OutputType::cuBERT_LOGITS
  );
}
