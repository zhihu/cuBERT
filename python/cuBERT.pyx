# cython: infer_types=True
from libc.stdlib cimport malloc, free
import numpy as np
cimport numpy as np
cimport cython
cimport _cuBERT

cdef bytes as_str(s):
    if isinstance(s, bytes):
        return s
    elif isinstance(s, unicode):
        return s.encode('utf8')
    raise TypeError('Cannot convert %s to string' % type(s))

class ComputeType:
    FLOAT = _cuBERT.cuBERT_ComputeType.cuBERT_COMPUTE_FLOAT
    HALF = _cuBERT.cuBERT_ComputeType.cuBERT_COMPUTE_HALF

class OutputType:
    LOGITS = _cuBERT.cuBERT_OutputType.cuBERT_LOGITS
    POOLED_OUTPUT = _cuBERT.cuBERT_OutputType.cuBERT_POOLED_OUTPUT
    SEQUENCE_OUTPUT = _cuBERT.cuBERT_OutputType.cuBERT_SEQUENCE_OUTPUT
    EMBEDDING_OUTPUT = _cuBERT.cuBERT_OutputType.cuBERT_EMBEDDING_OUTPUT

cdef class Model:
    cdef void* _c_model
    cdef void* _c_tokenizer
    cdef _cuBERT.cuBERT_ComputeType _c_compute_type

    def __cinit__(self, 
                  model_file,
                  int max_batch_size,
                  int seq_length,
                  int num_hidden_layers,
                  int num_attention_heads,
                  _cuBERT.cuBERT_ComputeType compute_type = _cuBERT.cuBERT_ComputeType.cuBERT_COMPUTE_FLOAT,
                  vocab_file = None,
                  int do_lower_case = 1):
        self._c_model = _cuBERT.cuBERT_open(as_str(model_file), 
                                            max_batch_size,
                                            seq_length,
                                            num_hidden_layers,
                                            num_attention_heads,
                                            compute_type)
        self._c_compute_type = compute_type
        if vocab_file:
            self._c_tokenizer = _cuBERT.cuBERT_open_tokenizer(as_str(vocab_file), do_lower_case)
    
    def __dealloc__(self):
        _cuBERT.cuBERT_close(self._c_model, self._c_compute_type)
        if self._c_tokenizer:
            _cuBERT.cuBERT_close_tokenizer(self._c_tokenizer)
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def compute(self, 
                int batch_size, 
                np.ndarray[np.int32_t, ndim=2, mode='c'] input_ids,
                np.ndarray[np.int8_t, ndim=2, mode='c'] input_mask,
                np.ndarray[np.int8_t, ndim=2, mode='c'] segment_ids,
                np.ndarray output,
                _cuBERT.cuBERT_OutputType output_type = _cuBERT.cuBERT_OutputType.cuBERT_LOGITS):
        if not input_ids.flags['C_CONTIGUOUS'] or not input_mask.flags['C_CONTIGUOUS'] or not segment_ids.flags['C_CONTIGUOUS'] or not output.flags['C_CONTIGUOUS']:
            raise ValueError('numpy array should be C_CONTIGUOUS')
        _cuBERT.cuBERT_compute(self._c_model,
                               batch_size,
                               <int*> input_ids.data,
                               <signed char*> input_mask.data,
                               <signed char*> segment_ids.data,
                               <void*> output.data,
                               output_type,
                               self._c_compute_type)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def tokenize_compute(self, 
                         int batch_size, 
                         list text_a, 
                         list text_b, 
                         np.ndarray output,
                         _cuBERT.cuBERT_OutputType output_type = _cuBERT.cuBERT_OutputType.cuBERT_LOGITS):
        if not output.flags['C_CONTIGUOUS']:
            raise ValueError('numpy array should be C_CONTIGUOUS')
        cdef list text_a_enc = [as_str(word) for word in text_a]
        cdef list text_b_enc = [as_str(word) for word in text_b]
        cdef const char** c_text_a = <const char**> malloc(batch_size * sizeof(char*))
        cdef const char** c_text_b = <const char**> malloc(batch_size * sizeof(char*))
        for i in range(batch_size):
            c_text_a[i] = text_a_enc[i]
            c_text_b[i] = text_b_enc[i]
        _cuBERT.cuBERT_tokenize_compute(self._c_model, 
                                        self._c_tokenizer,
                                        batch_size,
                                        c_text_a,
                                        c_text_b,
                                        <void*> output.data, 
                                        output_type,
                                        self._c_compute_type)
        free(c_text_b)
        free(c_text_a)
