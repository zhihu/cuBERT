# cython: infer_types=True
from libc.stdlib cimport malloc, free
import atexit
import numpy as np
cimport numpy as np
cimport cython
cimport _cuBERT

_cuBERT.cuBERT_initialize()
atexit.register(_cuBERT.cuBERT_finalize)

cdef bytes as_str(s):
    if isinstance(s, bytes):
        return s
    elif isinstance(s, unicode):
        return s.encode('utf8')
    raise TypeError('Cannot convert %s to string' % type(s))

cdef const char** string_array_to_ptr(list text):
    if not text:
        return NULL
    cdef const char** _c_text = <const char**> malloc(len(text) * sizeof(char*))
    for i in range(len(text)):
        _c_text[i] = text[i]
    return _c_text

class ComputeType:
    FLOAT = _cuBERT.cuBERT_ComputeType.cuBERT_COMPUTE_FLOAT
    HALF = _cuBERT.cuBERT_ComputeType.cuBERT_COMPUTE_HALF

class OutputType:
    LOGITS = _cuBERT.cuBERT_OutputType.cuBERT_LOGITS
    POOLED_OUTPUT = _cuBERT.cuBERT_OutputType.cuBERT_POOLED_OUTPUT
    SEQUENCE_OUTPUT = _cuBERT.cuBERT_OutputType.cuBERT_SEQUENCE_OUTPUT
    EMBEDDING_OUTPUT = _cuBERT.cuBERT_OutputType.cuBERT_EMBEDDING_OUTPUT
    PROBS = _cuBERT.cuBERT_OutputType.cuBERT_PROBS

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
                np.ndarray[np.int32_t, ndim=2, mode='c'] input_ids,
                np.ndarray[np.int8_t, ndim=2, mode='c'] input_mask,
                np.ndarray[np.int8_t, ndim=2, mode='c'] segment_ids,
                np.ndarray output,
                _cuBERT.cuBERT_OutputType output_type = _cuBERT.cuBERT_OutputType.cuBERT_LOGITS):
        if not input_ids.flags['C_CONTIGUOUS'] or \
                not input_mask.flags['C_CONTIGUOUS'] or \
                not segment_ids.flags['C_CONTIGUOUS'] or \
                not output.flags['C_CONTIGUOUS']:
            raise ValueError('numpy array should be C_CONTIGUOUS')
        cdef int batch_size = input_ids.shape[0]
        if batch_size != input_mask.shape[0] or \
                batch_size != segment_ids.shape[0] or \
                batch_size != output.shape[0]:
            raise ValueError('numpy shape mismatch')
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
                         list text_a, 
                         list text_b, 
                         np.ndarray output,
                         _cuBERT.cuBERT_OutputType output_type = _cuBERT.cuBERT_OutputType.cuBERT_LOGITS):
        if not output.flags['C_CONTIGUOUS']:
            raise ValueError('numpy array should be C_CONTIGUOUS')
        cdef int batch_size = len(text_a)
        if (text_b and batch_size != len(text_b)) or batch_size != output.shape[0]:
            raise ValueError('numpy shape mismatch')
        cdef list text_a_enc = [as_str(word) for word in text_a]
        cdef list text_b_enc = [as_str(word) for word in text_b] if text_b else None
        cdef const char** c_text_a = string_array_to_ptr(text_a_enc)  
        cdef const char** c_text_b = string_array_to_ptr(text_b_enc)
        _cuBERT.cuBERT_tokenize_compute(self._c_model, 
                                        self._c_tokenizer,
                                        batch_size,
                                        c_text_a,
                                        c_text_b,
                                        <void*> output.data, 
                                        output_type,
                                        self._c_compute_type)
        if text_b:
            free(c_text_b)
        free(c_text_a)
