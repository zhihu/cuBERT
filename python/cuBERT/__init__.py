#!/usr/bin/env python  # pylint: disable=import-error,invalid-name
# -*- coding: utf-8 -*-
#
# Copyright @2019 R&D, NTC Inc. (ntc.ai)
#
# Author: qinluo <eric.x.sun@gmail.com>
#

import numpy as np

from cuBERT_pybind import _cuBERT_compute
from cuBERT_pybind import _cuBERT_tokenize_compute
from cuBERT_pybind import cuBERT_close
from cuBERT_pybind import cuBERT_close_tokenizer
from cuBERT_pybind import cuBERT_ComputeType
from cuBERT_pybind import cuBERT_finalize
from cuBERT_pybind import cuBERT_initialize
from cuBERT_pybind import cuBERT_open
from cuBERT_pybind import cuBERT_open_tokenizer
from cuBERT_pybind import cuBERT_OutputType

_SUPPORTED_COMPUTE_TYPES = {
    cuBERT_ComputeType.cuBERT_COMPUTE_FLOAT,
    cuBERT_ComputeType.cuBERT_COMPUTE_HALF
}  # yapf: disable

_SUPPORTED_OUTPUT_TYPES = {
    cuBERT_OutputType.cuBERT_LOGITS,
    cuBERT_OutputType.cuBERT_POOLED_OUTPUT,
    cuBERT_OutputType.cuBERT_SEQUENCE_OUTPUT,
    cuBERT_OutputType.cuBERT_EMBEDDING_OUTPUT
}  # yapf: disable


def _output_size(output_type, batch_size, hidden_size=1, seq_length=1):
    """calculate the size of output based on output_type

    Args:
        output_type: (cuBERT_OutputType)
        batch_size: (int)
        hidden_size: (int), default 1
        seq_length: (int), default 1

    Returns:
        size of output: (int)

    Exceptions:
        ValueError: not supported output_type
    """
    if output_type not in _SUPPORTED_OUTPUT_TYPES:
        msg = 'not supported output_type: {0}, choose one from [{1}]'.format(
            output_type, ', '.join(_SUPPORTED_OUTPUT_TYPES)
        )
        raise ValueError(msg)

    bs = batch_size
    hs = hidden_size
    sl = seq_length

    if cuBERT_OutputType.cuBERT_LOGITS == output_type:
        hs = sl = 1
    elif cuBERT_OutputType.cuBERT_POOLED_OUTPUT == output_type:
        sl = 1
    # else: cuBERT_SEQUENCE_OUTPUT, cuBERT_EMBEDDING_OUTPUT => bs * hs * sl

    sz = bs * hs * sl

    if sz == 0:
        raise ValueError('output size should be greater than zero')

    return sz


def cuBERT_compute(  # pylint: disable=too-many-arguments
        batch_size, input_ids, input_mask, segment_ids, output_type, compute_type, model, hidden_size=1, seq_length=1):
    """

    Args:
        batch_size: (int)
        input_ids: (list of int) list of words in id
        input_mask: (list of int: 0 or 1) masks of input
        segment_ids: (list of int: 0 or 1)
        output_type: (cuBERT_OutputType)
        compute_type: (cuBERT_ComputeType)

        model: loaded by cuBERT_open
        hidden_size: (int) size of hidden unit defined in `model`, default 1
        seq_length: (int) length of sequence defined in `model`, default 1

    Returns:
        numpy array with value type

    Exceptions:
        ValueError: not supported compute_type
        ValueError: not support half precision on CPU
    """
    if compute_type in _SUPPORTED_COMPUTE_TYPES:
        output_size = _output_size(output_type, batch_size, hidden_size, seq_length)

        if cuBERT_ComputeType.cuBERT_COMPUTE_FLOAT == compute_type:
            output = np.zeros(output_size, dtype=np.float32)
        else:
            output = np.zeros(output_size, dtype=np.float16)

        output = _cuBERT_compute(model, batch_size, input_ids, input_mask, segment_ids, output, output_type)
        return output

    msg = 'not supported compute_type: {0}, choose one from [{1}]'.format(
        compute_type, ', '.join(_SUPPORTED_COMPUTE_TYPES)
    )
    raise ValueError(msg)


def cuBERT_tokenize_compute(  # pylint: disable=too-many-arguments
        batch_size, arr_text_a, arr_text_b, output_type, compute_type, model, tokenizer, hidden_size=1, seq_length=1):
    """

    Args:
        batch_size: (int)
        arr_text_a: (list of str)
        arr_text_b: (list of str)
        output_type: (cuBERT_OutputType)
        compute_type: (cuBERT_ComputeType)

        model: loaded by cuBERT_open
        tokenizer: loaded by cuBERT_open_tokenizer
        hidden_size: (int) size of hidden unit defined in `model`, default 1
        seq_length: (int) length of sequence defined in `model`, default 1

    Returns:
        numpy array with value type

    Exceptions:
        ValueError: not supported compute_type
        ValueError: not support half precision on CPU
    """
    if compute_type in _SUPPORTED_COMPUTE_TYPES:
        output_size = _output_size(output_type, batch_size, hidden_size, seq_length)

        if cuBERT_ComputeType.cuBERT_COMPUTE_FLOAT == compute_type:
            output = np.zeros(output_size, dtype=np.float32)
        else:
            output = np.zeros(output_size, dtype=np.float16)

        output = _cuBERT_tokenize_compute(model, tokenizer, batch_size, arr_text_a, arr_text_b, output, output_type)
        return output

    msg = 'not supported compute_type: {0}, choose one from [{1}]'.format(
        compute_type, ', '.join(_SUPPORTED_COMPUTE_TYPES)
    )
    raise ValueError(msg)
