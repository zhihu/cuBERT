# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
########################################################
# RELATIVE_PROTOBUF_GENERATE_CPP function
########################################################
# A variant of PROTOBUF_GENERATE_CPP that keeps the directory hierarchy.
# ROOT_DIR must be absolute, and proto paths must be relative to ROOT_DIR.
function(RELATIVE_PROTOBUF_GENERATE_CPP SRCS HDRS ROOT_DIR)
    if(NOT ARGN)
        message(SEND_ERROR "Error: RELATIVE_PROTOBUF_GENERATE_CPP() called without any proto files")
        return()
    endif()

    set(${SRCS})
    set(${HDRS})
    foreach(FIL ${ARGN})
        set(ABS_FIL ${ROOT_DIR}/${FIL})
        get_filename_component(FIL_WE ${FIL} NAME_WE)
        get_filename_component(FIL_DIR ${ABS_FIL} PATH)
        file(RELATIVE_PATH REL_DIR ${ROOT_DIR} ${FIL_DIR})

        list(APPEND ${SRCS} "${CMAKE_CURRENT_BINARY_DIR}/${REL_DIR}/${FIL_WE}.pb-c.c")
        list(APPEND ${HDRS} "${CMAKE_CURRENT_BINARY_DIR}/${REL_DIR}/${FIL_WE}.pb-c.h")

        add_custom_command(
                OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/${REL_DIR}/${FIL_WE}.pb-c.c"
                "${CMAKE_CURRENT_BINARY_DIR}/${REL_DIR}/${FIL_WE}.pb-c.h"
                COMMAND  ${protobuf-c_PROTOC_EXECUTABLE}
                ARGS --c_out  ${CMAKE_CURRENT_BINARY_DIR} -I ${ROOT_DIR} ${ABS_FIL} -I ${protobuf-c_INCLUDE_DIRS}
                DEPENDS ${ABS_FIL} protobuf-c
                COMMENT "Running C protocol buffer compiler on ${FIL}"
                VERBATIM )
    endforeach()

    set_source_files_properties(${${SRCS}} ${${HDRS}} PROPERTIES GENERATED TRUE)
    set(${SRCS} ${${SRCS}} PARENT_SCOPE)
    set(${HDRS} ${${HDRS}} PARENT_SCOPE)
endfunction()

########################################################
# tf_protos_cc library
########################################################

file(GLOB_RECURSE tf_protos_cc_srcs RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}
        "${CMAKE_CURRENT_SOURCE_DIR}/tensorflow/core/framework/*.proto")

RELATIVE_PROTOBUF_GENERATE_CPP(PROTO_SRCS PROTO_HDRS
        ${CMAKE_CURRENT_SOURCE_DIR} ${tf_protos_cc_srcs}
        )

add_library(tf_protos_cc ${PROTO_SRCS} ${PROTO_HDRS})
