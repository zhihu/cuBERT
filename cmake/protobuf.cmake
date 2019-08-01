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
include (ExternalProject)

set(PROTOBUF_URL https://github.com/google/protobuf.git)
set(PROTOBUF_TAG v3.6.0)

ExternalProject_Add(protobuf
        PREFIX protobuf
        GIT_REPOSITORY ${PROTOBUF_URL}
        GIT_TAG ${PROTOBUF_TAG}
        GIT_SHALLOW 1
        DOWNLOAD_DIR "${DOWNLOAD_LOCATION}"
        BUILD_IN_SOURCE 1
        SOURCE_DIR ${CMAKE_CURRENT_BINARY_DIR}/protobuf/src/protobuf
        CONFIGURE_COMMAND ${CMAKE_CURRENT_BINARY_DIR}/protobuf/src/protobuf/autogen.sh
        BUILD_COMMAND ${CMAKE_CURRENT_BINARY_DIR}/protobuf/src/protobuf/configure --prefix=${CMAKE_CURRENT_BINARY_DIR}/protobuf
        INSTALL_COMMAND make install
        )