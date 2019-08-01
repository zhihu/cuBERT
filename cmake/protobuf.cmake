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

set(Protobuf_LIBRARY ${CMAKE_CURRENT_BINARY_DIR}/protobuf/lib/libprotobuf.a)
set(Protobuf_PROTOC_LIBRARY ${CMAKE_CURRENT_BINARY_DIR}/protobuf/lib/libprotoc.a)
set(Protobuf_INCLUDE_DIR ${CMAKE_CURRENT_BINARY_DIR}/protobuf/include)
set(Protobuf_PROTOC_EXECUTABLE ${CMAKE_CURRENT_BINARY_DIR}/protobuf/bin/protoc)
set(Protobuf_LITE_LIBRARY ${CMAKE_CURRENT_BINARY_DIR}/protobuf/lib/libprotobuf-lite.a)

ExternalProject_Add(protobuf
        PREFIX protobuf
        GIT_REPOSITORY ${PROTOBUF_URL}
        GIT_TAG ${PROTOBUF_TAG}
        GIT_SHALLOW 1
        DOWNLOAD_DIR "${DOWNLOAD_LOCATION}"
        BUILD_IN_SOURCE 1
        SOURCE_DIR ${CMAKE_CURRENT_BINARY_DIR}/protobuf/src/protobuf
        # SOURCE_SUBDIR cmake/ # Requires CMake 3.7, this will allow removal of CONFIGURE_COMMAND
        # CONFIGURE_COMMAND resets some settings made in CMAKE_CACHE_ARGS and the generator used
        CONFIGURE_COMMAND ${CMAKE_COMMAND} cmake/
        -DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=ON
        -DCMAKE_BUILD_TYPE:STRING=Release
        -DCMAKE_VERBOSE_MAKEFILE:BOOL=OFF
        -Dprotobuf_BUILD_TESTS:BOOL=OFF
        -DCMAKE_C_FLAGS=-D_GLIBCXX_USE_CXX11_ABI=0
        -DCMAKE_CXX_FLAGS=-D_GLIBCXX_USE_CXX11_ABI=0
        -DCMAKE_INSTALL_PREFIX=${CMAKE_CURRENT_BINARY_DIR}/protobuf
        CMAKE_CACHE_ARGS
        -DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=ON
        -DCMAKE_BUILD_TYPE:STRING=Release
        -DCMAKE_VERBOSE_MAKEFILE:BOOL=OFF
        -Dprotobuf_BUILD_TESTS:BOOL=OFF
        -Dprotobuf_MSVC_STATIC_RUNTIME:BOOL=OFF
        -DCMAKE_C_FLAGS=-D_GLIBCXX_USE_CXX11_ABI=0
        -DCMAKE_CXX_FLAGS=-D_GLIBCXX_USE_CXX11_ABI=0
        -DCMAKE_INSTALL_PREFIX=${CMAKE_CURRENT_BINARY_DIR}/protobuf
        )