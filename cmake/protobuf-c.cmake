include (ExternalProject)

set(protobuf-c_INCLUDE_DIRS ${CMAKE_CURRENT_BINARY_DIR}/protobuf-c/include)
set(protobuf-c_STATIC_LIBRARIES ${CMAKE_CURRENT_BINARY_DIR}/protobuf-c/lib/libprotobuf-c.a)
set(protobuf-c_PROTOC_EXECUTABLE ${CMAKE_CURRENT_BINARY_DIR}/protobuf-c/bin/protoc-c)

ExternalProject_Add(protobuf-c
    PREFIX protobuf-c
    DEPENDS protobuf
    GIT_REPOSITORY https://github.com/protobuf-c/protobuf-c.git
    GIT_TAG v1.3.2
    GIT_SHALLOW 1
    DOWNLOAD_DIR "${DOWNLOAD_LOCATION}"
    BUILD_IN_SOURCE 1
    SOURCE_DIR ${CMAKE_CURRENT_BINARY_DIR}/protobuf-c/src/protobuf-c
    CONFIGURE_COMMAND ${CMAKE_COMMAND} build-cmake/
    -DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=ON
    -DCMAKE_BUILD_TYPE:STRING=Release
    -DCMAKE_VERBOSE_MAKEFILE:BOOL=OFF
    -DCMAKE_C_FLAGS=-std=c99
    -DCMAKE_CXX_FLAGS=-std=c++11
    -DCMAKE_INSTALL_PREFIX=${CMAKE_CURRENT_BINARY_DIR}/protobuf-c
    -DProtobuf_LIBRARY=${Protobuf_LIBRARY}
    -DProtobuf_PROTOC_LIBRARY=${Protobuf_PROTOC_LIBRARY}
    -DProtobuf_INCLUDE_DIR=${Protobuf_INCLUDE_DIR}
    -DProtobuf_PROTOC_EXECUTABLE=${Protobuf_PROTOC_EXECUTABLE}
    -DProtobuf_LITE_LIBRARY=${Protobuf_LITE_LIBRARY}
    CMAKE_CACHE_ARGS
    -DCMAKE_POSITION_INDEPENDENT_CODE:BOOL=ON
    -DCMAKE_BUILD_TYPE:STRING=Release
    -DCMAKE_VERBOSE_MAKEFILE:BOOL=OFF
    -DCMAKE_C_FLAGS=-std=c99
    -DCMAKE_CXX_FLAGS=-std=c++11
    -DCMAKE_INSTALL_PREFIX=${CMAKE_CURRENT_BINARY_DIR}/protobuf-c
    -DProtobuf_LIBRARY=${Protobuf_LIBRARY}
    -DProtobuf_PROTOC_LIBRARY=${Protobuf_PROTOC_LIBRARY}
    -DProtobuf_INCLUDE_DIR=${Protobuf_INCLUDE_DIR}
    -DProtobuf_PROTOC_EXECUTABLE=${Protobuf_PROTOC_EXECUTABLE}
    -DProtobuf_LITE_LIBRARY=${Protobuf_LITE_LIBRARY}
)
