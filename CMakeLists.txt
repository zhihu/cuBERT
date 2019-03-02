cmake_minimum_required(VERSION 3.1)
project(cuBERT LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/)

find_package(Threads REQUIRED)

# OpenMP Support
find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

include(cub)
include(protobuf)

set(cuBERT_EXTERNAL_LIBRARIES
        ${protobuf_STATIC_LIBRARIES})
set(cuBERT_EXTERNAL_DEPENDENCIES
        cub
        protobuf)

include_directories(
        # Source and generated code.
        src
        ${CMAKE_CURRENT_BINARY_DIR}
        # External dependencies.
        ${cub_INCLUDE_DIR}
        ${PROTOBUF_INCLUDE_DIRS})

# MKL Support
include(mkl)
list(APPEND cuBERT_EXTERNAL_LIBRARIES ${mkl_SHARED_LIBRARIES})
list(APPEND cuBERT_EXTERNAL_DEPENDENCIES mkl_copy_shared_to_destination)
include_directories(${mkl_INCLUDE_DIRS})

find_package(CUDA)
if (CUDA_FOUND)
    add_definitions(-DHAVE_CUDA)

    # use cmake internal CUDA_ARCH_NAME switch
    # e.g. CUDA_ARCH_NAME="Auto" will autodetect
    #      CUDA_ARCH_NAME="All"  will use all arches
    # cuda_select_nvcc_arch_flags(NVCC_ARCH_FLAGS ${CUDA_ARCH_NAME})
    # list(APPEND CUDA_NVCC_FLAGS ${NVCC_ARCH_FLAGS})
    # message(STATUS "Using CUDA arch flags: ${NVCC_ARCH_FLAGS_readable}")

    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_60,code=sm_60)
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_61,code=sm_61)
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_62,code=sm_62)
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_70,code=sm_70)

    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};--expt-relaxed-constexpr)
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-O3;-use_fast_math;-Xptxas -O3;-Xcompiler -O3)
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-ftz=true)  # Flush denormals to zero

    include_directories(${CUDA_INCLUDE_DIRS})
endif()

# Let's get to work!
include(tf_core_framework)

set(SOURCE_FILES
        src/cuBERT.cpp
        src/utf8proc/utf8proc.c
        src/cuBERT/common.cpp
        src/cuBERT/tokenization.cpp
        src/cuBERT/tensorflow/Graph.cpp
        src/cuBERT/op/GELU.cpp src/cuBERT/op/GELU.cu
        src/cuBERT/op/Dense.cpp
        src/cuBERT/op/Embedding.cpp src/cuBERT/op/Embedding.cu
        src/cuBERT/op/Softmax.cpp src/cuBERT/op/Softmax.cu
        src/cuBERT/op/LayerNorm.cpp src/cuBERT/op/LayerNorm.cu
        src/cuBERT/bert/BertEmbeddings.cpp
        src/cuBERT/bert/AttentionSelf.cpp
        src/cuBERT/bert/AttentionMask.cpp src/cuBERT/bert/AttentionMask.cu
        src/cuBERT/bert/BertPooler.cpp src/cuBERT/bert/BertPooler.cu
        src/cuBERT/bert/Transformer.cpp
        src/cuBERT/bert/Bert.cpp
        src/cuBERT/bert/BertBatchMatMul.cpp
        src/cuBERT/bert/AdditionalOutputLayer.cpp
        src/cuBERT/multi/BertM.cpp)

if (CUDA_FOUND)
    cuda_add_library(cuBERT SHARED ${SOURCE_FILES})
    set_target_properties(cuBERT PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
else ()
    add_library(cuBERT SHARED ${SOURCE_FILES})
endif ()
add_dependencies(cuBERT ${cuBERT_EXTERNAL_DEPENDENCIES})
target_link_libraries(cuBERT tf_protos_cc ${cuBERT_EXTERNAL_LIBRARIES})

add_executable(cuBERT_benchmark benchmark/benchmark_cu.cpp)
target_link_libraries(cuBERT_benchmark cuBERT)

add_executable(tfBERT_benchmark benchmark/benchmark_tf.cpp)
target_link_libraries(tfBERT_benchmark tensorflow)

enable_testing()
configure_file(test_vocab.txt ${CMAKE_CURRENT_BINARY_DIR}/test_vocab.txt COPYONLY)
include(googletest)
add_executable(cuBERT_test test/unit_test.cpp
        test/cuBERT/common_test.cpp
        test/cuBERT/tokenization_test.cpp
        test/cuBERT/op/DenseTest.cpp
        test/cuBERT/op/LayerNormTest.cpp
        test/cuBERT/op/SoftmaxTest.cpp
        test/cuBERT/op/GELUTest.cpp
        test/cuBERT/op/EmbeddingTest.cpp
        test/cuBERT/bert/BertBatchMatMulTest.cpp
        test/cuBERT/bert/AttentionSelfTest.cpp
        test/cuBERT/bert/TransformerTest.cpp
        test/cuBERT/bert/BertEmbeddingsTest.cpp
        test/cuBERT/bert/AttentionMaskTest.cpp
        test/cuBERT/bert/BertPoolerTest.cpp
        test/cuBERT/bert/AdditionalOutputLayerTest.cpp
        test/cuBERT/bert/BertTest.cpp
        test/cuBERT/multi/BertMTest.cpp
        test/cuBERT_test.cpp)
target_link_libraries(cuBERT_test cuBERT ${googletest_STATIC_LIBRARIES})
target_include_directories(cuBERT_test PUBLIC ${googletest_INCLUDE_DIRS})
add_test(UnitTest cuBERT_test)


install(TARGETS cuBERT
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib/static)
install(FILES ${CMAKE_SOURCE_DIR}/src/cuBERT.h DESTINATION include)

install(FILES ${mkl_SHARED_LIBRARIES} DESTINATION lib)
install(DIRECTORY ${mkl_INCLUDE_DIRS} DESTINATION .)
