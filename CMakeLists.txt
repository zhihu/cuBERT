cmake_minimum_required(VERSION 3.0.2)
project(cuBERT LANGUAGES C CXX)

if(NOT CMAKE_VERSION VERSION_LESS 3.1)
    set(CMAKE_CXX_STANDARD 11)
else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-std c++11)
endif()
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake/)

find_package(Threads REQUIRED)
find_package(CUDA)

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
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-O3;-use_fast_math;-Xptxas -O3;-Xcompiler -O3)  # Flush denormals to zero

include(mkl)

include_directories(src external ${CUDA_INCLUDE_DIRS} ${mkl_INCLUDE_DIRS})
add_subdirectory(external)

set(SOURCE_FILES
        cuBERT.cpp
        src/cuBERT/tf/Graph.cpp
        src/cuBERT/op/GELU.cu
        src/cuBERT/op/Dense.cpp
        src/cuBERT/op/Embedding.cpp src/cuBERT/op/Embedding.cu
        src/cuBERT/op/BatchMatMul.cpp
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
        src/cuBERT/mgpu/BertMGPU.cpp)

cuda_add_library(cuBERT SHARED ${SOURCE_FILES})
set_target_properties(cuBERT PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
add_dependencies(cuBERT mkl_copy_shared_to_destination)
target_link_libraries(cuBERT
        external
        ${CUDA_LIBRARIES}
        ${CUDA_CUBLAS_LIBRARIES}
        ${mkl_SHARED_LIBRARIES})

add_executable(cuBERT_benchmark benchmark.cpp)
target_link_libraries(cuBERT_benchmark cuBERT)

add_executable(tfBERT_benchmark benchmark_tf.cpp)
target_link_libraries(tfBERT_benchmark tensorflow)

enable_testing()
include(googletest)
add_executable(cuBERT_test test/unit_test.cpp
        test/cuBERT/op/DenseTest.cpp
        test/cuBERT/op/LayerNormTest.cpp
        test/cuBERT/op/BatchMatMulTest.cpp
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
        test/cuBERT/mgpu/BertMGPUTest.cpp
        test/cuBERT_test.cpp)
target_link_libraries(cuBERT_test cuBERT ${googletest_STATIC_LIBRARIES} pthread)
target_include_directories(cuBERT_test PUBLIC ${googletest_INCLUDE_DIRS})
add_test(UnitTest cuBERT_test)
