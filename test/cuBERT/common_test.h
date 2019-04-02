#ifndef CUBERT_COMMON_TEST_H
#define CUBERT_COMMON_TEST_H

#include "gtest/gtest.h"
#include <cstdlib>
#include <cstring>

#include "cuBERT/common.h"

class CommonTest : public ::testing::Test {
protected:
    void SetUp() override {
        cuBERT::initialize();
        handle = cuBERT::blas_create();
    }

    void TearDown() override {
        cuBERT::blas_destroy(handle);
        cuBERT::finalize();
    }

    void* handle;
};

#endif //CUBERT_COMMON_TEST_H
