/*
 * <gcuda/gcuda.h>
 *
 *  Created on: May 3, 2013
 *      Author: eaglenature@gmail.com
 *
 *  Description: Cuda C++ extension for Google C++ testing Framework
 */

#ifndef GCUDA_H_
#define GCUDA_H_

#include <gcuda/detail/gcuda.h>



/**
 * Assertions
 */
#define ASSERT_HOST_VECTOR_EQ(expected, actual) \
    gcuda::assertHostVectorEq((expected), (actual), __FILE__, __LINE__)

#define ASSERT_HOST_ARRAY_EQ(expected, actual, size) \
    gcuda::assertHostArrayEq((expected), (actual), (size), __FILE__, __LINE__)

#define ASSERT_HOST_VECTOR_NEAR(expected, actual, abs_error) \
    gcuda::assertHostVectorNear((expected), (actual), (abs_error), __FILE__, __LINE__)

#define ASSERT_HOST_ARRAY_NEAR(expected, actual, size, abs_error) \
    gcuda::assertHostArrayNear((expected), (actual), (size), (abs_error), __FILE__, __LINE__)


/**
 * Expectations
 */
#define EXPECT_HOST_VECTOR_EQ(expected, actual) \
    gcuda::expectHostVectorEq((expected), (actual), __FILE__, __LINE__)

#define EXPECT_HOST_ARRAY_EQ(expected, actual, size) \
    gcuda::expectHostArrayEq((expected), (actual), (size), __FILE__, __LINE__)

#define EXPECT_HOST_VECTOR_NEAR(expected, actual, abs_error) \
    gcuda::expectHostVectorNear((expected), (actual), (abs_error), __FILE__, __LINE__)

#define EXPECT_HOST_ARRAY_NEAR(expected, actual, size, abs_error) \
    gcuda::expectHostArrayNear((expected), (actual), (size), (abs_error), __FILE__, __LINE__)



#endif /* GCUDA_H_ */
