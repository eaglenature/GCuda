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

#ifndef __CUDACC__
#error  NVCC compiler required
#endif

#include <gcuda/detail/gcuda.h>


namespace gcuda
{

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




#define ASSERT_DEVICE_VECTOR_EQ(expected, actual) \
    gcuda::assertDeviceVectorEq((expected), (actual), __FILE__, __LINE__)

#define ASSERT_DEVICE_ARRAY_EQ(expected, actual, size) \
    gcuda::assertDeviceArrayEq((expected), (actual), (size), __FILE__, __LINE__)

#define ASSERT_DEVICE_VECTOR_NEAR(expected, actual, abs_error) \
    gcuda::assertDeviceVectorNear((expected), (actual), (abs_error), __FILE__, __LINE__)

#define ASSERT_DEVICE_ARRAY_NEAR(expected, actual, size, abs_error) \
    gcuda::assertDeviceArrayNear((expected), (actual), (size), (abs_error), __FILE__, __LINE__)



/**
 * Expects
 */
#define EXPECT_HOST_VECTOR_EQ(expected, actual) \
    gcuda::expectHostVectorEq((expected), (actual), __FILE__, __LINE__)

#define EXPECT_HOST_ARRAY_EQ(expected, actual, size) \
    gcuda::expectHostArrayEq((expected), (actual), (size), __FILE__, __LINE__)

#define EXPECT_HOST_VECTOR_NEAR(expected, actual, abs_error) \
    gcuda::expectHostVectorNear((expected), (actual), (abs_error), __FILE__, __LINE__)

#define EXPECT_HOST_ARRAY_NEAR(expected, actual, size, abs_error) \
    gcuda::expectHostArrayNear((expected), (actual), (size), (abs_error), __FILE__, __LINE__)



#define EXPECT_DEVICE_VECTOR_EQ(expected, actual) \
    gcuda::expectDeviceVectorEq((expected), (actual), __FILE__, __LINE__)

#define EXPECT_DEVICE_ARRAY_EQ(expected, actual, size) \
    gcuda::expectDeviceArrayEq((expected), (actual), (size), __FILE__, __LINE__)

#define EXPECT_DEVICE_VECTOR_NEAR(expected, actual, abs_error) \
    gcuda::expectDeviceVectorNear((expected), (actual), (abs_error), __FILE__, __LINE__)

#define EXPECT_DEVICE_ARRAY_NEAR(expected, actual, size, abs_error) \
    gcuda::expectDeviceArrayNear((expected), (actual), (size), (abs_error), __FILE__, __LINE__)



} // namespace gcuda

#endif /* GCUDA_H_ */
