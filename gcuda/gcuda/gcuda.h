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

#define ASSERT_HOST_VECTOR_EQ(expected, actual) \
    ASSERT_PRED_FORMAT2(gcuda::assertHostVectorEq, expected, actual)


#define ASSERT_HOST_ARRAY_EQ(expected, actual, count) \
    ASSERT_PRED_FORMAT3(gcuda::assertHostArrayEq, expected, actual, count)


#define ASSERT_HOST_VECTOR_NEAR(expected, actual, abs_error) \
    ASSERT_PRED_FORMAT3(gcuda::assertHostVectorNear, expected, actual, abs_error)


#define ASSERT_HOST_ARRAY_NEAR(expected, actual, count, abs_error) \
    ASSERT_PRED_FORMAT4(gcuda::assertHostArrayNear, expected, actual, count, abs_error)


#define ASSERT_DEVICE_VECTOR_EQ(expected, actual) \
    ASSERT_PRED_FORMAT2(gcuda::assertDeviceVectorEq, expected, actual)


#define ASSERT_DEVICE_ARRAY_EQ(expected, actual, count) \
    ASSERT_PRED_FORMAT3(gcuda::assertDeviceArrayEq, expected, actual, count)


#define ASSERT_DEVICE_VECTOR_NEAR(expected, actual, abs_error) \
    ASSERT_PRED_FORMAT3(gcuda::assertDeviceVectorNear, expected, actual, abs_error)


#define ASSERT_DEVICE_ARRAY_NEAR(expected, actual, count, abs_error) \
    ASSERT_PRED_FORMAT4(gcuda::assertDeviceArrayNear, expected, actual, count, abs_error)




#define EXPECT_HOST_VECTOR_EQ(expected, actual) \
    EXPECT_PRED_FORMAT2(gcuda::assertHostVectorEq, expected, actual)


#define EXPECT_HOST_ARRAY_EQ(expected, actual, count) \
    EXPECT_PRED_FORMAT3(gcuda::assertHostArrayEq, expected, actual, count)


#define EXPECT_HOST_VECTOR_NEAR(expected, actual, abs_error) \
    EXPECT_PRED_FORMAT3(gcuda::assertHostVectorNear, expected, actual, abs_error)


#define EXPECT_HOST_ARRAY_NEAR(expected, actual, count, abs_error) \
    EXPECT_PRED_FORMAT4(gcuda::assertHostArrayNear, expected, actual, count, abs_error)


#define EXPECT_DEVICE_VECTOR_EQ(expected, actual) \
    EXPECT_PRED_FORMAT2(gcuda::assertDeviceVectorEq, expected, actual)


#define EXPECT_DEVICE_ARRAY_EQ(expected, actual, count) \
    EXPECT_PRED_FORMAT3(gcuda::assertDeviceArrayEq, expected, actual, count)


#define EXPECT_DEVICE_VECTOR_NEAR(expected, actual, abs_error) \
    EXPECT_PRED_FORMAT3(gcuda::assertDeviceVectorNear, expected, actual, abs_error)


#define EXPECT_DEVICE_ARRAY_NEAR(expected, actual, count, abs_error) \
    EXPECT_PRED_FORMAT4(gcuda::assertDeviceArrayNear, expected, actual, count, abs_error)


} // namespace gcuda

#endif /* GCUDA_H_ */
