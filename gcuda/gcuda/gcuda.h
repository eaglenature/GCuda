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
/*!
 * Host side vector container assert.
 *
 *
 * STL vector example:
 * \code
 * #include <gcuda/gcuda.h>
 *
 * std::vector<int> a(size);
 * std::vector<int> b(size);
 *
 * ASSERT_HOST_VECTOR_EQ(a, b);
 *
 * \endcode
 *
 *
 * thrust vector example:
 * \code
 * #include <gcuda/gcuda.h>
 * #include <thrust/host_vector.h>
 *
 * thrust::host_vector<float4> a(size);
 * thrust::host_vector<float4> b(size);
 *
 * ASSERT_HOST_VECTOR_EQ(a, b);
 *
 * \endcode
 */
#define ASSERT_HOST_VECTOR_EQ(expected, actual) \
    ASSERT_PRED_FORMAT2(gcuda::assertHostVectorEq, expected, actual)


/*!
 * Host side raw array assert.
 *
 *
 * Raw host array example:
 * \code
 * #include <gcuda/gcuda.h>
 *
 * float* a = new float[size];
 * float* b = new float[size];
 *
 * ASSERT_HOST_VECTOR_EQ(a, b, size);
 *
 * \endcode
 */
#define ASSERT_HOST_ARRAY_EQ(expected, actual, size) \
    ASSERT_PRED_FORMAT3(gcuda::assertHostArrayEq, expected, actual, size)


/*!
 * Host side vector container assert.
 *
 * Compare floating-point data with given abs error.
 * Works only for floating-point data.
 *
 *
 * \code
 * #include <gcuda/gcuda.h>
 * #include <thrust/host_vector.h>
 *
 * thrust::host_vector<float4> a(size);
 * thrust::host_vector<float4> b(size);
 *
 * ASSERT_HOST_VECTOR_NEAR(a, b, 0.0001);
 *
 * \endcode
 */
#define ASSERT_HOST_VECTOR_NEAR(expected, actual, abs_error) \
    ASSERT_PRED_FORMAT3(gcuda::assertHostVectorNear, expected, actual, abs_error)


/*!
 * Host side raw array assert.
 * Compare floating-point data with given abs error.
 * Works only for floating-point data.
 *
 *
 * \code
 * #include <gcuda/gcuda.h>
 *
 * double3* a = new double3[size];
 * double3* b = new double3[size];
 *
 * ASSERT_HOST_ARRAY_NEAR(a, b, size, 0.0001);
 *
 * \endcode
 */
#define ASSERT_HOST_ARRAY_NEAR(expected, actual, size, abs_error) \
    ASSERT_PRED_FORMAT4(gcuda::assertHostArrayNear, expected, actual, size, abs_error)


/*!
 * Device side vector container assert.
 * Copy data back from device and compare on host side.
 *
 *
 * thrust vectors example:
 * \code
 * #include <gcuda/gcuda.h>
 * #include <thrust/device_vector.h>
 * #include <thrust/host_vector.h>
 *
 * thrust::host_vector<float4> a(size);
 * thrust::device_vector<float4> b(size);
 *
 * ASSERT_DEVICE_VECTOR_EQ(a, b);
 *
 * \endcode
 */
#define ASSERT_DEVICE_VECTOR_EQ(expected, actual) \
    ASSERT_PRED_FORMAT2(gcuda::assertDeviceVectorEq, expected, actual)


/*!
 * Device side raw array container assert.
 * Copy data back from device and compare on host side.
 *
 *
 * Compare with host vector example:
 * \code
 * #include <gcuda/gcuda.h>
 * #include <thrust/host_vector.h>
 *
 * thrust::host_vector<float4> a(size);
 * float4* d_data; // device pointer
 *
 * ASSERT_DEVICE_ARRAY_EQ(a, d_data, size);
 * \endcode
 *
 *
 * Compare with host raw pointer example:
 * \code
 * #include <gcuda/gcuda.h>
 *
 * float4  h_data; // host pointer
 * float4* d_data; // device pointer
 *
 * ASSERT_DEVICE_ARRAY_EQ(h_data, d_data, size);
 *
 * \endcode
 */
#define ASSERT_DEVICE_ARRAY_EQ(expected, actual, size) \
    ASSERT_PRED_FORMAT3(gcuda::assertDeviceArrayEq, expected, actual, size)


/*!
 * Device side vector container assert.
 * Compare floating-point data with given abs error.
 * Works only for floating-point data.
 *
 *
 * \code
 * #include <gcuda/gcuda.h>
 * #include <thrust/device_vector.h>
 * #include <thrust/host_vector.h>
 *
 * thrust::host_vector<float4> a(size);
 * thrust::device_vector<float4> b(size);
 *
 * ASSERT_DEVICE_VECTOR_NEAR(a, b, 0.0001);
 *
 * \endcode
 */
#define ASSERT_DEVICE_VECTOR_NEAR(expected, actual, abs_error) \
    ASSERT_PRED_FORMAT3(gcuda::assertDeviceVectorNear, expected, actual, abs_error)


/*!
 * Device side raw array container assert.
 * Copy data back from device and compare on host side.
 *
 *
 * Compare with host vector example:
 * \code
 * #include <gcuda/gcuda.h>
 * #include <thrust/host_vector.h>
 *
 * thrust::host_vector<float4> a(size);
 * float4* d_data; // device pointer
 *
 * ASSERT_DEVICE_ARRAY_NEAR(a, d_data, size, 0.00001);
 * \endcode
 *
 *
 * Compare with host raw pointer example:
 * \code
 * #include <gcuda/gcuda.h>
 *
 * float4  h_data; // host pointer
 * float4* d_data; // device pointer
 *
 * ASSERT_DEVICE_ARRAY_EQ(h_data, d_data, size, 0.0001);
 *
 * \endcode
 */
#define ASSERT_DEVICE_ARRAY_NEAR(expected, actual, size, abs_error) \
    ASSERT_PRED_FORMAT4(gcuda::assertDeviceArrayNear, expected, actual, size, abs_error)



/**
 * Expect does not hold test. Examples like in ASSERTS (TODO)
 */
#define EXPECT_HOST_VECTOR_EQ(expected, actual) \
    EXPECT_PRED_FORMAT2(gcuda::assertHostVectorEq, expected, actual)


#define EXPECT_HOST_ARRAY_EQ(expected, actual, size) \
    EXPECT_PRED_FORMAT3(gcuda::assertHostArrayEq, expected, actual, size)


#define EXPECT_HOST_VECTOR_NEAR(expected, actual, abs_error) \
    EXPECT_PRED_FORMAT3(gcuda::assertHostVectorNear, expected, actual, abs_error)


#define EXPECT_HOST_ARRAY_NEAR(expected, actual, size, abs_error) \
    EXPECT_PRED_FORMAT4(gcuda::assertHostArrayNear, expected, actual, size, abs_error)


#define EXPECT_DEVICE_VECTOR_EQ(expected, actual) \
    EXPECT_PRED_FORMAT2(gcuda::assertDeviceVectorEq, expected, actual)


#define EXPECT_DEVICE_ARRAY_EQ(expected, actual, size) \
    EXPECT_PRED_FORMAT3(gcuda::assertDeviceArrayEq, expected, actual, size)


#define EXPECT_DEVICE_VECTOR_NEAR(expected, actual, abs_error) \
    EXPECT_PRED_FORMAT3(gcuda::assertDeviceVectorNear, expected, actual, abs_error)


#define EXPECT_DEVICE_ARRAY_NEAR(expected, actual, size, abs_error) \
    EXPECT_PRED_FORMAT4(gcuda::assertDeviceArrayNear, expected, actual, size, abs_error)



} // namespace gcuda

#endif /* GCUDA_H_ */
