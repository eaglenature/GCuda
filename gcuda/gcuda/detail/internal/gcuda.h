/*
 * <gcuda/detail/internal/gcuda.h>
 *
 *  Created on: May 4, 2013
 *      Author: eaglenature@gmail.com
 */

#ifndef DETAIL_INTERNAL_GCUDA_H_
#define DETAIL_INTERNAL_GCUDA_H_


#include <gtest/gtest.h>
#include <gcuda/detail/internal/traits.h>



#ifndef GCUDA_INFO
#define GCUDA_INFO(index, file, line) \
    "At index: " << (index) << "\n" << (file) << ":" << (line) << '\n'
#else
#error GCUDA_INFO redefinition
#endif


namespace gcuda
{
namespace internal
{


/*******************************************
 *         ASSERT ARRAY EQUAL
 *******************************************/
template <typename T>
void assertArrayEq(
        const T*  expected,
        const T*  actual,
        const int size,
        const char* const file,
        const int line);


template <>
void assertArrayEq<char>(
        const char* expected,
        const char* actual,
        const int size,
        const char* const file,
        const int line)
{
    for (int i = 0; i < size; ++i)
    {
        ASSERT_EQ(expected[i], actual[i]) << GCUDA_INFO(i, file, line);
    }
}
template <>
void assertArrayEq<unsigned char>(
        const unsigned char* expected,
        const unsigned char* actual,
        const int size,
        const char* const file,
        const int line)
{
    for (int i = 0; i < size; ++i)
    {
        ASSERT_EQ(expected[i], actual[i]) << GCUDA_INFO(i, file, line);
    }
}
template <>
void assertArrayEq<short>(
        const short* expected,
        const short* actual,
        const int size,
        const char* const file,
        const int line)
{
    for (int i = 0; i < size; ++i)
    {
        ASSERT_EQ(expected[i], actual[i]) << GCUDA_INFO(i, file, line);
    }
}
template <>
void assertArrayEq<unsigned short>(
        const unsigned short* expected,
        const unsigned short* actual,
        const int size,
        const char* const file,
        const int line)
{
    for (int i = 0; i < size; ++i)
    {
        ASSERT_EQ(expected[i], actual[i]) << GCUDA_INFO(i, file, line);
    }
}
template <>
void assertArrayEq<int>(
        const int* expected,
        const int* actual,
        const int size,
        const char* const file,
        const int line)
{
    for (int i = 0; i < size; ++i)
    {
        ASSERT_EQ(expected[i], actual[i]) << GCUDA_INFO(i, file, line);
    }
}
template <>
void assertArrayEq<unsigned int>(
        const unsigned int* expected,
        const unsigned int* actual,
        const int size,
        const char* const file,
        const int line)
{
    for (int i = 0; i < size; ++i)
    {
        ASSERT_EQ(expected[i], actual[i]) << GCUDA_INFO(i, file, line);
    }
}
template <>
void assertArrayEq<long>(
        const long* expected,
        const long* actual,
        const int size,
        const char* const file,
        const int line)
{
    for (int i = 0; i < size; ++i)
    {
        ASSERT_EQ(expected[i], actual[i]) << GCUDA_INFO(i, file, line);
    }
}
template <>
void assertArrayEq<unsigned long>(
        const unsigned long* expected,
        const unsigned long* actual,
        const int size,
        const char* const file,
        const int line)
{
    for (int i = 0; i < size; ++i)
    {
        ASSERT_EQ(expected[i], actual[i]) << GCUDA_INFO(i, file, line);
    }
}
template <>
void assertArrayEq<char2>(
        const char2* expected,
        const char2* actual,
        const int size,
        const char* const file,
        const int line)
{
    for (int i = 0; i < size; ++i)
    {
        ASSERT_EQ(expected[i].x, actual[i].x) << GCUDA_INFO(i, file, line);
        ASSERT_EQ(expected[i].y, actual[i].y) << GCUDA_INFO(i, file, line);
    }
}
template <>
void assertArrayEq<uchar2>(
        const uchar2* expected,
        const uchar2* actual,
        const int size,
        const char* const file,
        const int line)
{
    for (int i = 0; i < size; ++i)
    {
        ASSERT_EQ(expected[i].x, actual[i].x) << GCUDA_INFO(i, file, line);
        ASSERT_EQ(expected[i].y, actual[i].y) << GCUDA_INFO(i, file, line);
    }
}
template <>
void assertArrayEq<short2>(
        const short2* expected,
        const short2* actual,
        const int size,
        const char* const file,
        const int line)
{
    for (int i = 0; i < size; ++i)
    {
        ASSERT_EQ(expected[i].x, actual[i].x) << GCUDA_INFO(i, file, line);
        ASSERT_EQ(expected[i].y, actual[i].y) << GCUDA_INFO(i, file, line);
    }
}
template <>
void assertArrayEq<ushort2>(
        const ushort2* expected,
        const ushort2* actual,
        const int size,
        const char* const file,
        const int line)
{
    for (int i = 0; i < size; ++i)
    {
        ASSERT_EQ(expected[i].x, actual[i].x) << GCUDA_INFO(i, file, line);
        ASSERT_EQ(expected[i].y, actual[i].y) << GCUDA_INFO(i, file, line);
    }
}
template <>
void assertArrayEq<int2>(
        const int2* expected,
        const int2* actual,
        const int size,
        const char* const file,
        const int line)
{
    for (int i = 0; i < size; ++i)
    {
        ASSERT_EQ(expected[i].x, actual[i].x) << GCUDA_INFO(i, file, line);
        ASSERT_EQ(expected[i].y, actual[i].y) << GCUDA_INFO(i, file, line);
    }
}
template <>
void assertArrayEq<uint2>(
        const uint2* expected,
        const uint2* actual,
        const int size,
        const char* const file,
        const int line)
{
    for (int i = 0; i < size; ++i)
    {
        ASSERT_EQ(expected[i].x, actual[i].x) << GCUDA_INFO(i, file, line);
        ASSERT_EQ(expected[i].y, actual[i].y) << GCUDA_INFO(i, file, line);
    }
}
template <>
void assertArrayEq<long2>(
        const long2* expected,
        const long2* actual,
        const int size,
        const char* const file,
        const int line)
{
    for (int i = 0; i < size; ++i)
    {
        ASSERT_EQ(expected[i].x, actual[i].x) << GCUDA_INFO(i, file, line);
        ASSERT_EQ(expected[i].y, actual[i].y) << GCUDA_INFO(i, file, line);
    }
}
template <>
void assertArrayEq<ulong2>(
        const ulong2* expected,
        const ulong2* actual,
        const int size,
        const char* const file,
        const int line)
{
    for (int i = 0; i < size; ++i)
    {
        ASSERT_EQ(expected[i].x, actual[i].x) << GCUDA_INFO(i, file, line);
        ASSERT_EQ(expected[i].y, actual[i].y) << GCUDA_INFO(i, file, line);
    }
}
template <>
void assertArrayEq<char3>(
        const char3* expected,
        const char3* actual,
        const int size,
        const char* const file,
        const int line)
{
    for (int i = 0; i < size; ++i)
    {
        ASSERT_EQ(expected[i].x, actual[i].x) << GCUDA_INFO(i, file, line);
        ASSERT_EQ(expected[i].y, actual[i].y) << GCUDA_INFO(i, file, line);
        ASSERT_EQ(expected[i].z, actual[i].z) << GCUDA_INFO(i, file, line);
    }
}
template <>
void assertArrayEq<uchar3>(
        const uchar3* expected,
        const uchar3* actual,
        const int size,
        const char* const file,
        const int line)
{
    for (int i = 0; i < size; ++i)
    {
        ASSERT_EQ(expected[i].x, actual[i].x) << GCUDA_INFO(i, file, line);
        ASSERT_EQ(expected[i].y, actual[i].y) << GCUDA_INFO(i, file, line);
        ASSERT_EQ(expected[i].z, actual[i].z) << GCUDA_INFO(i, file, line);
    }
}
template <>
void assertArrayEq<short3>(
        const short3* expected,
        const short3* actual,
        const int size,
        const char* const file,
        const int line)
{
    for (int i = 0; i < size; ++i)
    {
        ASSERT_EQ(expected[i].x, actual[i].x) << GCUDA_INFO(i, file, line);
        ASSERT_EQ(expected[i].y, actual[i].y) << GCUDA_INFO(i, file, line);
        ASSERT_EQ(expected[i].z, actual[i].z) << GCUDA_INFO(i, file, line);
    }
}
template <>
void assertArrayEq<ushort3>(
        const ushort3* expected,
        const ushort3* actual,
        const int size,
        const char* const file,
        const int line)
{
    for (int i = 0; i < size; ++i)
    {
        ASSERT_EQ(expected[i].x, actual[i].x) << GCUDA_INFO(i, file, line);
        ASSERT_EQ(expected[i].y, actual[i].y) << GCUDA_INFO(i, file, line);
        ASSERT_EQ(expected[i].z, actual[i].z) << GCUDA_INFO(i, file, line);
    }
}
template <>
void assertArrayEq<int3>(
        const int3* expected,
        const int3* actual,
        const int size,
        const char* const file,
        const int line)
{
    for (int i = 0; i < size; ++i)
    {
        ASSERT_EQ(expected[i].x, actual[i].x) << GCUDA_INFO(i, file, line);
        ASSERT_EQ(expected[i].y, actual[i].y) << GCUDA_INFO(i, file, line);
        ASSERT_EQ(expected[i].z, actual[i].z) << GCUDA_INFO(i, file, line);
    }
}
template <>
void assertArrayEq<uint3>(
        const uint3* expected,
        const uint3* actual,
        const int size,
        const char* const file,
        const int line)
{
    for (int i = 0; i < size; ++i)
    {
        ASSERT_EQ(expected[i].x, actual[i].x) << GCUDA_INFO(i, file, line);
        ASSERT_EQ(expected[i].y, actual[i].y) << GCUDA_INFO(i, file, line);
        ASSERT_EQ(expected[i].z, actual[i].z) << GCUDA_INFO(i, file, line);
    }
}
template <>
void assertArrayEq<long3>(
        const long3* expected,
        const long3* actual,
        const int size,
        const char* const file,
        const int line)
{
    for (int i = 0; i < size; ++i)
    {
        ASSERT_EQ(expected[i].x, actual[i].x) << GCUDA_INFO(i, file, line);
        ASSERT_EQ(expected[i].y, actual[i].y) << GCUDA_INFO(i, file, line);
        ASSERT_EQ(expected[i].z, actual[i].z) << GCUDA_INFO(i, file, line);
    }
}
template <>
void assertArrayEq<ulong3>(
        const ulong3* expected,
        const ulong3* actual,
        const int size,
        const char* const file,
        const int line)
{
    for (int i = 0; i < size; ++i)
    {
        ASSERT_EQ(expected[i].x, actual[i].x) << GCUDA_INFO(i, file, line);
        ASSERT_EQ(expected[i].y, actual[i].y) << GCUDA_INFO(i, file, line);
        ASSERT_EQ(expected[i].z, actual[i].z) << GCUDA_INFO(i, file, line);
    }
}
template <>
void assertArrayEq<char4>(
        const char4* expected,
        const char4* actual,
        const int size,
        const char* const file,
        const int line)
{
    for (int i = 0; i < size; ++i)
    {
        ASSERT_EQ(expected[i].x, actual[i].x) << GCUDA_INFO(i, file, line);
        ASSERT_EQ(expected[i].y, actual[i].y) << GCUDA_INFO(i, file, line);
        ASSERT_EQ(expected[i].z, actual[i].z) << GCUDA_INFO(i, file, line);
        ASSERT_EQ(expected[i].w, actual[i].w) << GCUDA_INFO(i, file, line);
    }
}
template <>
void assertArrayEq<uchar4>(
        const uchar4* expected,
        const uchar4* actual,
        const int size,
        const char* const file,
        const int line)
{
    for (int i = 0; i < size; ++i)
    {
        ASSERT_EQ(expected[i].x, actual[i].x) << GCUDA_INFO(i, file, line);
        ASSERT_EQ(expected[i].y, actual[i].y) << GCUDA_INFO(i, file, line);
        ASSERT_EQ(expected[i].z, actual[i].z) << GCUDA_INFO(i, file, line);
        ASSERT_EQ(expected[i].w, actual[i].w) << GCUDA_INFO(i, file, line);
    }
}
template <>
void assertArrayEq<short4>(
        const short4* expected,
        const short4* actual,
        const int size,
        const char* const file,
        const int line)
{
    for (int i = 0; i < size; ++i)
    {
        ASSERT_EQ(expected[i].x, actual[i].x) << GCUDA_INFO(i, file, line);
        ASSERT_EQ(expected[i].y, actual[i].y) << GCUDA_INFO(i, file, line);
        ASSERT_EQ(expected[i].z, actual[i].z) << GCUDA_INFO(i, file, line);
        ASSERT_EQ(expected[i].w, actual[i].w) << GCUDA_INFO(i, file, line);
    }
}
template <>
void assertArrayEq<ushort4>(
        const ushort4* expected,
        const ushort4* actual,
        const int size,
        const char* const file,
        const int line)
{
    for (int i = 0; i < size; ++i)
    {
        ASSERT_EQ(expected[i].x, actual[i].x) << GCUDA_INFO(i, file, line);
        ASSERT_EQ(expected[i].y, actual[i].y) << GCUDA_INFO(i, file, line);
        ASSERT_EQ(expected[i].z, actual[i].z) << GCUDA_INFO(i, file, line);
        ASSERT_EQ(expected[i].w, actual[i].w) << GCUDA_INFO(i, file, line);
    }
}
template <>
void assertArrayEq<int4>(
        const int4* expected,
        const int4* actual,
        const int size,
        const char* const file,
        const int line)
{
    for (int i = 0; i < size; ++i)
    {
        ASSERT_EQ(expected[i].x, actual[i].x) << GCUDA_INFO(i, file, line);
        ASSERT_EQ(expected[i].y, actual[i].y) << GCUDA_INFO(i, file, line);
        ASSERT_EQ(expected[i].z, actual[i].z) << GCUDA_INFO(i, file, line);
        ASSERT_EQ(expected[i].w, actual[i].w) << GCUDA_INFO(i, file, line);
    }
}
template <>
void assertArrayEq<uint4>(
        const uint4* expected,
        const uint4* actual,
        const int size,
        const char* const file,
        const int line)
{
    for (int i = 0; i < size; ++i)
    {
        ASSERT_EQ(expected[i].x, actual[i].x) << GCUDA_INFO(i, file, line);
        ASSERT_EQ(expected[i].y, actual[i].y) << GCUDA_INFO(i, file, line);
        ASSERT_EQ(expected[i].z, actual[i].z) << GCUDA_INFO(i, file, line);
        ASSERT_EQ(expected[i].w, actual[i].w) << GCUDA_INFO(i, file, line);
    }
}
template <>
void assertArrayEq<long4>(
        const long4* expected,
        const long4* actual,
        const int size,
        const char* const file,
        const int line)
{
    for (int i = 0; i < size; ++i)
    {
        ASSERT_EQ(expected[i].x, actual[i].x) << GCUDA_INFO(i, file, line);
        ASSERT_EQ(expected[i].y, actual[i].y) << GCUDA_INFO(i, file, line);
        ASSERT_EQ(expected[i].z, actual[i].z) << GCUDA_INFO(i, file, line);
        ASSERT_EQ(expected[i].w, actual[i].w) << GCUDA_INFO(i, file, line);
    }
}
template <>
void assertArrayEq<ulong4>(
        const ulong4* expected,
        const ulong4* actual,
        const int size,
        const char* const file,
        const int line)
{
    for (int i = 0; i < size; ++i)
    {
        ASSERT_EQ(expected[i].x, actual[i].x) << GCUDA_INFO(i, file, line);
        ASSERT_EQ(expected[i].y, actual[i].y) << GCUDA_INFO(i, file, line);
        ASSERT_EQ(expected[i].z, actual[i].z) << GCUDA_INFO(i, file, line);
        ASSERT_EQ(expected[i].w, actual[i].w) << GCUDA_INFO(i, file, line);
    }
}
template <>
void assertArrayEq<float>(
        const float* expected,
        const float* actual,
        const int size,
        const char* const file,
        const int line)
{
    for (int i = 0; i < size; ++i)
    {
        ASSERT_FLOAT_EQ(expected[i], actual[i]) << GCUDA_INFO(i, file, line);
    }
}
template <>
void assertArrayEq<double>(
        const double* expected,
        const double* actual,
        const int size,
        const char* const file,
        const int line)
{
    for (int i = 0; i < size; ++i)
    {
        ASSERT_DOUBLE_EQ(expected[i], actual[i]) << GCUDA_INFO(i, file, line);
    }
}
template <>
void assertArrayEq<float2>(
        const float2* expected,
        const float2* actual,
        const int size,
        const char* const file,
        const int line)
{
    for (int i = 0; i < size; ++i)
    {
        ASSERT_FLOAT_EQ(expected[i].x, actual[i].x) << GCUDA_INFO(i, file, line);
        ASSERT_FLOAT_EQ(expected[i].y, actual[i].y) << GCUDA_INFO(i, file, line);
    }
}
template <>
void assertArrayEq<float3>(
        const float3* expected,
        const float3* actual,
        const int size,
        const char* const file,
        const int line)
{
    for (int i = 0; i < size; ++i)
    {
        ASSERT_FLOAT_EQ(expected[i].x, actual[i].x) << GCUDA_INFO(i, file, line);
        ASSERT_FLOAT_EQ(expected[i].y, actual[i].y) << GCUDA_INFO(i, file, line);
        ASSERT_FLOAT_EQ(expected[i].z, actual[i].z) << GCUDA_INFO(i, file, line);
    }
}
template <>
void assertArrayEq<float4>(
        const float4* expected,
        const float4* actual,
        const int size,
        const char* const file,
        const int line)
{
    for (int i = 0; i < size; ++i)
    {
        ASSERT_FLOAT_EQ(expected[i].x, actual[i].x) << GCUDA_INFO(i, file, line);
        ASSERT_FLOAT_EQ(expected[i].y, actual[i].y) << GCUDA_INFO(i, file, line);
        ASSERT_FLOAT_EQ(expected[i].z, actual[i].z) << GCUDA_INFO(i, file, line);
        ASSERT_FLOAT_EQ(expected[i].w, actual[i].w) << GCUDA_INFO(i, file, line);
    }
}
template <>
void assertArrayEq<double2>(
        const double2* expected,
        const double2* actual,
        const int size,
        const char* const file,
        const int line)
{
    for (int i = 0; i < size; ++i)
    {
        ASSERT_DOUBLE_EQ(expected[i].x, actual[i].x) << GCUDA_INFO(i, file, line);
        ASSERT_DOUBLE_EQ(expected[i].y, actual[i].y) << GCUDA_INFO(i, file, line);
    }
}
template <>
void assertArrayEq<double3>(
        const double3* expected,
        const double3* actual,
        const int size,
        const char* const file,
        const int line)
{
    for (int i = 0; i < size; ++i)
    {
        ASSERT_DOUBLE_EQ(expected[i].x, actual[i].x) << GCUDA_INFO(i, file, line);
        ASSERT_DOUBLE_EQ(expected[i].y, actual[i].y) << GCUDA_INFO(i, file, line);
        ASSERT_DOUBLE_EQ(expected[i].z, actual[i].z) << GCUDA_INFO(i, file, line);
    }
}
template <>
void assertArrayEq<double4>(
        const double4* expected,
        const double4* actual,
        const int size,
        const char* const file,
        const int line)
{
    for (int i = 0; i < size; ++i)
    {
        ASSERT_DOUBLE_EQ(expected[i].x, actual[i].x) << GCUDA_INFO(i, file, line);
        ASSERT_DOUBLE_EQ(expected[i].y, actual[i].y) << GCUDA_INFO(i, file, line);
        ASSERT_DOUBLE_EQ(expected[i].z, actual[i].z) << GCUDA_INFO(i, file, line);
        ASSERT_DOUBLE_EQ(expected[i].w, actual[i].w) << GCUDA_INFO(i, file, line);
    }
}



/*******************************************
 *         ASSERT ARRAY NEAR
 *******************************************/
template <typename T>
void assertArrayNear(
        const T* expected,
        const T* actual,
        const int size,
        const double abs_error,
        const char* const file,
        const int line);



template <>
void assertArrayNear<float>(
        const float* expected,
        const float* actual,
        const int size,
        const double abs_error,
        const char* const file,
        const int line)
{
    for (int i = 0; i < size; ++i)
    {
        ASSERT_NEAR(expected[i], actual[i], abs_error) << GCUDA_INFO(i, file, line);
    }
}
template <>
void assertArrayNear<double>(
        const double* expected,
        const double* actual,
        const int size,
        const double abs_error,
        const char* const file,
        const int line)
{
    for (int i = 0; i < size; ++i)
    {
        ASSERT_NEAR(expected[i], actual[i], abs_error) << GCUDA_INFO(i, file, line);
    }
}
template <>
void assertArrayNear<float2>(
        const float2* expected,
        const float2* actual,
        const int size,
        const double abs_error,
        const char* const file,
        const int line)
{
    for (int i = 0; i < size; ++i)
    {
        ASSERT_NEAR(expected[i].x, actual[i].x, abs_error) << GCUDA_INFO(i, file, line);
        ASSERT_NEAR(expected[i].y, actual[i].y, abs_error) << GCUDA_INFO(i, file, line);
    }
}
template <>
void assertArrayNear<double2>(
        const double2* expected,
        const double2* actual,
        const int size,
        const double abs_error,
        const char* const file,
        const int line)
{
    for (int i = 0; i < size; ++i)
    {
        ASSERT_NEAR(expected[i].x, actual[i].x, abs_error) << GCUDA_INFO(i, file, line);
        ASSERT_NEAR(expected[i].y, actual[i].y, abs_error) << GCUDA_INFO(i, file, line);
    }
}

template <>
void assertArrayNear<float3>(
        const float3* expected,
        const float3* actual,
        const int size,
        const double abs_error,
        const char* const file,
        const int line)
{
    for (int i = 0; i < size; ++i)
    {
        ASSERT_NEAR(expected[i].x, actual[i].x, abs_error) << GCUDA_INFO(i, file, line);
        ASSERT_NEAR(expected[i].y, actual[i].y, abs_error) << GCUDA_INFO(i, file, line);
        ASSERT_NEAR(expected[i].z, actual[i].z, abs_error) << GCUDA_INFO(i, file, line);
    }
}
template <>
void assertArrayNear<double3>(
        const double3* expected,
        const double3* actual,
        const int size,
        const double abs_error,
        const char* const file,
        const int line)
{
    for (int i = 0; i < size; ++i)
    {
        ASSERT_NEAR(expected[i].x, actual[i].x, abs_error) << GCUDA_INFO(i, file, line);
        ASSERT_NEAR(expected[i].y, actual[i].y, abs_error) << GCUDA_INFO(i, file, line);
        ASSERT_NEAR(expected[i].z, actual[i].z, abs_error) << GCUDA_INFO(i, file, line);
    }
}

template <>
void assertArrayNear<float4>(
        const float4* expected,
        const float4* actual,
        const int size,
        const double abs_error,
        const char* const file,
        const int line)
{
    for (int i = 0; i < size; ++i)
    {
        ASSERT_NEAR(expected[i].x, actual[i].x, abs_error) << GCUDA_INFO(i, file, line);
        ASSERT_NEAR(expected[i].y, actual[i].y, abs_error) << GCUDA_INFO(i, file, line);
        ASSERT_NEAR(expected[i].z, actual[i].z, abs_error) << GCUDA_INFO(i, file, line);
        ASSERT_NEAR(expected[i].w, actual[i].w, abs_error) << GCUDA_INFO(i, file, line);
    }
}
template <>
void assertArrayNear<double4>(
        const double4* expected,
        const double4* actual,
        const int size,
        const double abs_error,
        const char* const file,
        const int line)
{
    for (int i = 0; i < size; ++i)
    {
        ASSERT_NEAR(expected[i].x, actual[i].x, abs_error) << GCUDA_INFO(i, file, line);
        ASSERT_NEAR(expected[i].y, actual[i].y, abs_error) << GCUDA_INFO(i, file, line);
        ASSERT_NEAR(expected[i].z, actual[i].z, abs_error) << GCUDA_INFO(i, file, line);
        ASSERT_NEAR(expected[i].w, actual[i].w, abs_error) << GCUDA_INFO(i, file, line);
    }
}



/*******************************************
 *         EXPECT ARRAY EQUAL
 *******************************************/
template <typename T>
void expectArrayEq(
        const T*  expected,
        const T*  actual,
        const int size,
        const char* const file,
        const int line);



template <>
void expectArrayEq<char>(
        const char* expected,
        const char* actual,
        const int size,
        const char* const file,
        const int line)
{
    for (int i = 0; i < size; ++i)
    {
        EXPECT_EQ(expected[i], actual[i]) << GCUDA_INFO(i, file, line);
    }
}
template <>
void expectArrayEq<unsigned char>(
        const unsigned char* expected,
        const unsigned char* actual,
        const int size,
        const char* const file,
        const int line)
{
    for (int i = 0; i < size; ++i)
    {
        EXPECT_EQ(expected[i], actual[i]) << GCUDA_INFO(i, file, line);
    }
}
template <>
void expectArrayEq<short>(
        const short* expected,
        const short* actual,
        const int size,
        const char* const file,
        const int line)
{
    for (int i = 0; i < size; ++i)
    {
        EXPECT_EQ(expected[i], actual[i]) << GCUDA_INFO(i, file, line);
    }
}
template <>
void expectArrayEq<unsigned short>(
        const unsigned short* expected,
        const unsigned short* actual,
        const int size,
        const char* const file,
        const int line)
{
    for (int i = 0; i < size; ++i)
    {
        EXPECT_EQ(expected[i], actual[i]) << GCUDA_INFO(i, file, line);
    }
}
template <>
void expectArrayEq<int>(
        const int* expected,
        const int* actual,
        const int size,
        const char* const file,
        const int line)
{
    for (int i = 0; i < size; ++i)
    {
        EXPECT_EQ(expected[i], actual[i]) << GCUDA_INFO(i, file, line);
    }
}
template <>
void expectArrayEq<unsigned int>(
        const unsigned int* expected,
        const unsigned int* actual,
        const int size,
        const char* const file,
        const int line)
{
    for (int i = 0; i < size; ++i)
    {
        EXPECT_EQ(expected[i], actual[i]) << GCUDA_INFO(i, file, line);
    }
}
template <>
void expectArrayEq<long>(
        const long* expected,
        const long* actual,
        const int size,
        const char* const file,
        const int line)
{
    for (int i = 0; i < size; ++i)
    {
        EXPECT_EQ(expected[i], actual[i]) << GCUDA_INFO(i, file, line);
    }
}
template <>
void expectArrayEq<unsigned long>(
        const unsigned long* expected,
        const unsigned long* actual,
        const int size,
        const char* const file,
        const int line)
{
    for (int i = 0; i < size; ++i)
    {
        EXPECT_EQ(expected[i], actual[i]) << GCUDA_INFO(i, file, line);
    }
}
template <>
void expectArrayEq<char2>(
        const char2* expected,
        const char2* actual,
        const int size,
        const char* const file,
        const int line)
{
    for (int i = 0; i < size; ++i)
    {
        EXPECT_EQ(expected[i].x, actual[i].x) << GCUDA_INFO(i, file, line);
        EXPECT_EQ(expected[i].y, actual[i].y) << GCUDA_INFO(i, file, line);
    }
}
template <>
void expectArrayEq<uchar2>(
        const uchar2* expected,
        const uchar2* actual,
        const int size,
        const char* const file,
        const int line)
{
    for (int i = 0; i < size; ++i)
    {
        EXPECT_EQ(expected[i].x, actual[i].x) << GCUDA_INFO(i, file, line);
        EXPECT_EQ(expected[i].y, actual[i].y) << GCUDA_INFO(i, file, line);
    }
}
template <>
void expectArrayEq<short2>(
        const short2* expected,
        const short2* actual,
        const int size,
        const char* const file,
        const int line)
{
    for (int i = 0; i < size; ++i)
    {
        EXPECT_EQ(expected[i].x, actual[i].x) << GCUDA_INFO(i, file, line);
        EXPECT_EQ(expected[i].y, actual[i].y) << GCUDA_INFO(i, file, line);
    }
}
template <>
void expectArrayEq<ushort2>(
        const ushort2* expected,
        const ushort2* actual,
        const int size,
        const char* const file,
        const int line)
{
    for (int i = 0; i < size; ++i)
    {
        EXPECT_EQ(expected[i].x, actual[i].x) << GCUDA_INFO(i, file, line);
        EXPECT_EQ(expected[i].y, actual[i].y) << GCUDA_INFO(i, file, line);
    }
}
template <>
void expectArrayEq<int2>(
        const int2* expected,
        const int2* actual,
        const int size,
        const char* const file,
        const int line)
{
    for (int i = 0; i < size; ++i)
    {
        EXPECT_EQ(expected[i].x, actual[i].x) << GCUDA_INFO(i, file, line);
        EXPECT_EQ(expected[i].y, actual[i].y) << GCUDA_INFO(i, file, line);
    }
}
template <>
void expectArrayEq<uint2>(
        const uint2* expected,
        const uint2* actual,
        const int size,
        const char* const file,
        const int line)
{
    for (int i = 0; i < size; ++i)
    {
        EXPECT_EQ(expected[i].x, actual[i].x) << GCUDA_INFO(i, file, line);
        EXPECT_EQ(expected[i].y, actual[i].y) << GCUDA_INFO(i, file, line);
    }
}
template <>
void expectArrayEq<long2>(
        const long2* expected,
        const long2* actual,
        const int size,
        const char* const file,
        const int line)
{
    for (int i = 0; i < size; ++i)
    {
        EXPECT_EQ(expected[i].x, actual[i].x) << GCUDA_INFO(i, file, line);
        EXPECT_EQ(expected[i].y, actual[i].y) << GCUDA_INFO(i, file, line);
    }
}
template <>
void expectArrayEq<ulong2>(
        const ulong2* expected,
        const ulong2* actual,
        const int size,
        const char* const file,
        const int line)
{
    for (int i = 0; i < size; ++i)
    {
        EXPECT_EQ(expected[i].x, actual[i].x) << GCUDA_INFO(i, file, line);
        EXPECT_EQ(expected[i].y, actual[i].y) << GCUDA_INFO(i, file, line);
    }
}
template <>
void expectArrayEq<char3>(
        const char3* expected,
        const char3* actual,
        const int size,
        const char* const file,
        const int line)
{
    for (int i = 0; i < size; ++i)
    {
        EXPECT_EQ(expected[i].x, actual[i].x) << GCUDA_INFO(i, file, line);
        EXPECT_EQ(expected[i].y, actual[i].y) << GCUDA_INFO(i, file, line);
        EXPECT_EQ(expected[i].z, actual[i].z) << GCUDA_INFO(i, file, line);
    }
}
template <>
void expectArrayEq<uchar3>(
        const uchar3* expected,
        const uchar3* actual,
        const int size,
        const char* const file,
        const int line)
{
    for (int i = 0; i < size; ++i)
    {
        EXPECT_EQ(expected[i].x, actual[i].x) << GCUDA_INFO(i, file, line);
        EXPECT_EQ(expected[i].y, actual[i].y) << GCUDA_INFO(i, file, line);
        EXPECT_EQ(expected[i].z, actual[i].z) << GCUDA_INFO(i, file, line);
    }
}
template <>
void expectArrayEq<short3>(
        const short3* expected,
        const short3* actual,
        const int size,
        const char* const file,
        const int line)
{
    for (int i = 0; i < size; ++i)
    {
        EXPECT_EQ(expected[i].x, actual[i].x) << GCUDA_INFO(i, file, line);
        EXPECT_EQ(expected[i].y, actual[i].y) << GCUDA_INFO(i, file, line);
        EXPECT_EQ(expected[i].z, actual[i].z) << GCUDA_INFO(i, file, line);
    }
}
template <>
void expectArrayEq<ushort3>(
        const ushort3* expected,
        const ushort3* actual,
        const int size,
        const char* const file,
        const int line)
{
    for (int i = 0; i < size; ++i)
    {
        EXPECT_EQ(expected[i].x, actual[i].x) << GCUDA_INFO(i, file, line);
        EXPECT_EQ(expected[i].y, actual[i].y) << GCUDA_INFO(i, file, line);
        EXPECT_EQ(expected[i].z, actual[i].z) << GCUDA_INFO(i, file, line);
    }
}
template <>
void expectArrayEq<int3>(
        const int3* expected,
        const int3* actual,
        const int size,
        const char* const file,
        const int line)
{
    for (int i = 0; i < size; ++i)
    {
        EXPECT_EQ(expected[i].x, actual[i].x) << GCUDA_INFO(i, file, line);
        EXPECT_EQ(expected[i].y, actual[i].y) << GCUDA_INFO(i, file, line);
        EXPECT_EQ(expected[i].z, actual[i].z) << GCUDA_INFO(i, file, line);
    }
}
template <>
void expectArrayEq<uint3>(
        const uint3* expected,
        const uint3* actual,
        const int size,
        const char* const file,
        const int line)
{
    for (int i = 0; i < size; ++i)
    {
        EXPECT_EQ(expected[i].x, actual[i].x) << GCUDA_INFO(i, file, line);
        EXPECT_EQ(expected[i].y, actual[i].y) << GCUDA_INFO(i, file, line);
        EXPECT_EQ(expected[i].z, actual[i].z) << GCUDA_INFO(i, file, line);
    }
}
template <>
void expectArrayEq<long3>(
        const long3* expected,
        const long3* actual,
        const int size,
        const char* const file,
        const int line)
{
    for (int i = 0; i < size; ++i)
    {
        EXPECT_EQ(expected[i].x, actual[i].x) << GCUDA_INFO(i, file, line);
        EXPECT_EQ(expected[i].y, actual[i].y) << GCUDA_INFO(i, file, line);
        EXPECT_EQ(expected[i].z, actual[i].z) << GCUDA_INFO(i, file, line);
    }
}
template <>
void expectArrayEq<ulong3>(
        const ulong3* expected,
        const ulong3* actual,
        const int size,
        const char* const file,
        const int line)
{
    for (int i = 0; i < size; ++i)
    {
        EXPECT_EQ(expected[i].x, actual[i].x) << GCUDA_INFO(i, file, line);
        EXPECT_EQ(expected[i].y, actual[i].y) << GCUDA_INFO(i, file, line);
        EXPECT_EQ(expected[i].z, actual[i].z) << GCUDA_INFO(i, file, line);
    }
}
template <>
void expectArrayEq<char4>(
        const char4* expected,
        const char4* actual,
        const int size,
        const char* const file,
        const int line)
{
    for (int i = 0; i < size; ++i)
    {
        EXPECT_EQ(expected[i].x, actual[i].x) << GCUDA_INFO(i, file, line);
        EXPECT_EQ(expected[i].y, actual[i].y) << GCUDA_INFO(i, file, line);
        EXPECT_EQ(expected[i].z, actual[i].z) << GCUDA_INFO(i, file, line);
        EXPECT_EQ(expected[i].w, actual[i].w) << GCUDA_INFO(i, file, line);
    }
}
template <>
void expectArrayEq<uchar4>(
        const uchar4* expected,
        const uchar4* actual,
        const int size,
        const char* const file,
        const int line)
{
    for (int i = 0; i < size; ++i)
    {
        EXPECT_EQ(expected[i].x, actual[i].x) << GCUDA_INFO(i, file, line);
        EXPECT_EQ(expected[i].y, actual[i].y) << GCUDA_INFO(i, file, line);
        EXPECT_EQ(expected[i].z, actual[i].z) << GCUDA_INFO(i, file, line);
        EXPECT_EQ(expected[i].w, actual[i].w) << GCUDA_INFO(i, file, line);
    }
}
template <>
void expectArrayEq<short4>(
        const short4* expected,
        const short4* actual,
        const int size,
        const char* const file,
        const int line)
{
    for (int i = 0; i < size; ++i)
    {
        EXPECT_EQ(expected[i].x, actual[i].x) << GCUDA_INFO(i, file, line);
        EXPECT_EQ(expected[i].y, actual[i].y) << GCUDA_INFO(i, file, line);
        EXPECT_EQ(expected[i].z, actual[i].z) << GCUDA_INFO(i, file, line);
        EXPECT_EQ(expected[i].w, actual[i].w) << GCUDA_INFO(i, file, line);
    }
}
template <>
void expectArrayEq<ushort4>(
        const ushort4* expected,
        const ushort4* actual,
        const int size,
        const char* const file,
        const int line)
{
    for (int i = 0; i < size; ++i)
    {
        EXPECT_EQ(expected[i].x, actual[i].x) << GCUDA_INFO(i, file, line);
        EXPECT_EQ(expected[i].y, actual[i].y) << GCUDA_INFO(i, file, line);
        EXPECT_EQ(expected[i].z, actual[i].z) << GCUDA_INFO(i, file, line);
        EXPECT_EQ(expected[i].w, actual[i].w) << GCUDA_INFO(i, file, line);
    }
}
template <>
void expectArrayEq<int4>(
        const int4* expected,
        const int4* actual,
        const int size,
        const char* const file,
        const int line)
{
    for (int i = 0; i < size; ++i)
    {
        EXPECT_EQ(expected[i].x, actual[i].x) << GCUDA_INFO(i, file, line);
        EXPECT_EQ(expected[i].y, actual[i].y) << GCUDA_INFO(i, file, line);
        EXPECT_EQ(expected[i].z, actual[i].z) << GCUDA_INFO(i, file, line);
        EXPECT_EQ(expected[i].w, actual[i].w) << GCUDA_INFO(i, file, line);
    }
}
template <>
void expectArrayEq<uint4>(
        const uint4* expected,
        const uint4* actual,
        const int size,
        const char* const file,
        const int line)
{
    for (int i = 0; i < size; ++i)
    {
        EXPECT_EQ(expected[i].x, actual[i].x) << GCUDA_INFO(i, file, line);
        EXPECT_EQ(expected[i].y, actual[i].y) << GCUDA_INFO(i, file, line);
        EXPECT_EQ(expected[i].z, actual[i].z) << GCUDA_INFO(i, file, line);
        EXPECT_EQ(expected[i].w, actual[i].w) << GCUDA_INFO(i, file, line);
    }
}
template <>
void expectArrayEq<long4>(
        const long4* expected,
        const long4* actual,
        const int size,
        const char* const file,
        const int line)
{
    for (int i = 0; i < size; ++i)
    {
        EXPECT_EQ(expected[i].x, actual[i].x) << GCUDA_INFO(i, file, line);
        EXPECT_EQ(expected[i].y, actual[i].y) << GCUDA_INFO(i, file, line);
        EXPECT_EQ(expected[i].z, actual[i].z) << GCUDA_INFO(i, file, line);
        EXPECT_EQ(expected[i].w, actual[i].w) << GCUDA_INFO(i, file, line);
    }
}
template <>
void expectArrayEq<ulong4>(
        const ulong4* expected,
        const ulong4* actual,
        const int size,
        const char* const file,
        const int line)
{
    for (int i = 0; i < size; ++i)
    {
        EXPECT_EQ(expected[i].x, actual[i].x) << GCUDA_INFO(i, file, line);
        EXPECT_EQ(expected[i].y, actual[i].y) << GCUDA_INFO(i, file, line);
        EXPECT_EQ(expected[i].z, actual[i].z) << GCUDA_INFO(i, file, line);
        EXPECT_EQ(expected[i].w, actual[i].w) << GCUDA_INFO(i, file, line);
    }
}
template <>
void expectArrayEq<float>(
        const float* expected,
        const float* actual,
        const int size,
        const char* const file,
        const int line)
{
    for (int i = 0; i < size; ++i)
    {
        EXPECT_FLOAT_EQ(expected[i], actual[i]) << GCUDA_INFO(i, file, line);
    }
}
template <>
void expectArrayEq<double>(
        const double* expected,
        const double* actual,
        const int size,
        const char* const file,
        const int line)
{
    for (int i = 0; i < size; ++i)
    {
        EXPECT_DOUBLE_EQ(expected[i], actual[i]) << GCUDA_INFO(i, file, line);
    }
}
template <>
void expectArrayEq<float2>(
        const float2* expected,
        const float2* actual,
        const int size,
        const char* const file,
        const int line)
{
    for (int i = 0; i < size; ++i)
    {
        EXPECT_FLOAT_EQ(expected[i].x, actual[i].x) << GCUDA_INFO(i, file, line);
        EXPECT_FLOAT_EQ(expected[i].y, actual[i].y) << GCUDA_INFO(i, file, line);
    }
}
template <>
void expectArrayEq<float3>(
        const float3* expected,
        const float3* actual,
        const int size,
        const char* const file,
        const int line)
{
    for (int i = 0; i < size; ++i)
    {
        EXPECT_FLOAT_EQ(expected[i].x, actual[i].x) << GCUDA_INFO(i, file, line);
        EXPECT_FLOAT_EQ(expected[i].y, actual[i].y) << GCUDA_INFO(i, file, line);
        EXPECT_FLOAT_EQ(expected[i].z, actual[i].z) << GCUDA_INFO(i, file, line);
    }
}
template <>
void expectArrayEq<float4>(
        const float4* expected,
        const float4* actual,
        const int size,
        const char* const file,
        const int line)
{
    for (int i = 0; i < size; ++i)
    {
        EXPECT_FLOAT_EQ(expected[i].x, actual[i].x) << GCUDA_INFO(i, file, line);
        EXPECT_FLOAT_EQ(expected[i].y, actual[i].y) << GCUDA_INFO(i, file, line);
        EXPECT_FLOAT_EQ(expected[i].z, actual[i].z) << GCUDA_INFO(i, file, line);
        EXPECT_FLOAT_EQ(expected[i].w, actual[i].w) << GCUDA_INFO(i, file, line);
    }
}
template <>
void expectArrayEq<double2>(
        const double2* expected,
        const double2* actual,
        const int size,
        const char* const file,
        const int line)
{
    for (int i = 0; i < size; ++i)
    {
        EXPECT_DOUBLE_EQ(expected[i].x, actual[i].x) << GCUDA_INFO(i, file, line);
        EXPECT_DOUBLE_EQ(expected[i].y, actual[i].y) << GCUDA_INFO(i, file, line);
    }
}
template <>
void expectArrayEq<double3>(
        const double3* expected,
        const double3* actual,
        const int size,
        const char* const file,
        const int line)
{
    for (int i = 0; i < size; ++i)
    {
        EXPECT_DOUBLE_EQ(expected[i].x, actual[i].x) << GCUDA_INFO(i, file, line);
        EXPECT_DOUBLE_EQ(expected[i].y, actual[i].y) << GCUDA_INFO(i, file, line);
        EXPECT_DOUBLE_EQ(expected[i].z, actual[i].z) << GCUDA_INFO(i, file, line);
    }
}
template <>
void expectArrayEq<double4>(
        const double4* expected,
        const double4* actual,
        const int size,
        const char* const file,
        const int line)
{
    for (int i = 0; i < size; ++i)
    {
        EXPECT_DOUBLE_EQ(expected[i].x, actual[i].x) << GCUDA_INFO(i, file, line);
        EXPECT_DOUBLE_EQ(expected[i].y, actual[i].y) << GCUDA_INFO(i, file, line);
        EXPECT_DOUBLE_EQ(expected[i].z, actual[i].z) << GCUDA_INFO(i, file, line);
        EXPECT_DOUBLE_EQ(expected[i].w, actual[i].w) << GCUDA_INFO(i, file, line);
    }
}



/*******************************************
 *         EXPECT ARRAY NEAR
 *******************************************/
template <typename T>
void expectArrayNear(
        const T*  expected,
        const T*  actual,
        const int size,
        const double abs_error,
        const char* const file,
        const int line);



template <>
void expectArrayNear<float>(
        const float* expected,
        const float* actual,
        const int size,
        const double abs_error,
        const char* const file,
        const int line)
{
    for (int i = 0; i < size; ++i)
    {
        EXPECT_NEAR(expected[i], actual[i], abs_error) << GCUDA_INFO(i, file, line);
    }
}
template <>
void expectArrayNear<double>(
        const double* expected,
        const double* actual,
        const int size,
        const double abs_error,
        const char* const file,
        const int line)
{
    for (int i = 0; i < size; ++i)
    {
        EXPECT_NEAR(expected[i], actual[i], abs_error) << GCUDA_INFO(i, file, line);
    }
}
template <>
void expectArrayNear<float2>(
        const float2* expected,
        const float2* actual,
        const int size,
        const double abs_error,
        const char* const file,
        const int line)
{
    for (int i = 0; i < size; ++i)
    {
        EXPECT_NEAR(expected[i].x, actual[i].x, abs_error) << GCUDA_INFO(i, file, line);
        EXPECT_NEAR(expected[i].y, actual[i].y, abs_error) << GCUDA_INFO(i, file, line);
    }
}
template <>
void expectArrayNear<double2>(
        const double2* expected,
        const double2* actual,
        const int size,
        const double abs_error,
        const char* const file,
        const int line)
{
    for (int i = 0; i < size; ++i)
    {
        EXPECT_NEAR(expected[i].x, actual[i].x, abs_error) << GCUDA_INFO(i, file, line);
        EXPECT_NEAR(expected[i].y, actual[i].y, abs_error) << GCUDA_INFO(i, file, line);
    }
}
template <>
void expectArrayNear<float3>(
        const float3* expected,
        const float3* actual,
        const int size,
        const double abs_error,
        const char* const file,
        const int line)
{
    for (int i = 0; i < size; ++i)
    {
        EXPECT_NEAR(expected[i].x, actual[i].x, abs_error) << GCUDA_INFO(i, file, line);
        EXPECT_NEAR(expected[i].y, actual[i].y, abs_error) << GCUDA_INFO(i, file, line);
        EXPECT_NEAR(expected[i].z, actual[i].z, abs_error) << GCUDA_INFO(i, file, line);
    }
}
template <>
void expectArrayNear<double3>(
        const double3* expected,
        const double3* actual,
        const int size,
        const double abs_error,
        const char* const file,
        const int line)
{
    for (int i = 0; i < size; ++i)
    {
        EXPECT_NEAR(expected[i].x, actual[i].x, abs_error) << GCUDA_INFO(i, file, line);
        EXPECT_NEAR(expected[i].y, actual[i].y, abs_error) << GCUDA_INFO(i, file, line);
        EXPECT_NEAR(expected[i].z, actual[i].z, abs_error) << GCUDA_INFO(i, file, line);
    }
}

template <>
void expectArrayNear<float4>(
        const float4* expected,
        const float4* actual,
        const int size,
        const double abs_error,
        const char* const file,
        const int line)
{
    for (int i = 0; i < size; ++i)
    {
        EXPECT_NEAR(expected[i].x, actual[i].x, abs_error) << GCUDA_INFO(i, file, line);
        EXPECT_NEAR(expected[i].y, actual[i].y, abs_error) << GCUDA_INFO(i, file, line);
        EXPECT_NEAR(expected[i].z, actual[i].z, abs_error) << GCUDA_INFO(i, file, line);
        EXPECT_NEAR(expected[i].w, actual[i].w, abs_error) << GCUDA_INFO(i, file, line);
    }
}
template <>
void expectArrayNear<double4>(
        const double4* expected,
        const double4* actual,
        const int size,
        const double abs_error,
        const char* const file,
        const int line)
{
    for (int i = 0; i < size; ++i)
    {
        EXPECT_NEAR(expected[i].x, actual[i].x, abs_error) << GCUDA_INFO(i, file, line);
        EXPECT_NEAR(expected[i].y, actual[i].y, abs_error) << GCUDA_INFO(i, file, line);
        EXPECT_NEAR(expected[i].z, actual[i].z, abs_error) << GCUDA_INFO(i, file, line);
        EXPECT_NEAR(expected[i].w, actual[i].w, abs_error) << GCUDA_INFO(i, file, line);
    }
}


} // namespace internal
} // namespace gcuda

#endif /* DETAIL_INTERNAL_GCUDA_H_ */
