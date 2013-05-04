/*
 * <gcuda/detail/internal/gcuda.h>
 *
 *  Created on: May 4, 2013
 *      Author: eaglenature@gmail.com
 */

#ifndef DETAIL_INTERNAL_GCUDA_H_
#define DETAIL_INTERNAL_GCUDA_H_

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include <gcuda/detail/internal/traits.h>

namespace gcuda
{
namespace internal
{


enum Expected { Integral, FloatingPoint, DoubleFloatingPoint, Unknown };


template <typename HostVector>
void assertHostVectorEq(
        const HostVector& expected,
        const HostVector& actual,
        const char* const file,
        const int line,
        int2Type<Integral>)
{
    for (int i = 0; i < actual.size(); ++i)
    {
        ASSERT_EQ(expected[i], actual[i]) << " Element: " << i << "\n" << file << ":" << line << '\n';
    }
}

template <typename HostVector>
void assertHostVectorEq(
        const HostVector& expected,
        const HostVector& actual,
        const char* const file,
        const int line,
        int2Type<FloatingPoint>)
{
    for (int i = 0; i < actual.size(); ++i)
    {
        ASSERT_FLOAT_EQ(expected[i], actual[i]) << " Element: " << i << "\n" << file << ":" << line << '\n';
    }
}

template <typename HostVector>
void assertHostVectorEq(
        const HostVector& expected,
        const HostVector& actual,
        const char* const file,
        const int line,
        int2Type<DoubleFloatingPoint>)
{
    for (int i = 0; i < actual.size(); ++i)
    {
        ASSERT_DOUBLE_EQ(expected[i], actual[i]) << " Element: " << i << "\n" << file << ":" << line << '\n';
    }
}


template <typename HostVector>
void assertHostVectorNear(
        const HostVector& expected,
        const HostVector& actual,
        const double abs_error,
        const char* const file,
        const int line,
        int2Type<FloatingPoint>)
{
    for (int i = 0; i < actual.size(); ++i)
    {
        ASSERT_NEAR(expected[i], actual[i], abs_error) << " Element: " << i << "\n" << file << ":" << line << '\n';
    }
}





template <typename T>
void assertHostArrayEq(
        const T*  expected,
        const T*  actual,
        const int size,
        const char* const file,
        const int line,
        int2Type<Integral>)
{
    for (int i = 0; i < size; ++i)
    {
        ASSERT_EQ(expected[i], actual[i]) << " Element: " << i << "\n" << file << ":" << line << '\n';
    }
}

template <typename T>
void assertHostArrayEq(
        const T*  expected,
        const T*  actual,
        const int size,
        const char* const file,
        const int line,
        int2Type<FloatingPoint>)
{
    for (int i = 0; i < size; ++i)
    {
        ASSERT_FLOAT_EQ(expected[i], actual[i]) << " Element: " << i << "\n" << file << ":" << line << '\n';
    }
}

template <typename T>
void assertHostArrayEq(
        const T*  expected,
        const T*  actual,
        const int size,
        const char* const file,
        const int line,
        int2Type<DoubleFloatingPoint>)
{
    for (int i = 0; i < size; ++i)
    {
        ASSERT_DOUBLE_EQ(expected[i], actual[i]) << " Element: " << i << "\n" << file << ":" << line << '\n';
    }
}


template <typename T>
void assertHostArrayNear(
        const T*  expected,
        const T*  actual,
        const int size,
        const double abs_error,
        const char* const file,
        const int line,
        int2Type<FloatingPoint>)
{
    for (int i = 0; i < size; ++i)
    {
        ASSERT_NEAR(expected[i], actual[i], abs_error) << " Element: " << i << "\n" << file << ":" << line << '\n';
    }
}




template <typename HostVector>
void expectHostVectorEq(
        const HostVector& expected,
        const HostVector& actual,
        const char* const file,
        const int line,
        int2Type<Integral>)
{
    for (size_t i = 0; i < actual.size(); ++i)
    {
        EXPECT_EQ(expected[i], actual[i]) << " Element: " << i << "\n" << file << ":" << line << '\n';
    }
}

template <typename HostVector>
void expectHostVectorEq(
        const HostVector& expected,
        const HostVector& actual,
        const char* const file,
        const int line,
        int2Type<FloatingPoint>)
{
    for (size_t i = 0; i < actual.size(); ++i)
    {
        EXPECT_FLOAT_EQ(expected[i], actual[i]) << " Element: " << i << "\n" << file << ":" << line << '\n';
    }
}

template <typename HostVector>
void expectHostVectorEq(
        const HostVector& expected,
        const HostVector& actual,
        const char* const file,
        const int line,
        int2Type<DoubleFloatingPoint>)
{
    for (size_t i = 0; i < actual.size(); ++i)
    {
        EXPECT_DOUBLE_EQ(expected[i], actual[i]) << " Element: " << i << "\n" << file << ":" << line << '\n';
    }
}


template <typename HostVector>
void expectHostVectorNear(
        const HostVector& expected,
        const HostVector& actual,
        const double abs_error,
        const char* const file,
        const int line,
        int2Type<FloatingPoint>)
{
    for (int i = 0; i < actual.size(); ++i)
    {
        EXPECT_NEAR(expected[i], actual[i], abs_error) << " Element: " << i << "\n" << file << ":" << line << '\n';
    }
}




template <typename T>
void expectHostArrayEq(
        const T*  expected,
        const T*  actual,
        const int size,
        const char* const file,
        const int line,
        int2Type<Integral>)
{
    for (int i = 0; i < size; ++i)
    {
        EXPECT_EQ(expected[i], actual[i]) << " Element: " << i << "\n" << file << ":" << line << '\n';
    }
}

template <typename T>
void expectHostArrayEq(
        const T*  expected,
        const T*  actual,
        const int size,
        const char* const file,
        const int line,
        int2Type<FloatingPoint>)
{
    for (int i = 0; i < size; ++i)
    {
        EXPECT_FLOAT_EQ(expected[i], actual[i]) << " Element: " << i << "\n" << file << ":" << line << '\n';
    }
}


template <typename T>
void expectHostArrayEq(
        const T*  expected,
        const T*  actual,
        const int size,
        const char* const file,
        const int line,
        int2Type<DoubleFloatingPoint>)
{
    for (int i = 0; i < size; ++i)
    {
        EXPECT_DOUBLE_EQ(expected[i], actual[i]) << " Element: " << i << "\n" << file << ":" << line << '\n';
    }
}

template <typename T>
void expectHostArrayNear(
        const T*  expected,
        const T*  actual,
        const int size,
        const double abs_error,
        const char* const file,
        const int line,
        int2Type<FloatingPoint>)
{
    for (int i = 0; i < size; ++i)
    {
        EXPECT_NEAR(expected[i], actual[i], abs_error) << " Element: " << i << "\n" << file << ":" << line << '\n';
    }
}



} // namespace internal
} // namespace gcuda

#endif /* DETAIL_INTERNAL_GCUDA_H_ */
