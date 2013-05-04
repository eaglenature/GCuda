/*
 * <gcuda/detail/gcuda.h>
 *
 *  Created on: May 3, 2013
 *      Author: eaglenature@gmail.com
 */

#ifndef DETAIL_GCUDA_H_
#define DETAIL_GCUDA_H_


#include <gcuda/detail/internal/gcuda.h>


namespace gcuda
{


template <typename HostVector>
void assertHostVectorEq(
        const HostVector& expected,
        const HostVector& actual,
        const char* const file,
        const int line)
{
    internal::assertArrayEq(expected.data(), actual.data(), actual.size(), file, line);
}



template <typename HostVector>
void assertHostVectorNear(
        const HostVector& expected,
        const HostVector& actual,
        const double abs_error,
        const char* const file,
        const int line)
{
    internal::assertArrayNear(expected.data(), actual.data(), actual.size(), abs_error, file, line);
}



template <typename T>
void assertHostArrayEq(
        const T*  expected,
        const T*  actual,
        const int size,
        const char* const file,
        const int line)
{
    internal::assertArrayEq(expected, actual, size, file, line);
}



template <typename T>
void assertHostArrayNear(
        const T*  expected,
        const T*  actual,
        const int size,
        const double abs_error,
        const char* const file,
        const int line)
{
    internal::assertArrayNear(expected, actual, size, abs_error, file, line);
}


template <typename HostVector>
void expectHostVectorEq(
        const HostVector& expected,
        const HostVector& actual,
        const char* const file,
        const int line)
{
    internal::expectArrayEq(expected.data(), actual.data(), actual.size(), file, line);
}


template <typename HostVector>
void expectHostVectorNear(
        const HostVector& expected,
        const HostVector& actual,
        const double abs_error,
        const char* const file,
        const int line)
{
    internal::expectArrayNear(expected.data(), actual.data(), actual.size(), abs_error, file, line);
}


template <typename T>
void expectHostArrayEq(
        const T*  expected,
        const T*  actual,
        const int size,
        const char* const file,
        const int line)
{
    internal::expectArrayEq(expected, actual, size, file, line);
}


template <typename T>
void expectHostArrayNear(
        const T*  expected,
        const T*  actual,
        const int size,
        const double abs_error,
        const char* const file,
        const int line)
{
    internal::expectArrayNear(expected, actual, size, abs_error, file, line);
}


template <typename HostVector,
          typename DeviceVector>
void assertDeviceVectorEq(
        const HostVector& expected,
        const DeviceVector& actual,
        const char* const file,
        const int line)
{
    HostVector actualCopy = actual;
    internal::assertArrayEq(expected.data(), actualCopy.data(), actualCopy.size(), file, line);
}


template <typename HostVector>
void assertDeviceArrayEq(
        const HostVector& expected,
        const typename HostVector::value_type* actual,
        const int size,
        const char* const file,
        const int line)
{
    typedef typename HostVector::value_type T;
    HostVector actualCopy(size);
    ASSERT_EQ(cudaMemcpy(actualCopy.data(), actual, sizeof(T) * size, cudaMemcpyDeviceToHost), cudaSuccess);
    internal::assertArrayEq(expected.data(), actualCopy.data(), actualCopy.size(), file, line);
}


template <typename T>
void assertDeviceArrayEq(
        const T* expected,
        const T* actual,
        const int size,
        const char* const file,
        const int line)
{
    T* const actualCopy = new T[size];
    ASSERT_TRUE(actualCopy != NULL);
    ASSERT_EQ(cudaMemcpy(actualCopy, actual, sizeof(T) * size, cudaMemcpyDeviceToHost), cudaSuccess);
    internal::assertArrayEq(expected, actualCopy, size, file, line);
    delete [] actualCopy;
}


template <typename HostVector,
          typename DeviceVector>
void assertDeviceVectorNear(
        const HostVector& expected,
        const DeviceVector& actual,
        const double abs_error,
        const char* const file,
        const int line)
{
    HostVector actualCopy = actual;
    internal::assertArrayNear(expected.data(), actualCopy.data(), actualCopy.size(), abs_error, file, line);
}


template <typename HostVector>
void assertDeviceArrayNear(
        const HostVector& expected,
        const typename HostVector::value_type* actual,
        const int size,
        const double abs_error,
        const char* const file,
        const int line)
{
    typedef typename HostVector::value_type T;
    HostVector actualCopy(size);
    ASSERT_EQ(cudaMemcpy(actualCopy.data(), actual, sizeof(T) * size, cudaMemcpyDeviceToHost), cudaSuccess);
    internal::assertArrayNear(expected.data(), actualCopy.data(), actualCopy.size(), abs_error, file, line);
}


template <typename T>
void assertDeviceArrayNear(
        const T* expected,
        const T* actual,
        const int size,
        const double abs_error,
        const char* const file,
        const int line)
{
    T* const actualCopy = new T[size];
    ASSERT_TRUE(actualCopy != NULL);
    ASSERT_EQ(cudaMemcpy(actualCopy, actual, sizeof(T) * size, cudaMemcpyDeviceToHost), cudaSuccess);
    internal::assertArrayNear(expected, actualCopy, size, abs_error, file, line);
    delete [] actualCopy;
}


template <typename HostVector,
          typename DeviceVector>
void expectDeviceVectorEq(
        const HostVector& expected,
        const DeviceVector& actual,
        const char* const file,
        const int line)
{
    HostVector actualCopy = actual;
    internal::expectArrayEq(expected.data(), actualCopy.data(), actualCopy.size(), file, line);
}


template <typename HostVector>
void expectDeviceArrayEq(
        const HostVector& expected,
        const typename HostVector::value_type* actual,
        const int size,
        const char* const file,
        const int line)
{
    typedef typename HostVector::value_type T;
    HostVector actualCopy(size);
    ASSERT_EQ(cudaMemcpy(actualCopy.data(), actual, sizeof(T) * size, cudaMemcpyDeviceToHost), cudaSuccess);
    internal::expectArrayEq(expected.data(), actualCopy.data(), actualCopy.size(), file, line);
}


template <typename T>
void expectDeviceArrayEq(
        const T* expected,
        const T* actual,
        const int size,
        const char* const file,
        const int line)
{
    T* const actualCopy = new T[size];
    ASSERT_TRUE(actualCopy != NULL);
    ASSERT_EQ(cudaMemcpy(actualCopy, actual, sizeof(T) * size, cudaMemcpyDeviceToHost), cudaSuccess);
    internal::expectArrayEq(expected, actualCopy, size, file, line);
    delete [] actualCopy;
}


template <typename HostVector,
          typename DeviceVector>
void expectDeviceVectorNear(
        const HostVector& expected,
        const DeviceVector& actual,
        const double abs_error,
        const char* const file,
        const int line)
{
    HostVector actualCopy = actual;
    internal::expectArrayNear(expected.data(), actualCopy.data(), actualCopy.size(), abs_error, file, line);
}


template <typename HostVector>
void expectDeviceArrayNear(
        const HostVector& expected,
        const typename HostVector::value_type* actual,
        const int size,
        const double abs_error,
        const char* const file,
        const int line)
{
    typedef typename HostVector::value_type T;
    HostVector actualCopy(size);
    ASSERT_EQ(cudaMemcpy(actualCopy.data(), actual, sizeof(T) * size, cudaMemcpyDeviceToHost), cudaSuccess);
    internal::expectArrayNear(expected.data(), actualCopy.data(), actualCopy.size(), abs_error, file, line);
}


template <typename T>
void expectDeviceArrayNear(
        const T* expected,
        const T* actual,
        const int size,
        const double abs_error,
        const char* const file,
        const int line)
{
    T* const actualCopy = new T[size];
    ASSERT_TRUE(actualCopy != NULL);
    ASSERT_EQ(cudaMemcpy(actualCopy, actual, sizeof(T) * size, cudaMemcpyDeviceToHost), cudaSuccess);
    internal::expectArrayNear(expected, actualCopy, size, abs_error, file, line);
    delete [] actualCopy;
}

} // namespace gcuda

#endif /* DETAIL_GCUDA_H_ */
