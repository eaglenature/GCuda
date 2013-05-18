/*
 * <gcuda/detail/internal/assert_device.h>
 *
 *  Created on: May 18, 2013
 *      Author: eaglenature@gmail.com
 */

#ifndef DETAIL_INTERNAL_ASSERT_DEVICE_H_
#define DETAIL_INTERNAL_ASSERT_DEVICE_H_

#include <gcuda/detail/internal/utility/message.h>
#include <gcuda/detail/internal/assert.h>

namespace gcuda
{
namespace detail
{

//-------------------------------------------------------------//
//                                                             //
//                 Device array assertions                     //
//                                                             //
//-------------------------------------------------------------//

template <class HostVector,
          class DeviceVector>
::testing::AssertionResult assertDeviceVectorEq(
        const char*         expected_expr,
        const char*         actual_expr,
        const HostVector&   expected,
        const DeviceVector& actual)
{
    typedef std::pair<size_t, bool> ResultPair;

    HostVector actualCopy = actual;

    const ResultPair resultPair = detail::assertArrayEq(expected.data(), actualCopy.data(), actual.size());
    if (resultPair.second)
    {
        return ::testing::AssertionSuccess();
    }
    return ::testing::AssertionFailure() << detail::message(expected_expr, actual_expr, resultPair.first, expected, actual);
}



template <class HostVector>
::testing::AssertionResult assertDeviceArrayEq(
        const char*       expected_expr,
        const char*       actual_expr,
        const char*       size_expr,
        const HostVector& expected,
        const typename HostVector::value_type* actual,
        const size_t      size)
{
    typedef typename HostVector::value_type RawType;
    typedef std::pair<size_t, bool> ResultPair;

    HostVector actualCopy(size);
    ASSERT_EQ(cudaMemcpy(actualCopy.data(), actual, sizeof(RawType) * size, cudaMemcpyDeviceToHost), cudaSuccess);

    const ResultPair resultPair = detail::assertArrayEq(expected, actual, size);
    if (resultPair.second)
    {
        return ::testing::AssertionSuccess();
    }
    return ::testing::AssertionFailure() << detail::message(expected_expr, actual_expr, resultPair.first, expected, actual);
}



template <class T>
::testing::AssertionResult assertDeviceArrayEq(
        const char*  expected_expr,
        const char*  actual_expr,
        const char*  size_expr,
        const T*     expected,
        const T*     actual,
        const size_t size)
{
    typedef std::pair<size_t, bool> ResultPair;

    std::vector<T> actualCopy(size);
    ASSERT_EQ(cudaMemcpy(actualCopy.data(), actual, sizeof(T) * size, cudaMemcpyDeviceToHost), cudaSuccess);

    const ResultPair resultPair = detail::assertArrayEq(expected, actualCopy.data(), size);
    if (resultPair.second)
    {
        return ::testing::AssertionSuccess();
    }
    return ::testing::AssertionFailure() << detail::message(expected_expr, actual_expr, resultPair.first, expected, actual);
}



template <class HostVector,
          class DeviceVector>
::testing::AssertionResult assertDeviceVectorNear(
        const char*       expected_expr,
        const char*       actual_expr,
        const char*       abs_error_expr,
        const HostVector& expected,
        const HostVector& actual,
        const double      abs_error)
{
    typedef std::pair<size_t, bool> ResultPair;

    HostVector actualCopy = actual;

    const ResultPair resultPair = detail::assertArrayNear(expected.data(), actualCopy.data(), actual.size(), abs_error);
    if (resultPair.second)
    {
        return ::testing::AssertionSuccess();
    }
    return ::testing::AssertionFailure() << detail::message(expected_expr, actual_expr, resultPair.first, expected, actual);
}



template <class HostVector>
::testing::AssertionResult assertDeviceArrayNear(
        const char*       expected_expr,
        const char*       actual_expr,
        const char*       size_expr,
        const char*       abs_error_expr,
        const HostVector& expected,
        const typename HostVector::value_type* actual,
        const size_t      size,
        const double      abs_error)
{
    typedef typename HostVector::value_type RawType;
    typedef std::pair<size_t, bool> ResultPair;

    HostVector actualCopy(size);
    ASSERT_EQ(cudaMemcpy(actualCopy.data(), actual, sizeof(RawType) * size, cudaMemcpyDeviceToHost), cudaSuccess);

    const ResultPair resultPair = detail::assertArrayNear(expected, actual, size, abs_error);
    if (resultPair.second)
    {
        return ::testing::AssertionSuccess();
    }
    return ::testing::AssertionFailure() << detail::message(expected_expr, actual_expr, resultPair.first, expected, actual);
}



template <class T>
::testing::AssertionResult assertDeviceArrayNear(
        const char*  expected_expr,
        const char*  actual_expr,
        const char*  size_expr,
        const char*  abs_error_expr,
        const T*     expected,
        const T*     actual,
        const size_t size,
        const double abs_error)
{
    typedef std::pair<size_t, bool> ResultPair;

    std::vector<T> actualCopy(size);
    ASSERT_EQ(cudaMemcpy(actualCopy.data(), actual, sizeof(T) * size, cudaMemcpyDeviceToHost), cudaSuccess);

    const ResultPair resultPair = detail::assertArrayNear(expected, actualCopy.data(), size, abs_error);
    if (resultPair.second)
    {
        return ::testing::AssertionSuccess();
    }
    return ::testing::AssertionFailure() << detail::message(expected_expr, actual_expr, resultPair.first, expected, actual);
}

} // namespace detail
} // namespace gcuda

#endif /* DETAIL_INTERNAL_ASSERT_DEVICE_H_ */
