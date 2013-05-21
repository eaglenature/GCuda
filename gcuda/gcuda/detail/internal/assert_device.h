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
    typedef std::pair<std::size_t, bool> ResultPair;
    typedef typename HostVector::value_type T;

    HostVector actualCopy = actual;

    const ResultPair resultPair = detail::assertArrayEq(expected.data(), actualCopy.data(), actual.size());
    if (resultPair.second)
    {
        return ::testing::AssertionSuccess();
    }
    return ::testing::AssertionFailure() << detail::message<T>(expected_expr, actual_expr, resultPair.first, expected.data(), actualCopy.data());
}



template <class HostVector>
::testing::AssertionResult assertDeviceArrayEq(
        const char*       expected_expr,
        const char*       actual_expr,
        const char*       count_expr,
        const HostVector& expected,
        const typename HostVector::value_type* actual,
        const std::size_t count)
{
    typedef typename HostVector::value_type RawType;
    typedef std::pair<std::size_t, bool> ResultPair;

    HostVector actualCopy(count);
    EXPECT_EQ(cudaMemcpy(actualCopy.data(), actual, sizeof(RawType) * count, cudaMemcpyDeviceToHost), cudaSuccess);

    const ResultPair resultPair = detail::assertArrayEq(expected.data(), actualCopy.data(), count);
    if (resultPair.second)
    {
        return ::testing::AssertionSuccess();
    }
    return ::testing::AssertionFailure() << detail::message<RawType>(expected_expr, actual_expr, resultPair.first, expected.data(), actualCopy.data());
}



template <class T>
::testing::AssertionResult assertDeviceArrayEq(
        const char*  expected_expr,
        const char*  actual_expr,
        const char*  count_expr,
        const T*     expected,
        const T*     actual,
        const std::size_t count)
{
    typedef std::pair<std::size_t, bool> ResultPair;

    std::vector<T> actualCopy(count);
    EXPECT_EQ(cudaMemcpy(actualCopy.data(), actual, sizeof(T) * count, cudaMemcpyDeviceToHost), cudaSuccess);

    const ResultPair resultPair = detail::assertArrayEq(expected, actualCopy.data(), count);
    if (resultPair.second)
    {
        return ::testing::AssertionSuccess();
    }
    return ::testing::AssertionFailure() << detail::message<T>(expected_expr, actual_expr, resultPair.first, expected, actualCopy.data());
}



template <class HostVector,
          class DeviceVector>
::testing::AssertionResult assertDeviceVectorNear(
        const char*         expected_expr,
        const char*         actual_expr,
        const char*         abs_error_expr,
        const HostVector&   expected,
        const DeviceVector& actual,
        const double        abs_error)
{
    typedef typename HostVector::value_type RawType;
    typedef std::pair<std::size_t, bool> ResultPair;

    HostVector actualCopy = actual;

    const ResultPair resultPair = detail::assertArrayNear(expected.data(), actualCopy.data(), actual.size(), abs_error);
    if (resultPair.second)
    {
        return ::testing::AssertionSuccess();
    }
    return ::testing::AssertionFailure() << detail::message<RawType>(expected_expr, actual_expr, resultPair.first, expected.data(), actualCopy.data());
}



template <class HostVector>
::testing::AssertionResult assertDeviceArrayNear(
        const char*       expected_expr,
        const char*       actual_expr,
        const char*       count_expr,
        const char*       abs_error_expr,
        const HostVector& expected,
        const typename HostVector::value_type* actual,
        const std::size_t count,
        const double      abs_error)
{
    typedef typename HostVector::value_type RawType;
    typedef std::pair<std::size_t, bool> ResultPair;

    HostVector actualCopy(count);
    EXPECT_EQ(cudaMemcpy(actualCopy.data(), actual, sizeof(RawType) * count, cudaMemcpyDeviceToHost), cudaSuccess);

    const ResultPair resultPair = detail::assertArrayNear(expected.data(), actualCopy.data(), count, abs_error);
    if (resultPair.second)
    {
        return ::testing::AssertionSuccess();
    }
    return ::testing::AssertionFailure() << detail::message<RawType>(expected_expr, actual_expr, resultPair.first, expected.data(), actualCopy.data());
}



template <class T>
::testing::AssertionResult assertDeviceArrayNear(
        const char*  expected_expr,
        const char*  actual_expr,
        const char*  count_expr,
        const char*  abs_error_expr,
        const T*     expected,
        const T*     actual,
        const std::size_t count,
        const double abs_error)
{
    typedef std::pair<size_t, bool> ResultPair;

    std::vector<T> actualCopy(count);
    EXPECT_EQ(cudaMemcpy(actualCopy.data(), actual, sizeof(T) * count, cudaMemcpyDeviceToHost), cudaSuccess);

    const ResultPair resultPair = detail::assertArrayNear(expected, actualCopy.data(), count, abs_error);
    if (resultPair.second)
    {
        return ::testing::AssertionSuccess();
    }
    return ::testing::AssertionFailure() << detail::message<T>(expected_expr, actual_expr, resultPair.first, expected, actualCopy.data());
}

} // namespace detail
} // namespace gcuda

#endif /* DETAIL_INTERNAL_ASSERT_DEVICE_H_ */
