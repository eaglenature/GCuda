/*
 * <gcuda/detail/internal/assert_host.h>
 *
 *  Created on: May 18, 2013
 *      Author: eaglenature@gmail.com
 */

#ifndef DETAIL_INTERNAL_ASSERT_HOST_H_
#define DETAIL_INTERNAL_ASSERT_HOST_H_

#include <gcuda/detail/internal/utility/message.h>
#include <gcuda/detail/internal/assert.h>

namespace gcuda
{
namespace detail
{

//-------------------------------------------------------------//
//                                                             //
//                 Host array assertions                       //
//                                                             //
//-------------------------------------------------------------//

template <class HostVector>
::testing::AssertionResult assertHostVectorEq(
        const char*       expected_expr,
        const char*       actual_expr,
        const HostVector& expected,
        const HostVector& actual)
{
    typedef typename HostVector::value_type RawType;
    typedef std::pair<std::size_t, bool> ResultPair;

    const ResultPair resultPair = detail::assertArrayEq(expected.data(), actual.data(), actual.size());
    if (resultPair.second)
    {
        return ::testing::AssertionSuccess();
    }
    return ::testing::AssertionFailure() << detail::message<RawType>(expected_expr, actual_expr, resultPair.first, expected.data(), actual.data());
}



template <class T>
::testing::AssertionResult assertHostArrayEq(
        const char*  expected_expr,
        const char*  actual_expr,
        const char*  count_expr,
        const T*     expected,
        const T*     actual,
        const std::size_t count)
{
    typedef std::pair<std::size_t, bool> ResultPair;

    const ResultPair resultPair = detail::assertArrayEq(expected, actual, count);
    if (resultPair.second)
    {
        return ::testing::AssertionSuccess();
    }
    return ::testing::AssertionFailure() << detail::message<T>(expected_expr, actual_expr, resultPair.first, expected, actual);
}



template <class HostVector>
::testing::AssertionResult assertHostVectorNear(
        const char*       expected_expr,
        const char*       actual_expr,
        const char*       abs_error_expr,
        const HostVector& expected,
        const HostVector& actual,
        const double      abs_error)
{
    typedef typename HostVector::value_type RawType;
    typedef std::pair<std::size_t, bool> ResultPair;

    const ResultPair resultPair = detail::assertArrayNear(expected.data(), actual.data(), actual.size(), abs_error);
    if (resultPair.second)
    {
        return ::testing::AssertionSuccess();
    }
    return ::testing::AssertionFailure() << detail::message<RawType>(expected_expr, actual_expr, resultPair.first, expected.data(), actual.data());
}



template <class T>
::testing::AssertionResult assertHostArrayNear(
        const char*  expected_expr,
        const char*  actual_expr,
        const char*  count_expr,
        const char*  abs_error_expr,
        const T*     expected,
        const T*     actual,
        const std::size_t count,
        const double abs_error)
{
    typedef std::pair<std::size_t, bool> ResultPair;

    const ResultPair resultPair = detail::assertArrayNear(expected, actual, count, abs_error);
    if (resultPair.second)
    {
        return ::testing::AssertionSuccess();
    }
    return ::testing::AssertionFailure() << detail::message<T>(expected_expr, actual_expr, resultPair.first, expected, actual);
}

} // namespace detail
} // namespace gcuda

#endif /* DETAIL_INTERNAL_ASSERT_HOST_H_ */
