/*
 * <gcuda/detail/gcuda.h>
 *
 *  Created on: May 3, 2013
 *      Author: eaglenature@gmail.com
 */

#ifndef DETAIL_GCUDA_H_
#define DETAIL_GCUDA_H_

#include <gcuda/detail/internal/assert_device.h>
#include <gcuda/detail/internal/assert_host.h>


namespace gcuda
{

template <class HostVector>
::testing::AssertionResult assertHostVectorEq(
        const char*       expected_expr,
        const char*       actual_expr,
        const HostVector& expected,
        const HostVector& actual)
{
    return detail::assertHostVectorEq(expected_expr, actual_expr, expected, actual);
}

template <class T>
::testing::AssertionResult assertHostArrayEq(
        const char*  expected_expr,
        const char*  actual_expr,
        const char*  size_expr,
        const T*     expected,
        const T*     actual,
        const size_t size)
{
    return detail::assertHostArrayEq(expected_expr, actual_expr, size_expr, expected, actual, size);
}

template <class HostVector>
::testing::AssertionResult assertHostVectorNear(
        const char*       expected_expr,
        const char*       actual_expr,
        const HostVector& expected,
        const HostVector& actual,
        const double      abs_error)
{
    return detail::assertHostVectorNear(expected_expr, actual_expr, expected, actual, abs_error);
}

template <class T>
::testing::AssertionResult assertHostArrayNear(
        const char*  expected_expr,
        const char*  actual_expr,
        const char*  size_expr,
        const T*     expected,
        const T*     actual,
        const size_t size,
        const double abs_error)
{
    return detail::assertHostArrayNear(expected_expr, actual_expr, size_expr, expected, actual, size, abs_error);
}

template <class HostVector,
          class DeviceVector>
::testing::AssertionResult assertDeviceVectorEq(
        const char*         expected_expr,
        const char*         actual_expr,
        const HostVector&   expected,
        const DeviceVector& actual)
{
    return detail::assertDeviceVectorEq(expected_expr, actual_expr, expected, actual);
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
    return detail::assertDeviceArrayEq(expected_expr, actual_expr, size_expr, expected, actual, size);
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
    return detail::assertDeviceArrayEq(expected_expr, actual_expr, size_expr, expected, actual, size);
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
    return detail::assertDeviceVectorNear(expected_expr, actual_expr, abs_error_expr, expected, actual, abs_error);
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
    return detail::assertDeviceArrayNear(expected_expr, actual_expr, size_expr, abs_error_expr, expected, actual, size, abs_error);
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
    return detail::assertDeviceArrayNear(expected_expr, actual_expr, size_expr, abs_error_expr, expected, actual, size, abs_error);
}

} // namespace gcuda

#endif /* DETAIL_GCUDA_H_ */
