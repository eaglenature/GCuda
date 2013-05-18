/*
 * <gcuda/detail/internal/assert.h>
 *
 *  Created on: May , 2013
 *      Author: eaglenature@gmail.com
 */

#ifndef DETAIL_INTERNAL_ASSERT_H_
#define DETAIL_INTERNAL_ASSERT_H_

#include <gtest/gtest.h>
#include <gcuda/detail/internal/utility/compare.h>


namespace gcuda
{
namespace detail
{


template <typename T>
std::pair<size_t, bool> assertArrayEq(
        const T*     expected,
        const T*     actual,
        const size_t size)
{
    for (size_t n = 0; n < size; ++n)
    {
        if (!detail::compareEq(expected[n], actual[n]))
        {
            return std::make_pair(n, false);
        }
    }
    return std::make_pair(0, true);
}



template <typename T>
std::pair<size_t, bool> assertArrayNear(
        const T*     expected,
        const T*     actual,
        const size_t size,
        const double abs_error)
{
    for (size_t n = 0; n < size; ++n)
    {
        if (!detail::compareNear(expected[n], actual[n], abs_error))
        {
            return std::make_pair(n, false);
        }
    }
    return std::make_pair(0, true);
}


} // namespace detail
} // namespace gcuda

#endif /* DETAIL_INTERNAL_ASSERT_H_ */