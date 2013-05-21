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


template <class T>
std::pair<std::size_t, bool> assertArrayEq(
        const T*     expected,
        const T*     actual,
        const std::size_t count)
{
    for (std::size_t n = 0; n < count; ++n)
    {
        if (!detail::compareEq(expected[n], actual[n]))
        {
            return std::make_pair(n, false);
        }
    }
    return std::make_pair(0, true);
}



template <class T>
std::pair<std::size_t, bool> assertArrayNear(
        const T*     expected,
        const T*     actual,
        const std::size_t count,
        const double abs_error)
{
    for (std::size_t n = 0; n < count; ++n)
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
