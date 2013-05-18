/*
 * <gcuda/detail/internal/utility/predicate.h>
 *
 *  Created on: May 18, 2013
 *      Author: eaglenature@gmail.com
 */
#ifndef DETAIL_INTERNAL_UTILITY_PREDICATE_H_
#define DETAIL_INTERNAL_UTILITY_PREDICATE_H_

#include <gcuda/detail/internal/type_traits/vector_component.h>
#include <gtest/internal/gtest-internal.h>

namespace gcuda
{
namespace detail
{

template <class T, class ComponentTag>
struct predicate
{
    static bool is_equal(const T& lhs, const T& rhs)
    {
        return lhs == rhs;
    }
};

template <class T>
struct predicate<T, single_prec_float_component_tag>
{
    static bool is_equal(const T& lhs, const T& rhs)
    {
        const ::testing::internal::FloatingPoint<T> a(lhs), b(rhs);
        return a.AlmostEquals(b);
    }
    static bool is_near(const T& lhs, const T& rhs, double abs_error)
    {
        return std::fabs(lhs - rhs) < abs_error; // TODO correct solution
    }
};

template <class T>
struct predicate<T, double_prec_float_component_tag>
{
    static bool is_equal(const T& lhs, const T& rhs)
    {
        const ::testing::internal::FloatingPoint<T> a(lhs), b(rhs);
        return a.AlmostEquals(b);
    }
    static bool is_near(const T& lhs, const T& rhs, double abs_error)
    {
        return std::fabs(lhs - rhs) < abs_error; // TODO correct solution
    }
};

} // namespace detail
} // namespace gcuda

#endif /* DETAIL_INTERNAL_UTILITY_PREDICATE_H_ */
