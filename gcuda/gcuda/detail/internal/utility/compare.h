/*
 * <gcuda/detail/internal/utility/compare.h>
 *
 *  Created on: May 18, 2013
 *      Author: eaglenature@gmail.com
 */

#ifndef DETAIL_INTERNAL_UTILITY_COMPARE_H_
#define DETAIL_INTERNAL_UTILITY_COMPARE_H_

#include <gcuda/detail/internal/type_traits/vector_component.h>
#include <gcuda/detail/internal/type_traits/vector_dimension.h>
#include <gcuda/detail/internal/utility/predicate.h>

namespace gcuda
{
namespace detail
{


namespace
{
template <class RawType>
bool compareEqImpl(const RawType& lhs, const RawType& rhs, dimension0)
{
    typedef typename vector_component_tag<RawType>::type ComponentTag;
    typedef typename vector_component_base<RawType>::type ComponentBase;
    return predicate<ComponentBase, ComponentTag>::is_equal(lhs, rhs);
}

template <class RawType>
bool compareEqImpl(const RawType& lhs, const RawType& rhs, dimension1)
{
    typedef typename vector_component_tag<RawType>::type ComponentTag;
    typedef typename vector_component_base<RawType>::type ComponentBase;
    return predicate<ComponentBase, ComponentTag>::is_equal(lhs.x, rhs.x);
}

template <class RawType>
bool compareEqImpl(const RawType& lhs, const RawType& rhs, dimension2)
{
    typedef typename vector_component_tag<RawType>::type ComponentTag;
    typedef typename vector_component_base<RawType>::type ComponentBase;
    return predicate<ComponentBase, ComponentTag>::is_equal(lhs.x, rhs.x) &&
           predicate<ComponentBase, ComponentTag>::is_equal(lhs.y, rhs.y);
}

template <class RawType>
bool compareEqImpl(const RawType& lhs, const RawType& rhs, dimension3)
{
    typedef typename vector_component_tag<RawType>::type ComponentTag;
    typedef typename vector_component_base<RawType>::type ComponentBase;
    return predicate<ComponentBase, ComponentTag>::is_equal(lhs.x, rhs.x) &&
           predicate<ComponentBase, ComponentTag>::is_equal(lhs.y, rhs.y) &&
           predicate<ComponentBase, ComponentTag>::is_equal(lhs.z, rhs.z);
}

template <class RawType>
bool compareEqImpl(const RawType& lhs, const RawType& rhs, dimension4)
{
    typedef typename vector_component_tag<RawType>::type ComponentTag;
    typedef typename vector_component_base<RawType>::type ComponentBase;
    return predicate<ComponentBase, ComponentTag>::is_equal(lhs.x, rhs.x) &&
           predicate<ComponentBase, ComponentTag>::is_equal(lhs.y, rhs.y) &&
           predicate<ComponentBase, ComponentTag>::is_equal(lhs.z, rhs.z) &&
           predicate<ComponentBase, ComponentTag>::is_equal(lhs.w, rhs.w);
}
} // namespace

/**
 * Compare equal interface
 */
template <class RawType>
bool compareEq(const RawType& lhs, const RawType& rhs)
{
    typedef typename vector_dimension<RawType>::type DimensionTag;
    return compareEqImpl(lhs, rhs, DimensionTag());
}




namespace
{
template <class RawType>
bool compareNearImpl(const RawType& lhs, const RawType& rhs, const double abs_error, dimension0)
{
    typedef typename vector_component_tag<RawType>::type ComponentTag;
    typedef typename vector_component_base<RawType>::type ComponentBase;
    return predicate<ComponentBase, ComponentTag>::is_near(lhs, rhs, abs_error);
}

template <class RawType>
bool compareNearImpl(const RawType& lhs, const RawType& rhs, const double abs_error, dimension1)
{
    typedef typename vector_component_tag<RawType>::type ComponentTag;
    typedef typename vector_component_base<RawType>::type ComponentBase;
    return predicate<ComponentBase, ComponentTag>::is_near(lhs.x, rhs.x, abs_error);
}

template <class RawType>
bool compareNearImpl(const RawType& lhs, const RawType& rhs, const double abs_error, dimension2)
{
    typedef typename vector_component_tag<RawType>::type ComponentTag;
    typedef typename vector_component_base<RawType>::type ComponentBase;
    return predicate<ComponentBase, ComponentTag>::is_near(lhs.x, rhs.x, abs_error) &&
           predicate<ComponentBase, ComponentTag>::is_near(lhs.y, rhs.y, abs_error);
}

template <class RawType>
bool compareNearImpl(const RawType& lhs, const RawType& rhs, const double abs_error, dimension3)
{
    typedef typename vector_component_tag<RawType>::type ComponentTag;
    typedef typename vector_component_base<RawType>::type ComponentBase;
    return predicate<ComponentBase, ComponentTag>::is_near(lhs.x, rhs.x, abs_error) &&
           predicate<ComponentBase, ComponentTag>::is_near(lhs.y, rhs.y, abs_error) &&
           predicate<ComponentBase, ComponentTag>::is_near(lhs.z, rhs.z, abs_error);
}

template <class RawType>
bool compareNearImpl(const RawType& lhs, const RawType& rhs, const double abs_error, dimension4)
{
    typedef typename vector_component_tag<RawType>::type ComponentTag;
    typedef typename vector_component_base<RawType>::type ComponentBase;
    return predicate<ComponentBase, ComponentTag>::is_near(lhs.x, rhs.x, abs_error) &&
           predicate<ComponentBase, ComponentTag>::is_near(lhs.y, rhs.y, abs_error) &&
           predicate<ComponentBase, ComponentTag>::is_near(lhs.z, rhs.z, abs_error) &&
           predicate<ComponentBase, ComponentTag>::is_near(lhs.w, rhs.w, abs_error);
}
} // namespace


/**
 * Compare near interface
 */
template <class RawType>
bool compareNear(const RawType& lhs, const RawType& rhs, const double abs_error)
{
    typedef typename vector_dimension<RawType>::type DimensionTag;
    return compareNearImpl(lhs, rhs, abs_error, DimensionTag());
}


} // namespace detail
} // namespace gcuda

#endif /* DETAIL_INTERNAL_UTILITY_COMPARE_H_ */
