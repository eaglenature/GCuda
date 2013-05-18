/*
 * <gcuda/detail/internal/utility/format.h>
 *
 *  Created on: May 18, 2013
 *      Author: eaglenature@gmail.com
 */

#ifndef DETAIL_INTERNAL_UTILITY_FORMAT_H_
#define DETAIL_INTERNAL_UTILITY_FORMAT_H_

#include <gcuda/detail/internal/type_traits/vector_component.h>
#include <gcuda/detail/internal/type_traits/vector_dimension.h>
#include <gcuda/detail/internal/utility/modify.h>
#include <sstream>

namespace gcuda
{
namespace detail
{

namespace
{

template <class RawType>
std::string formatImpl(const RawType& v, dimension0)
{
    typedef typename vector_component<RawType>::type ComponentTag;
    std::stringstream ss;
    ss << modify<ComponentTag> << v;
    return ss.str();
}

template <class RawType>
std::string formatImpl(const RawType& v, dimension1)
{
    typedef typename vector_component<RawType>::type ComponentTag;
    std::stringstream ss;
    ss << modify<ComponentTag> << v.x;
    return ss.str();
}

template <class RawType>
std::string formatImpl(const RawType& v, dimension2)
{
    typedef typename vector_component<RawType>::type ComponentTag;
    std::stringstream ss;
    ss << modify<ComponentTag> << v.x << " " << v.y;
    return ss.str();
}

template <class RawType>
std::string formatImpl(const RawType& v, dimension3)
{
    typedef typename vector_component<RawType>::type ComponentTag;
    std::stringstream ss;
    ss << modify<ComponentTag> << v.x << " " << v.y << " " << v.z;
    return ss.str();
}

template <class RawType>
std::string formatImpl(const RawType& v, dimension4)
{
    typedef typename vector_component<RawType>::type ComponentTag;
    std::stringstream ss;
    ss << modify<ComponentTag> << v.x << " " << v.y << " " << v.z << " " << v.w;
    return ss.str();
}

} // namespace


/**
 * Format interface
 */
template <class RawType>
std::string format(const RawType& v)
{
    typedef typename vector_dimension<RawType>::type DimensionTag;
    return formatImpl(v, DimensionTag());
}

} // namespace detail
} // namespace gcuda

#endif /* DETAIL_INTERNAL_UTILITY_FORMAT_H_ */