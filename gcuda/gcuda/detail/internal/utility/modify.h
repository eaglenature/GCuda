/*
 * <gcuda/detail/internal/utility/modify.h>
 *
 *  Created on: May 18, 2013
 *      Author: eaglenature@gmail.com
 */

#ifndef DETAIL_INTERNAL_UTILITY_MODIFY_H_
#define DETAIL_INTERNAL_UTILITY_MODIFY_H_

#include <gcuda/detail/internal/type_traits/vector_component.h>
#include <ostream>
#include <iomanip>
#include <limits>

namespace gcuda
{
namespace detail
{

template <class ComponentTag>
std::ostream& modify(std::ostream& os)
{
    os << std::right
       << std::setw(10);
    return os;
}

template <>
std::ostream& modify<single_prec_float_component_tag>(std::ostream& os)
{
    os << std::setprecision(std::numeric_limits<float>::digits10 + 2)
       << std::right
       << std::setw(16)
       << std::showpoint;
    return os;
}

template <>
std::ostream& modify<double_prec_float_component_tag>(std::ostream& os)
{
    os << std::setprecision(std::numeric_limits<double>::digits10 + 2)
       << std::right
       << std::setw(16)
       << std::showpoint;
    return os;
}

} // namespace detail
} // namespace gcuda

#endif /* DETAIL_INTERNAL_UTILITY_MODIFY_H_ */
