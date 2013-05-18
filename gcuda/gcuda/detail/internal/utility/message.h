/*
 * <gcuda/detail/internal/utility/message.h>
 *
 *  Created on: May 18, 2013
 *      Author: eaglenature@gmail.com
 */
#ifndef DETAIL_INTERNAL_UTILITY_MESSAGE_H_
#define DETAIL_INTERNAL_UTILITY_MESSAGE_H_

#include <gcuda/detail/internal/utility/format.h>

namespace gcuda
{
namespace detail
{

template <class T>
struct message
{
    const char*  expr1;
    const char*  expr2;
    const size_t index;
    const T*     expect;
    const T*     actual;
    message(const char* expr1, const char* expr2, size_t index, const T* expect, const T* actual)
        : expr1(expr1), expr2(expr2), index(index), expect(expect), actual(actual)
    {};
};

template <class T>
std::ostream& operator<<(std::ostream& os, const message<T>& msg)
{
    os << "Compare " << msg.expr1 << " and " << msg.expr2
       << "\nIndex :   [" << msg.index << "]"
       << "\nExpect:   [" << detail::format(msg.expect[msg.index]) << " ]"
       << "\nActual:   [" << detail::format(msg.actual[msg.index]) << " ]";
    return os;
}

} // namespace detail
} // namespace gcuda

#endif /* DETAIL_INTERNAL_UTILITY_MESSAGE_H_ */
