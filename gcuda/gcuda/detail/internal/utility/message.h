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
    const char* a;
    const char* b;
    const size_t index;
    const T* expected;
    const T* actual;
    message(const char* a, const char* b, size_t index, const T* const expected, const T* const actual)
        : a(a), b(b), index(index), expected(expected), actual(actual)
    {}
};

template <class T>
std::ostream& operator<<(std::ostream& os, const message<T>& msg)
{
    os << msg.a << " and " << msg.b
       << " Element: "   << msg.index
       << "\nExpected: " << format(msg.expected[msg.index])
       << "\nActual:   " << format(msg.actual[msg.index]);
    return os;
}

} // namespace detail
} // namespace gcuda

#endif /* DETAIL_INTERNAL_UTILITY_MESSAGE_H_ */
