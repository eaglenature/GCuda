/*
 * <gcuda/detail/gcuda.h>
 *
 *  Created on: May 3, 2013
 *      Author: eaglenature@gmail.com
 */

#ifndef DETAIL_GCUDA_H_
#define DETAIL_GCUDA_H_

#include <gcuda/detail/internal/gcuda.h>


namespace gcuda
{

template <typename HostVector>
void assertHostVectorEq(
        const HostVector& expected,
        const HostVector& actual,
        const char* const file,
        const int line)
{
    typedef typename HostVector::value_type T;

    enum { ExpectedType = internal::isDoubleFloatingPointType<T>::value ?
            internal::DoubleFloatingPoint : (internal::isFloatingPointType<T>::value ?
                    internal::FloatingPoint : (internal::isIntegralType<T>::value ?
                            internal::Integral : internal::Unknown)) };
//    enum { ExpectedType = internal::isDoubleFloatingPointType<T>::value ?
//            internal::DoubleFloatingPoint : (internal::isFloatingPointType<T>::value ?
//                    internal::FloatingPoint : (internal::isIntegralType<T>::value ?
//                            internal::Integral : internal::Unknown)) };

    internal::assertHostVectorEq(expected, actual, file, line, internal::int2Type<ExpectedType>());
}


template <typename T>
void assertHostArrayEq(
        const T*  expected,
        const T*  actual,
        const int size,
        const char* const file,
        const int line)
{
    enum { ExpectedType = internal::isDoubleFloatingPointType<T>::value ?
            internal::DoubleFloatingPoint : (internal::isFloatingPointType<T>::value ?
                    internal::FloatingPoint : (internal::isIntegralType<T>::value ?
                            internal::Integral : internal::Unknown) ) };

//    enum { ExpectedType = internal::isFloatingPointType<T>::value ?
//            internal::FloatingPoint : (internal::isIntegralType<T>::value ?
//                    internal::Integral : internal::Unknown) };

    internal::assertHostArrayEq(expected, actual, size, file, line, internal::int2Type<ExpectedType>());
}


template <typename HostVector>
void expectHostVectorEq(
        const HostVector& expected,
        const HostVector& actual,
        const char* const file,
        const int line)
{
    typedef typename HostVector::value_type T;

    enum { ExpectedType = internal::isDoubleFloatingPointType<T>::value ?
            internal::DoubleFloatingPoint : (internal::isFloatingPointType<T>::value ?
                    internal::FloatingPoint : (internal::isIntegralType<T>::value ?
                            internal::Integral : internal::Unknown) ) };

//    enum { ExpectedType = internal::isFloatingPointType<T>::value ?
//            internal::FloatingPoint : (internal::isIntegralType<T>::value ?
//                    internal::Integral : internal::Unknown) };

    internal::expectHostVectorEq(expected, actual, file, line, internal::int2Type<ExpectedType>());
}


template <typename T>
void expectHostArrayEq(
        const T*  expected,
        const T*  actual,
        const int size,
        const char* const file,
        const int line)
{
    enum { ExpectedType = internal::isDoubleFloatingPointType<T>::value ?
            internal::DoubleFloatingPoint : (internal::isFloatingPointType<T>::value ?
                    internal::FloatingPoint : (internal::isIntegralType<T>::value ?
                            internal::Integral : internal::Unknown) ) };

//    enum { ExpectedType = internal::isFloatingPointType<T>::value ?
//            internal::FloatingPoint : (internal::isIntegralType<T>::value ?
//                    internal::Integral : internal::Unknown) };

    internal::expectHostArrayEq(expected, actual, size, file, line, internal::int2Type<ExpectedType>());
}


} // namespace gcuda

#endif /* DETAIL_GCUDA_H_ */
