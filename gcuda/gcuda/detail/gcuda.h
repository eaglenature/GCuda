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

/**
 * Assertions
 */
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

    internal::assertHostVectorEq(expected, actual, file, line, internal::int2Type<ExpectedType>());
}

template <typename HostVector>
void assertHostVectorNear(
        const HostVector& expected,
        const HostVector& actual,
        const double abs_error,
        const char* const file,
        const int line)
{
    typedef typename HostVector::value_type T;

    enum { ExpectedType = (internal::isDoubleFloatingPointType<T>::value || internal::isFloatingPointType<T>::value) ?
                    internal::FloatingPoint : internal::Unknown };

    internal::assertHostVectorNear(expected, actual, abs_error, file, line, internal::int2Type<ExpectedType>());
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

    internal::assertHostArrayEq(expected, actual, size, file, line, internal::int2Type<ExpectedType>());
}

template <typename T>
void assertHostArrayNear(
        const T*  expected,
        const T*  actual,
        const int size,
        const double abs_error,
        const char* const file,
        const int line)
{
    enum { ExpectedType = (internal::isDoubleFloatingPointType<T>::value || internal::isFloatingPointType<T>::value) ?
                    internal::FloatingPoint : internal::Unknown };

    internal::assertHostArrayNear(expected, actual, size, abs_error, file, line, internal::int2Type<ExpectedType>());
}


/**
 * Expectations
 */
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

    internal::expectHostVectorEq(expected, actual, file, line, internal::int2Type<ExpectedType>());
}

template <typename HostVector>
void expectHostVectorNear(
        const HostVector& expected,
        const HostVector& actual,
        const double abs_error,
        const char* const file,
        const int line)
{
    typedef typename HostVector::value_type T;

    enum { ExpectedType = (internal::isDoubleFloatingPointType<T>::value || internal::isFloatingPointType<T>::value) ?
                    internal::FloatingPoint : internal::Unknown };

    internal::expectHostVectorNear(expected, actual, abs_error, file, line, internal::int2Type<ExpectedType>());
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

    internal::expectHostArrayEq(expected, actual, size, file, line, internal::int2Type<ExpectedType>());
}

template <typename T>
void expectHostArrayNear(
        const T*  expected,
        const T*  actual,
        const int size,
        const double abs_error,
        const char* const file,
        const int line)
{
    enum { ExpectedType = (internal::isDoubleFloatingPointType<T>::value || internal::isFloatingPointType<T>::value) ?
                    internal::FloatingPoint : internal::Unknown };

    internal::expectHostArrayNear(expected, actual, size, abs_error, file, line, internal::int2Type<ExpectedType>());
}


} // namespace gcuda

#endif /* DETAIL_GCUDA_H_ */
