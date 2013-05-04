/*
 * <gcuda/detail/internal/traits.h>
 *
 *  Created on: May 4, 2013
 *      Author: eaglenature@gmail.com
 */

#ifndef DETAIL_INTERNAL_TRAITS_H_
#define DETAIL_INTERNAL_TRAITS_H_


namespace gcuda
{
namespace internal
{

/**
 * If T is integral type (char, unsigned char, short, ushort, int, uint, long, ulong)
 * provide member constant value equal true. For any other type value is false.
 */
template <typename T>
struct isIntegralType
{ static const bool value = false; };

template <>
struct isIntegralType<char>
{ static const bool value = true; };

template <>
struct isIntegralType<unsigned char>
{ static const bool value = true; };

template <>
struct isIntegralType<short>
{ static const bool value = true; };

template <>
struct isIntegralType<unsigned short>
{ static const bool value = true; };

template <>
struct isIntegralType<int>
{ static const bool value = true; };

template <>
struct isIntegralType<unsigned int>
{ static const bool value = true; };

template <>
struct isIntegralType<long>
{ static const bool value = true; };

template <>
struct isIntegralType<unsigned long>
{ static const bool value = true; };


/**
 * If T is floating-point type (float, double)
 * provide member constant value equal true. For any other type value is false.
 */
template <typename T>
struct isFloatingPointType
{ static const bool value = false; };

template <>
struct isFloatingPointType<float>
{ static const bool value = true; };

//template <>
//struct isFloatingPointType<double>
//{ static const bool value = true; };


/**
 * If T is double floating-point type (double)
 * provide member constant value equal true. For any other type value is false.
 */
template <typename T>
struct isDoubleFloatingPointType
{ static const bool value = false; };

template <>
struct isDoubleFloatingPointType<double>
{ static const bool value = true; };


/**
 * Integral value to type map
 */
template <int Value>
struct int2Type
{ static const int value = Value; };


} // namespace internal
} // namespace gcuda

#endif /* GCUDA_DETAIL_INTERNAL_TRAITS_H_ */
