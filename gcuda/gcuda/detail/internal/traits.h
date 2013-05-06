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
 * If T is single floating-point type (float)
 * provide member constant value equal true. For any other type value is false.
 */
template <typename T>
struct isFloatingPointType
{ static const bool value = false; };

template <>
struct isFloatingPointType<float>
{ static const bool value = true; };


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



template <typename T>
struct lengthOf;

template <> struct lengthOf<char>    { static const int value = 1; };
template <> struct lengthOf<unsigned char> { static const int value = 1; };
template <> struct lengthOf<short>   { static const int value = 1; };
template <> struct lengthOf<ushort>  { static const int value = 1; };
template <> struct lengthOf<int>     { static const int value = 1; };
template <> struct lengthOf<uint>    { static const int value = 1; };
template <> struct lengthOf<long>    { static const int value = 1; };
template <> struct lengthOf<ulong>   { static const int value = 1; };
template <> struct lengthOf<float>   { static const int value = 1; };
template <> struct lengthOf<double>  { static const int value = 1; };

template <> struct lengthOf<char2>   { static const int value = 2; };
template <> struct lengthOf<uchar2>  { static const int value = 2; };
template <> struct lengthOf<short2>  { static const int value = 2; };
template <> struct lengthOf<ushort2> { static const int value = 2; };
template <> struct lengthOf<int2>    { static const int value = 2; };
template <> struct lengthOf<uint2>   { static const int value = 2; };
template <> struct lengthOf<long2>   { static const int value = 2; };
template <> struct lengthOf<ulong2>  { static const int value = 2; };
template <> struct lengthOf<float2>  { static const int value = 2; };
template <> struct lengthOf<double2> { static const int value = 2; };

template <> struct lengthOf<char3>   { static const int value = 3; };
template <> struct lengthOf<uchar3>  { static const int value = 3; };
template <> struct lengthOf<short3>  { static const int value = 3; };
template <> struct lengthOf<ushort3> { static const int value = 3; };
template <> struct lengthOf<int3>    { static const int value = 3; };
template <> struct lengthOf<uint3>   { static const int value = 3; };
template <> struct lengthOf<long3>   { static const int value = 3; };
template <> struct lengthOf<ulong3>  { static const int value = 3; };
template <> struct lengthOf<float3>  { static const int value = 3; };
template <> struct lengthOf<double3> { static const int value = 3; };

template <> struct lengthOf<char4>   { static const int value = 4; };
template <> struct lengthOf<uchar4>  { static const int value = 4; };
template <> struct lengthOf<short4>  { static const int value = 4; };
template <> struct lengthOf<ushort4> { static const int value = 4; };
template <> struct lengthOf<int4>    { static const int value = 4; };
template <> struct lengthOf<uint4>   { static const int value = 4; };
template <> struct lengthOf<long4>   { static const int value = 4; };
template <> struct lengthOf<ulong4>  { static const int value = 4; };
template <> struct lengthOf<float4>  { static const int value = 4; };
template <> struct lengthOf<double4> { static const int value = 4; };

struct IntegralType {};
struct FloatingPointSinglePrecType {};
struct FloatingPointDoublePrecType {};

template <typename T> struct numericType;

template <> struct numericType<char>    { typedef IntegralType type; };
template <> struct numericType<char2>   { typedef IntegralType type; };
template <> struct numericType<char3>   { typedef IntegralType type; };
template <> struct numericType<char4>   { typedef IntegralType type; };

template <> struct numericType<unsigned char> { typedef IntegralType type; };
template <> struct numericType<uchar2>  { typedef IntegralType type; };
template <> struct numericType<uchar3>  { typedef IntegralType type; };
template <> struct numericType<uchar4>  { typedef IntegralType type; };

template <> struct numericType<short>   { typedef IntegralType type; };
template <> struct numericType<short2>  { typedef IntegralType type; };
template <> struct numericType<short3>  { typedef IntegralType type; };
template <> struct numericType<short4>  { typedef IntegralType type; };

template <> struct numericType<ushort>  { typedef IntegralType type; };
template <> struct numericType<ushort2> { typedef IntegralType type; };
template <> struct numericType<ushort3> { typedef IntegralType type; };
template <> struct numericType<ushort4> { typedef IntegralType type; };

template <> struct numericType<int>     { typedef IntegralType type; };
template <> struct numericType<int2>    { typedef IntegralType type; };
template <> struct numericType<int3>    { typedef IntegralType type; };
template <> struct numericType<int4>    { typedef IntegralType type; };

template <> struct numericType<uint>    { typedef IntegralType type; };
template <> struct numericType<uint2>   { typedef IntegralType type; };
template <> struct numericType<uint3>   { typedef IntegralType type; };
template <> struct numericType<uint4>   { typedef IntegralType type; };

template <> struct numericType<long>    { typedef IntegralType type; };
template <> struct numericType<long2>   { typedef IntegralType type; };
template <> struct numericType<long3>   { typedef IntegralType type; };
template <> struct numericType<long4>   { typedef IntegralType type; };

template <> struct numericType<ulong>   { typedef IntegralType type; };
template <> struct numericType<ulong2>  { typedef IntegralType type; };
template <> struct numericType<ulong3>  { typedef IntegralType type; };
template <> struct numericType<ulong4>  { typedef IntegralType type; };


template <> struct numericType<float>   { typedef FloatingPointSinglePrecType type; };
template <> struct numericType<float2>  { typedef FloatingPointSinglePrecType type; };
template <> struct numericType<float3>  { typedef FloatingPointSinglePrecType type; };
template <> struct numericType<float4>  { typedef FloatingPointSinglePrecType type; };

template <> struct numericType<double>  { typedef FloatingPointDoublePrecType type; };
template <> struct numericType<double2> { typedef FloatingPointDoublePrecType type; };
template <> struct numericType<double3> { typedef FloatingPointDoublePrecType type; };
template <> struct numericType<double4> { typedef FloatingPointDoublePrecType type; };

} // namespace internal
} // namespace gcuda


#endif /* GCUDA_DETAIL_INTERNAL_TRAITS_H_ */
