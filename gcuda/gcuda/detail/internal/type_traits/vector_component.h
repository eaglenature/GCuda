/*
 * <gcuda/detail/internal/type_traits/vector_component.h>
 *
 *  Created on: May 18, 2013
 *      Author: eaglenature@gmail.com
 */

#ifndef DETAIL_INTERNAL_TYPE_TRAITS_VECTOR_COMPONENT_H_
#define DETAIL_INTERNAL_TYPE_TRAITS_VECTOR_COMPONENT_H_

namespace gcuda
{
namespace detail
{

struct integral_component_tag {};
struct single_prec_float_component_tag {};
struct double_prec_float_component_tag {};

template <class RawType>
struct vector_component;

template <> struct vector_component< char  >   { typedef integral_component_tag type; typedef char value_type; };
template <> struct vector_component< char1 >   { typedef integral_component_tag type; typedef char value_type; };
template <> struct vector_component< char2 >   { typedef integral_component_tag type; typedef char value_type; };
template <> struct vector_component< char3 >   { typedef integral_component_tag type; typedef char value_type; };
template <> struct vector_component< char4 >   { typedef integral_component_tag type; typedef char value_type; };

template <> struct vector_component< unsigned char > { typedef integral_component_tag type; typedef unsigned char value_type; };
template <> struct vector_component< uchar1 >  { typedef integral_component_tag type; typedef unsigned char value_type; };
template <> struct vector_component< uchar2 >  { typedef integral_component_tag type; typedef unsigned char value_type; };
template <> struct vector_component< uchar3 >  { typedef integral_component_tag type; typedef unsigned char value_type; };
template <> struct vector_component< uchar4 >  { typedef integral_component_tag type; typedef unsigned char value_type; };

template <> struct vector_component< short  >  { typedef integral_component_tag type; typedef short value_type; };
template <> struct vector_component< short1 >  { typedef integral_component_tag type; typedef short value_type; };
template <> struct vector_component< short2 >  { typedef integral_component_tag type; typedef short value_type; };
template <> struct vector_component< short3 >  { typedef integral_component_tag type; typedef short value_type; };
template <> struct vector_component< short4 >  { typedef integral_component_tag type; typedef short value_type; };

template <> struct vector_component< unsigned short > { typedef integral_component_tag type; typedef unsigned short value_type; };
template <> struct vector_component< ushort1 > { typedef integral_component_tag type; typedef unsigned short value_type; };
template <> struct vector_component< ushort2 > { typedef integral_component_tag type; typedef unsigned short value_type; };
template <> struct vector_component< ushort3 > { typedef integral_component_tag type; typedef unsigned short value_type; };
template <> struct vector_component< ushort4 > { typedef integral_component_tag type; typedef unsigned short value_type; };

template <> struct vector_component< int  >    { typedef integral_component_tag type; typedef int value_type; };
template <> struct vector_component< int1 >    { typedef integral_component_tag type; typedef int value_type; };
template <> struct vector_component< int2 >    { typedef integral_component_tag type; typedef int value_type; };
template <> struct vector_component< int3 >    { typedef integral_component_tag type; typedef int value_type; };
template <> struct vector_component< int4 >    { typedef integral_component_tag type; typedef int value_type; };

template <> struct vector_component< uint  >   { typedef integral_component_tag type; typedef uint value_type; };
template <> struct vector_component< uint1 >   { typedef integral_component_tag type; typedef uint value_type; };
template <> struct vector_component< uint2 >   { typedef integral_component_tag type; typedef uint value_type; };
template <> struct vector_component< uint3 >   { typedef integral_component_tag type; typedef uint value_type; };
template <> struct vector_component< uint4 >   { typedef integral_component_tag type; typedef uint value_type; };

template <> struct vector_component< long  >   { typedef integral_component_tag type; typedef long value_type; };
template <> struct vector_component< long1 >   { typedef integral_component_tag type; typedef long value_type; };
template <> struct vector_component< long2 >   { typedef integral_component_tag type; typedef long value_type; };
template <> struct vector_component< long3 >   { typedef integral_component_tag type; typedef long value_type; };
template <> struct vector_component< long4 >   { typedef integral_component_tag type; typedef long value_type; };

template <> struct vector_component< unsigned long > { typedef integral_component_tag type; typedef unsigned long value_type; };
template <> struct vector_component< ulong1 >  { typedef integral_component_tag type; typedef unsigned long value_type; };
template <> struct vector_component< ulong2 >  { typedef integral_component_tag type; typedef unsigned long value_type; };
template <> struct vector_component< ulong3 >  { typedef integral_component_tag type; typedef unsigned long value_type; };
template <> struct vector_component< ulong4 >  { typedef integral_component_tag type; typedef unsigned long value_type; };

template <> struct vector_component< float  >  { typedef single_prec_float_component_tag type; typedef float value_type; };
template <> struct vector_component< float1 >  { typedef single_prec_float_component_tag type; typedef float value_type; };
template <> struct vector_component< float2 >  { typedef single_prec_float_component_tag type; typedef float value_type; };
template <> struct vector_component< float3 >  { typedef single_prec_float_component_tag type; typedef float value_type; };
template <> struct vector_component< float4 >  { typedef single_prec_float_component_tag type; typedef float value_type; };

template <> struct vector_component< double  > { typedef double_prec_float_component_tag type; typedef double value_type; };
template <> struct vector_component< double1 > { typedef double_prec_float_component_tag type; typedef double value_type; };
template <> struct vector_component< double2 > { typedef double_prec_float_component_tag type; typedef double value_type; };
template <> struct vector_component< double3 > { typedef double_prec_float_component_tag type; typedef double value_type; };
template <> struct vector_component< double4 > { typedef double_prec_float_component_tag type; typedef double value_type; };

} // namespace detail
} // namespace gcuda

#endif /* DETAIL_INTERNAL_TYPE_TRAITS_VECTOR_COMPONENT_H_ */
