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
struct vector_component_tag;

template <> struct vector_component_tag< char  >   { typedef integral_component_tag type; };
template <> struct vector_component_tag< char1 >   { typedef integral_component_tag type; };
template <> struct vector_component_tag< char2 >   { typedef integral_component_tag type; };
template <> struct vector_component_tag< char3 >   { typedef integral_component_tag type; };
template <> struct vector_component_tag< char4 >   { typedef integral_component_tag type; };

template <> struct vector_component_tag< unsigned char > { typedef integral_component_tag type; };
template <> struct vector_component_tag< uchar1 >  { typedef integral_component_tag type; };
template <> struct vector_component_tag< uchar2 >  { typedef integral_component_tag type; };
template <> struct vector_component_tag< uchar3 >  { typedef integral_component_tag type; };
template <> struct vector_component_tag< uchar4 >  { typedef integral_component_tag type; };

template <> struct vector_component_tag< short  >  { typedef integral_component_tag type; };
template <> struct vector_component_tag< short1 >  { typedef integral_component_tag type; };
template <> struct vector_component_tag< short2 >  { typedef integral_component_tag type; };
template <> struct vector_component_tag< short3 >  { typedef integral_component_tag type; };
template <> struct vector_component_tag< short4 >  { typedef integral_component_tag type; };

template <> struct vector_component_tag< unsigned short > { typedef integral_component_tag type; };
template <> struct vector_component_tag< ushort1 > { typedef integral_component_tag type; };
template <> struct vector_component_tag< ushort2 > { typedef integral_component_tag type; };
template <> struct vector_component_tag< ushort3 > { typedef integral_component_tag type; };
template <> struct vector_component_tag< ushort4 > { typedef integral_component_tag type; };

template <> struct vector_component_tag< int  >    { typedef integral_component_tag type; };
template <> struct vector_component_tag< int1 >    { typedef integral_component_tag type; };
template <> struct vector_component_tag< int2 >    { typedef integral_component_tag type; };
template <> struct vector_component_tag< int3 >    { typedef integral_component_tag type; };
template <> struct vector_component_tag< int4 >    { typedef integral_component_tag type; };

template <> struct vector_component_tag< uint  >   { typedef integral_component_tag type; };
template <> struct vector_component_tag< uint1 >   { typedef integral_component_tag type; };
template <> struct vector_component_tag< uint2 >   { typedef integral_component_tag type; };
template <> struct vector_component_tag< uint3 >   { typedef integral_component_tag type; };
template <> struct vector_component_tag< uint4 >   { typedef integral_component_tag type; };

template <> struct vector_component_tag< long  >   { typedef integral_component_tag type; };
template <> struct vector_component_tag< long1 >   { typedef integral_component_tag type; };
template <> struct vector_component_tag< long2 >   { typedef integral_component_tag type; };
template <> struct vector_component_tag< long3 >   { typedef integral_component_tag type; };
template <> struct vector_component_tag< long4 >   { typedef integral_component_tag type; };

template <> struct vector_component_tag< unsigned long > { typedef integral_component_tag type; };
template <> struct vector_component_tag< ulong1 >  { typedef integral_component_tag type; };
template <> struct vector_component_tag< ulong2 >  { typedef integral_component_tag type; };
template <> struct vector_component_tag< ulong3 >  { typedef integral_component_tag type; };
template <> struct vector_component_tag< ulong4 >  { typedef integral_component_tag type; };

template <> struct vector_component_tag< float  >  { typedef single_prec_float_component_tag type; };
template <> struct vector_component_tag< float1 >  { typedef single_prec_float_component_tag type; };
template <> struct vector_component_tag< float2 >  { typedef single_prec_float_component_tag type; };
template <> struct vector_component_tag< float3 >  { typedef single_prec_float_component_tag type; };
template <> struct vector_component_tag< float4 >  { typedef single_prec_float_component_tag type; };

template <> struct vector_component_tag< double  > { typedef double_prec_float_component_tag type; };
template <> struct vector_component_tag< double1 > { typedef double_prec_float_component_tag type; };
template <> struct vector_component_tag< double2 > { typedef double_prec_float_component_tag type; };
template <> struct vector_component_tag< double3 > { typedef double_prec_float_component_tag type; };
template <> struct vector_component_tag< double4 > { typedef double_prec_float_component_tag type; };



template <class RawType>
struct vector_component_base;

template <> struct vector_component_base< char  >   { typedef char type; };
template <> struct vector_component_base< char1 >   { typedef char type; };
template <> struct vector_component_base< char2 >   { typedef char type; };
template <> struct vector_component_base< char3 >   { typedef char type; };
template <> struct vector_component_base< char4 >   { typedef char type; };

template <> struct vector_component_base< unsigned char > { typedef unsigned char type; };
template <> struct vector_component_base< uchar1 >  { typedef unsigned char type; };
template <> struct vector_component_base< uchar2 >  { typedef unsigned char type; };
template <> struct vector_component_base< uchar3 >  { typedef unsigned char type; };
template <> struct vector_component_base< uchar4 >  { typedef unsigned char type; };

template <> struct vector_component_base< short  >  { typedef short type; };
template <> struct vector_component_base< short1 >  { typedef short type; };
template <> struct vector_component_base< short2 >  { typedef short type; };
template <> struct vector_component_base< short3 >  { typedef short type; };
template <> struct vector_component_base< short4 >  { typedef short type; };

template <> struct vector_component_base< unsigned short > { typedef unsigned short type; };
template <> struct vector_component_base< ushort1 > { typedef unsigned short type; };
template <> struct vector_component_base< ushort2 > { typedef unsigned short type; };
template <> struct vector_component_base< ushort3 > { typedef unsigned short type; };
template <> struct vector_component_base< ushort4 > { typedef unsigned short type; };

template <> struct vector_component_base< int  >    { typedef int type; };
template <> struct vector_component_base< int1 >    { typedef int type; };
template <> struct vector_component_base< int2 >    { typedef int type; };
template <> struct vector_component_base< int3 >    { typedef int type; };
template <> struct vector_component_base< int4 >    { typedef int type; };

template <> struct vector_component_base< uint  >   { typedef uint type; };
template <> struct vector_component_base< uint1 >   { typedef uint type; };
template <> struct vector_component_base< uint2 >   { typedef uint type; };
template <> struct vector_component_base< uint3 >   { typedef uint type; };
template <> struct vector_component_base< uint4 >   { typedef uint type; };

template <> struct vector_component_base< long  >   { typedef long type; };
template <> struct vector_component_base< long1 >   { typedef long type; };
template <> struct vector_component_base< long2 >   { typedef long type; };
template <> struct vector_component_base< long3 >   { typedef long type; };
template <> struct vector_component_base< long4 >   { typedef long type; };

template <> struct vector_component_base< unsigned long > { typedef unsigned long type; };
template <> struct vector_component_base< ulong1 >  { typedef unsigned long type; };
template <> struct vector_component_base< ulong2 >  { typedef unsigned long type; };
template <> struct vector_component_base< ulong3 >  { typedef unsigned long type; };
template <> struct vector_component_base< ulong4 >  { typedef unsigned long type; };

template <> struct vector_component_base< float  >  { typedef float type; };
template <> struct vector_component_base< float1 >  { typedef float type; };
template <> struct vector_component_base< float2 >  { typedef float type; };
template <> struct vector_component_base< float3 >  { typedef float type; };
template <> struct vector_component_base< float4 >  { typedef float type; };

template <> struct vector_component_base< double  > { typedef double type; };
template <> struct vector_component_base< double1 > { typedef double type; };
template <> struct vector_component_base< double2 > { typedef double type; };
template <> struct vector_component_base< double3 > { typedef double type; };
template <> struct vector_component_base< double4 > { typedef double type; };

} // namespace detail
} // namespace gcuda

#endif /* DETAIL_INTERNAL_TYPE_TRAITS_VECTOR_COMPONENT_H_ */
