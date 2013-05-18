/*
 * <gcuda/detail/internal/type_traits/vector_dimension.h>
 *
 *  Created on: May 18, 2013
 *      Author: eaglenature@gmail.com
 */

#ifndef DETAIL_INTERNAL_TYPE_TRAITS_VECTOR_DIMENSION_H_
#define DETAIL_INTERNAL_TYPE_TRAITS_VECTOR_DIMENSION_H_

namespace gcuda
{
namespace detail
{

struct dimension0 {};
struct dimension1 {};
struct dimension2 {};
struct dimension3 {};
struct dimension4 {};

template <class RawType>
struct vector_dimension;

template <> struct vector_dimension< char  >   { typedef dimension0 type; };
template <> struct vector_dimension< char1 >   { typedef dimension1 type; };
template <> struct vector_dimension< char2 >   { typedef dimension2 type; };
template <> struct vector_dimension< char3 >   { typedef dimension3 type; };
template <> struct vector_dimension< char4 >   { typedef dimension4 type; };

template <> struct vector_dimension< unsigned char > { typedef dimension0 type; };
template <> struct vector_dimension< uchar1 >  { typedef dimension1 type; };
template <> struct vector_dimension< uchar2 >  { typedef dimension2 type; };
template <> struct vector_dimension< uchar3 >  { typedef dimension3 type; };
template <> struct vector_dimension< uchar4 >  { typedef dimension4 type; };

template <> struct vector_dimension< short  >  { typedef dimension0 type; };
template <> struct vector_dimension< short1 >  { typedef dimension1 type; };
template <> struct vector_dimension< short2 >  { typedef dimension2 type; };
template <> struct vector_dimension< short3 >  { typedef dimension3 type; };
template <> struct vector_dimension< short4 >  { typedef dimension4 type; };

template <> struct vector_dimension< unsigned short > { typedef dimension0 type; };
template <> struct vector_dimension< ushort1 > { typedef dimension1 type; };
template <> struct vector_dimension< ushort2 > { typedef dimension2 type; };
template <> struct vector_dimension< ushort3 > { typedef dimension3 type; };
template <> struct vector_dimension< ushort4 > { typedef dimension4 type; };

template <> struct vector_dimension< int  >    { typedef dimension0 type; };
template <> struct vector_dimension< int1 >    { typedef dimension1 type; };
template <> struct vector_dimension< int2 >    { typedef dimension2 type; };
template <> struct vector_dimension< int3 >    { typedef dimension3 type; };
template <> struct vector_dimension< int4 >    { typedef dimension4 type; };

template <> struct vector_dimension< uint  >   { typedef dimension0 type; };
template <> struct vector_dimension< uint1 >   { typedef dimension1 type; };
template <> struct vector_dimension< uint2 >   { typedef dimension2 type; };
template <> struct vector_dimension< uint3 >   { typedef dimension3 type; };
template <> struct vector_dimension< uint4 >   { typedef dimension4 type; };

template <> struct vector_dimension< long  >   { typedef dimension0 type; };
template <> struct vector_dimension< long1 >   { typedef dimension1 type; };
template <> struct vector_dimension< long2 >   { typedef dimension2 type; };
template <> struct vector_dimension< long3 >   { typedef dimension3 type; };
template <> struct vector_dimension< long4 >   { typedef dimension4 type; };

template <> struct vector_dimension< unsigned long > { typedef dimension0 type; };
template <> struct vector_dimension< ulong1 >  { typedef dimension1 type; };
template <> struct vector_dimension< ulong2 >  { typedef dimension2 type; };
template <> struct vector_dimension< ulong3 >  { typedef dimension3 type; };
template <> struct vector_dimension< ulong4 >  { typedef dimension4 type; };

template <> struct vector_dimension< float  >  { typedef dimension0 type; };
template <> struct vector_dimension< float1 >  { typedef dimension1 type; };
template <> struct vector_dimension< float2 >  { typedef dimension2 type; };
template <> struct vector_dimension< float3 >  { typedef dimension3 type; };
template <> struct vector_dimension< float4 >  { typedef dimension4 type; };

template <> struct vector_dimension< double  > { typedef dimension0 type; };
template <> struct vector_dimension< double1 > { typedef dimension1 type; };
template <> struct vector_dimension< double2 > { typedef dimension2 type; };
template <> struct vector_dimension< double3 > { typedef dimension3 type; };
template <> struct vector_dimension< double4 > { typedef dimension4 type; };

} // namespace detail
} // namespace gcuda

#endif /* DETAIL_INTERNAL_TYPE_TRAITS_VECTOR_DIMENSION_H_ */
