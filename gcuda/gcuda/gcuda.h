/*
 * <gcuda/gcuda.h>
 *
 *  Created on: May 3, 2013
 *      Author: eaglenature@gmail.com
 *
 *  Description: Cuda C++ extension for Google C++ testing Framework
 */

#ifndef GCUDA_H_
#define GCUDA_H_

#include <gcuda/detail/gcuda.h>

#define ASSERT_HOST_ARRAY_EQ(expected, actual)
#define EXPECT_HOST_ARRAY_EQ(expected, actual)

#define ASSERT_DEVICE_ARRAY_EQ(expected, actual)
#define EXPECT_DEVICE_ARRAY_EQ(expected, actual)

#define ASSERT_HOST_DEVICE_ARRAY_EQ(expected, actual)
#define EXPECT_HOST_DEVICE_ARRAY_EQ(expected, actual)

#endif /* GCUDA_H_ */
