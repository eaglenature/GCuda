/*
 * <gcuda/detail/internal/gcuda.h>
 *
 *  Created on: May 4, 2013
 *      Author: eaglenature@gmail.com
 */

#ifndef DETAIL_INTERNAL_GCUDA_H_
#define DETAIL_INTERNAL_GCUDA_H_


#include <gtest/gtest.h>
#include <gcuda/detail/internal/traits.h>



#ifndef GCUDA_MESSAGE
#define GCUDA_MESSAGE(index, file, line) \
    "At index: " << (index) << "\n" << (file) << ":" << (line) << '\n'
#else
#error GCUDA_MESSAGE redefinition
#endif



namespace gcuda
{
namespace internal
{

template <typename NumericType, int Length>
struct AssertArray;

template <>
struct AssertArray<IntegralType, 1>
{
    template <typename T>
    static void Equal(
            const T* expected,
            const T* actual,
            const int size,
            const char* const file,
            const int line)
    {
        for (int i = 0; i < size; ++i)
        {
            ASSERT_EQ(expected[i], actual[i]) << GCUDA_MESSAGE(i, file, line);
        }
    }
};

template <>
struct AssertArray<IntegralType, 2>
{
    template <typename T>
    static void Equal(
            const T* expected,
            const T* actual,
            const int size,
            const char* const file,
            const int line)
    {
        for (int i = 0; i < size; ++i)
        {
            ASSERT_EQ(expected[i].x, actual[i].x) << GCUDA_MESSAGE(i, file, line);
            ASSERT_EQ(expected[i].y, actual[i].y) << GCUDA_MESSAGE(i, file, line);
        }
    }
};

template <>
struct AssertArray<IntegralType, 3>
{
    template <typename T>
    static void Equal(
            const T* expected,
            const T* actual,
            const int size,
            const char* const file,
            const int line)
    {
        for (int i = 0; i < size; ++i)
        {
            ASSERT_EQ(expected[i].x, actual[i].x) << GCUDA_MESSAGE(i, file, line);
            ASSERT_EQ(expected[i].y, actual[i].y) << GCUDA_MESSAGE(i, file, line);
            ASSERT_EQ(expected[i].z, actual[i].z) << GCUDA_MESSAGE(i, file, line);
        }
    }
};

template <>
struct AssertArray<IntegralType, 4>
{
    template <typename T>
    static void Equal(
            const T* expected,
            const T* actual,
            const int size,
            const char* const file,
            const int line)
    {
        for (int i = 0; i < size; ++i)
        {
            ASSERT_EQ(expected[i].x, actual[i].x) << GCUDA_MESSAGE(i, file, line);
            ASSERT_EQ(expected[i].y, actual[i].y) << GCUDA_MESSAGE(i, file, line);
            ASSERT_EQ(expected[i].z, actual[i].z) << GCUDA_MESSAGE(i, file, line);
            ASSERT_EQ(expected[i].w, actual[i].w) << GCUDA_MESSAGE(i, file, line);
        }
    }
};

template <>
struct AssertArray<FloatingPointSinglePrecType, 1>
{
    template <typename T>
    static void Equal(
            const T* expected,
            const T* actual,
            const int size,
            const char* const file,
            const int line)
    {
        for (int i = 0; i < size; ++i)
        {
            ASSERT_FLOAT_EQ(expected[i], actual[i]) << GCUDA_MESSAGE(i, file, line);
        }
    }

    template <typename T>
    static void Near(
            const T* expected,
            const T* actual,
            const int size,
            const double abs_error,
            const char* const file,
            const int line)
    {
        for (int i = 0; i < size; ++i)
        {
            ASSERT_NEAR(expected[i], actual[i], abs_error) << GCUDA_MESSAGE(i, file, line);
        }
    }
};

template <>
struct AssertArray<FloatingPointSinglePrecType, 2>
{
    template <typename T>
    static void Equal(
            const T* expected,
            const T* actual,
            const int size,
            const char* const file,
            const int line)
    {
        for (int i = 0; i < size; ++i)
        {
            ASSERT_FLOAT_EQ(expected[i].x, actual[i].x) << GCUDA_MESSAGE(i, file, line);
            ASSERT_FLOAT_EQ(expected[i].y, actual[i].y) << GCUDA_MESSAGE(i, file, line);
        }
    }

    template <typename T>
    static void Near(
            const T* expected,
            const T* actual,
            const int size,
            const double abs_error,
            const char* const file,
            const int line)
    {
        for (int i = 0; i < size; ++i)
        {
            ASSERT_NEAR(expected[i].x, actual[i].x, abs_error) << GCUDA_MESSAGE(i, file, line);
            ASSERT_NEAR(expected[i].y, actual[i].y, abs_error) << GCUDA_MESSAGE(i, file, line);
        }
    }
};

template <>
struct AssertArray<FloatingPointSinglePrecType, 3>
{
    template <typename T>
    static void Equal(
            const T* expected,
            const T* actual,
            const int size,
            const char* const file,
            const int line)
    {
        for (int i = 0; i < size; ++i)
        {
            ASSERT_FLOAT_EQ(expected[i].x, actual[i].x) << GCUDA_MESSAGE(i, file, line);
            ASSERT_FLOAT_EQ(expected[i].y, actual[i].y) << GCUDA_MESSAGE(i, file, line);
            ASSERT_FLOAT_EQ(expected[i].z, actual[i].z) << GCUDA_MESSAGE(i, file, line);
        }
    }

    template <typename T>
    static void Near(
            const T* expected,
            const T* actual,
            const int size,
            const double abs_error,
            const char* const file,
            const int line)
    {
        for (int i = 0; i < size; ++i)
        {
            ASSERT_NEAR(expected[i].x, actual[i].x, abs_error) << GCUDA_MESSAGE(i, file, line);
            ASSERT_NEAR(expected[i].y, actual[i].y, abs_error) << GCUDA_MESSAGE(i, file, line);
            ASSERT_NEAR(expected[i].z, actual[i].z, abs_error) << GCUDA_MESSAGE(i, file, line);
        }
    }
};

template <>
struct AssertArray<FloatingPointSinglePrecType, 4>
{
    template <typename T>
    static void Equal(
            const T* expected,
            const T* actual,
            const int size,
            const char* const file,
            const int line)
    {
        for (int i = 0; i < size; ++i)
        {
            ASSERT_FLOAT_EQ(expected[i].x, actual[i].x) << GCUDA_MESSAGE(i, file, line);
            ASSERT_FLOAT_EQ(expected[i].y, actual[i].y) << GCUDA_MESSAGE(i, file, line);
            ASSERT_FLOAT_EQ(expected[i].z, actual[i].z) << GCUDA_MESSAGE(i, file, line);
            ASSERT_FLOAT_EQ(expected[i].w, actual[i].w) << GCUDA_MESSAGE(i, file, line);
        }
    }

    template <typename T>
    static void Near(
            const T* expected,
            const T* actual,
            const int size,
            const double abs_error,
            const char* const file,
            const int line)
    {
        for (int i = 0; i < size; ++i)
        {
            ASSERT_NEAR(expected[i].x, actual[i].x, abs_error) << GCUDA_MESSAGE(i, file, line);
            ASSERT_NEAR(expected[i].y, actual[i].y, abs_error) << GCUDA_MESSAGE(i, file, line);
            ASSERT_NEAR(expected[i].z, actual[i].z, abs_error) << GCUDA_MESSAGE(i, file, line);
            ASSERT_NEAR(expected[i].w, actual[i].w, abs_error) << GCUDA_MESSAGE(i, file, line);
        }
    }
};

template <>
struct AssertArray<FloatingPointDoublePrecType, 1>
{
    template <typename T>
    static void Equal(
            const T* expected,
            const T* actual,
            const int size,
            const char* const file,
            const int line)
    {
        for (int i = 0; i < size; ++i)
        {
            ASSERT_DOUBLE_EQ(expected[i], actual[i]) << GCUDA_MESSAGE(i, file, line);
        }
    }

    template <typename T>
    static void Near(
            const T* expected,
            const T* actual,
            const int size,
            const double abs_error,
            const char* const file,
            const int line)
    {
        for (int i = 0; i < size; ++i)
        {
            ASSERT_NEAR(expected[i], actual[i], abs_error) << GCUDA_MESSAGE(i, file, line);
        }
    }
};

template <>
struct AssertArray<FloatingPointDoublePrecType, 2>
{
    template <typename T>
    static void Equal(
            const T* expected,
            const T* actual,
            const int size,
            const char* const file,
            const int line)
    {
        for (int i = 0; i < size; ++i)
        {
            ASSERT_DOUBLE_EQ(expected[i].x, actual[i].x) << GCUDA_MESSAGE(i, file, line);
            ASSERT_DOUBLE_EQ(expected[i].y, actual[i].y) << GCUDA_MESSAGE(i, file, line);
        }
    }

    template <typename T>
    static void Near(
            const T* expected,
            const T* actual,
            const int size,
            const double abs_error,
            const char* const file,
            const int line)
    {
        for (int i = 0; i < size; ++i)
        {
            ASSERT_NEAR(expected[i].x, actual[i].x, abs_error) << GCUDA_MESSAGE(i, file, line);
            ASSERT_NEAR(expected[i].y, actual[i].y, abs_error) << GCUDA_MESSAGE(i, file, line);
        }
    }
};

template <>
struct AssertArray<FloatingPointDoublePrecType, 3>
{
    template <typename T>
    static void Equal(
            const T* expected,
            const T* actual,
            const int size,
            const char* const file,
            const int line)
    {
        for (int i = 0; i < size; ++i)
        {
            ASSERT_DOUBLE_EQ(expected[i].x, actual[i].x) << GCUDA_MESSAGE(i, file, line);
            ASSERT_DOUBLE_EQ(expected[i].y, actual[i].y) << GCUDA_MESSAGE(i, file, line);
            ASSERT_DOUBLE_EQ(expected[i].z, actual[i].z) << GCUDA_MESSAGE(i, file, line);
        }
    }

    template <typename T>
    static void Near(
            const T* expected,
            const T* actual,
            const int size,
            const double abs_error,
            const char* const file,
            const int line)
    {
        for (int i = 0; i < size; ++i)
        {
            ASSERT_NEAR(expected[i].x, actual[i].x, abs_error) << GCUDA_MESSAGE(i, file, line);
            ASSERT_NEAR(expected[i].y, actual[i].y, abs_error) << GCUDA_MESSAGE(i, file, line);
            ASSERT_NEAR(expected[i].z, actual[i].z, abs_error) << GCUDA_MESSAGE(i, file, line);
        }
    }
};

template <>
struct AssertArray<FloatingPointDoublePrecType, 4>
{
    template <typename T>
    static void Equal(
            const T* expected,
            const T* actual,
            const int size,
            const char* const file,
            const int line)
    {
        for (int i = 0; i < size; ++i)
        {
            ASSERT_DOUBLE_EQ(expected[i].x, actual[i].x) << GCUDA_MESSAGE(i, file, line);
            ASSERT_DOUBLE_EQ(expected[i].y, actual[i].y) << GCUDA_MESSAGE(i, file, line);
            ASSERT_DOUBLE_EQ(expected[i].z, actual[i].z) << GCUDA_MESSAGE(i, file, line);
            ASSERT_DOUBLE_EQ(expected[i].w, actual[i].w) << GCUDA_MESSAGE(i, file, line);
        }
    }

    template <typename T>
    static void Near(
            const T* expected,
            const T* actual,
            const int size,
            const double abs_error,
            const char* const file,
            const int line)
    {
        for (int i = 0; i < size; ++i)
        {
            ASSERT_NEAR(expected[i].x, actual[i].x, abs_error) << GCUDA_MESSAGE(i, file, line);
            ASSERT_NEAR(expected[i].y, actual[i].y, abs_error) << GCUDA_MESSAGE(i, file, line);
            ASSERT_NEAR(expected[i].z, actual[i].z, abs_error) << GCUDA_MESSAGE(i, file, line);
            ASSERT_NEAR(expected[i].w, actual[i].w, abs_error) << GCUDA_MESSAGE(i, file, line);
        }
    }
};


template <typename NumericType, int Length>
struct ExpectArray;

template <>
struct ExpectArray<IntegralType, 1>
{
    template <typename T>
    static void Equal(
            const T* expected,
            const T* actual,
            const int size,
            const char* const file,
            const int line)
    {
        for (int i = 0; i < size; ++i)
        {
            EXPECT_EQ(expected[i], actual[i]) << GCUDA_MESSAGE(i, file, line);
        }
    }
};

template <>
struct ExpectArray<IntegralType, 2>
{
    template <typename T>
    static void Equal(
            const T* expected,
            const T* actual,
            const int size,
            const char* const file,
            const int line)
    {
        for (int i = 0; i < size; ++i)
        {
            EXPECT_EQ(expected[i].x, actual[i].x) << GCUDA_MESSAGE(i, file, line);
            EXPECT_EQ(expected[i].y, actual[i].y) << GCUDA_MESSAGE(i, file, line);
        }
    }
};

template <>
struct ExpectArray<IntegralType, 3>
{
    template <typename T>
    static void Equal(
            const T* expected,
            const T* actual,
            const int size,
            const char* const file,
            const int line)
    {
        for (int i = 0; i < size; ++i)
        {
            EXPECT_EQ(expected[i].x, actual[i].x) << GCUDA_MESSAGE(i, file, line);
            EXPECT_EQ(expected[i].y, actual[i].y) << GCUDA_MESSAGE(i, file, line);
            EXPECT_EQ(expected[i].z, actual[i].z) << GCUDA_MESSAGE(i, file, line);
        }
    }
};

template <>
struct ExpectArray<IntegralType, 4>
{
    template <typename T>
    static void Equal(
            const T* expected,
            const T* actual,
            const int size,
            const char* const file,
            const int line)
    {
        for (int i = 0; i < size; ++i)
        {
            EXPECT_EQ(expected[i].x, actual[i].x) << GCUDA_MESSAGE(i, file, line);
            EXPECT_EQ(expected[i].y, actual[i].y) << GCUDA_MESSAGE(i, file, line);
            EXPECT_EQ(expected[i].z, actual[i].z) << GCUDA_MESSAGE(i, file, line);
            EXPECT_EQ(expected[i].w, actual[i].w) << GCUDA_MESSAGE(i, file, line);
        }
    }
};

template <>
struct ExpectArray<FloatingPointSinglePrecType, 1>
{
    template <typename T>
    static void Equal(
            const T* expected,
            const T* actual,
            const int size,
            const char* const file,
            const int line)
    {
        for (int i = 0; i < size; ++i)
        {
            EXPECT_FLOAT_EQ(expected[i], actual[i]) << GCUDA_MESSAGE(i, file, line);
        }
    }

    template <typename T>
    static void Near(
            const T* expected,
            const T* actual,
            const int size,
            const double abs_error,
            const char* const file,
            const int line)
    {
        for (int i = 0; i < size; ++i)
        {
            EXPECT_NEAR(expected[i], actual[i], abs_error) << GCUDA_MESSAGE(i, file, line);
        }
    }
};

template <>
struct ExpectArray<FloatingPointSinglePrecType, 2>
{
    template <typename T>
    static void Equal(
            const T* expected,
            const T* actual,
            const int size,
            const char* const file,
            const int line)
    {
        for (int i = 0; i < size; ++i)
        {
            EXPECT_FLOAT_EQ(expected[i].x, actual[i].x) << GCUDA_MESSAGE(i, file, line);
            EXPECT_FLOAT_EQ(expected[i].y, actual[i].y) << GCUDA_MESSAGE(i, file, line);
        }
    }

    template <typename T>
    static void Near(
            const T* expected,
            const T* actual,
            const int size,
            const double abs_error,
            const char* const file,
            const int line)
    {
        for (int i = 0; i < size; ++i)
        {
            EXPECT_NEAR(expected[i].x, actual[i].x, abs_error) << GCUDA_MESSAGE(i, file, line);
            EXPECT_NEAR(expected[i].y, actual[i].y, abs_error) << GCUDA_MESSAGE(i, file, line);
        }
    }
};

template <>
struct ExpectArray<FloatingPointSinglePrecType, 3>
{
    template <typename T>
    static void Equal(
            const T* expected,
            const T* actual,
            const int size,
            const char* const file,
            const int line)
    {
        for (int i = 0; i < size; ++i)
        {
            EXPECT_FLOAT_EQ(expected[i].x, actual[i].x) << GCUDA_MESSAGE(i, file, line);
            EXPECT_FLOAT_EQ(expected[i].y, actual[i].y) << GCUDA_MESSAGE(i, file, line);
            EXPECT_FLOAT_EQ(expected[i].z, actual[i].z) << GCUDA_MESSAGE(i, file, line);
        }
    }

    template <typename T>
    static void Near(
            const T* expected,
            const T* actual,
            const int size,
            const double abs_error,
            const char* const file,
            const int line)
    {
        for (int i = 0; i < size; ++i)
        {
            EXPECT_NEAR(expected[i].x, actual[i].x, abs_error) << GCUDA_MESSAGE(i, file, line);
            EXPECT_NEAR(expected[i].y, actual[i].y, abs_error) << GCUDA_MESSAGE(i, file, line);
            EXPECT_NEAR(expected[i].z, actual[i].z, abs_error) << GCUDA_MESSAGE(i, file, line);
        }
    }
};

template <>
struct ExpectArray<FloatingPointSinglePrecType, 4>
{
    template <typename T>
    static void Equal(
            const T* expected,
            const T* actual,
            const int size,
            const char* const file,
            const int line)
    {
        for (int i = 0; i < size; ++i)
        {
            EXPECT_FLOAT_EQ(expected[i].x, actual[i].x) << GCUDA_MESSAGE(i, file, line);
            EXPECT_FLOAT_EQ(expected[i].y, actual[i].y) << GCUDA_MESSAGE(i, file, line);
            EXPECT_FLOAT_EQ(expected[i].z, actual[i].z) << GCUDA_MESSAGE(i, file, line);
            EXPECT_FLOAT_EQ(expected[i].w, actual[i].w) << GCUDA_MESSAGE(i, file, line);
        }
    }

    template <typename T>
    static void Near(
            const T* expected,
            const T* actual,
            const int size,
            const double abs_error,
            const char* const file,
            const int line)
    {
        for (int i = 0; i < size; ++i)
        {
            EXPECT_NEAR(expected[i].x, actual[i].x, abs_error) << GCUDA_MESSAGE(i, file, line);
            EXPECT_NEAR(expected[i].y, actual[i].y, abs_error) << GCUDA_MESSAGE(i, file, line);
            EXPECT_NEAR(expected[i].z, actual[i].z, abs_error) << GCUDA_MESSAGE(i, file, line);
            EXPECT_NEAR(expected[i].w, actual[i].w, abs_error) << GCUDA_MESSAGE(i, file, line);
        }
    }
};

template <>
struct ExpectArray<FloatingPointDoublePrecType, 1>
{
    template <typename T>
    static void Equal(
            const T* expected,
            const T* actual,
            const int size,
            const char* const file,
            const int line)
    {
        for (int i = 0; i < size; ++i)
        {
            EXPECT_DOUBLE_EQ(expected[i], actual[i]) << GCUDA_MESSAGE(i, file, line);
        }
    }

    template <typename T>
    static void Near(
            const T* expected,
            const T* actual,
            const int size,
            const double abs_error,
            const char* const file,
            const int line)
    {
        for (int i = 0; i < size; ++i)
        {
            EXPECT_NEAR(expected[i], actual[i], abs_error) << GCUDA_MESSAGE(i, file, line);
        }
    }
};

template <>
struct ExpectArray<FloatingPointDoublePrecType, 2>
{
    template <typename T>
    static void Equal(
            const T* expected,
            const T* actual,
            const int size,
            const char* const file,
            const int line)
    {
        for (int i = 0; i < size; ++i)
        {
            EXPECT_DOUBLE_EQ(expected[i].x, actual[i].x) << GCUDA_MESSAGE(i, file, line);
            EXPECT_DOUBLE_EQ(expected[i].y, actual[i].y) << GCUDA_MESSAGE(i, file, line);
        }
    }

    template <typename T>
    static void Near(
            const T* expected,
            const T* actual,
            const int size,
            const double abs_error,
            const char* const file,
            const int line)
    {
        for (int i = 0; i < size; ++i)
        {
            EXPECT_NEAR(expected[i].x, actual[i].x, abs_error) << GCUDA_MESSAGE(i, file, line);
            EXPECT_NEAR(expected[i].y, actual[i].y, abs_error) << GCUDA_MESSAGE(i, file, line);
        }
    }
};

template <>
struct ExpectArray<FloatingPointDoublePrecType, 3>
{
    template <typename T>
    static void Equal(
            const T* expected,
            const T* actual,
            const int size,
            const char* const file,
            const int line)
    {
        for (int i = 0; i < size; ++i)
        {
            EXPECT_DOUBLE_EQ(expected[i].x, actual[i].x) << GCUDA_MESSAGE(i, file, line);
            EXPECT_DOUBLE_EQ(expected[i].y, actual[i].y) << GCUDA_MESSAGE(i, file, line);
            EXPECT_DOUBLE_EQ(expected[i].z, actual[i].z) << GCUDA_MESSAGE(i, file, line);
        }
    }

    template <typename T>
    static void Near(
            const T* expected,
            const T* actual,
            const int size,
            const double abs_error,
            const char* const file,
            const int line)
    {
        for (int i = 0; i < size; ++i)
        {
            EXPECT_NEAR(expected[i].x, actual[i].x, abs_error) << GCUDA_MESSAGE(i, file, line);
            EXPECT_NEAR(expected[i].y, actual[i].y, abs_error) << GCUDA_MESSAGE(i, file, line);
            EXPECT_NEAR(expected[i].z, actual[i].z, abs_error) << GCUDA_MESSAGE(i, file, line);
        }
    }
};

template <>
struct ExpectArray<FloatingPointDoublePrecType, 4>
{
    template <typename T>
    static void Equal(
            const T* expected,
            const T* actual,
            const int size,
            const char* const file,
            const int line)
    {
        for (int i = 0; i < size; ++i)
        {
            EXPECT_DOUBLE_EQ(expected[i].x, actual[i].x) << GCUDA_MESSAGE(i, file, line);
            EXPECT_DOUBLE_EQ(expected[i].y, actual[i].y) << GCUDA_MESSAGE(i, file, line);
            EXPECT_DOUBLE_EQ(expected[i].z, actual[i].z) << GCUDA_MESSAGE(i, file, line);
            EXPECT_DOUBLE_EQ(expected[i].w, actual[i].w) << GCUDA_MESSAGE(i, file, line);
        }
    }

    template <typename T>
    static void Near(
            const T* expected,
            const T* actual,
            const int size,
            const double abs_error,
            const char* const file,
            const int line)
    {
        for (int i = 0; i < size; ++i)
        {
            EXPECT_NEAR(expected[i].x, actual[i].x, abs_error) << GCUDA_MESSAGE(i, file, line);
            EXPECT_NEAR(expected[i].y, actual[i].y, abs_error) << GCUDA_MESSAGE(i, file, line);
            EXPECT_NEAR(expected[i].z, actual[i].z, abs_error) << GCUDA_MESSAGE(i, file, line);
            EXPECT_NEAR(expected[i].w, actual[i].w, abs_error) << GCUDA_MESSAGE(i, file, line);
        }
    }
};






template <typename T>
void assertArrayEq(
        const T* expected,
        const T* actual,
        const int size,
        const char* const file,
        const int line)
{
    AssertArray<typename internal::numericType<T>::type, internal::lengthOf<T>::value>::Equal(
            expected,
            actual,
            size,
            file,
            line);
}


template <typename T>
void assertArrayNear(
        const T* expected,
        const T* actual,
        const int size,
        const double abs_error,
        const char* const file,
        const int line)
{
    AssertArray<typename internal::numericType<T>::type, internal::lengthOf<T>::value>::Near(
            expected,
            actual,
            size,
            abs_error,
            file,
            line);
}


template <typename T>
void expectArrayEq(
        const T* expected,
        const T* actual,
        const int size,
        const char* const file,
        const int line)
{
    ExpectArray<typename internal::numericType<T>::type, internal::lengthOf<T>::value>::Equal(
            expected,
            actual,
            size,
            file,
            line);
}


template <typename T>
void expectArrayNear(
        const T* expected,
        const T* actual,
        const int size,
        const double abs_error,
        const char* const file,
        const int line)
{
    ExpectArray<typename internal::numericType<T>::type, internal::lengthOf<T>::value>::Near(
            expected,
            actual,
            size,
            abs_error,
            file,
            line);
}


} // namespace internal
} // namespace gcuda

#endif /* DETAIL_INTERNAL_GCUDA_H_ */
