#include <gcuda/gcuda.h>

//-------------------------------------------------------------//
//                                                             //
//                 ASSERT_HOST_ARRAY_EQ                        //
//                                                             //
//-------------------------------------------------------------//
TEST(HostArrayEqual, Int)
{
    typedef int T;
    const int numElements = 10;
    std::vector<T> a(numElements);
    std::vector<T> b(numElements);
    for (int i = 0; i < numElements; ++i)
    {
        a[i] = i;
        b[i] = i;
    }
    ASSERT_HOST_ARRAY_EQ(a.data(), b.data(), numElements);
}

TEST(HostArrayEqual, Int2)
{
    typedef int2 T;
    const int numElements = 10;
    std::vector<T> a(numElements);
    std::vector<T> b(numElements);
    for (int i = 0; i < numElements; ++i)
    {
        a[i] = make_int2(i, -i);
        b[i] = make_int2(i, -i);
    }
    //b[7] = make_int2(131, 123123);
    ASSERT_HOST_ARRAY_EQ(a.data(), b.data(), numElements);
}

TEST(HostArrayEqual, Float3)
{
    typedef float3 T;
    const int numElements = 10;
    std::vector<T> a(numElements);
    std::vector<T> b(numElements);
    for (int i = 0; i < numElements; ++i)
    {
        a[i] = make_float3(i * 3.14f, (-i) * 3.14f, 0.01234f);
        b[i] = make_float3(i * 3.14f, (-i) * 3.14f, 0.01234f);
    }
    //b[2] = make_float3(131, 555123123.03f, 0.01235f);
    ASSERT_HOST_ARRAY_EQ(a.data(), b.data(), numElements);
}

TEST(HostArrayEqual, Double4)
{
    typedef double4 T;
    const int numElements = 10;
    std::vector<T> a(numElements);
    std::vector<T> b(numElements);
    for (int i = 0; i < numElements; ++i)
    {
        a[i] = make_double4(i * 3.1412312, (-i) * 0.0314, 0.01234, 0.000012);
        b[i] = make_double4(i * 3.1412312, (-i) * 0.0314, 0.01234,  0.000012);
    }
    //b[2] = make_double4(131, 123123, 0.01235f, -123.12345);
    ASSERT_HOST_ARRAY_EQ(a.data(), b.data(), numElements);
}

//-------------------------------------------------------------//
//                                                             //
//                 ASSERT_HOST_ARRAY_NEAR                      //
//                                                             //
//-------------------------------------------------------------//
TEST(HostArrayNear, Float)
{
    typedef float T;
    const int numElements = 10;
    std::vector<T> a(numElements);
    std::vector<T> b(numElements);
    for (int i = 0; i < numElements; ++i)
    {
        a[i] = i + 0.0007f;
        b[i] = i + 0.0002f;
    }
    const double absError = 0.001;
    ASSERT_HOST_ARRAY_NEAR(a.data(), b.data(), numElements, absError);
}

TEST(HostArrayNear, Double)
{
    typedef double T;
    const int numElements = 10;
    std::vector<T> a(numElements);
    std::vector<T> b(numElements);
    for (int i = 0; i < numElements; ++i)
    {
        a[i] = 0.0000007;
        b[i] = 0.0000007;;
    }
    const double absError = 0.000001;
    ASSERT_HOST_ARRAY_NEAR(a.data(), b.data(), numElements, absError);
}

TEST(HostArrayNear, Float2)
{
    typedef float2 T;
    const int numElements = 10;
    std::vector<T> a(numElements);
    std::vector<T> b(numElements);
    for (int i = 0; i < numElements; ++i)
    {
        a[i] = make_float2(0.0007f, -0.0006f);
        b[i] = make_float2(0.0007f, -0.0007f);
    }
    //b[7] = make_float2(131, 123123);
    const double absError = 0.001;
    ASSERT_HOST_ARRAY_NEAR(a.data(), b.data(), numElements, absError);
}

TEST(HostArrayNear, Float3)
{
    typedef float3 T;
    const int numElements = 10;
    std::vector<T> a(numElements);
    std::vector<T> b(numElements);
    for (int i = 0; i < numElements; ++i)
    {
        a[i] = make_float3(0.01234f, -0.01234f, 3.14f);
        b[i] = make_float3(0.01231f, -0.01231f, 3.14f);
    }
    //b[2] = make_float3(131.31f, 123123.03f, 0.01235f);
    const double absError = 0.001;
    ASSERT_HOST_ARRAY_NEAR(a.data(), b.data(), numElements, absError);
}

TEST(HostArrayNear, Double4)
{
    typedef double4 T;
    const int numElements = 10;
    std::vector<T> a(numElements);
    std::vector<T> b(numElements);
    for (int i = 0; i < numElements; ++i)
    {
        a[i] = make_double4(3.1412312, -0.0991234, 3.00014, 0.0);
        b[i] = make_double4(3.1412312, -0.0991231, 3.00014, 0.0);
    }
    //b[2] = make_double4(131, 123123, 0.01235f, -123.12345);
    const double absError = 0.00001;
    ASSERT_HOST_ARRAY_NEAR(a.data(), b.data(), numElements, absError);
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
