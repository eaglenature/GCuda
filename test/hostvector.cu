#include <gcuda/gcuda.h>
#include <thrust/host_vector.h>

//-------------------------------------------------------------//
//                                                             //
//                 ASSERT_HOST_VECTOR_EQ                       //
//                                                             //
//-------------------------------------------------------------//
TEST(AssertHostVectorEqual, StdVectorInt)
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
    ASSERT_HOST_VECTOR_EQ(a, b);
}
TEST(AssertHostVectorEqual, ThrustVectorInt2)
{
    typedef int2 T;
    const int numElements = 10;
    thrust::host_vector<T> a(numElements);
    thrust::host_vector<T> b(numElements);
    for (int i = 0; i < numElements; ++i)
    {
        a[i] = make_int2(i, -i);
        b[i] = make_int2(i, -i);
    }
    //b[7] = make_int2(131, 123123);
    ASSERT_HOST_VECTOR_EQ(a, b);
}
TEST(AssertHostVectorEqual, Float3)
{
    typedef float3 T;
    const int numElements = 10;
    thrust::host_vector<T> a(numElements);
    thrust::host_vector<T> b(numElements);
    for (int i = 0; i < numElements; ++i)
    {
        a[i] = make_float3(i * 3.14f, (-i) * 3.14f, 0.01234f);
        b[i] = make_float3(i * 3.14f, (-i) * 3.14f, 0.01234f);
    }
    //b[2] = make_float3(131.31f, 123123.03f, 0.01235f);
    ASSERT_HOST_VECTOR_EQ(a, b);
}
TEST(AssertHostVectorEqual, Double4)
{
    typedef double4 T;
    const int numElements = 10;
    thrust::host_vector<T> a(numElements);
    thrust::host_vector<T> b(numElements);
    for (int i = 0; i < numElements; ++i)
    {
        a[i] = make_double4(i * 3.1412312, (-i) * 0.0314, 0.01234f, 0.000012);
        b[i] = make_double4(i * 3.1412312, (-i) * 0.0314, 0.01234f,  0.000012);
    }
    //b[2] = make_double4(131, 123123, 0.01235f, -123.12345);
    ASSERT_HOST_VECTOR_EQ(a, b);
}


//-------------------------------------------------------------//
//                                                             //
//                 ASSERT_HOST_VECTOR_NEAR                     //
//                                                             //
//-------------------------------------------------------------//
TEST(AssertHostVectorNear, Float)
{
    typedef float T;
    const int numElements = 10;
    thrust::host_vector<T> a(numElements);
    thrust::host_vector<T> b(numElements);
    for (int i = 0; i < numElements; ++i)
    {
        a[i] = i + 0.0007f;
        b[i] = i + 0.0002f;
    }
    const double absError = 0.001;
    ASSERT_HOST_VECTOR_NEAR(a, b, absError);
}
TEST(AssertHostVectorNear, Double)
{
    typedef double T;
    const int numElements = 10;
    std::vector<T> a(numElements);
    std::vector<T> b(numElements);
    for (int i = 0; i < numElements; ++i)
    {
        a[i] = 0.0000007;
        b[i] = 0.0000006;;
    }
    const double absError = 0.000001;
    ASSERT_HOST_VECTOR_NEAR(a, b, absError);
}
TEST(AssertHostVectorNear, Float2)
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
    ASSERT_HOST_VECTOR_NEAR(a, b, absError);
}
TEST(AssertHostVectorNear, Float3)
{
    typedef float3 T;
    const int numElements = 10;
    thrust::host_vector<T> a(numElements);
    thrust::host_vector<T> b(numElements);
    for (int i = 0; i < numElements; ++i)
    {
        a[i] = make_float3(0.01234f, -0.01234f, 3.14f);
        b[i] = make_float3(0.01231f, -0.01231f, 3.14f);
    }
    //b[2] = make_float3(131.31f, 123123.03f, 0.01235f);
    const double absError = 0.001;
    ASSERT_HOST_VECTOR_NEAR(a, b, absError);
}
TEST(AssertHostVectorNear, Double4)
{
    typedef double4 T;
    const int numElements = 10;
    thrust::host_vector<T> a(numElements);
    thrust::host_vector<T> b(numElements);
    for (int i = 0; i < numElements; ++i)
    {
        a[i] = make_double4(3.1412312, -0.0991234, 3.00014, 0.0);
        b[i] = make_double4(3.1412312, -0.0991231, 3.00014, 0.0);
    }
    //b[2] = make_double4(131, 123123, 0.01235f, -123.12345);
    const double absError = 0.00001;
    ASSERT_HOST_VECTOR_NEAR(a, b, absError);
}


//-------------------------------------------------------------//
//                                                             //
//                 EXPECT_HOST_VECTOR_EQ                       //
//                                                             //
//-------------------------------------------------------------//
TEST(ExpectHostVectorEqual, StdVectorInt)
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
    EXPECT_HOST_VECTOR_EQ(a, b);
}
TEST(ExpectHostVectorEqual, ThrustVectorInt2)
{
    typedef int2 T;
    const int numElements = 10;
    thrust::host_vector<T> a(numElements);
    thrust::host_vector<T> b(numElements);
    for (int i = 0; i < numElements; ++i)
    {
        a[i] = make_int2(i, -i);
        b[i] = make_int2(i, -i);
    }
    //b[7] = make_int2(131, 123123);
    EXPECT_HOST_VECTOR_EQ(a, b);
}
TEST(ExpectHostVectorEqual, Float3)
{
    typedef float3 T;
    const int numElements = 10;
    thrust::host_vector<T> a(numElements);
    thrust::host_vector<T> b(numElements);
    for (int i = 0; i < numElements; ++i)
    {
        a[i] = make_float3(i * 3.14f, (-i) * 3.14f, 0.01234f);
        b[i] = make_float3(i * 3.14f, (-i) * 3.14f, 0.01234f);
    }
    //b[2] = make_float3(131.31f, 123123.03f, 0.01235f);
    EXPECT_HOST_VECTOR_EQ(a, b);
}
TEST(ExpectHostVectorEqual, Double4)
{
    typedef double4 T;
    const int numElements = 10;
    thrust::host_vector<T> a(numElements);
    thrust::host_vector<T> b(numElements);
    for (int i = 0; i < numElements; ++i)
    {
        a[i] = make_double4(i * 3.1412312, (-i) * 0.0314, 0.01234f, 0.000012);
        b[i] = make_double4(i * 3.1412312, (-i) * 0.0314, 0.01234f,  0.000012);
    }
    //b[2] = make_double4(131, 123123, 0.01235f, -123.12345);
    EXPECT_HOST_VECTOR_EQ(a, b);
}


//-------------------------------------------------------------//
//                                                             //
//                 EXPECT_HOST_VECTOR_NEAR                     //
//                                                             //
//-------------------------------------------------------------//
TEST(ExpectHostVectorNear, Float)
{
    typedef float T;
    const int numElements = 10;
    thrust::host_vector<T> a(numElements);
    thrust::host_vector<T> b(numElements);
    for (int i = 0; i < numElements; ++i)
    {
        a[i] = i + 0.0007f;
        b[i] = i + 0.0002f;
    }
    const double absError = 0.001;
    EXPECT_HOST_VECTOR_NEAR(a, b, absError);
}
TEST(ExpectHostVectorNear, Double)
{
    typedef double T;
    const int numElements = 10;
    std::vector<T> a(numElements);
    std::vector<T> b(numElements);
    for (int i = 0; i < numElements; ++i)
    {
        a[i] = 0.0000007;
        b[i] = 0.0000006;;
    }
    const double absError = 0.000001;
    EXPECT_HOST_VECTOR_NEAR(a, b, absError);
}
TEST(ExpectHostVectorNear, Float2)
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
    EXPECT_HOST_VECTOR_NEAR(a, b, absError);
}
TEST(ExpectHostVectorNear, Float3)
{
    typedef float3 T;
    const int numElements = 10;
    thrust::host_vector<T> a(numElements);
    thrust::host_vector<T> b(numElements);
    for (int i = 0; i < numElements; ++i)
    {
        a[i] = make_float3(0.01234f, -0.01234f, 3.14f);
        b[i] = make_float3(0.01231f, -0.01231f, 3.14f);
    }
    //b[2] = make_float3(131.31f, 123123.03f, 0.01235f);
    const double absError = 0.001;
    EXPECT_HOST_VECTOR_NEAR(a, b, absError);
}
TEST(ExpectHostVectorNear, Double4)
{
    typedef double4 T;
    const int numElements = 10;
    thrust::host_vector<T> a(numElements);
    thrust::host_vector<T> b(numElements);
    for (int i = 0; i < numElements; ++i)
    {
        a[i] = make_double4(3.1412312, -0.0991234, 3.00014, 0.0);
        b[i] = make_double4(3.1412312, -0.0991231, 3.00014, 0.0);
    }
    //b[2] = make_double4(131, 123123, 0.01235f, -123.12345);
    const double absError = 0.00001;
    EXPECT_HOST_VECTOR_NEAR(a, b, absError);
}


int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
