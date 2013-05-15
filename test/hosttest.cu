#include <gcuda/gcuda.h>

#include <vector>
#include <thrust/host_vector.h>
#include <thrust/transform.h>

TEST(STLVector, AssertTestChar)
{
    {
        typedef char T;
        const int numElements = 10;
        std::vector<T> a(numElements);
        std::vector<T> b(numElements);
        for (int i = 0; i < numElements; ++i)
        {
            a[i] = i;
            b[i] = i;
        }
        //b[5] = 1;
        ASSERT_HOST_VECTOR_EQ(a, b);
    }
    {
        typedef unsigned char T;
        const int numElements = 10;
        std::vector<T> a(numElements);
        std::vector<T> b(numElements);
        for (int i = 0; i < numElements; ++i)
        {
            a[i] = i;
            b[i] = i;
        }
        //b[5] = 1;
        ASSERT_HOST_VECTOR_EQ(a, b);
    }
}

TEST(STLVector, AssertTestShort)
{
    {
        typedef short T;
        const int numElements = 10;
        std::vector<T> a(numElements);
        std::vector<T> b(numElements);
        for (int i = 0; i < numElements; ++i)
        {
            a[i] = i;
            b[i] = i;
        }
        //b[5] = 1;
        ASSERT_HOST_VECTOR_EQ(a, b);
    }
    {
        typedef unsigned short T;
        const int numElements = 10;
        std::vector<T> a(numElements);
        std::vector<T> b(numElements);
        for (int i = 0; i < numElements; ++i)
        {
            a[i] = i;
            b[i] = i;
        }
        //b[5] = 1;
        ASSERT_HOST_VECTOR_EQ(a, b);
    }
}

TEST(STLVector, AssertTestInt)
{
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
        //b[5] = 1;
        ASSERT_HOST_VECTOR_EQ(a, b);
    }
    {
        typedef unsigned int T;
        const int numElements = 10;
        std::vector<T> a(numElements);
        std::vector<T> b(numElements);
        for (int i = 0; i < numElements; ++i)
        {
            a[i] = i;
            b[i] = i;
        }
        //b[5] = 1;
        ASSERT_HOST_VECTOR_EQ(a, b);
    }
}


TEST(STLVector, AssertTestLong)
{
    {
        typedef long T;
        const int numElements = 10;
        std::vector<T> a(numElements);
        std::vector<T> b(numElements);
        for (int i = 0; i < numElements; ++i)
        {
            a[i] = i;
            b[i] = i;
        }
        //b[5] = 1;
        ASSERT_HOST_VECTOR_EQ(a, b);
    }
    {
        typedef unsigned long T;
        const int numElements = 10;
        std::vector<T> a(numElements);
        std::vector<T> b(numElements);
        for (int i = 0; i < numElements; ++i)
        {
            a[i] = i;
            b[i] = i;
        }
        //b[5] = 1;
        ASSERT_HOST_VECTOR_EQ(a, b);
    }
}

TEST(STLVector, AssertTestFloat)
{
    {
        typedef float T;
        const int numElements = 10;
        std::vector<T> a(numElements);
        std::vector<T> b(numElements);
        for (int i = 0; i < numElements; ++i)
        {
            a[i] = i;
            b[i] = i;
        }
        //b[5] = 1.32f;
        ASSERT_HOST_VECTOR_EQ(a, b);
    }
}

TEST(STLVector, AssertTestDouble)
{
    {
        typedef double T;
        const int numElements = 10;
        std::vector<T> a(numElements);
        std::vector<T> b(numElements);
        for (int i = 0; i < numElements; ++i)
        {
            a[i] = i;
            b[i] = i;
        }
        //b[5] = 1.322345;
        ASSERT_HOST_VECTOR_EQ(a, b);
    }
}



TEST(ThrustVector, AssertTestChar)
{
    {
        typedef char T;
        const int numElements = 10;
        thrust::host_vector<T> a(numElements);
        thrust::host_vector<T> b(numElements);
        for (int i = 0; i < numElements; ++i)
        {
            a[i] = i;
            b[i] = i;
        }
        //b[5] = 1;
        ASSERT_HOST_VECTOR_EQ(a, b);
    }
    {
        typedef unsigned char T;
        const int numElements = 10;
        thrust::host_vector<T> a(numElements);
        thrust::host_vector<T> b(numElements);
        for (int i = 0; i < numElements; ++i)
        {
            a[i] = i;
            b[i] = i;
        }
        //b[5] = 1;
        ASSERT_HOST_VECTOR_EQ(a, b);
    }
}

TEST(ThrustVector, AssertTestShort)
{
    {
        typedef short T;
        const int numElements = 10;
        thrust::host_vector<T> a(numElements);
        thrust::host_vector<T> b(numElements);
        for (int i = 0; i < numElements; ++i)
        {
            a[i] = i;
            b[i] = i;
        }
        //b[5] = 1;
        ASSERT_HOST_VECTOR_EQ(a, b);
    }
    {
        typedef unsigned short T;
        const int numElements = 10;
        thrust::host_vector<T> a(numElements);
        thrust::host_vector<T> b(numElements);
        for (int i = 0; i < numElements; ++i)
        {
            a[i] = i;
            b[i] = i;
        }
        //b[5] = 1;
        ASSERT_HOST_VECTOR_EQ(a, b);
    }
}

TEST(ThrustVector, AssertTestInt)
{
    {
        typedef int T;
        const int numElements = 10;
        thrust::host_vector<T> a(numElements);
        thrust::host_vector<T> b(numElements);
        for (int i = 0; i < numElements; ++i)
        {
            a[i] = i;
            b[i] = i;
        }
        //b[5] = 1;
        ASSERT_HOST_VECTOR_EQ(a, b);
    }
    {
        typedef unsigned int T;
        const int numElements = 10;
        thrust::host_vector<T> a(numElements);
        thrust::host_vector<T> b(numElements);
        for (int i = 0; i < numElements; ++i)
        {
            a[i] = i;
            b[i] = i;
        }
        //b[5] = 1;
        ASSERT_HOST_VECTOR_EQ(a, b);
    }
}


TEST(ThrustVector, AssertTestInt2)
{
    {
        typedef int2 T;
        const int numElements = 10;
        thrust::host_vector<T> a(numElements);
        thrust::host_vector<T> b(numElements);
        for (int i = 0; i < numElements; ++i)
        {
            a[i] = make_int2(i, i);
            b[i] = make_int2(i, i);
        }
        //b[5] = make_int2(3, 8);
        ASSERT_HOST_VECTOR_EQ(a, b);
    }
    {
        typedef uint2 T;
        const int numElements = 10;
        thrust::host_vector<T> a(numElements);
        thrust::host_vector<T> b(numElements);
        for (int i = 0; i < numElements; ++i)
        {
            a[i] = make_uint2(i, i);
            b[i] = make_uint2(i, i);
        }
        //b[5] = make_uint2(5, 4);
        ASSERT_HOST_VECTOR_EQ(a, b);
    }
}


TEST(ThrustVector, AssertTestLong)
{
    {
        typedef long T;
        const int numElements = 10;
        thrust::host_vector<T> a(numElements);
        thrust::host_vector<T> b(numElements);
        for (int i = 0; i < numElements; ++i)
        {
            a[i] = i;
            b[i] = i;
        }
        //b[5] = 1;
        ASSERT_HOST_VECTOR_EQ(a, b);
    }
    {
        typedef unsigned long T;
        const int numElements = 10;
        thrust::host_vector<T> a(numElements);
        thrust::host_vector<T> b(numElements);
        for (int i = 0; i < numElements; ++i)
        {
            a[i] = i;
            b[i] = i;
        }
        //b[5] = 1;
        ASSERT_HOST_VECTOR_EQ(a, b);
    }
}

TEST(ThrustVector, AssertTestFloat)
{
    {
        typedef float T;
        const int numElements = 10;
        thrust::host_vector<T> a(numElements);
        thrust::host_vector<T> b(numElements);
        for (int i = 0; i < numElements; ++i)
        {
            a[i] = i;
            b[i] = i;
        }
        //b[5] = 1.32f;
        ASSERT_HOST_VECTOR_EQ(a, b);
    }
}

TEST(ThrustVector, AssertTestFloat2)
{
    {
        typedef float2 T;
        const int numElements = 10;
        thrust::host_vector<T> a(numElements);
        thrust::host_vector<T> b(numElements);
        for (int i = 0; i < numElements; ++i)
        {
            a[i] = make_float2(i, i);
            b[i] = make_float2(i, i);;
        }
        //b[5] = make_float2(5.0f, 1.43f);
        ASSERT_HOST_VECTOR_EQ(a, b);
    }
}

TEST(ThrustVector, AssertTestFloat3)
{
    {
        typedef float3 T;
        const int numElements = 10;
        thrust::host_vector<T> a(numElements);
        thrust::host_vector<T> b(numElements);
        for (int i = 0; i < numElements; ++i)
        {
            a[i] = make_float3(i, i, i);
            b[i] = make_float3(i, i, i);;
        }
        //b[5] = make_float3(5.0f, 1.43f, 5.0f);
        ASSERT_HOST_VECTOR_EQ(a, b);
    }
}

TEST(ThrustVector, AssertTestFloat4)
{
    {
        typedef float4 T;
        const int numElements = 10;
        thrust::host_vector<T> a(numElements);
        thrust::host_vector<T> b(numElements);
        for (int i = 0; i < numElements; ++i)
        {
            a[i] = make_float4(i, i, i, i);
            b[i] = make_float4(i, i, i, i);;
        }
        //b[5] = make_float4(5.0f, 5.0f, 1.43f, 5.0f);
        ASSERT_HOST_VECTOR_EQ(a, b);
    }
}

TEST(ThrustVector, AssertTestDouble)
{
    {
        typedef double T;
        const int numElements = 10;
        thrust::host_vector<T> a(numElements);
        thrust::host_vector<T> b(numElements);
        for (int i = 0; i < numElements; ++i)
        {
            a[i] = i;
            b[i] = i;
        }
        //b[5] = 1.322345;
        ASSERT_HOST_VECTOR_EQ(a, b);
    }
}

TEST(ThrustVector, AssertTestDouble2)
{
    {
        typedef double2 T;
        const int numElements = 10;
        thrust::host_vector<T> a(numElements);
        thrust::host_vector<T> b(numElements);
        for (int i = 0; i < numElements; ++i)
        {
            a[i] = make_double2(i, i);
            b[i] = make_double2(i, i);;
        }
        //b[5] = make_double2(5.0f, 1.43f);
        ASSERT_HOST_VECTOR_EQ(a, b);
    }
}

TEST(ThrustVector, AssertTestDouble3)
{
    {
        typedef double3 T;
        const int numElements = 10;
        thrust::host_vector<T> a(numElements);
        thrust::host_vector<T> b(numElements);
        for (int i = 0; i < numElements; ++i)
        {
            a[i] = make_double3(i, i, i);
            b[i] = make_double3(i, i, i);;
        }
        //b[5] = make_double3(5.0f, 1.43f, 5.0f);
        ASSERT_HOST_VECTOR_EQ(a, b);
    }
}

TEST(ThrustVector, AssertTestDouble4)
{
    {
        typedef double4 T;
        const int numElements = 10;
        thrust::host_vector<T> a(numElements);
        thrust::host_vector<T> b(numElements);
        for (int i = 0; i < numElements; ++i)
        {
            a[i] = make_double4(i, i, i, i);
            b[i] = make_double4(i, i, i, i);;
        }
        //b[5] = make_double4(5.0f, 5.0f, 1.43f, 5.0f);
        ASSERT_HOST_VECTOR_EQ(a, b);
    }
}




TEST(ThrustVector, AssertNearTestDouble)
{
    {
        typedef double T;
        const int numElements = 10;
        thrust::host_vector<T> a(numElements);
        thrust::host_vector<T> b(numElements);
        for (int i = 0; i < numElements; ++i)
        {
            a[i] = i;
            b[i] = i;
        }
        b[5] = 5.00005;
        ASSERT_HOST_VECTOR_NEAR(a, b, 0.0001);
    }
}

TEST(ThrustVector, AssertNearTestFloat)
{
    {
        typedef float T;
        const int numElements = 10;
        thrust::host_vector<T> a(numElements);
        thrust::host_vector<T> b(numElements);
        for (int i = 0; i < numElements; ++i)
        {
            a[i] = i;
            b[i] = i;
        }
        b[5] = 5.005;
        ASSERT_HOST_VECTOR_NEAR(a, b, 0.006);
    }
}

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
