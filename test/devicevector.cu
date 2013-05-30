#include <gcuda/gcuda.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>

//-------------------------------------------------------------//
//                                                             //
//                 ASSERT_DEVICE_VECTOR_EQ                     //
//                                                             //
//-------------------------------------------------------------//
TEST(AssertDeviceVectorEqual, Int)
{
    typedef int T;
    const int numElements = 10;
    thrust::host_vector<T> a(numElements);
    for (int i = 0; i < numElements; ++i)
    {
        a[i] = i;
    }
    thrust::device_vector<T> b = a;
    thrust::sort(a.begin(), a.end());
    thrust::sort(b.begin(), b.end());
    ASSERT_DEVICE_VECTOR_EQ(a, b);
}

TEST(AssertDeviceVectorEqual, Float)
{
    typedef float T;
    const int numElements = 10;
    thrust::host_vector<T> a(numElements);
    for (int i = 0; i < numElements; ++i)
    {
        a[i] = i;
    }
    thrust::device_vector<T> b = a;
    thrust::sort(a.begin(), a.end());
    thrust::sort(b.begin(), b.end());
    ASSERT_DEVICE_VECTOR_EQ(a, b);
}

//-------------------------------------------------------------//
//                                                             //
//                 ASSERT_DEVICE_VECTOR_NEAR                   //
//                                                             //
//-------------------------------------------------------------//
TEST(AssertDeviceVectorNear, Float)
{
    typedef float T;
    const int numElements = 10;
    thrust::host_vector<T> a(numElements);
    for (int i = 0; i < numElements; ++i)
    {
        a[i] = i;
    }
    thrust::device_vector<T> b = a;
    thrust::sort(a.begin(), a.end());
    thrust::sort(b.begin(), b.end());
    a[5] += 0.001f;
    const double absError = 0.002;
    ASSERT_DEVICE_VECTOR_NEAR(a, b, absError);
}

TEST(AssertDeviceVectorNear, Double)
{
    typedef double T;
    const int numElements = 10;
    thrust::host_vector<T> a(numElements);
    for (int i = 0; i < numElements; ++i)
    {
        a[i] = i;
    }
    thrust::device_vector<T> b = a;
    thrust::sort(a.begin(), a.end());
    thrust::sort(b.begin(), b.end());
    a[5] += 0.00001f;
    const double absError = 0.00002;
    ASSERT_DEVICE_VECTOR_NEAR(a, b, absError);
}


//-------------------------------------------------------------//
//                                                             //
//                 EXPECT_DEVICE_VECTOR_EQ                     //
//                                                             //
//-------------------------------------------------------------//
TEST(ExpectDeviceVectorEqual, Int)
{
    typedef int T;
    const int numElements = 10;
    thrust::host_vector<T> a(numElements);
    for (int i = 0; i < numElements; ++i)
    {
        a[i] = i;
    }
    thrust::device_vector<T> b = a;
    thrust::sort(a.begin(), a.end());
    thrust::sort(b.begin(), b.end());
    EXPECT_DEVICE_VECTOR_EQ(a, b);
}

TEST(ExpectDeviceVectorEqual, Float)
{
    typedef float T;
    const int numElements = 10;
    thrust::host_vector<T> a(numElements);
    for (int i = 0; i < numElements; ++i)
    {
        a[i] = i;
    }
    thrust::device_vector<T> b = a;
    thrust::sort(a.begin(), a.end());
    thrust::sort(b.begin(), b.end());
    EXPECT_DEVICE_VECTOR_EQ(a, b);
}



//-------------------------------------------------------------//
//                                                             //
//                 EXPECT_DEVICE_VECTOR_NEAR                   //
//                                                             //
//-------------------------------------------------------------//
TEST(ExpectDeviceVectorNear, Float)
{
    typedef float T;
    const int numElements = 10;
    thrust::host_vector<T> a(numElements);
    for (int i = 0; i < numElements; ++i)
    {
        a[i] = i;
    }
    thrust::device_vector<T> b = a;
    thrust::sort(a.begin(), a.end());
    thrust::sort(b.begin(), b.end());
    a[5] += 0.001f;
    const double absError = 0.002;
    EXPECT_DEVICE_VECTOR_NEAR(a, b, absError);
}

TEST(ExpectDeviceVectorNear, Double)
{
    typedef double T;
    const int numElements = 10;
    thrust::host_vector<T> a(numElements);
    for (int i = 0; i < numElements; ++i)
    {
        a[i] = i;
    }
    thrust::device_vector<T> b = a;
    thrust::sort(a.begin(), a.end());
    thrust::sort(b.begin(), b.end());
    a[5] += 0.00001f;
    const double absError = 0.00002;
    EXPECT_DEVICE_VECTOR_NEAR(a, b, absError);
}


int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
