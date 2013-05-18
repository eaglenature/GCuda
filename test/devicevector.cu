#include <gcuda/gcuda.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>

//-------------------------------------------------------------//
//                                                             //
//                 ASSERT_DEVICE_VECTOR_EQ                     //
//                                                             //
//-------------------------------------------------------------//
TEST(DeviceVectorEqual, Int)
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

TEST(DeviceVectorEqual, Float)
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
TEST(DeviceVectorNear, Float)
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

TEST(DeviceVectorNear, Double)
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

int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
