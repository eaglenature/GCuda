/*
 * main.cpp
 *
 *  Created on: May 3, 2013
 *      Author: eaglenature@gmail.com
 *
 *  Library features unit testing code.
 */

#include <gcuda/gcuda.h>

#include <thrust/host_vector.h>

TEST(STLVector, AssertTest0)
{
    const int numElements = 10;
    std::vector<double> a(numElements);
    std::vector<double> b(numElements);
    for (int i = 0; i < numElements; ++i)
    {
        a[i] = i;
        b[i] = i;
    }
    ASSERT_HOST_VECTOR_EQ(a, b);
}

TEST(STLVector, ExpectTest0)
{
    const int numElements = 10;
    std::vector<int> a(numElements);
    std::vector<int> b(numElements);
    for (int i = 0; i < numElements; ++i)
    {
        a[i] = i;
        b[i] = i;
    }
    EXPECT_HOST_VECTOR_EQ(a, b);
}


TEST(HostVector, AssertThrustHostVector0)
{
    const int numElements = 10;
    thrust::host_vector<double> a(numElements);
    thrust::host_vector<double> b(numElements);
    for (int i = 0; i < numElements; ++i)
    {
        a[i] = i;
        b[i] = i;
    }
    ASSERT_HOST_VECTOR_EQ(a, b);
}

TEST(HostVector, ExpectThrustHostVector0)
{
    const int numElements = 10;
    thrust::host_vector<int> a(numElements);
    thrust::host_vector<int> b(numElements);
    for (int i = 0; i < numElements; ++i)
    {
        a[i] = i;
        b[i] = i;
    }
    EXPECT_HOST_VECTOR_EQ(a, b);
}


TEST(RawArray, AssertTest0)
{
    const int numElements = 10;
    double* a = new double[numElements];
    double* b = new double[numElements];
    for (int i = 0; i < numElements; ++i)
    {
        a[i] = i;
        b[i] = i;
    }
    ASSERT_HOST_ARRAY_EQ(a, b, numElements);
    delete [] a;
    delete [] b;
}


TEST(RawArray, ExpectTest1)
{
    const int numElements = 10;
    int* a = new int[numElements];
    int* b = new int[numElements];
    for (int i = 0; i < numElements; ++i)
    {
        a[i] = i;
        b[i] = i;
    }
    EXPECT_HOST_ARRAY_EQ(a, b, numElements);
    delete [] a;
    delete [] b;
}


int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
