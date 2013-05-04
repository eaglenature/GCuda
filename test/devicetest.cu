#include <gcuda/gcuda.h>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/sort.h>


template <typename T>
__global__ void incrementKernel(T* data, int size)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < size)
    {
        int d = data[tid];
        data[tid] = d + 1;
    }
}

TEST(DeviceVector, ThrustVectors)
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
TEST(DeviceVector, ThrustVectorsNear)
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
    ASSERT_DEVICE_VECTOR_NEAR(a, b, 0.002);
}
TEST(DeviceVector, HostVectorVsDeviceRawPointer)
{
    typedef int T;
    const int numElements = 16 * 128;
    thrust::host_vector<T> a(numElements);
    thrust::host_vector<T> ref(numElements);
    for (int i = 0; i < numElements; ++i)
    {
        a[i] = i;
        ref[i] = i + 1;
    }
    T* b;
    cudaMalloc((void**)&b, sizeof(int) * numElements);
    cudaMemcpy(b, a.data(), sizeof(int) * numElements, cudaMemcpyHostToDevice);
    int threads = 128;
    int blocks = (numElements + threads - 1)/threads;
    incrementKernel<T><<<blocks, threads>>>(b, numElements);
    ASSERT_DEVICE_ARRAY_EQ(ref, b, numElements);
    cudaFree(b);
}
TEST(DeviceVector, HostRawVsDeviceRawPointer)
{
    typedef int T;
    const int numElements = 16 * 128;
    std::vector<T> a(numElements);
    std::vector<T> ref(numElements);
    for (int i = 0; i < numElements; ++i)
    {
        a[i] = i;
        ref[i] = i + 1;
    }
    T* b;
    cudaMalloc((void**)&b, sizeof(int) * numElements);
    cudaMemcpy(b, a.data(), sizeof(int) * numElements, cudaMemcpyHostToDevice);
    int threads = 128;
    int blocks = (numElements + threads - 1)/threads;
    incrementKernel<T><<<blocks, threads>>>(b, numElements);
    ASSERT_DEVICE_ARRAY_EQ(ref.data(), b, numElements);
    cudaFree(b);
}


int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
