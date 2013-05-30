#include <gcuda/gcuda.h>
#include <thrust/host_vector.h>

template <typename T>
__global__ void incrementKernel(T* data, int size)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < size)
    {
        T d = data[tid];
        data[tid] = d + T(1);
    }
}

template <typename T>
__global__ void incrementWithErrorKernel(T* data, int size, double error)
{
    int tid = blockDim.x * blockIdx.x + threadIdx.x;
    if (tid < size)
    {
        T d = data[tid];
        data[tid] = d + error;
    }
}

//-------------------------------------------------------------//
//                                                             //
//                 ASSERT_DEVICE_ARRAY_EQ                      //
//                                                             //
//-------------------------------------------------------------//
TEST(AssertDeviceArrayEqual, Int)
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
    cudaMalloc((void**)&b, sizeof(T) * numElements);
    cudaMemcpy(b, a.data(), sizeof(T) * numElements, cudaMemcpyHostToDevice);
    int threads = 128;
    int blocks = (numElements + threads - 1)/threads;
    incrementKernel<T><<<blocks, threads>>>(b, numElements);
    //ref[4] = 12;
    ASSERT_DEVICE_ARRAY_EQ(ref, b, numElements);
    cudaFree(b);
}

TEST(AssertDeviceArrayEqual, Float)
{
    typedef float T;
    const int numElements = 16 * 128;
    std::vector<T> a(numElements);
    std::vector<T> ref(numElements);
    for (int i = 0; i < numElements; ++i)
    {
        a[i] = i;
        ref[i] = i + 1;
    }
    T* b;
    cudaMalloc((void**)&b, sizeof(T) * numElements);
    cudaMemcpy(b, a.data(), sizeof(T) * numElements, cudaMemcpyHostToDevice);
    int threads = 128;
    int blocks = (numElements + threads - 1)/threads;
    incrementKernel<T><<<blocks, threads>>>(b, numElements);
    ASSERT_DEVICE_ARRAY_EQ(ref.data(), b, numElements);
    cudaFree(b);
}


//-------------------------------------------------------------//
//                                                             //
//                 ASSERT_DEVICE_ARRAY_NEAR                    //
//                                                             //
//-------------------------------------------------------------//
TEST(AssertDeviceArrayNear, Float)
{
    typedef float T;
    const int numElements = 16 * 128;
    const double error = 0.00001;
    thrust::host_vector<T> a(numElements);
    thrust::host_vector<T> ref(numElements);
    for (int i = 0; i < numElements; ++i)
    {
        a[i] = i;
        ref[i] = i + error;
    }
    T* b;
    cudaMalloc((void**)&b, sizeof(T) * numElements);
    cudaMemcpy(b, a.data(), sizeof(T) * numElements, cudaMemcpyHostToDevice);
    int threads = 128;
    int blocks = (numElements + threads - 1)/threads;
    incrementWithErrorKernel<T><<<blocks, threads>>>(b, numElements, error);
    const double absError = error * 1.1;
    ASSERT_DEVICE_ARRAY_NEAR(ref, b, numElements, absError);
    cudaFree(b);
}

TEST(AssertDeviceArrayNear, Double)
{
    typedef double T;
    const int numElements = 16 * 128;
    const double error = 0.0000001;
    std::vector<T> a(numElements);
    std::vector<T> ref(numElements);
    for (int i = 0; i < numElements; ++i)
    {
        a[i] = i;
        ref[i] = i + error;
    }
    T* b;
    cudaMalloc((void**)&b, sizeof(T) * numElements);
    cudaMemcpy(b, a.data(), sizeof(T) * numElements, cudaMemcpyHostToDevice);
    int threads = 128;
    int blocks = (numElements + threads - 1)/threads;
    incrementWithErrorKernel<T><<<blocks, threads>>>(b, numElements, error);
    const double absError = error * 1.01;
    ASSERT_DEVICE_ARRAY_NEAR(ref.data(), b, numElements, absError);
    cudaFree(b);
}


//-------------------------------------------------------------//
//                                                             //
//                 EXPECT_DEVICE_ARRAY_EQ                      //
//                                                             //
//-------------------------------------------------------------//
TEST(ExpectDeviceArrayEqual, Int)
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
    cudaMalloc((void**)&b, sizeof(T) * numElements);
    cudaMemcpy(b, a.data(), sizeof(T) * numElements, cudaMemcpyHostToDevice);
    int threads = 128;
    int blocks = (numElements + threads - 1)/threads;
    incrementKernel<T><<<blocks, threads>>>(b, numElements);
    //ref[4] = 12;
    EXPECT_DEVICE_ARRAY_EQ(ref, b, numElements);
    cudaFree(b);
}

TEST(ExpectDeviceArrayEqual, Float)
{
    typedef float T;
    const int numElements = 16 * 128;
    std::vector<T> a(numElements);
    std::vector<T> ref(numElements);
    for (int i = 0; i < numElements; ++i)
    {
        a[i] = i;
        ref[i] = i + 1;
    }
    T* b;
    cudaMalloc((void**)&b, sizeof(T) * numElements);
    cudaMemcpy(b, a.data(), sizeof(T) * numElements, cudaMemcpyHostToDevice);
    int threads = 128;
    int blocks = (numElements + threads - 1)/threads;
    incrementKernel<T><<<blocks, threads>>>(b, numElements);
    EXPECT_DEVICE_ARRAY_EQ(ref.data(), b, numElements);
    cudaFree(b);
}


//-------------------------------------------------------------//
//                                                             //
//                 EXPECT_DEVICE_ARRAY_NEAR                    //
//                                                             //
//-------------------------------------------------------------//
TEST(ExpectDeviceArrayNear, Float)
{
    typedef float T;
    const int numElements = 16 * 128;
    const double error = 0.00001;
    thrust::host_vector<T> a(numElements);
    thrust::host_vector<T> ref(numElements);
    for (int i = 0; i < numElements; ++i)
    {
        a[i] = i;
        ref[i] = i + error;
    }
    T* b;
    cudaMalloc((void**)&b, sizeof(T) * numElements);
    cudaMemcpy(b, a.data(), sizeof(T) * numElements, cudaMemcpyHostToDevice);
    int threads = 128;
    int blocks = (numElements + threads - 1)/threads;
    incrementWithErrorKernel<T><<<blocks, threads>>>(b, numElements, error);
    const double absError = error * 1.1;
    EXPECT_DEVICE_ARRAY_NEAR(ref, b, numElements, absError);
    cudaFree(b);
}

TEST(ExpectDeviceArrayNear, Double)
{
    typedef double T;
    const int numElements = 16 * 128;
    const double error = 0.0000001;
    std::vector<T> a(numElements);
    std::vector<T> ref(numElements);
    for (int i = 0; i < numElements; ++i)
    {
        a[i] = i;
        ref[i] = i + error;
    }
    T* b;
    cudaMalloc((void**)&b, sizeof(T) * numElements);
    cudaMemcpy(b, a.data(), sizeof(T) * numElements, cudaMemcpyHostToDevice);
    int threads = 128;
    int blocks = (numElements + threads - 1)/threads;
    incrementWithErrorKernel<T><<<blocks, threads>>>(b, numElements, error);
    const double absError = error * 1.01;
    EXPECT_DEVICE_ARRAY_NEAR(ref.data(), b, numElements, absError);
    cudaFree(b);
}



int main(int argc, char **argv)
{
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
